"""Equivalence gate between the PaddleScience ClimaX port and PaddleCFD.

Runs three comparisons, each side in an isolated subprocess with its own
PYTHONPATH (PaddleScience root / PaddleCFD root, plus the respective
examples/climax directory):

  gate 1  data pipeline: same-index samples of both ClimateBench datasets
          (train/val/test, 4 indices each incl. first and last), bit-exact
  gate 2  forward: same-source climax_initial.pdparams strict-loaded on both
          models ("net." prefix stripped, no load_mae_weights cleaning), eval
          mode, synthetic RandomState(42) input and the real first test
          batch, prediction max_abs_diff target < 1e-6
  gate 3  single-batch loss: real forward output through the ppsci loss
          expressions (examples/climax/utils.py) vs the PaddleCFD module
          training_step / validation_step paths, values equal

Usage:
    python tools/compare_ppsci_ppcfd.py [--workdir /tmp/climax08_gate]

Exits 0 when all gates pass, 1 otherwise. Local verification asset, not part
of either upstream repo.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

PYTHON = sys.executable
PPSCI_ROOT = "/home/lkyu/baidu/PaddleScience"
PPCFD_ROOT = "/home/lkyu/baidu/PaddleCFD"
DATA_ROOT = "/home/lkyu/baidu/CLIMAX/dataset/climatebench/5.625deg"
CKPT_PATH = "/home/lkyu/baidu/CLIMAX/models/climax_paddle/climax_initial.pdparams"

# ClimaXClimateBench construction shared by both sides (climax.yaml MODEL
# section / climate_projection.yaml net.init_args).
MODEL_KW = """    default_vars=["CO2", "SO2", "CH4", "BC"],
    out_vars="tas",
    img_size=[32, 64],
    time_history=10,
    patch_size=2,
    embed_dim=1024,
    depth=8,
    num_heads=16,
    mlp_ratio=4,
    drop_path=0.1,
    drop_rate=0.1,
    parallel_patch_embed=False,
    freeze_encoder=True,"""

STRICT_LOAD = """stripped = {k[len("net."):]: v for k, v in ckpt.items() if k.startswith("net.")}
state = net.state_dict()
net.set_state_dict(stripped)
net.eval()
result["forward"]["strict_load"] = {
    "ckpt_keys": len(stripped),
    "model_keys": len(state),
    "missing": sorted(set(state) - set(stripped)),
    "unexpected": sorted(set(stripped) - set(state)),
    "shape_mismatch": sorted(
        k
        for k in state
        if k in stripped and tuple(stripped[k].shape) != tuple(state[k].shape)
    ),
}"""

SIDE_PPSCI = '''"""ppsci side of the equivalence gate: dataset samples, strict-load eval
forward, loss expressions from examples/climax/utils.py."""
import json
import sys

import numpy as np
import paddle

paddle.set_device("cpu")

from ppsci.arch.climax import ClimaXClimateBench  # noqa: E402
from ppsci.data.dataset.climatebench_dataset import (  # noqa: E402
    ClimateBenchDataset,
)

import utils  # noqa: E402

DATA_ROOT = "__DATA_ROOT__"
CKPT_PATH = "__CKPT_PATH__"


def build_dataset(partition, inp_transform=None, out_transform=None):
    return ClimateBenchDataset(
        root_dir=DATA_ROOT,
        input_keys=("x", "lead_times"),
        label_keys=("tas",),
        partition=partition,
        seed=42,
        inp_transform=inp_transform,
        out_transform=out_transform,
    )


def probe_indices(n):
    return sorted(set([0, 1, n - 2, n - 1]))


result = {"data": {}, "forward": {}, "loss": {}}

# gate 1: dataset samples; the train partition shares its transforms with
# val/test, mirroring the reference datamodule's set_normalize calls
ds_train = build_dataset("train")
ds_val = build_dataset("val", ds_train.inp_transform, ds_train.out_transform)
ds_test = build_dataset("test", ds_train.inp_transform, ds_train.out_transform)
for partition, ds in (("train", ds_train), ("val", ds_val), ("test", ds_test)):
    samples = []
    for i in probe_indices(len(ds)):
        inp, out, _ = ds[i]
        samples.append(
            {
                "index": i,
                "x": inp["x"].tolist(),
                "lead_times": inp["lead_times"].tolist(),
                "y": out["tas"].tolist(),
            }
        )
    result["data"][partition] = {"length": len(ds), "samples": samples}

# gate 2: strict-load eval forward
net = ClimaXClimateBench(
    input_keys=("x", "lead_times"),
    output_keys=("tas",),
__MODEL_KW__
)
ckpt = paddle.load(CKPT_PATH)["state_dict"]
__STRICT_LOAD__


def run_forward(x, lead):
    with paddle.no_grad():
        out = net({"x": paddle.to_tensor(x), "lead_times": paddle.to_tensor(lead)})
    return out["tas"].numpy()


rs = np.random.RandomState(42)
x_syn = rs.randn(1, 10, 4, 32, 64).astype("float32")
lead_syn = np.zeros((1, 1), dtype="float32")
inp0, out0, _ = ds_test[0]
x_real = inp0["x"][None]
lead_real = inp0["lead_times"][None]
y_real = out0["tas"][None]

preds_syn = run_forward(x_syn, lead_syn)
preds_real = run_forward(x_real, lead_real)
result["forward"]["synthetic"] = preds_syn.tolist()
result["forward"]["real"] = preds_real.tolist()

# gate 3: single-batch loss on the real forward output
w_mse_expr = utils.make_w_mse_loss(ds_test.lat, log_postfix=None)
with paddle.no_grad():
    output_dict = {"tas": paddle.to_tensor(preds_real)}
    label_dict = {"tas": paddle.to_tensor(y_real)}
    result["loss"]["train_mse"] = float(utils.mse_loss(output_dict, label_dict)["mse"])
    result["loss"]["w_mse"] = float(w_mse_expr(output_dict, label_dict)["w_mse"])
result["loss"]["clim"] = float(ds_test.y_normalization)

with open(sys.argv[1], "w") as f:
    json.dump(result, f, sort_keys=True)
print("ppsci side ok")
'''

SIDE_PPCFD = '''"""ppcfd side of the equivalence gate: reference datamodule/dataset, model,
and the module loss paths (examples/climax)."""
import json
import sys

import numpy as np
import paddle

paddle.set_device("cpu")

from datamodule import ClimateBenchDataModule  # noqa: E402
from module import ClimateProjectionModule  # noqa: E402
from ppcfd.models.climax.climatebench import (  # noqa: E402
    ClimaXClimateBench,
)

DATA_ROOT = "__DATA_ROOT__"
CKPT_PATH = "__CKPT_PATH__"


def probe_indices(n):
    return sorted(set([0, 1, n - 2, n - 1]))


np.random.seed(42)
dm = ClimateBenchDataModule(root_dir=DATA_ROOT, history=10, batch_size=1)

result = {"data": {}, "forward": {}, "loss": {}}

# gate 1: dataset samples
for partition, ds in (
    ("train", dm.dataset_train),
    ("val", dm.dataset_val),
    ("test", dm.dataset_test),
):
    samples = []
    for i in probe_indices(len(ds)):
        inp, out, lead_times, _, _ = ds[i]
        samples.append(
            {
                "index": i,
                "x": inp.tolist(),
                "lead_times": lead_times.tolist(),
                "y": out.tolist(),
            }
        )
    result["data"][partition] = {"length": len(ds), "samples": samples}

# gate 2: strict-load eval forward
net = ClimaXClimateBench(
__MODEL_KW__
)
ckpt = paddle.load(CKPT_PATH)["state_dict"]
__STRICT_LOAD__


def run_forward(x, lead, y):
    with paddle.no_grad():
        _, preds = net.forward(
            paddle.to_tensor(x),
            paddle.to_tensor(y),
            paddle.to_tensor(lead),
            dm.variables,
            dm.out_variables,
            None,
            dm.lat,
        )
    return preds.numpy()


rs = np.random.RandomState(42)
x_syn = rs.randn(1, 10, 4, 32, 64).astype("float32")
lead_syn = np.zeros((1,), dtype="float32")
x_item, y_item, lead_real, _, _ = dm.dataset_test[0]
x_real = x_item.numpy()[None]
y_real = y_item.numpy()[None]

preds_syn = run_forward(x_syn, lead_syn, y_real)
preds_real = run_forward(x_real, lead_real, y_real)
result["forward"]["synthetic"] = preds_syn.tolist()
result["forward"]["real"] = preds_real.tolist()

# gate 3: module training_step / validation_step paths on the real batch
clim = dm.get_test_clim()
module = ClimateProjectionModule(net)
module.set_lat_lon(dm.lat, dm.lon)
module.set_denormalization(np.array([0.0]), np.array([1.0]))
module.set_val_clim(clim)
module.set_test_clim(clim)
batch = (
    paddle.to_tensor(x_real),
    paddle.to_tensor(y_real),
    lead_real,
    dm.variables,
    dm.out_variables,
)
with paddle.no_grad():
    result["loss"]["train_mse"] = float(module.training_step(batch, 0))
    val_dict = module.validation_step(batch, 0)
    result["loss"]["w_mse"] = float(val_dict["w_mse"])
result["loss"]["clim"] = float(clim)

with open(sys.argv[1], "w") as f:
    json.dump(result, f, sort_keys=True)
print("ppcfd side ok")
'''


def materialize(template, path):
    source = template.replace("__DATA_ROOT__", DATA_ROOT)
    source = source.replace("__CKPT_PATH__", CKPT_PATH)
    source = source.replace("__MODEL_KW__", MODEL_KW)
    source = source.replace("__STRICT_LOAD__", STRICT_LOAD)
    with open(path, "w") as f:
        f.write(source)


def run_side(side, workdir):
    script = os.path.join(workdir, f"side_{side}.py")
    out_json = os.path.join(workdir, f"out_{side}.json")
    if side == "ppsci":
        paths = [os.path.join(PPSCI_ROOT, "examples/climax"), PPSCI_ROOT]
    else:
        paths = [os.path.join(PPCFD_ROOT, "examples/climax"), PPCFD_ROOT]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(paths)
    subprocess.run([PYTHON, script, out_json], env=env, check=True)
    with open(out_json) as f:
        return json.load(f)


def diff_stats(a, b):
    diff = np.abs(np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64"))
    return float(diff.mean()), float(diff.max())


def gate1(ppsci, ppcfd, report):
    print("== gate 1: data pipeline (same-index dataset samples) ==")
    ok = True
    for part in ("train", "val", "test"):
        a, b = ppsci["data"][part], ppcfd["data"][part]
        part_max = 0.0
        len_ok = a["length"] == b["length"]
        ok = ok and len_ok
        for sa, sb in zip(a["samples"], b["samples"]):
            assert sa["index"] == sb["index"], f"{part}: index mismatch"
            for key in ("x", "y", "lead_times"):
                _, mx = diff_stats(sa[key], sb[key])
                part_max = max(part_max, mx)
        ok = ok and part_max == 0.0
        print(
            f"  {part}: len {a['length']} vs {b['length']} "
            f"(equal={len_ok}), 4 indices x/y/lead_times "
            f"max_abs_diff={part_max}"
        )
        report[f"gate1.{part}.max_abs_diff"] = part_max
        report[f"gate1.{part}.length"] = a["length"]
    return ok


def gate2(ppsci, ppcfd, report):
    print("== gate 2: forward (strict load, eval mode) ==")
    ok = True
    for side in ("ppsci", "ppcfd"):
        sl = (ppsci if side == "ppsci" else ppcfd)["forward"]["strict_load"]
        clean = not (sl["missing"] or sl["unexpected"] or sl["shape_mismatch"])
        ok = ok and clean
        print(
            f"  {side} strict_load: ckpt {sl['ckpt_keys']} / model "
            f"{sl['model_keys']} keys, missing={sl['missing']} "
            f"unexpected={sl['unexpected']} shape_mismatch={sl['shape_mismatch']}"
        )
        report[f"gate2.{side}.ckpt_keys"] = sl["ckpt_keys"]
        report[f"gate2.{side}.model_keys"] = sl["model_keys"]
    for case in ("synthetic", "real"):
        mean_abs, max_abs = diff_stats(ppsci["forward"][case], ppcfd["forward"][case])
        passed = max_abs < 1e-6
        ok = ok and passed
        print(
            f"  preds[{case}]: mean_abs_diff={mean_abs:.3e} "
            f"max_abs_diff={max_abs:.3e} (<1e-6: {passed})"
        )
        report[f"gate2.preds.{case}.mean_abs_diff"] = mean_abs
        report[f"gate2.preds.{case}.max_abs_diff"] = max_abs
    return ok


def gate3(ppsci, ppcfd, report):
    print("== gate 3: single-batch loss (real forward output) ==")
    ok = True
    for key in ("train_mse", "w_mse", "clim"):
        a, b = ppsci["loss"][key], ppcfd["loss"][key]
        abs_diff = abs(a - b)
        passed = abs_diff == 0.0
        ok = ok and passed
        print(f"  {key}: ppsci={a!r} ppcfd={b!r} abs_diff={abs_diff} ({passed})")
        report[f"gate3.{key}.ppsci"] = a
        report[f"gate3.{key}.ppcfd"] = b
        report[f"gate3.{key}.abs_diff"] = abs_diff
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        default="/tmp/climax08_gate",
        help="directory for side scripts and JSON outputs",
    )
    args = parser.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    materialize(SIDE_PPSCI, os.path.join(args.workdir, "side_ppsci.py"))
    materialize(SIDE_PPCFD, os.path.join(args.workdir, "side_ppcfd.py"))

    report = {}
    ppsci = run_side("ppsci", args.workdir)
    ppcfd = run_side("ppcfd", args.workdir)

    ok = gate1(ppsci, ppcfd, report)
    ok = gate2(ppsci, ppcfd, report) and ok
    ok = gate3(ppsci, ppcfd, report) and ok

    with open(os.path.join(args.workdir, "report.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print("RESULT:", "ALL GATES PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
