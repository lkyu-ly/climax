"""Forward alignment verification between the torch and paddle ClimaX ports.

Task 8 acceptance gate. Runs both frameworks on one fixed random input under
identical initial weights and compares the predicted ``tas`` field.

Contract (see task brief):
  * fixed input: ``np.random.RandomState(42)`` -> x=(1,10,4,32,64),
    y=(1,1,32,64), stored float64 in ``exps/task8_forward_align/inputs.npz``
    (both sides cast to float32 after loading); lat read from the real
    ClimateBench nc file; lead_times = zeros(1).
  * weights: BOTH sides load the full *initial* state directly --
    torch: ``module.load_state_dict(torch.load(climax_initial.pt), strict=True)``
    paddle: ``net.set_state_dict(climax_initial.pdparams['state_dict'])``
    (missing/unexpected must be empty). Never the ``load_mae_weights``
    cleaning path (it drops token_embeds/head keys and would desynchronize
    the freshly initialized heads).
  * eval mode, metric=None, forward returns (loss, preds) with preds shaped
    (1, 1, 32, 64).
  * acceptance: mean_abs_error < 1e-5 and mean_rel_error at 1e-6 scale.
    Five metrics reported: max/mean_abs_error, rmse, max/mean_rel_error.

The torch and paddle forwards run in separate subprocesses (each side gets
its own PYTHONPATH/cwd; only numpy crosses the boundary via npz), so
``all`` is the one-command entry point:

    python compare_forward.py all [--trace]

Subcommands: setup | torch | paddle | compare | trace-diff | all.
``--trace`` additionally dumps layer-by-layer intermediates (token_embeds
out, after var aggregation, after pos embed, after time/lead embed, every
2 blocks, after norm, after GAP, after time_agg, preds) for divergence
localization; ``trace-diff`` compares two trace files.

CPU only; pure-numpy comparison needs no framework in the parent process.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TORCH_SIDE = os.path.join(REPO_ROOT, "climax_torch")
PADDLE_SIDE = os.path.join(REPO_ROOT, "climax_paddle")
EXP_DIR = os.path.join(REPO_ROOT, "exps", "task8_forward_align")
CONFIG = os.path.join(
    PADDLE_SIDE, "configs", "climate_projection.yaml"
)  # model.net.init_args, identical semantics on both sides
LAT_NC = os.path.join(
    REPO_ROOT, "dataset", "climatebench", "5.625deg", "train_val",
    "outputs_ssp126.nc",
)
TORCH_CKPT = os.path.join(REPO_ROOT, "models", "climax_torch", "climax_initial.pt")
PADDLE_CKPT = os.path.join(
    REPO_ROOT, "models", "climax_paddle", "climax_initial.pdparams"
)
INPUTS_NPZ = os.path.join(EXP_DIR, "inputs.npz")
TORCH_OUT = os.path.join(EXP_DIR, "torch_out.npz")
PADDLE_OUT = os.path.join(EXP_DIR, "paddle_out.npz")
TORCH_TRACE = os.path.join(EXP_DIR, "torch_trace.npz")
PADDLE_TRACE = os.path.join(EXP_DIR, "paddle_trace.npz")

VARIABLES = ("CO2", "SO2", "CH4", "BC")
OUT_VARIABLES = ("tas",)

# acceptance thresholds (mean_rel_error at 1e-6 scale => well below 1e-5)
ABS_TOL = 1e-5
REL_TOL = 1e-5


def load_init_args():
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(CONFIG).model.net.init_args)


# ---------------------------------------------------------------- setup ----
def cmd_setup(_args):
    import xarray as xr

    rng = np.random.RandomState(42)
    x = rng.randn(1, 10, 4, 32, 64)  # float64 source of truth
    y = rng.randn(1, 1, 32, 64)
    lat = np.asarray(xr.open_dataset(LAT_NC)["lat"].values, dtype=np.float64)
    os.makedirs(EXP_DIR, exist_ok=True)
    np.savez(INPUTS_NPZ, x=x, y=y, lat=lat)
    print("[setup] RandomState(42): x%s y%s float64; lat%s from %s"
          % (x.shape, y.shape, lat.shape,
             os.path.relpath(LAT_NC, REPO_ROOT)))
    print("[setup] lat[:3]=%s ... lat[-2:]=%s" % (lat[:3], lat[-2:]))
    print("[setup] saved -> %s" % INPUTS_NPZ)


# ---------------------------------------------------------------- torch ----
def trace_torch(net, x, lead_times, variables):
    """Unrolled forward_encoder mirroring torch arch.py, capturing named
    intermediates for cross-framework divergence localization."""
    import torch

    out = {}
    b, t = x.shape[0], x.shape[1]
    xf = x.flatten(0, 1)  # BxT, V, H, W
    var_ids = net.get_var_ids(tuple(variables), xf.device)
    embeds = [net.token_embeds[int(i)](xf[:, i: i + 1]) for i in range(len(var_ids))]
    xe = torch.stack(embeds, dim=1)  # BxT, V, L, D
    out["token_embeds"] = xe
    var_embed = net.get_var_emb(net.var_embed, tuple(variables))
    x1 = xe + var_embed.unsqueeze(2)
    x2 = net.aggregate_variables(x1)
    out["after_var_agg"] = x2
    x3 = x2 + net.pos_embed
    out["after_pos_embed"] = x3
    x4 = x3.unflatten(0, sizes=(b, t)) + net.time_pos_embed.unsqueeze(2)
    lead_time_emb = net.lead_time_embed(lead_times.unsqueeze(-1))
    x5 = (x4 + lead_time_emb.unsqueeze(1).unsqueeze(2)).flatten(0, 1)
    out["after_time_lead_embed"] = x5
    h = net.pos_drop(x5)
    for i, blk in enumerate(net.blocks):
        h = blk(h)
        if (i + 1) % 2 == 0:
            out["after_block%d" % (i + 1)] = h
    hn = net.norm(h)
    out["after_norm"] = hn
    g = hn.unflatten(0, sizes=(b, t)).mean(-2)  # B, T, D
    out["after_gap"] = g
    time_query = net.time_query.repeat_interleave(g.shape[0], dim=0)
    ta, _ = net.time_agg(time_query, g, g)  # B, 1, D
    out["after_time_agg"] = ta
    preds = net.head(ta).reshape(-1, 1, net.img_size[0], net.img_size[1])
    out["preds"] = preds
    return out


def cmd_torch(args):
    import torch
    from climax.climate_projection.arch import ClimaXClimateBench
    from climax.climate_projection.module import ClimateProjectionModule

    init_args = load_init_args()
    print("[torch] net init_args: %s" % init_args)
    net = ClimaXClimateBench(**init_args)
    # no pretrained_path => no load_mae_weights cleaning (would drop
    # token_embeds/head); we load the FULL exported initial state instead
    module = ClimateProjectionModule(net=net)
    sd = torch.load(TORCH_CKPT, map_location="cpu")
    missing, unexpected = module.load_state_dict(sd, strict=True)
    print("[torch] module.load_state_dict(strict=True): missing=%s unexpected=%s"
          % (missing, unexpected))
    assert not missing and not unexpected, "strict load failed on torch side"
    module.eval()

    d = np.load(INPUTS_NPZ)
    x = torch.from_numpy(d["x"].astype(np.float32))
    y = torch.from_numpy(d["y"].astype(np.float32))
    lat = torch.from_numpy(d["lat"].astype(np.float32))
    lead_times = torch.zeros(1)
    with torch.inference_mode():
        loss, preds = net.forward(
            x, y, lead_times, VARIABLES, OUT_VARIABLES, metric=None, lat=lat
        )
    print("[torch] preds shape=%s dtype=%s |loss|=%s"
          % (tuple(preds.shape), preds.dtype, loss))
    np.savez(TORCH_OUT, preds=preds.numpy().astype(np.float64))
    print("[torch] saved -> %s" % TORCH_OUT)

    if args.trace:
        with torch.inference_mode():
            tr = trace_torch(net, x, lead_times, VARIABLES)
        np.savez(TORCH_TRACE, **{k: v.numpy().astype(np.float64)
                                 for k, v in tr.items()})
        print("[torch] trace saved -> %s (%d layers)" % (TORCH_TRACE, len(tr)))


# --------------------------------------------------------------- paddle ----
def trace_paddle(net, x, lead_times, variables):
    """Unrolled forward_encoder mirroring paddle arch.py (same stations as
    trace_torch)."""
    import paddle

    out = {}
    b, t = x.shape[0], x.shape[1]
    xf = x.flatten(0, 1)
    var_ids = net.get_var_ids(tuple(variables), xf.device)
    embeds = [net.token_embeds[int(i)](xf[:, i: i + 1]) for i in range(len(var_ids))]
    xe = paddle.stack(embeds, dim=1)
    out["token_embeds"] = xe
    var_embed = net.get_var_emb(net.var_embed, tuple(variables))
    x1 = xe + var_embed.unsqueeze(2)
    x2 = net.aggregate_variables(x1)
    out["after_var_agg"] = x2
    x3 = x2 + net.pos_embed
    out["after_pos_embed"] = x3
    x4 = x3.unflatten(0, sizes=(b, t)) + net.time_pos_embed.unsqueeze(2)
    lead_time_emb = net.lead_time_embed(lead_times.unsqueeze(-1))
    x5 = (x4 + lead_time_emb.unsqueeze(1).unsqueeze(2)).flatten(0, 1)
    out["after_time_lead_embed"] = x5
    h = net.pos_drop(x5)
    for i, blk in enumerate(net.blocks):
        h = blk(h)
        if (i + 1) % 2 == 0:
            out["after_block%d" % (i + 1)] = h
    hn = net.norm(h)
    out["after_norm"] = hn
    g = hn.unflatten(0, sizes=(b, t)).mean(-2)
    out["after_gap"] = g
    time_query = net.time_query.repeat_interleave(g.shape[0], dim=0)
    ta, _ = net.time_agg(time_query, g, g)
    out["after_time_agg"] = ta
    preds = net.head(ta).reshape(-1, 1, net.img_size[0], net.img_size[1])
    out["preds"] = preds
    return out


def cmd_paddle(args):
    import paddle
    from climax.climate_projection.arch import ClimaXClimateBench

    paddle.set_device("cpu")
    init_args = load_init_args()
    print("[paddle] net init_args: %s" % init_args)
    net = ClimaXClimateBench(**init_args)
    # full strict load of the converted initial state; keys in the pdparams
    # carry the lightning "net." prefix, strip to ClimaXClimateBench level
    ckpt = paddle.load(PADDLE_CKPT)["state_dict"]
    full = {k[len("net."):]: v for k, v in ckpt.items() if k.startswith("net.")}
    missing, unexpected = net.set_state_dict(full)
    print("[paddle] net.set_state_dict: missing=%s unexpected=%s"
          % (sorted(missing), sorted(unexpected)))
    assert not missing and not unexpected, "strict load failed on paddle side"
    net.eval()

    d = np.load(INPUTS_NPZ)
    x = paddle.to_tensor(d["x"].astype(np.float32))
    y = paddle.to_tensor(d["y"].astype(np.float32))
    lat = paddle.to_tensor(d["lat"].astype(np.float32))
    lead_times = paddle.zeros((1,), dtype="float32")
    loss, preds = net.forward(
        x, y, lead_times, VARIABLES, OUT_VARIABLES, metric=None, lat=lat
    )
    print("[paddle] preds shape=%s dtype=%s |loss|=%s"
          % (tuple(preds.shape), preds.dtype, loss))
    np.savez(PADDLE_OUT, preds=preds.numpy().astype(np.float64))
    print("[paddle] saved -> %s" % PADDLE_OUT)

    if args.trace:
        tr = trace_paddle(net, x, lead_times, VARIABLES)
        np.savez(PADDLE_TRACE, **{k: v.numpy().astype(np.float64)
                                  for k, v in tr.items()})
        print("[paddle] trace saved -> %s (%d layers)" % (PADDLE_TRACE, len(tr)))


# -------------------------------------------------------------- compare ----
def five_metrics(a, b):
    """max/mean abs error, rmse, max/mean rel error (rel vs |torch|)."""
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-12)  # guard zero crossings
    rel = diff / denom
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "max_rel_error": float(rel.max()),
        "mean_rel_error": float(rel.mean()),
    }


def cmd_compare(_args):
    a = np.load(TORCH_OUT)["preds"]
    b = np.load(PADDLE_OUT)["preds"]
    print("[compare] torch%s paddle%s dtype=%s"
          % (a.shape, b.shape, a.dtype))
    assert a.shape == b.shape == (1, 1, 32, 64), "unexpected preds shape"
    print("[compare] |torch| stats: min=%.6g max=%.6g mean=%.6g"
          % (np.abs(a).min(), np.abs(a).max(), np.abs(a).mean()))
    m = five_metrics(a, b)
    for k in ("max_abs_error", "mean_abs_error", "rmse",
              "max_rel_error", "mean_rel_error"):
        print("[compare] %-15s = %.6g" % (k, m[k]))
    passed = m["mean_abs_error"] < ABS_TOL and m["mean_rel_error"] < REL_TOL
    print("[compare] %s (acceptance: mean_abs_error < %g and mean_rel_error "
          "< %g, i.e. 1e-6 scale)" % ("PASS" if passed else "FAIL",
                                      ABS_TOL, REL_TOL))
    return 0 if passed else 1


def cmd_trace_diff(_args):
    ta = np.load(TORCH_TRACE)
    tb = np.load(PADDLE_TRACE)
    keys = [k for k in ta.files if k in tb.files]
    print("[trace-diff] %d common layers" % len(keys))
    first_bad = None
    for k in keys:
        m = five_metrics(ta[k], tb[k])
        flag = ""
        if m["mean_rel_error"] > 1e-3 and first_bad is None:
            first_bad = k
            flag = "  <-- first big divergence"
        print("[trace-diff] %-22s max_abs=%.4g mean_abs=%.4g mean_rel=%.4g%s"
              % (k, m["max_abs_error"], m["mean_abs_error"],
                 m["mean_rel_error"], flag))
    if first_bad:
        print("[trace-diff] divergence localizes at: %s" % first_bad)
    return 0


# ------------------------------------------------------------------ all ----
def run_side(side, extra):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    if side == "torch":
        env["PYTHONPATH"] = TORCH_SIDE
        cwd = REPO_ROOT
    else:
        env["PYTHONPATH"] = PADDLE_SIDE
        cwd = PADDLE_SIDE
    print("[all] === %s side (cwd=%s) ===" % (side, cwd))
    subprocess.run([sys.executable, os.path.abspath(__file__), side] + extra,
                   env=env, cwd=cwd, check=True)


def cmd_all(args):
    extra = ["--trace"] if args.trace else []
    cmd_setup(args)
    run_side("torch", extra)
    run_side("paddle", extra)
    rc = cmd_compare(args)
    if rc != 0 and args.trace:
        cmd_trace_diff(args)
    return rc


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="generate the fixed input npz")
    for name in ("torch", "paddle"):
        p = sub.add_parser(name, help="run the %s forward and save preds" % name)
        p.add_argument("--trace", action="store_true",
                       help="also dump layer-by-layer intermediates")
    sub.add_parser("compare", help="five-metric comparison + acceptance")
    sub.add_parser("trace-diff", help="compare two trace files layer by layer")
    p_all = sub.add_parser("all", help="setup -> torch -> paddle -> compare")
    p_all.add_argument("--trace", action="store_true",
                       help="dump and diff intermediates (divergence hunt)")
    args = parser.parse_args()

    # keep parent/child output interleaved in order when piped to tee
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # pragma: no cover
        pass

    if args.command == "setup":
        cmd_setup(args)
        return 0
    if args.command == "torch":
        cmd_torch(args)
        return 0
    if args.command == "paddle":
        cmd_paddle(args)
        return 0
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "trace-diff":
        return cmd_trace_diff(args)
    if args.command == "all":
        return cmd_all(args)


if __name__ == "__main__":
    sys.exit(main())
