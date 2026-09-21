"""Convert the exported torch initial state to a paddle pdparams checkpoint.

Reads ``models/climax_torch/climax_initial.pt`` (produced by
export_torch_initial_state.py; keys carry the ``net.`` prefix), applies the
name-driven transpose rules below, and writes
``models/climax_paddle/climax_initial.pdparams`` as
``{"state_dict": {net.-prefixed keys: paddle tensors}}`` — the exact contract
of paddle ClimateProjectionModule.load_mae_weights.

Transpose rules (measured on this environment, paddle 3.4):
  * ``net.blocks.<i>.attn.{qkv,proj}.weight`` and ``net.blocks.<i>.mlp.{fc1,fc2}.weight``
    — native paddle.nn.Linear inside thirdparty.timm Block stores weight as
    [in, out]; torch stores [out, in] -> transpose. NOTE attn.proj.weight is
    square [1024, 1024]: a missed transpose is NOT caught by shape checks,
    so the rule is driven by key names only, never by shapes.
  * everything else copies verbatim (same layout on both sides):
      - lead_time_embed / head: paddle.compat.nn.Linear keeps the torch
        [out, in] weight layout (verified numerically, forward diff 0.0);
      - var_agg / time_agg: paddle.compat.nn.MultiheadAttention state_dict is
        identical to torch nn.MultiheadAttention (in_proj_weight [3E, E],
        in_proj_bias [3E], out_proj.weight [E, E], out_proj.bias [E];
        verified numerically, forward diff ~1e-6);
      - token_embeds.*.proj: Conv2d weight [out, in, kh, kw] on both sides;
      - LayerNorm / Embedding-like parameters: same name, same shape.

The script self-checks: key-set diff vs a freshly built paddle model must be
empty and every post-conversion shape must match the paddle state_dict, then
it round-trips through ClimateProjectionModule.load_mae_weights (the Task 8/9
load path) and additionally runs a strict full set_state_dict.

Usage (works from any cwd; pure CPU):
    python convert_torch_to_paddle.py [--input PATH] [--config PATH] [--out PATH] [--skip-verify]
"""
import argparse
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # pure-CPU conversion

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PADDLE_SIDE = os.path.join(REPO_ROOT, "climax_paddle")
if PADDLE_SIDE not in sys.path:
    # resolve "climax"/"thirdparty.timm" to the paddle side even from other cwds
    sys.path.insert(0, PADDLE_SIDE)

import numpy as np  # noqa: E402
import paddle  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from climax.climate_projection.arch import ClimaXClimateBench  # noqa: E402
from climax.climate_projection.module import ClimateProjectionModule  # noqa: E402

# name-driven transpose rule (see module docstring; shapes are never consulted)
TRANSPOSE_RE = re.compile(
    r"^net\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2))\.weight$"
)

BUCKETS = [
    ("token_embeds", lambda k: k.startswith("net.token_embeds.")),
    ("blocks", lambda k: k.startswith("net.blocks.")),
    ("var_agg", lambda k: k.startswith("net.var_")),
    ("head", lambda k: k.startswith("net.head.")),
    ("time_agg", lambda k: k.startswith("net.time_")),
    (
        "other (pos_embed/lead_time_embed/norm)",
        lambda k: k
        in ("net.pos_embed", "net.lead_time_embed.weight", "net.lead_time_embed.bias",
            "net.norm.weight", "net.norm.bias"),
    ),
]


def bucket_of(key):
    for name, pred in BUCKETS:
        if pred(key):
            return name
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join(
            REPO_ROOT, "models", "climax_torch", "climax_initial.pt"
        ),
        help="torch state_dict (.pt) exported by export_torch_initial_state.py",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(
            REPO_ROOT, "climax_torch", "configs", "climate_projection.yaml"
        ),
        help="torch-side yaml providing model.net.init_args (same on both sides)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            REPO_ROOT, "models", "climax_paddle", "climax_initial.pdparams"
        ),
        help="output pdparams path",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the load-path round-trip verification (debug only)",
    )
    args = parser.parse_args()

    torch_sd = torch.load(args.input, map_location="cpu")
    init_args = OmegaConf.to_container(OmegaConf.load(args.config).model.net.init_args)

    net = ClimaXClimateBench(**init_args)
    # net.state_dict() is net-level (no prefix); re-key to the module-level
    # "net."-prefixed namespace of the exported torch state / pdparams contract
    paddle_sd = {"net." + k: v for k, v in net.state_dict().items()}

    # ---- Step 2: key-set and shape comparison table ----
    t_keys, p_keys = set(torch_sd), set(paddle_sd)
    only_torch, only_paddle = sorted(t_keys - p_keys), sorted(p_keys - t_keys)
    print("[convert] key-set diff vs paddle model: torch-only=%d, paddle-only=%d"
          % (len(only_torch), len(only_paddle)))
    if only_torch or only_paddle:
        for k in only_torch:
            print("  torch-only: %s %s" % (k, tuple(torch_sd[k].shape)))
        for k in only_paddle:
            print("  paddle-only: %s %s" % (k, tuple(paddle_sd[k].shape)))
        raise SystemExit("key sets differ; aborting")

    n_transposed = 0
    shape_mismatch = []
    converted = {}
    print("[convert] full key/shape comparison (T = transposed on transfer):")
    for k in sorted(torch_sd.keys()):
        w = torch_sd[k]
        if TRANSPOSE_RE.match(k):
            arr = w.numpy().T
            n_transposed += 1
            rule = "T"
        else:
            arr = w.numpy()
            rule = " "
        converted[k] = arr
        t_shape, p_shape = tuple(arr.shape), tuple(paddle_sd[k].shape)
        if t_shape != p_shape:
            shape_mismatch.append((k, t_shape, p_shape))
        print("  [%s] %-45s torch%s -> paddle%s%s"
              % (rule, k, tuple(w.shape), p_shape,
                 "" if t_shape == p_shape else "  <-- MISMATCH"))
    if shape_mismatch:
        for k, ts, ps in shape_mismatch:
            print("  shape mismatch: %s converted%s vs paddle%s" % (k, ts, ps))
        raise SystemExit("shape mismatches after conversion; aborting")
    print("[convert] keys: %d total, %d transposed, %d verbatim, all shapes match"
          % (len(converted), n_transposed, len(converted) - n_transposed))

    # ---- Step 3: save pdparams ----
    out_dict = {"state_dict": {k: paddle.to_tensor(v) for k, v in converted.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    paddle.save(out_dict, args.out)
    counts = {}
    for k in converted:
        counts[bucket_of(k)] = counts.get(bucket_of(k), 0) + 1
    for name, _ in BUCKETS:
        if name in counts:
            print("[convert] bucket %-35s %3d tensors" % (name, counts[name]))
    print("[convert] saved -> %s" % args.out)

    if args.skip_verify:
        return

    # ---- Step 4: round-trip through the Task 8/9 load path ----
    # module construction runs load_mae_weights, which prints its
    # (missing_keys, unexpected_keys) message; missing must be exactly the
    # re-initialized token_embeds/head keys. The programmatic check below is
    # stronger: every other converted tensor must be bit-identical in the
    # loaded net (the conversion is exact copy/transpose, no arithmetic).
    print("[verify] ClimateProjectionModule.load_mae_weights round-trip:")
    module = ClimateProjectionModule(
        net=ClimaXClimateBench(**init_args), pretrained_path=args.out
    )
    checkpoint = paddle.load(args.out)["state_dict"]
    kept = {
        k[len("net."):]: v
        for k, v in checkpoint.items()
        if k.startswith("net.") and "token_embeds" not in k and "head" not in k
    }
    net_sd = module.net.state_dict()
    assert set(kept) <= set(net_sd), "kept keys missing from the loaded net"
    for k, v in kept.items():
        assert np.array_equal(v.numpy(), net_sd[k].numpy()), (
            "weight mismatch at %s after load_mae_weights" % k
        )
    print("[verify] %d converted tensors bit-identical in the loaded net; "
          "re-initialized from scratch: %d token_embeds/head keys"
          % (len(kept), len(checkpoint) - len(kept)))

    # strict check: full converted dict must load with zero missing/unexpected
    strict_net = ClimaXClimateBench(**init_args)
    full = {k[len("net."):]: v for k, v in checkpoint.items()}
    s_missing, s_unexpected = strict_net.set_state_dict(full)
    print("[verify] strict set_state_dict: missing=%s unexpected=%s"
          % (sorted(s_missing), sorted(s_unexpected)))
    assert not s_missing and not s_unexpected, "strict check failed"
    print("[verify] OK")


if __name__ == "__main__":
    main()
