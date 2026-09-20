"""Export the torch-side initial state (cleaned MAE load) for the paddle port.

Replicates the torch baseline assembly of ``configs/climate_projection.yaml``
(``model.net.init_args`` with seed 42), constructs the Lightning
``ClimateProjectionModule`` with the local MAE checkpoint so that
``load_mae_weights`` performs the exact baseline cleaning (pos-embed
interpolation + channel->var rename + token_embeds/head removal + shape
filtering), then saves ``module.state_dict()`` (keys carry the ``net.``
prefix) to ``models/climax_torch/climax_initial.pt``.

The exported state is the *full post-cleaning* state: the MAE backbone plus
the randomly initialized new heads (token_embeds, head, var/time aggregation),
so the torch and paddle sides can be compared under strictly identical
weights (the new heads are seeded here, then transferred by conversion).

Usage (works from any cwd; the script fixes up sys.path itself):
    python export_torch_initial_state.py [--config PATH] [--ckpt PATH] [--out PATH] [--seed N]
"""
import argparse
import os
import random
import sys

import numpy as np

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TORCH_SIDE = os.path.join(REPO_ROOT, "climax_torch")
if TORCH_SIDE not in sys.path:
    # resolve "climax" to the torch side even when PYTHONPATH points elsewhere
    sys.path.insert(0, TORCH_SIDE)

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from climax.climate_projection.arch import ClimaXClimateBench  # noqa: E402
from climax.climate_projection.module import ClimateProjectionModule  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(
            REPO_ROOT, "climax_torch", "configs", "climate_projection.yaml"
        ),
        help="torch-side yaml providing model.net.init_args",
    )
    parser.add_argument(
        "--ckpt",
        default=os.path.join(REPO_ROOT, "models", "climax_torch", "5.625deg.ckpt"),
        help="local MAE checkpoint loaded via module.load_mae_weights",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            REPO_ROOT, "models", "climax_torch", "climax_initial.pt"
        ),
        help="output .pt path for module.state_dict()",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # replicate pytorch_lightning.seed_everything(seed) (CPU export)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    init_args = OmegaConf.to_container(cfg.model.net.init_args)
    print("[export] net init_args: %s" % init_args)

    net = ClimaXClimateBench(**init_args)
    # __init__ runs load_mae_weights (interpolate_pos_embed + cleaning);
    # pretrained_path is deliberately taken from --ckpt (local file), not the
    # HF url in the yaml.
    module = ClimateProjectionModule(net=net, pretrained_path=args.ckpt)

    state = module.state_dict()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(state, args.out)

    n_params = sum(p.numel() for p in module.parameters())
    n_tensors = sum(v.numel() for v in state.values())
    print("[export] total parameters (module): %s" % format(n_params, ","))
    print("[export] total elements in state_dict: %s" % format(n_tensors, ","))
    print("[export] state_dict keys: %d" % len(state))
    print("[export] saved -> %s" % args.out)


if __name__ == "__main__":
    main()
