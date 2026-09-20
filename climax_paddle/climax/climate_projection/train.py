# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import ast
import os
import random

import numpy as np
import paddle
from omegaconf import OmegaConf

from climax.climate_projection.arch import ClimaXClimateBench
from climax.climate_projection.datamodule import ClimateBenchDataModule
from climax.climate_projection.module import ClimateProjectionModule


def _parse_value(raw):
    """Convert a --key=value string into bool/int/float/None/list/str."""
    s = raw.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass
    return s


def _apply_overrides(cfg, tokens):
    """Apply --key=value dot-path overrides, omegaconf merge style."""
    for token in tokens:
        if not token.startswith("--") or "=" not in token:
            raise SystemExit(f"Unrecognized argument: {token} (expected --key=value)")
        key, value = token[2:].split("=", 1)
        keys = [k for k in key.split(".") if k]
        if not keys:
            raise SystemExit(f"Invalid override key: {token}")
        override = cur = {}
        for k in keys[:-1]:
            cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = _parse_value(value)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override))
    return cfg


def _iter_limited(loader, limit_batches):
    """Iterate a dataloader, optionally taking only the first N batches (dryrun)."""
    if limit_batches is None:
        yield from loader
        return
    for i, batch in enumerate(loader):
        if i >= limit_batches:
            break
        yield batch


def _epoch_means(module, prefix, start_lengths):
    """Mean over the log_history entries appended since start_lengths was taken."""
    means = {}
    for name, values in module.log_history.items():
        if not name.startswith(prefix):
            continue
        seg = values[start_lengths.get(name, 0):]
        if seg:
            means[name] = sum(seg) / len(seg)
    return means


def _print_metric_table(metrics, header="Test metric", source="DataLoader 0"):
    """Print a lightning-style two-column metric table (torch baseline format).

    Matches the torch baseline table (exps/torch_baseline_train.log): two
    columns of equal width, no leading empty column, cells centered with
    format-spec ``^`` (odd padding puts the extra space on the right, like
    lightning). Width is floored at 27 so the current metric key set renders
    exactly at the baseline's 27/27; longer keys/values widen the columns.
    """
    longest = max(
        [len(header), len(source)] + [len(k) for k in metrics] +
        [len(str(v)) for v in metrics.values()]
    )
    width = max(27, longest + 2)
    print("┏" + "━" * width + "┳" + "━" * width + "┓")
    print(f"┃{header:^{width}}┃{source:^{width}}┃")
    print("┡" + "━" * width + "╇" + "━" * width + "┩")
    for name in sorted(metrics.keys()):
        print(f"│{name:^{width}}│{metrics[name]:^{width}}│")
    print("└" + "─" * width + "┴" + "─" * width + "┘")


def main():
    parser = argparse.ArgumentParser(
        description="Train ClimaX climate_projection with paddle (custom loop)"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/climate_projection.yaml",
        help="Path to the yaml config",
    )
    args, overrides = parser.parse_known_args()

    cfg = _apply_overrides(OmegaConf.load(args.config), overrides)
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", 42)
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    root_dir = cfg.train.default_root_dir
    os.makedirs(root_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(root_dir, "config.yaml"))

    datamodule = ClimateBenchDataModule(
        root_dir=cfg.data.root_dir,
        history=cfg.data.history,
        list_train_simu=list(cfg.data.list_train_simu),
        list_test_simu=list(cfg.data.list_test_simu),
        variables=list(cfg.data.variables),
        out_variables=cfg.data.out_variables,
        train_ratio=cfg.data.train_ratio,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    net = ClimaXClimateBench(**cfg.model.net.init_args)

    # --model.init_state_path: load the FULL initial state (backbone + heads)
    # directly, bypassing load_mae_weights (whose cleaning drops token_embeds/
    # head keys and would leave the heads freshly initialized instead of shared
    # with the torch baseline). Same load path as tools/compare_forward.py.
    init_state_path = cfg.model.get("init_state_path", "")
    if init_state_path:
        ckpt = paddle.load(init_state_path)["state_dict"]
        full = {k[len("net."):]: v for k, v in ckpt.items() if k.startswith("net.")}
        missing, unexpected = net.set_state_dict(full)
        print(
            "Loaded full initial state from %s: missing=%s unexpected=%s"
            % (init_state_path, sorted(missing), sorted(unexpected))
        )
        if missing or unexpected:
            raise RuntimeError("strict initial state load failed")
    module = ClimateProjectionModule(
        net=net,
        pretrained_path="" if init_state_path else cfg.model.get("pretrained_path", ""),
        lr=cfg.model.get("lr", 5e-4),
        beta_1=cfg.model.get("beta_1", 0.9),
        beta_2=cfg.model.get("beta_2", 0.99),
        weight_decay=cfg.model.get("weight_decay", 1e-5),
        warmup_epochs=cfg.model.get("warmup_epochs", 60),
        max_epochs=cfg.model.get("max_epochs", 600),
        warmup_start_lr=cfg.model.get("warmup_start_lr", 1e-8),
        eta_min=cfg.model.get("eta_min", 1e-8),
    )

    # assembly sequence replicated from torch main()
    normalization = datamodule.dataset_train.out_transform
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    module.set_denormalization(mean_denorm, std_denorm)
    module.set_lat_lon(*datamodule.get_lat_lon())
    module.set_pred_range(0)
    module.set_val_clim(None)
    module.set_test_clim(datamodule.get_test_clim())

    optimizer, lr_scheduler = module.configure_optimizers()

    max_epochs = cfg.train.get("max_epochs", 50)
    patience = int(cfg.train.get("patience", 5))
    # --train.limit_batches=N: first N batches of train/val/test each (dryrun)
    limit_batches = cfg.train.get("limit_batches", None)
    monitor_key = "val/w_mse"

    best_path = os.path.join(root_dir, "best.pdparams")
    last_path = os.path.join(root_dir, "last.pdparams")
    best_metric = float("inf")
    bad_epochs = 0

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    for epoch in range(max_epochs):
        module.net.train()
        for i, batch in enumerate(_iter_limited(train_loader, limit_batches)):
            loss = module.training_step(batch, i)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()
            if (i + 1) % 50 == 0:
                print(
                    f"Epoch {epoch} step {i + 1}: train/loss = {float(loss):.4f}"
                )

        # validation: epoch means of the metrics logged by validation_step
        module.net.eval()
        start_lengths = {
            name: len(vals)
            for name, vals in module.log_history.items()
            if name.startswith("val/")
        }
        with paddle.no_grad():
            for i, batch in enumerate(_iter_limited(val_loader, limit_batches)):
                module.validation_step(batch, i)
        val_metrics = _epoch_means(module, "val/", start_lengths)
        print(
            f"Epoch {epoch}: "
            + ", ".join(f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
        )

        # checkpoints: best on monitor + last at every epoch end (torch yaml:
        # ModelCheckpoint monitor=val/w_mse mode=min save_last=True)
        paddle.save(module.net.state_dict(), last_path)
        monitor = val_metrics.get(monitor_key)
        if monitor is None:
            raise RuntimeError(
                f"Monitored metric {monitor_key} missing from validation logs"
            )
        if monitor < best_metric:
            best_metric = monitor
            bad_epochs = 0
            paddle.save(module.net.state_dict(), best_path)
            print(f"Epoch {epoch}: {monitor_key} improved to {monitor:.6f}")
        else:
            bad_epochs += 1
            print(
                f"Epoch {epoch}: no {monitor_key} improvement for "
                f"{bad_epochs}/{patience} epochs"
            )
            # EarlyStopping (torch yaml: patience=5, min_delta=0., mode=min)
            if bad_epochs >= patience:
                print("Early stopping triggered")
                break

    # test the trained model with the best checkpoint (torch main():
    # ckpt_path='best')
    if os.path.exists(best_path):
        print(f"Restoring states from the checkpoint path at {best_path}")
        module.net.set_state_dict(paddle.load(best_path))
    else:
        print(f"No checkpoint found at {best_path}, testing current weights")

    module.net.eval()
    test_loader = datamodule.test_dataloader()
    start_lengths = {
        name: len(vals)
        for name, vals in module.log_history.items()
        if name.startswith("test/")
    }
    with paddle.no_grad():
        for i, batch in enumerate(_iter_limited(test_loader, limit_batches)):
            module.test_step(batch, i)
    test_metrics = _epoch_means(module, "test/", start_lengths)
    _print_metric_table(test_metrics)
    return test_metrics


if __name__ == "__main__":
    main()
