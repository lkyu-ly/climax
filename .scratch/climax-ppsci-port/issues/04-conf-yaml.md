# 04: examples/climax/conf/climax.yaml

**Type:** task
**Status:** resolved
**Blocked by:** None (can start immediately)

## What to build

hydra defaults 组合链形态的 ClimaX 配置文件就位，compose 冒烟通过。

## 实施要点

- 模板：`examples/climateformer/conf/climateformer.yaml`（79 行，deep-dive-conventions.md §6 已录全字段）
- defaults 链：`ppsci_default / TRAIN:train_default / EVAL:eval_default / _self_`（TRAIN/ema、TRAIN/swa 省略——可选组）
- MODEL 段：ClimaX 全部常量（default_vars/out_vars/img_size=[32,64]/time_history=10/patch_size=2/embed_dim=1024/depth=8/num_heads=16/mlp_ratio=4/drop_path=0.1/drop_rate=0.1/freeze_encoder=true + input_keys/output_keys）
- TRAIN/EVAL 段字段对照 climateformer.yaml；`eval_during_train: true` + `eval_freq`（单位 epoch）
- 编排值（lr/beta/wd/warmup 60/max_epochs 50/seed 42/batch_size 1 等）从 PaddleCFD feat/climax 的 `examples/climax/configs/climate_projection.yaml` 逐项照抄（该 yaml 已对照原版逐项恢复过）
- 数据路径占位 `./data/climatebench/5.625deg`；输出占位 `./exps/climate_projection`；init_state_path 默认 ""（yaml 键存在）
- `to_static: false` 默认（CINN 由 env 联动，05 票实现）

## Acceptance criteria

- [x] hydra compose 冒烟（`@hydra.main` 同款路径解析成功，无缺键）
- [x] 编排值与 ppcfd 版逐项核对一致（记录核对清单）
- [x] 无 Lightning 残留字段

## Answer

### 做了什么

新增 `/home/lkyu/baidu/PaddleScience/examples/climax/conf/climax.yaml`（93 行，唯一改动文件，工作区未提交）。结构：

- defaults：`ppsci_default / TRAIN:train_default / EVAL:eval_default / hydra/job/config/override_dirname/exclude_keys:exclude_keys_default / _self_`（无 ema/swa/INFER——example 仅 train/eval 两模式；exclude_keys 为 climateformer/fuxi 共有惯例）
- hydra 段照 climateformer（`job.name: ${mode}`、`chdir: false`、InitCallback callback、sweep），`run.dir: ./exps/climate_projection`（对位 ppcfd `train.default_root_dir`）
- 顶层：`mode: train / seed: 42 / output_dir: ${hydra:run.dir} / log_freq: 20 / to_static: false`（显式写出，CINN 由 main.py 的 `CLIMAX_USE_CINN` 联动）
- `DATA` 块（新增段，字段名照 ppcfd `data:` 段）：root_dir/history/list_train_simu/list_test_simu/variables/out_variables/train_ratio/num_workers；`output_dir` 即 `./exps/climate_projection` 路径由 hydra.run.dir 承载
- `MODEL` 块 = ppcfd `model.net.init_args` 14 项逐项照抄 + ppsci 约定新增 `input_keys: ["x","lead_times"]` / `output_keys: ["tas"]`（票 02 `ClimaXClimateBench(**cfg.MODEL)` 直用，无 class_path）
- `TRAIN` 块：字段名/结构照 climateformer（epochs/save_freq/eval_during_train/eval_freq/batch_size/pretrained_model_path/checkpoint_path + lr_scheduler）；`lr_scheduler` 键名对齐 `ppsci.optimizer.lr_scheduler.Cosine` 签名（`epochs: 600 / learning_rate: 5e-4 / eta_min: 1e-8 / warmup_epoch: 60 / warmup_start_lr: 1e-8 / by_epoch: true`）；平铺 `beta_1/beta_2/weight_decay`（brusselator3d/deepcfd 的 `TRAIN.weight_decay` 平铺先例）与 `init_state_path: ""`/`pretrained_path: ""`（键存在，直读 `cfg.TRAIN.init_state_path`）
- `EVAL` 块：pretrained_model_path/compute_metric_by_batch: true（票 06 keep_batch 要求）/eval_with_no_grad: true/batch_size: 1
- iters_per_epoch 不写（train_default 默认 20 残留但被 main.py 以 dataloader 长度覆盖，climateformer 同款）

### 关键证据（命令均真实执行）

1. compose 冒烟（/tmp/climax_conf_smoke.py，`hydra.initialize_config_dir + compose(return_hydra_config=True) + HydraConfig.instance().set_config(cfg)`——与 `@hydra.main` 相同的解析机制，`${hydra:run.dir}` 可解）：42 项断言全过，末行 `ALL ASSERTIONS PASSED`，EXIT=0；compose 结果确认 ppsci_default 合入（use_vdl/device/amp_level 等默认键在）、TRAIN 组 ema/swa 为 null、输出目录解析为 `./exps/climate_projection`
2. 逐项核对（/tmp/climax_conf_crosscheck.py，机械比对 `git -C PaddleCFD show feat/climax:examples/climax/configs/climate_projection.yaml` vs compose 结果）：`total=39 pass=39 fail=0`，EXIT=0
3. `Cosine(**cfg.TRAIN.lr_scheduler, iters_per_epoch=106)()` 构造成功，首 lr=1e-08（= warmup_start_lr，符合预期），EXIT=0
4. Lightning 残留扫描：对 compose 全树（除 hydra 自身节）遍历 25 个 Lightning trainer 键（default_root_dir/max_steps/accumulate_grad_batches/precision/strategy/…）零命中

### 核对表（ppcfd → climax.yaml，39 项全 PASS，摘要）

| ppcfd | → climax.yaml |
|---|---|
| model.lr 5e-4 | TRAIN.lr_scheduler.learning_rate |
| model.beta_1 0.9 / beta_2 0.999 / weight_decay 1e-5 | TRAIN.beta_1 / beta_2 / weight_decay |
| model.warmup_epochs 60 | TRAIN.lr_scheduler.warmup_epoch（Cosine 用单数形） |
| model.max_epochs 600 | TRAIN.lr_scheduler.epochs（lr 周期地平线，非训练轮数） |
| model.warmup_start_lr / eta_min 1e-8 | TRAIN.lr_scheduler.* |
| model.init_state_path / pretrained_path "" | TRAIN.init_state_path / pretrained_path |
| model.net.init_args 全 14 项（default_vars/out_vars/img_size [32,64]/time_history 10/patch_size 2/embed_dim 1024/depth 8/num_heads 16/mlp_ratio 4/drop_path 0.1/drop_rate 0.1/parallel_patch_embed False/freeze_encoder True） | MODEL 同名逐项 |
| data.root_dir ./data/climatebench/5.625deg | DATA.root_dir |
| data.history/list_train_simu/list_test_simu/variables/out_variables/train_ratio/num_workers | DATA 同名 |
| data.batch_size 1 | TRAIN.batch_size + EVAL.batch_size |
| train.max_epochs 50 | TRAIN.epochs |
| train.default_root_dir ./exps/climate_projection | hydra.run.dir（= output_dir） |
| seed 42 | seed |

### 有意偏差（4 项，均已记录在核对脚本输出）

- `model.net.class_path` 不搬：arch 类由 main.py 选定（`ppsci.arch.ClimaXClimateBench`）
- `data.pin_memory` 不搬：paddle DataLoader 无此参数（ppcfd 自身仅"API parity"保留，为死旋钮）
- `train.patience: 5` 不搬：ppsci Solver 无 early-stopping，训练跑满 TRAIN.epochs=50
- MODEL 新增 `input_keys/output_keys`：ppsci Arch dict 进 dict 出约定
- 注：pyyaml(1.1) 会把 `5e-4` 读成字符串而 omegaconf 读成 float——核对脚本已做数值化归一后比对；port 侧类型为正确 float

### 遗留事项

- 票 05 main.py 读键约定：`cfg.DATA.*`（数据集装配）、`cfg.MODEL`（整体 kwargs 传 arch）、`cfg.TRAIN.{beta_1,beta_2,weight_decay,init_state_path,pretrained_path,lr_scheduler,batch_size,epochs,eval_during_train,eval_freq,save_freq}`、`cfg.EVAL.*`、顶层 `seed/to_static/output_dir/log_freq/mode`
- `save_freq: 5` 票面未指定，取 climateformer 同值；如需逐 epoch 存档可命令行覆盖
- /tmp/climax_conf_smoke.py 与 /tmp/climax_conf_crosscheck.py 留存于 /tmp 作复验脚本（未进两仓）

**勘误（2026-09-26，票 16）**：`model.warmup_epochs/max_epochs → lr_scheduler.warmup_epoch/epochs + by_epoch: true` 的映射是时间尺度错位——原版调度为 **step 单位**（interval="step"），by_epoch=true 使 epoch-0 lr 恒 1e-8、头未训练（val/w_mse 停在 1.7675）。该映射已由票 16 的平移方案取代：`TRAIN.lr_scheduler` 现为 `LinearWarmupCosineAnnealingLR` 签名（learning_rate/warmup_epochs/max_epochs/warmup_start_lr/eta_min，step 单位，无 by_epoch）。
