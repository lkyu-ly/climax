ClimaX迁移实验记录

## 2026-09-19

### 前期调研与建仓

- 读调研报告，范围锁定：ClimaX 主干 + ClimateBench + 5.625° + 单输出变量 tas（输入 CO2/SO2/CH4/BC），不做 CMIP6 预训练 / ERA5 全球 / 区域预报
- 读五个历史模型实验记录（mpp/phye2e/poseidon/gfno/prose-fd），复用六阶段流程与受控对齐方法（共享初始权重 + 固定 batch 顺序）
- 验证本地源码 research/climax 完整（HEAD 6d5d354，与调研报告记录一致，工作区干净）
- 新仓 /home/lkyu/baidu/CLIMAX（远程 lkyu-ly/climax）：
  - commit1 88a8243：抽取扁平化 climax_torch/（climax 包 12 个 py：主干 arch/parallelpatchembed + climate_projection 链路 + utils 三个，加配置/重采样脚本/environment.yml/tests/pyproject/LICENSE），逐文件 diff 校验与源一致
  - commit2 64c6cbf：pyproject `where=["src"]`→`["."]`、删 readme 行；.gitignore 追加 .spec-workflow/。Python 源码零修改（内部 import 全为 climax.\* 绝对路径，依赖闭包 grep 验证）
  - 不带：pretrain/global_forecast/regional_forecast、utils/data_utils.py（climate_projection 链路零引用）、snakemake_configs、其余 configs 与数据脚本、docs 等

## 2026-09-20

### 数据与权重（先做最小版本调研）

调研结论：

- ClimateBench v1.0（zenodo 7064308，含历次修复）共 3 个文件，其中 CMIP6.zip（1.5G，关联原始 CMIP6 文件）ClimaX 用不到，只下 train_val.tar.gz（839MB）+ test.tar.gz（74MB）
- 无现成 5.625° 版本可下载（WeatherBench 的 5.625° 是 ERA5 另一数据集），需本地重采样
- 权重仅两个：5.625deg.ckpt（412MB）即任务匹配的最小选择

执行：

- 下载到位，字节数与官方一致：dataset/climatebench/raw/{train_val,test}.tar.gz、models/climax_torch/5.625deg.ckpt
- tar 内容为平铺结构，按 datamodule 期望的 root_dir/{train_val,test} 分目录解压（test=ssp245 保留情景）
- 原始结构实测：96×144 网格（约 1.9°×2.5°，NorESM2）；inputs 4 变量（CO2/CH4 为广播成场的全球量，ClimateBench 累积口径）；outputs 4 变量 × 3 集合成员；historical 165 年（1850-2014），ssp 至 2100
- 重采样：新建 conda 环境 climdat（python3.10 + xesmf + netcdf4 + click），跑原版 regrid_climatebench.py 生成 dataset/climatebench/5.625deg/（32×64，95MB，6 情景 train_val + ssp245 test）
- 踩坑：conda 本地 pkgs 缓存连续损坏（pandas、numpy 先后报 CondaVerificationError），`conda clean --packages` 后重建成功
- 产物验证：inputs (165,32,64)、tas 距平范围 -5.2~8.3K、lat/lon 坐标 5.625° 网格
- .gitignore 追加 dataset/、models/（改动留在工作区未提交）

### 模型与数据结构讲解

- 交付完整讲解：任务背景 / 数据结构（滑窗样本构造、归一化口径）/ 模型结构（图块嵌入-变量聚合-时空 Transformer-线性头）/ 前向逐步形状流转 / 训练与权重更新（预训练加载清洗、freeze_encoder、AdamW 双参数组、预热+余弦退火）
- 讲解确立术语规范：公认英文缩写可保留，其余用官方中文译名，禁止自造缩写

### torch 侧依赖确定（paddletorch 环境，uv pip 安装）

- 缺失安装：pytorch-lightning==1.9.5、jsonargparse[signatures]、netCDF4（torchmetrics 被连带升到 1.9.0）
- 踩坑：环境中原有 wandb 0.25.1 在 protobuf 7.36（随 paddle 3.4 升级而来）下 import 即崩，连带 lightning logger 初始化失败；升级 wandb 0.30.0 修复（不绕过、修好为止）
- timm 1.0.24 实测：Block/PatchEmbed/trunc*normal* 三件套可用；timm.models.layers.helpers 已移除 → parallelpatchembed.py 改用 timm.layers.helpers（1 行）

### torch 训练基线跑通（limit_batches=1 完整循环，问题-修复循环）

按序遇到的坑：

1. jsonargparse 4.18 的 RichProgressBar 校验 bug（theme 参数默认实例过不了校验）→ 换原仓库 pin 的 4.19.0
2. `Unrecognized arguments: fit`：我误加了子命令。根因：train.py 传 run=False，LightningCLI 不注册子命令（setup_parser(add_subcommands=run)），usage.md 原始命令本就没有 fit。之前"4.19/4.25 子命令失效"全是此误用的同因假象
3. timm 1.0 Block 签名变更：drop→proj_drop → arch.py 改 1 行
4. numpy 新版移除 np.float 别名 → pos_embed.py 改 dtype=float（1 行）
5. test 目录缺 inputs/outputs_historical.nc：load_x_y 的 ssp 分支需拼接历史期（usage.md 未提此隐含要求）→ 从 train_val 软链接补齐
6. fast_dev_run 与 train.py 里 ckpt_path='best' 不兼容（不保存检查点）→ 改用 max_epochs=1 + limit_train/val/test_batches=1，顺带把检查点保存-恢复链路也验证了
7. yaml 的 `${oc.env:AMLT_OUTPUT_DIR,/home/tungnd/...}` 插值先于 CLI 覆盖展开，PermissionError → 运行时设 AMLT_OUTPUT_DIR 环境变量（零文件改动）
8. 新版 rich 与 lightning 1.9.5 的 RichProgressBar 运行时不兼容（clear_live 空栈崩溃）→ yaml 删除 RichProgressBar 回调（仅进度条美化）

torch 侧累计源码适配 3 处 + 配置 1 处：parallelpatchembed.py（timm 路径）、arch.py（proj_drop）、pos_embed.py（np.float）、climate_projection.yaml（删 RichProgressBar）

最终跑通命令：

```bash
cd /home/lkyu/baidu/CLIMAX/climax_torch
AMLT_OUTPUT_DIR=/home/lkyu/baidu/CLIMAX/exps/torch_baseline_dryrun \
PYTHONPATH=. /home/lkyu/miniconda3/envs/paddletorch/bin/python \
  climax/climate_projection/train.py --config configs/climate_projection.yaml \
  --trainer.max_epochs=1 --trainer.limit_train_batches=1 --trainer.limit_val_batches=1 \
  --trainer.limit_test_batches=1 --trainer.precision=32 --trainer.strategy=auto \
  --trainer.default_root_dir=/home/lkyu/baidu/CLIMAX/exps/torch_baseline_dryrun \
  --data.root_dir=/home/lkyu/baidu/CLIMAX/dataset/climatebench/5.625deg \
  --model.pretrained_path=/home/lkyu/baidu/CLIMAX/models/climax_torch/5.625deg.ckpt
```

关键 log（RTX 4060 Ti，FP32 单卡，数据/权重加载 + 训练/验证/测试各 1 批，全程约 1 分钟）：

```
Global seed set to 42
FrozenMappingWarningOnValuesAccess({'time': 251, ...}) ssp585 / historical / hist-GHG / hist-aer / ssp245（六情景全部加载）
Epoch 0: train/loss=4.420, train/tas=4.420
Trainer.fit stopped: max_epochs=1 reached.
Restoring states from the checkpoint path at .../checkpoints/epoch_000.ckpt
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ test/w_mse    ┃ 4.5597243309021   ┃
┃ test/w_rmse   ┃ 2.1353511810302734┃
┃ w_nrmses_tas  ┃ 1.0301862955093384┃
┃ w_nrmseg_tas  ┃ 0.8652374148368835┃
┗━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━┛
```

结论：torch 侧训练基线跑通（只训 1 步，数值为预训练主干+随机新头的初始水平，量级合理），检查点保存/恢复与纬度加权指标链路全部验证。

### torch 正式基线短训（5 epoch 全量批次）

命令（去掉全部 limit\_\*\_batches，max_epochs=5，nohup 后台跑，全程 22 分 35 秒，RTX 4060 Ti FP32 单卡 batch_size=1）：

```bash
cd /home/lkyu/baidu/CLIMAX/climax_torch
AMLT_OUTPUT_DIR=/home/lkyu/baidu/CLIMAX/exps/torch_baseline PYTHONPATH=. nohup \
  /home/lkyu/miniconda3/envs/paddletorch/bin/python climax/climate_projection/train.py \
  --config configs/climate_projection.yaml --trainer.max_epochs=5 --trainer.precision=32 \
  --trainer.strategy=auto --trainer.default_root_dir=/home/lkyu/baidu/CLIMAX/exps/torch_baseline \
  --data.root_dir=/home/lkyu/baidu/CLIMAX/dataset/climatebench/5.625deg \
  --model.pretrained_path=/home/lkyu/baidu/CLIMAX/models/climax_torch/5.625deg.ckpt \
  > /home/lkyu/baidu/CLIMAX/exps/torch_baseline_train.log 2>&1 &
```

实际规模：每 epoch 进度条 882 步 = 793 训练 + 89 验证，训练 4:27/epoch + 验证 0:11/epoch（约 3.3 it/s），远低于 8 分钟阈值，未降级、跑满 5 epoch。训练结束用 best 检查点 epoch_004.ckpt 跑 test（21 批）。

每 epoch 末尾 train/loss（进度条末值，单批）与验证指标（tensorboard events，val 验证不打控制台表）：

```
epoch  train/loss  val/w_mse  val/w_rmse
0      0.145       0.2503     0.4787
1      0.300       0.2210     0.4483
2      0.252       0.3964     0.5750
3      0.212       0.2287     0.4548
4      0.408       0.2118     0.4383   <- best，保存 epoch_004.ckpt
```

（train/loss 单批波动大，每 50 步采样的 epoch 均值：0.313 / 0.379 / 0.267 / 0.267 / 0.288；对照 dryrun 初始 4.42，训练后稳定在 0.2 量级）

最终 test 指标（epoch_004 检查点）：

```
test/w_mse        0.1390    （dryrun 初始 4.560）
test/w_rmse       0.3687    （dryrun 初始 2.135）
test/w_nrmse_tas  0.3408
test/w_nrmses_tas 0.1779
test/w_nrmseg_tas 0.0326
```

产物：exps/torch_baseline/{checkpoints/epoch_004.ckpt,last.ckpt, logs/version_0/events.out.tfevents.\*}，日志 exps/torch_baseline_train.log。结论：正式基线短训完成，loss 与 val 指标较初始大幅收敛（val/w_mse 5 epoch 内 0.25→0.21，中期有波动），test w_rmse 0.369 K 量级，可作为 torch 侧基线锚点。

### paddle 侧移植（writing-plans + subagent-driven-development，计划：docs/superpowers/plans/2026-09-20-climax-paddle-port.md）

paconvert v3.3.1 转换（commit 281e56e 为未修改基准，便于看 diff）：307 个 torch API 自动转 291 个（94.79%），16 处手动（timm 7 / Lightning 3 / torchvision Normalize 3 / \_LRScheduler 1 / xarray 误报 2）。Lightning 经确认彻底舍弃（torch 侧仅作对照基线），paddle 侧普通类 + 自建训练循环。

按任务记录（审查均通过，含 2 轮 fix）：

- timm*paddle 最小闭包（timm 1.0.24 函数级抽取，244bfd1）：state_dict 键名契约逐键一致，Block 前向自检 max_abs 4.77e-07。坑：paddle 3.4 的 `Tensor.mul*/add*` 不收 Python float 标量 → trunc_normal* 重写为 functional；本环境 paddle sdpa 数值异常（与手工分支差 ~2.4，torch sdpa 为 3.6e-07）→ fused 分支默认关闭走手工分支
- 模型层接线 + utils（87e1fd7/b39caad/750647f）：参数量 torch=paddle=111,831,040；调度器 601 步逐点 abs_err=0.0；Normalize 与 torchvision 逐点 0 误差。坑：`nn.LayerList` 不接受 0-dim Tensor 索引（torch ModuleList 接受）→ `int(id)`；paddle 组字典 `learning_rate` 仅构造期读作缩放因子 → 调度器 step() 内 set_lr 受控传播；numpy 广播把 `(C,)` 对到尾维 → Normalize 按 torchvision 语义 reshape 到 dim -3
- 数据侧（8fe1b57）：dataset/datamodule 去 Lightning/torchvision，与 torch 侧同 index 张量 max_abs_diff=0.0（数据管线逐位一致）
- module+train 自建循环（9b52570/2c397d3）：dryrun（真实数据）EXIT=0，测试指标表与 torch 基线逐字符一致；paddle `set_state_dict` 无 strict 形参、默认宽松、返回 (missing, unexpected)
- 最小跑通验收：随机头初始 test/w_mse=4.545 vs torch 初始 4.560（该值由目标场统计主导，两侧近同值交叉验证数据管线与 eval 行为一致）
- 权重转换（f0eee81，tools/ 两脚本）：123 键全对照（键集合 diff 0/0、形状 123/123），32 键转置（timm_paddle Block 内 qkv/proj/fc1/fc2 weight，键名规则驱动，方阵漏转由前向数值对比背书——反事实测试分离度 5.1e+01）+ 91 键直拷；实测 `paddle.compat.nn.MultiheadAttention/Linear` 键名布局与 torch 完全一致免转置；seed 42 导出 bit 级可重建
- 前向对齐（e6cddfc，tools/compare_forward.py）：两侧 strict 加载同源完整初始权重（含随机新头，绝不可走 load_mae_weights 清洗路径——其会删 token_embeds/head 键），RandomState(42) 固定输入，eval 前向：**mean_abs_error=2.09e-08、mean_rel_error=3.01e-06、max_abs=1.01e-07**（验收 1e-5/1e-6 量级均过，优于历史五模型水准），12 层中间量 trace 全程 1e-6~1e-5 平稳无发散，残差归因 MHA float32 归约与 8 层堆叠的已知无害噪声

### paddle 侧 5 epoch 短训对照（对齐通过）

train.py 增加 `--model.init_state_path`（d27cdb1）：完整初始状态直载（strict、绕过清洗路径，与 torch 基线共享主干+头权重）。

```bash
cd /home/lkyu/baidu/CLIMAX/climax_paddle && \
PYTHONPATH=/home/lkyu/baidu/CLIMAX/climax_paddle PYTHONUNBUFFERED=1 nohup \
/home/lkyu/miniconda3/envs/paddletorch/bin/python climax/climate_projection/train.py \
configs/climate_projection.yaml \
--model.init_state_path=/home/lkyu/baidu/CLIMAX/models/climax_paddle/climax_initial.pdparams \
> /home/lkyu/baidu/CLIMAX/exps/paddle_baseline_train.log 2>&1
```

（同款口径：seed 42、FP32、batch_size=1、5 epoch 全量 793+89 批/epoch、单卡，无崩溃无修复）

val/w_mse 对照：

```
epoch   torch    paddle   相对偏差
0       0.2503   0.2601   +3.9%
1       0.2210   0.2278   +3.1%
2       0.3964   0.3058   -22.9%（两侧同现中期波动）
3       0.2287   0.2430   +6.3%
4       0.2118   0.2010   -5.1%（两侧同取最优，best=epoch4）
```

最终 test 对照（best checkpoint）：

```
指标             torch       paddle      相对偏差
test/w_mse       0.13900     0.13394     -3.6%
test/w_rmse      0.36871     0.36180     -1.9%
test/w_nrmse_tas 0.34081     0.34945     +2.5%
test/w_nrmses    0.17788     0.17455     -1.9%
test/w_nrmseg    0.03258     0.03498     +7.3%
```

总耗时 23m13s vs torch 22m35s（+2.8%，预期 30-50% 减速未出现）。epoch 级 train/loss 点值偏差 ±20~45%（初始权重同源但 shuffle 顺序与 dropout 随机流不同，属预期，简报已预判）；50 步采样 epoch 均值两侧同落 0.2~0.4 量级带、末 epoch 几乎相同（0.288 vs 0.2872）。

判定：**对齐通过**——最终 test 五项相对偏差全部 <10%（最大 7.3%），曲线量级与收敛趋势一致，无需受控对齐。产物 exps/paddle_baseline/{best,last}.pdparams（各 447MB）。

## 2026-09-21

### 结构调整

- timm_paddle → climax_paddle/thirdparty/timm（git rename，import 与注释同步）；timm 抽取的 torch 源删除（可由 timm 1.0.24 官方源 + 计划文档抽取表重建，git 历史留档）；paconvert 日志清理
- paddle 侧零 torch 依赖双重验证：静态 grep 无任何 torch/torchvision/lightning/timm import；运行时 meta_path 阻断 torch 全家后完整 import 链 + 模型构造（参数量 111,831,040）通过；改名后单批 dryrun 初始 test/w_rmse 与 torch 逐位一致（2.1354）

### CINN 对齐（第三次测试：CINN 前向/训练 + 加速比，计划 docs/superpowers/plans/2026-09-21-climax-cinn.md）

接入方式（踩坑手册坑四模式，commit fd337c4）：`CLIMAX_USE_CINN` 单开关（默认 0，保护基线）——开 = FLAGS 三件套 true（import paddle 前设置）+ `paddle.jit.to_static(net, full_graph=True)`（权重加载后包裹）；`CLIMAX_FULL_GRAPH` 备降级未动用（full_graph=True 一次通过）。paddle 3.4 实测：to_static 的 full_graph 移至 kwargs、backend 默认即 CINN。手册九坑预检：六坑免疫（trace 路径无兼容层方法、forward 显式参数、tuple 返回、无 einops、无复数、数据无生成线程）；五个待实测点（einsum/repeat_interleave/无参 squeeze/lru_cache/compat MHA 训练态）全部未阻塞；唯一实际修复 = forward_encoder 的 ModuleList 0-dim Tensor 索引改 Python 侧 var_map 查找（数值严格等价，与此前 int(id) 修复同族）。

前向（commit 1899dc2，compare_forward 增加 CINN GPU 分支与 robust rel 门禁）——三组对照（torch 恒 CPU）：

```
paddle 侧          mean_abs    mean_rel(raw)  mean_rel(robust*)
CPU 动态(基线复现)  2.087e-08   3.012e-06      1.360e-06
GPU 动态           2.194e-08   1.261e-05      1.368e-06
GPU CINN           2.237e-08   1.093e-05      1.442e-06   <- 验收 PASS(<1e-5)
```

eval 前向确证真实走 CINN（日志 `add_cinn_pass.cc:334 Compiling subgraph with CINN backend`，eval 图编译约 34s，CINN 输出三次运行 md5 一致）。mean_rel 跨 GPU 升高归因近零点 ulp 噪声（|torch|<1e-3 的 23/2048 点贡献 ~89% rel 质量，分桶 mean_abs 均匀 ~2e-08）——据此验收门禁增补 robust rel（分母 |torch|≥1e-3；mean_abs 主门禁不变）。

训练（commit 61f9d41，train.py 增 per-step 计时，两侧同口径）——CINN 5 epoch 全量（同源初始权重、seed 42、FP32、batch_size=1）：最终 test 五项 vs torch 基线最大偏差 5.06%（w_mse 0.13416 vs 0.13900、w_rmse 0.36218 vs 0.36871）、vs paddle 动态基线最大 2.14%，均 <10% 通过；val/w_mse 轨迹同形（两侧同 epoch-2 尖峰、同 epoch-4 最优 0.2047/0.2010），epoch-2 尖峰幅度放大（0.4094 vs 0.3058）归因 batch_size=1 下融合改变浮点求和顺序的轨迹分叉（两次独立动态图运行 epoch-1 偏差仅 0.18%、epoch-0 完全可复现，佐证分叉源于 CINN 数值路径而非随机性）。

加速比（确定性数据）：

```
主指标（第 2 epoch 纯训练段墙钟，剔除首次编译与 GPU 预热）：
  动态图 267.69s -> CINN 253.78s = +5.20%
辅指标：稳态步中位数 +2.77%；50 步块均值中位数 +3.65%（CINN 4 个稳态 epoch 方差 <1%）
首次编译 90.1s 全程仅 1 次（训练图）；epoch0 因编译反慢 83.6s
5 epoch 规模累计仍略亏，约第 7 epoch 起净收益（短跑看稳态吞吐、长训练才净赚）
```

结论：CINN 全流程（开关/前向/训练）对齐通过，加速 +5.2%（历史 to_static 模式区间 6.3%~28.1% 的下沿，如实报告）。产物：exps/{cinn_smoke,speed_dynamic,cinn_baseline}*、models 侧无新增。

## 2026-09-26

### PaddleScience 移植（reviewer 意见转仓；wayfinder 地图 + 16 票 + 5 波次 subagent，全程不 commit 等用户审查）

调研先行：结构对照报告 + 三路深挖（solver 管线 / ClimaX 侧 / 惯例，`docs/superpowers/research/2026-09-26-*` 四份）；grilling 九项决策（Q1-Q9，地图 `.scratch/climax-ppsci-port/`）。

落地（PaddleScience `feat/climax`，自 develop `d57561b1`，官方 upstream 无前进）：**34 文件 +3016 行**——`ppsci/arch/climax/` 706（ClimaX/ClimaXClimateBench 改 base.Arch dict 进出，loss/evaluate 移出，lead_times 内部 flatten [B,1]→[B]）；`ppsci/arch/paddle_timm/` 542（vendored，SPDX 署名，weight_init 换 `ppsci.utils.initializer.trunc_normal_` 复用）；`ppsci/data/dataset/climatebench_dataset.py` 334（三元组形态，seed 参数保划分可重放）；`ppsci/optimizer/lr_scheduler.py` +81（见下）；`examples/climax/` 758（main.py hydra mode=train/eval + Solver：FunctionalLoss mse/w_mse、FunctionalMetric 指标族、AdamW 双参数组直构、权重 Solver 前手动加载、CINN FLAGS+to_static 联动、validator Val 置末位使 best 监控 Val/w_mse 对齐原版）；双语文档 578 + 注册 11（mkdocs nav/api×5/index×2）。

等价门（tools/compare_ppsci_ppcfd.py，08 票）：数据三分区/同源权重前向/单样本 loss 三层 vs ppcfd 版**全部 diff=0.0 逐位一致**。

踩坑一例（10 票判负→16 票修复）：1 epoch 验证 epoch-0 val/w_mse=1.7675 vs 锚点 0.2601（+580%）。根因闭环：ClimaX 原版 warmup_epochs=60/max_epochs=600 配 interval="step" 是 **step 单位**（60 步升满 lr、SGDR 重启），内置 Cosine by_epoch 两态均按 epoch 解释（true：epoch-0 lr 恒 1e-8 实测；false：47580 步升满）→ 头未训练、val 停在初始权重水平。深挖报告"内置同构"结论系公式形态对比、漏时间单位换算（调研误判）。修复 F1（用户拍板）：`LinearWarmupCosineAnnealingLR` 平移入主包（paddle LRScheduler 子类化，递推逐字），**601 步逐点 abs_err=0.0**（含边界/重启点）。

终验（1 epoch 全量双模式，793+89 批+test，Q2 口径"与既往无差异即等价"）：动态 val/w_mse **0.26845**（锚点 0.2601，+3.2%<10% 过门）、test/w_mse 0.12991、loss 4.41→0.71 单调降；CINN 编译 1 次 53.7s、与动态舍入级一致（793 点仅 1 点差 1e-5）、**本负载稳态无加速**（0.319 vs 0.318 s/iter，如实呈现；+5.2% 属 ppcfd 自建循环口径不进 PR 正文）。

终审（15 票）修复 8 处后全绿：内部参照字样清除（utils/main/lr_scheduler docstring 改 URL 署名式）、双语文档过时口径改写（3.2.4 调度器节、CINN 节）、**"within 5.1%" 对源修正为 7.3%**（5.1% 实为 val best 偏差，test 五项最大 7.3%，PR 与 docs 统一）、pycache 残留清理、lr_scheduler 的 api 文档注册补齐。pre-commit 全钩绿（black 88/isort/ruff，仅 regrid 一文件需重排）。

PR message 草稿：`/home/lkyu/baidu/test/ppsci-climax-pr.md`（三段式，待用户润色建 PR）。等价工具留 `tools/compare_ppsci_ppcfd.py`（不进上游）。

