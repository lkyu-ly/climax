ClimaX迁移实验记录

## 2026-09-19

### 前期调研与建仓

- 读调研报告，范围锁定：ClimaX 主干 + ClimateBench + 5.625° + 单输出变量 tas（输入 CO2/SO2/CH4/BC），不做 CMIP6 预训练 / ERA5 全球 / 区域预报
- 读五个历史模型实验记录（mpp/phye2e/poseidon/gfno/prose-fd），复用六阶段流程与受控对齐方法（共享初始权重 + 固定 batch 顺序）
- 验证本地源码 research/climax 完整（HEAD 6d5d354，与调研报告记录一致，工作区干净）
- 新仓 /home/lkyu/baidu/CLIMAX（远程 lkyu-ly/climax）：
  - commit1 88a8243：抽取扁平化 climax_torch/（climax 包 12 个 py：主干 arch/parallelpatchembed + climate_projection 链路 + utils 三个，加配置/重采样脚本/environment.yml/tests/pyproject/LICENSE），逐文件 diff 校验与源一致
  - commit2 64c6cbf：pyproject `where=["src"]`→`["."]`、删 readme 行；.gitignore 追加 .spec-workflow/。Python 源码零修改（内部 import 全为 climax.* 绝对路径，依赖闭包 grep 验证）
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
- timm 1.0.24 实测：Block/PatchEmbed/trunc_normal_ 三件套可用；timm.models.layers.helpers 已移除 → parallelpatchembed.py 改用 timm.layers.helpers（1 行）

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

命令（去掉全部 limit_*_batches，max_epochs=5，nohup 后台跑，全程 22 分 35 秒，RTX 4060 Ti FP32 单卡 batch_size=1）：

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

产物：exps/torch_baseline/{checkpoints/epoch_004.ckpt,last.ckpt, logs/version_0/events.out.tfevents.*}，日志 exps/torch_baseline_train.log。结论：正式基线短训完成，loss 与 val 指标较初始大幅收敛（val/w_mse 5 epoch 内 0.25→0.21，中期有波动），test w_rmse 0.369 K 量级，可作为 torch 侧基线锚点。

### paddle 侧移植（writing-plans + subagent-driven-development，计划：docs/superpowers/plans/2026-09-20-climax-paddle-port.md）

paconvert v3.3.1 转换（commit 281e56e 为未修改基准，便于看 diff）：307 个 torch API 自动转 291 个（94.79%），16 处手动（timm 7 / Lightning 3 / torchvision Normalize 3 / _LRScheduler 1 / xarray 误报 2）。Lightning 经确认彻底舍弃（torch 侧仅作对照基线），paddle 侧普通类 + 自建训练循环。

按任务记录（审查均通过，含 2 轮 fix）：

- timm_paddle 最小闭包（timm 1.0.24 函数级抽取，244bfd1）：state_dict 键名契约逐键一致，Block 前向自检 max_abs 4.77e-07。坑：paddle 3.4 的 `Tensor.mul_/add_` 不收 Python float 标量 → trunc_normal_ 重写为 functional；本环境 paddle sdpa 数值异常（与手工分支差 ~2.4，torch sdpa 为 3.6e-07）→ fused 分支默认关闭走手工分支
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



