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

