# PaddleScience 深挖：数据呈现、文档骨架、配置与流程惯例精确事实（ClimaX 移植第二轮调研）

- 调研日期：2026-09-26
- 调研对象：`/home/lkyu/baidu/PaddleScience`（develop，HEAD `d57561b1`）
- 定位：补充并修正 `2026-09-26-ppcfd-vs-ppsci-structure.md`；本文所有结论带 file:line 证据，未确证处显式标注。
- 本仓库只读，未修改 PaddleScience 任何文件。

## Q1 数据源直链先例：ppsci 并非一律 bcebos（修正既有报告）

全库 grep `zenodo`（docs/ examples/ ppsci/ jointContribution/ README.md mkdocs.yml）共 16 处命中，气象/地球科学相关直链先例齐全。**结论：ppsci 对"让用户从原始出处下载数据"有充分先例，bcebos 只是主流而非强制。**

### 原始出处直链（无 bcebos）先例清单

| example | 呈现方式 | 证据 |
|---|---|---|
| **UTAE**（遥感） | note 提示块："请在 [PASTIS官网](https://zenodo.org/records/5012942) 中下载PASTIS数据集，并将其放在 `./UTAE/data/` 文件夹下"——**zenodo 直下+自行摆放，零预处理** | `docs/zh/examples/UTAE.md:5`（en 镜像 `docs/en/examples/UTAE.md:5`）；正文再述 `UTAE.md:116` |
| **FourCastNet**（气象） | "请先下载[数据集](https://app.globus.org/file-manager?origin_id=945b3c9e-...)";正文 "该数据集可以从[此处](globus 同链)下载"——**NVIDIA 处理好的 ERA5 走 globus，不走 bcebos**；仅 pdparams 权重在 bcebos | `docs/zh/examples/fourcastnet.md:3`、`:151` |
| **earthformer** | "请先下载以下数据集"：ICAR-ENSO（tianchi.aliyun.com/dataset/98942）+ SEVIR（MIT-AI-Accelerator 教程链） | `docs/zh/examples/earthformer.md:3-7` |
| **transformer4sr** | 命令 tab 内 `git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_*`；权重仍 bcebos | `docs/zh/examples/transformer4sr.md:18-23`、正文 `:94-100` |
| **DGMR** | "开源的英国训练数据集已镜像到 [HuggingFace 数据集](https://huggingface.co/datasets/openclimatefix/nimrod-uk-1km)，用户可以自行下载使用" | `docs/zh/examples/dgmr.md:98` |
| **fuxi** | `# Download sample input data and model weight from https://pan.baidu.com/s/1PDeb-nwUprYtu9AKGnWnNw?pwd=fuxi` ——**百度网盘+提取码先例** | `docs/zh/examples/fuxi.md:19` |

### "原始出处 + bcebos 镜像并存"先例（最贴近 ClimaX 呈现）

- **deepcfd**：wget 两条 bcebos `dataX.pkl/dataY.pkl` 直链后，另注一行"数据集原始下载地址为：<https://zenodo.org/record/3666056/files/DeepCFD.zip?download=1>"（`docs/zh/examples/deepcfd.md:95-104`）。
- **lorenz/rossler/cylinder2d_unsteady_transformer_physx**（Transformer-PhysX 系）：bcebos hdf5 表格下载 + 表下一行"数据集官网为：<https://zenodo.org/record/5148524...>"（`docs/zh/examples/lorenz.md:103`）。
- **climateformer / meteoformer / preformer**（与 ClimaX 同为 ERA5 系）：开头"请下载[ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download)数据集文件"（训练数据指向 Copernicus 原始出处）+ "用于评估的数据集已保存，可通过以下链接下载"（评估子集放 bcebos）+ 权重 bcebos —— **"训练数据原始出处 + 评估子集/权重 bcebos"三段式，就是 ClimaX 可直接套用的模板**。证据：`docs/zh/examples/climateformer.md:3-10`、`docs/zh/examples/meteoformer.md:3-8`、`docs/zh/examples/preformer.md:3-8`。

### 关键否定性事实

- 全库（docs/ examples/ ppsci/ jointContribution/）grep `xesmf|regrid|重采样` **零命中**——ppsci 无任何"下载后本地重采样/再加工"的文档先例。
- 唯一的"数据转换脚本"先例是 `examples/pangu_weather/convert_data.py`（npy→NetCDF，**输出侧可视化转换**，非输入预处理；上游 ref HaxyMoly/Pangu-Weather-ReadyToGo），文档在"结果可视化"节引导 `python convert_data.py`（`docs/zh/examples/pangu_weather.md:75-88`）。ClimaX 的 `regrid_climatebench.py` 若保留，呈现位置可仿此（放可视化/准备节），或仿 deepcfd 附原始出处链接。

**对 ClimateBench(zero) 的直接推论**：照 UTAE/deepcfd 先例，文档可写"请从 [ClimateBench 官网/zenodo] 下载并运行 `regrid_climatebench.py` 重采样到 5.625° 放入 `./data/`"，无需把数据传 bcebos；评估子集与 pdparams 走 bcebos（climateformer 三段式）是更稳的组合。

## Q2 bcebos 数据/checkpoint 贡献流程：无成文流程（既有报告结论精确化）

- `docs/zh/development.md`（972 行）全文 grep `bcebos|上传|网盘|托管|下载链接` 仅命中一处示例 yaml 里的 `pretrained_model_path: "https://paddle-org.bj.bcebos.com/paddlescience/models/viv/viv_pretrained.pdparams"`（`docs/zh/development.md:783`，讲解字段语义，非贡献流程）。
- `docs/zh/development.md` §3"编写文档"（:890-935）只讲 mkdocs 预览与 nav 注册；§4"整理代码并提交"（:949-975）只讲分支/PR 操作。**没有任何"贡献者如何上传数据/权重到 bcebos"的说明。**
- 唯一成文要求：`docs/zh/reproduction.md:197` "文档末尾附上参考论文、参考代码网址、复现训练好的模型参数下载链接"——只要求"有链接"，不规定托管方。
- PR 先例：`git show b2e35d87:docs/zh/examples/climateformer.md` 在**合并时就已含** `datasets/climateformer/2018.h5`、`models/climateformer/climateformer.pdparams` 等完整 bcebos 直链（该文件第 8-10、21 行）——即贡献者 PR 文档直接写最终 bcebos 链接，上传动作发生在 PR 过程中由谁执行**未确认**（推断为 Paddle 官方 bucket、维护者或贡献者代传，仓库内无记载）。

## Q3 climateformer.md 与 fourcastnet.md 骨架；en 镜像同构性

### docs/zh/examples/climateformer.md（268 行）

| 行 | 章节 | 内容形态 |
|---|---|---|
| :1 | `# Climateformer` | 标题 |
| :3-10 | （无名下载段） | ERA5 Copernicus 链接 + bcebos 评估子集三文件直链 |
| :12-22 | `=== "模型训练命令"` / `=== "模型评估命令"` | mkdocs-material tab，各一条 sh 命令 |
| :24 | `## 1. 背景简介` | 两段中文 |
| :30-95 | `## 2. 模型原理`（2.1 编码器 :34 / 2.2 演变器 :44 / 2.3 解码器 :54 / 2.4 模型结构 :64） | 每小节文字+ ``` py linenums="243" title="ppsci/arch/climateformer.py"` + `--8<--\nppsci/arch/climateformer.py:243:277\n--8<--` 源码行级嵌入（示例 :36-41）；2.4 末 bcebos 架构图 figure |
| :97-225 | `## 3. 模型训练` | 3.1 数据集介绍 :99；3.2 模型训练 :105（3.2.1 模型构建 :107 → 3.2.2 约束器构建 :117 → 3.2.3 评估器构建 :137 → 3.2.4 学习率与优化器构建 :157 → 3.2.5 模型训练 :167 → 3.2.6 训练时评估 :177）；3.3 评估模型 :187（3.3.1 :189 / 3.3.2 加载模型并进行评估 :209）。每小节 = 一段讲解 + main.py 对应行区间 `--8<-- examples/climateformer/main.py:a:b` 嵌入 |
| :227-259 | `## 4. 完整代码` | 整文件嵌入四件：`ppsci/data/dataset/era5climate_dataset.py`、`ppsci/arch/climateformer.py`、`examples/climateformer/main.py`、`examples/climateformer/conf/climateformer.yaml`（均 ``` py linenums="1" title=路径） |
| :261-268 | `## 5. 结果展示` | 一段说明 + bcebos 托管 result.png figure |
| （无） | 无"参考资料"章 | climateformer 无 bib/参考节 |

### docs/zh/examples/fourcastnet.md（593 行）

`# FourCastNet` :1 → AI Studio 按钮 :3 → globus 下载提示 :5 → 四个命令 tab（训练/评估/导出/推理）:7-65 → `## 1. 背景简介` :67 → `## 2. 模型原理` :71（2.1 风速 :86 / 2.2 降水量 :111）→ `## 3. 风速模型实现` :129（3.1 数据集介绍 :137；3.2 模型预训练 :160，内含 3.2.1 约束构建 :170 …；3.3 模型微调 :290，内含测试集评估 :326、可视化器构建 :352）→ `## 4. 降水量模型实现` :394（4.1 约束构建 :404 … 4.7 可视化器构建 :517）→ `## 5. 完整代码` :559 → `## 6. 结果展示` :579。含 ACC/RMSE 结果表（既有报告已录）。

### en 镜像同构性：**逐行同构确认**

`docs/en/examples/climateformer.md` 与 zh 同为 **268 行**，全部标题行号逐一相同（1/24/30/34/44/54/64/97/99/105/107/117/137/157/167/177/187/189/209/227/261，实测 grep 对比）；fourcastnet zh 593 行 vs en 595 行（近似）。en 的 3.2.3 译作 "Validator Construction"（zh "评估器构建"），语义一致。

## Q4 四处注册的精确格式（含一处对既有报告的重要修正）

1. **mkdocs.yml nav**（唯一需手改的 nav）：`mkdocs.yml:106` `- 地球科学(AI for Earth Science):` 分组，组内一行一例，气象区相邻行：
   ```yaml
   - 地球科学(AI for Earth Science):
     - KMCast: examples/kmcast.md
     - Extformer-MoE: examples/extformer_moe.md
     - FourCastNet: examples/fourcastnet.md
     ...
     - MeteoFormer: examples/meteoformer.md
     - Preformer: examples/preformer.md
     - ClimateFormer: examples/climateformer.md   # :116
   ```
   （`mkdocs.yml:106-130`，ClimaX 插在 FuXi/UNetFormer 附近即可。）
2. **根 README.md 案例大表——并非强制注册点（修正）**：表头 7 列 `| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |`（`README.md:133`），气象区行如 `| 天气预报 | [FourCastNet 气象预报](https://paddlescience-docs.readthedocs.io/zh-cn/latest//examples/fourcastnet) | 数据驱动 | AFNO | 监督学习 | [ERA5](globus...) | [Paper](...) |`（`README.md:137`）。**但 `grep -ci climateformer README.md` = 0**，meteoformer/preformer 同样缺席——Climateformer PR（b2e35d87）压根没改 README.md。README 大表 87 行 vs docs 表 71 行，本身就不同步。真正被 Climateformer PR 注册的案例总表是 **`docs/zh/index.md`（旧 `docs/index.md`）**：`| 天气预报 | [Climateformer 气候预测](./examples/climateformer.md) | 数据驱动 | Transformer | 监督学习 | [ERA5](https://cds.climate.copernicus.eu/...) | - |`（`docs/zh/index.md:174`）。
3. **docs/zh/api/arch.md**：mkdocstrings 成员清单，`::: ppsci.arch` + `handler: python` + `options: members:` 字母序列表内加一行 `        - Climateformer`（`docs/zh/api/arch.md:12`）。新 dataset 另加 `docs/zh/api/data/dataset.md` 同款一行（b2e35d87 含此文件 +1）。
4. **docs/en 对应位置**：`docs/en/api/arch.md:12` 同款一行。**en 的 examples 导航无需另注册**——mkdocs 用 i18n 插件 `docs_structure: folder`（`mkdocs.yml:230-236`，zh default、en build，另有 nav_translations），docs/en/examples/climateformer.md 按目录镜像自动映射。
5. 官方成文要求只有 mkdocs 一处：`docs/zh/development.md:912` "需要修改 `PaddleScience/mkdocs.yml`。将 `your_example.md` 的相对路径仿照其他案例，添加到列表中"。

**修正结论**：单模型 PR 的实际注册集 = mkdocs.yml nav 1 行 + docs/zh/en/examples/<name>.md 两文件 + api/arch.md（+dataset.md）各 1 行 + docs/zh/index.md 案例表 1 行；README.md 大表为"多数派惯例但非强制"（climateformer/meteoformer/preformer 三个近期气象 PR 均未登记）。

## Q5 examples/fuxi/requirements.txt 与 example 级依赖先例

`examples/fuxi/requirements.txt` 全文 5 行（无版本号）：
```
bottleneck
cartopy
dask
netCDF4
xarray
```
文档引用处：命令 tab 内 `pip install -r requirements.txt`（`docs/zh/examples/fuxi.md:24`、`docs/en/examples/fuxi.md:24`，均位于 `cd examples/fuxi` 之后）。

**example 级 requirements.txt 共 11 个先例（修正"仅 fuxi"的印象）**：ifm、CNN_UTS、fuxi、transolver、synthemol、tadf、moflow、extformer_moe、amgnet、perovskite_solar_cells、smc_reac（`find examples -name requirements.txt` 实测）。ClimaX 的 xarray/netCDF4/xesmf 栈放 `examples/climax/requirements.txt` 完全合规且是多数做法。

## Q6 hydra 配置：ppsci_default schema 与 climateformer.yaml 全文

### psci_default 注册段（ppsci/utils/config.py）

`cs.store(name="ppsci_default", node=SolverConfig)` :380；`TRAIN:train_default` :384；`TRAIN/ema:ema_default` :388；`TRAIN/swa:swa_default` :392；`EVAL:eval_default` :396；`INFER:infer_default` :400；`hydra/job/config/override_dirname/exclude_keys:exclude_keys_default` :444-448（exclude 列表 :402-443）。

**SolverConfig 全字段（:305-328）**：
```python
mode: Union[Literal["train","eval","export","infer"], str] = "train"
output_dir: Optional[str] = None
log_freq: int = 20
seed: int = 42
use_vdl: bool = False
use_tbd: bool = False
wandb_config: Mapping = {}
use_wandb: bool = False
device: Literal["cpu","gpu","xpu","sdaa",None] = None
use_amp: bool = False
amp_level: Literal["O0","O1","O2","OD"] = "O1"
to_static: bool = False
prim: bool = False
log_level: Literal["debug","info","warning","error"] = "info"
trace: bool = False
TRAIN/EVAL/INFER: Optional[...] = None
```
TrainConfig（:87-102）：`epochs=1, iters_per_epoch=20, update_freq=1, save_freq=0, eval_during_train=False, start_eval_epoch=1, eval_freq=1, checkpoint_path=None, pretrained_model_path=None, ema=None, swa=None`（校验器：iters_per_epoch 允许 -1 :114-126；ema/swa 互斥 :166-173）。EMAConfig（:38-41）：`use_ema=False, decay=0.9, avg_freq=1`。SWAConfig（:60-63）：`use_swa=False, avg_freq=1, avg_range=None`。EvalConfig（:188-196）：`pretrained_model_path=None, eval_with_no_grad=False, compute_metric_by_batch=False, batch_size=256`。InferConfig（:206-225）：`pretrained_model_path, export_path="./inference", pdmodel_path, pdiparams_path, onnx_path, device="cpu", engine="native", precision="fp32", ir_optim=True, min_subgraph_size=30, gpu_mem=2000, gpu_id=0, max_batch_size=1024, num_cpu_threads=10, batch_size=256`。

### TRAIN/ema、TRAIN/swa 是否可选：**可选**

126 个 `examples/*/conf/*.yaml` 中 11 个不含 `TRAIN/ema`（即 fuxi/fengwu/pangu_weather/nsfnet×3/CNN_UTS/xrdmatch 系，实测列表：`examples/fuxi/conf/fuxi*.yaml`、`examples/fengwu/conf/fengwu.yaml`、`examples/pangu_weather/conf/pangu_weather.yaml`、`examples/nsfnet/conf/VP_NSFNet{1,2,3}.yaml`、`examples/CNN_UTS/conf/resnet.yaml`、`examples/xrdmatch/conf/xrdmatch.yaml`）。全 examples 中 `use_ema: true` 仅 `examples/adv/conf/adv_cvit.yaml` 一处；`use_swa: true` **零处**。`to_static` 出现于唯一 yaml `examples/euler_beam/conf/euler_beam.yaml:31`（值 false）。

### examples/climateformer/conf/climateformer.yaml 全文 79 行（ClimaX yaml 最小充分参照）

- defaults（:1-9）：`ppsci_default / TRAIN:train_default / TRAIN/ema:ema_default / TRAIN/swa:swa_default / EVAL:eval_default / INFER:infer_default / hydra/job/config/override_dirname/exclude_keys:exclude_keys_default / _self_`
- hydra（:11-24）：`run.dir: outputs_climateformer/${now:%Y-%m-%d}/${now:%H-%M-%S}`；`job.name: ${mode}`；`job.chdir: false`；`callbacks.init_callback._target_: ppsci.utils.callbacks.InitCallback`；`sweep.dir/subdir`
- general（:26-30）：`mode: train`、`seed: 1024`、`output_dir: ${hydra:run.dir}`、`log_freq: 20`
- 业务超参顶层自由命名（:32-48）：`SQ_LEN/IMG_H/IMG_W/USE_SAMPLED_DATA/TRAIN_YEARS/EVAL_YEARS/TRAIN_FILE_PATH/DATA_MEAN_PATH/DATA_STD_PATH/VALID_FILE_PATH`（本地路径直写 `/data/ERA5/`）
- MODEL（:50-58）：`input_keys: ["input"]`、`output_keys: ["output"]`、`shape_in` 嵌套 `${IMG_H}` 引用
- TRAIN（:60-72）：`epochs/save_freq/eval_during_train/eval_freq/lr_scheduler{epochs:${TRAIN.epochs},learning_rate,by_epoch}/batch_size/pretrained_model_path/checkpoint_path`
- EVAL（:74-79）：`pretrained_model_path/compute_metric_by_batch/eval_with_no_grad/batch_size`

入口：`@hydra.main(version_base=None, config_path="./conf", config_name="climateformer.yaml")`（`examples/climateformer/main.py:172`），`def train(cfg)` :22 / `def evaluate(cfg)` :128。

## Q7 python 版本强制力：仅打包元数据，无 CI 矩阵（修正既有报告"CI 矩阵"未知项）

- `pyproject.toml:12` `requires-python = ">=3.8"`；classifiers 3.8/3.9/3.10（:25-28）。
- **`setup.py` 全文无 `python_requires`**（grep 'python' 零命中）。
- **无任何 CI 矩阵/lint job**：`.github/workflows/` 仅 `github-actions-demo.yml`（on push，全部步骤是 echo/ls 的官方模板 demo）。无 travis/circle/azure/jenkins 配置。
- 代码实际语法水平：218 个 ppsci py 中 149 个带 `from __future__ import annotations`（68%）；运行时位未发现 walrus `:=`、`match`、`X | Y` 运行时求值联合、`removeprefix`、`zip(strict=)` 等 3.9+/3.10+ 特性（grep 探测仅命中误报字符串）。类型注解普遍 `Optional[/Union[`（110 文件）。
- **结论**：事实约束 = "声明 3.8 + 写法保守（future import）"。ClimaX 代码若含 3.10+ 语法（如运行时 `X | Y`），要么降级写法要么加 future import；无 CI 会拦，风险只在 review（是否强制**未确认**）。

## Q8 jointContribution 收录判据与 examples 归属

`jointContribution/README.md` 全文一行：
> This directory is mainly used for sample libraries and model reproduction

无任何正式收录判据。子目录 15 个（AI_Disease_Climate/CFDGCN/CHGNet/Deep-Spatio-Temporal/DU_CNN/gencast/graphcast/graphGalerkin/HighResolution/IJCAI_2024/mattersim/PIDeepONet-LBM/PINO/PIRBN/XPINNs；修正既有报告"17 个"）。注意 graphcast **同时存在** `jointContribution/graphcast/`（原始复现形态，带 README.md）与 `examples/graphcast/`（规范化形态）——jointContribution→examples 的迁移路径先例；gencast 仅在 jointContribution。

**examples/ 是单模型+单 example 贡献的正确去处**（确证）：Climateformer PR #1211 = 13 文件 1181 行；Pangu-Weather #1089（96eb5ba6）更小 = 5 文件 459 行（docs 112 + yaml 44 + convert_data 159 + predict 143 + mkdocs 1，无 arch 改动）。约 2600 行的 ClimaX 是 climateformer 规模的 ~2.2 倍，仍在 examples 惯例内。

## Q9 规模参照：git show --stat b2e35d87 完整清单

commit `b2e35d87` "Add Climateformer for climate prediction (#1211)"（squash，作者 Zhang Pu，2025-10-24），**13 files changed, 1181 insertions(+), 3 deletions(-)**：

| 文件 | 行数 | 类别 |
|---|---|---|
| ppsci/arch/climateformer.py | +435 | arch（新增） |
| ppsci/data/dataset/era5climate_dataset.py | +183 | dataset（新增） |
| ppsci/data/dataset/__init__.py | +2 | dataset 注册 |
| ppsci/data/__init__.py | +2/-1 | 版本注释行 |
| ppsci/arch/__init__.py | +4/-1 | arch 注册 |
| examples/climateformer/main.py | +183 | example |
| examples/climateformer/conf/climateformer.yaml | +79 | example |
| examples/climateformer/utils.py | +22 | example |
| docs/zh/examples/climateformer.md | +268 | 文档 |
| docs/index.md（今 docs/zh/index.md） | +1 | 案例总表注册 |
| docs/zh/api/arch.md | +1 | API 注册 |
| docs/zh/api/data/dataset.md | +1 | API 注册 |
| mkdocs.yml | +3/-1 | nav 注册 |

分布：arch 435 / dataset 185 / example 284 / docs+注册 277。**未改根 README.md、未含 docs/en**（en 全量镜像由后续 `2d2d855b` "[Doc] Add en docs (#1261)" 一次性补齐，132 files +29271）——即"先 zh 合入、en 后补"也是可接受路径（是否被允许照搬**未确认**，但成例在）。

## Q10 .pre-commit-config.yaml 全文与 lint 强制

```yaml
repos:
  - repo: https://github.com/PyCQA/isort
    rev: 5.11.5
    hooks:
      - id: isort
        args: ["--multi-line=7", "--sl", "--profile", "black", "--filter-files"]

  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: "v0.0.272"
    hooks:
      - id: ruff

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: a11d9314b22d8f8c7556443875b731ef05965464
    hooks:
      - id: check-merge-conflict
      - id: check-symlinks
      - id: detect-private-key
        files: (?!.*paddle)^.*$
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-case-conflict
      - id: check-yaml
        exclude: "mkdocs.yml|recipe/meta.yaml"
      - id: pretty-format-json
        args: [--autofix]
      - id: requirements-txt-fixer

  - repo: https://github.com/Lucas-C/pre-commit-hooks
    rev: v1.0.1
    hooks:
      - id: forbid-crlf
        files: \.md$
      - id: remove-crlf
        files: \.md$
      - id: forbid-tabs
        files: \.md$
      - id: remove-tabs
        files: \.md$

  - repo: local
    hooks:
      - id: clang-format
        name: clang-format
        description: Format files with ClangFormat
        entry: bash .clang_format.hook -i
        language: system
        files: \.(c|cc|cxx|cpp|cu|h|hpp|hxx|cuh|proto)$

exclude: |
  ^jointContribution/
```

要点：black **22.3.0 无任何 args/行宽覆盖 → 默认 88**（pyproject 亦无 `[tool.black]`，仅 `[tool.ruff] line-length=88, ignore=["E501","E741","E731"]`（`pyproject.toml:53-55`）与 `[tool.isort] profile="black"`（:70-71））；isort `--multi-line=7 --sl`（单行模式）；md 文件强制无 CRLF/无 tab；jointContribution 整体豁免。

**CI lint 强制**：仓库内 workflow 仅 demo（见 Q7），**无 in-repo lint job**；强制力来自 PaddlePaddle 平台侧 code-style 检测——`docs/zh/development.md:53-58` 原文："在 commit 您的代码之前，请务必先执行以下命令安装 pre-commit，否则提交的 PR 会被 code-style 检测到代码未格式化而无法合入"。Climateformer commit body 亦含独立 "* pre-commit 使用pre-commit对代码格式化" 提交记录（b2e35d87 message，贡献者实践佐证）。

---

## 与既有报告差异汇总（供决策）

| # | 既有报告结论 | 本文修正/精确化 |
|---|---|---|
| 1 | "ppsci 惯例全部走 bcebos 直链"（§移植风险 5） | 非一律：zenodo/globus/huggingface/tianchi/pan.baidu/Copernicus 原始出处直链先例各≥1（Q1 表）；climateformer 等 ERA5 系本就采用"训练数据原始出处+评估子集 bcebos"三段式 |
| 2 | "文档四点注册缺一不可（含 README 大表）"（§风险 10） | README 非强制：climateformer/meteoformer/preformer 均不在 README；实际注册集 = mkdocs nav + zh/en md + api/arch(+dataset).md + docs/zh/index.md 表（Q4） |
| 3 | en 镜像"同构"（推测） | 逐行同构实测确认（268 行、标题行号全同）；en 由 i18n 插件 folder 结构自动映射，nav 只注册一次（Q3/Q4） |
| 4 | fuxi 是 example 级依赖先例 | 共 11 个 example 带 requirements.txt（Q5） |
| 5 | python 3.8 兼容"未确认 reviewer 是否强制" | 补充：无 CI 矩阵、setup.py 无 python_requires、无 in-repo lint job；强制力仅 pyproject 元数据 + 平台侧 code-style（Q7/Q10） |
| 6 | jointContribution 含 17 个子项目 | 15 个；graphcast 在 jointContribution 与 examples 双存在（迁移路径先例）（Q8） |
| 7 | bcebos 贡献流程"待定" | 确认无成文流程；reproduction.md:197 只要求文档末尾给出参数下载链接；PR 合并时文档已带最终 bcebos 链接（Q2） |
