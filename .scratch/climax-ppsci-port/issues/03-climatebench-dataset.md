# 03: climatebench dataset 进主包

**Type:** task
**Status:** resolved
**Blocked by:** None (can start immediately)

## What to build

`ppsci/data/dataset/climatebench_dataset.py` 提供三元组形态的 ClimateBench 数据集并注册进主包；同 index 样本与 PaddleCFD 版逐位一致。

## 实施要点

- 主包 dataset import xarray 有 6 文件先例（era5climate_dataset.py 等，见 deep-dive-conventions.md），无依赖障碍
- `__getitem__` 返回 `(input_dict, label_dict, weight_dict)`：input_dict = {"x": inp ndarray (T,V,H,W), "lead_times": np.asarray([lead], dtype="float32")}；label_dict = {"tas": out ndarray}；weight_dict = None（atmospheric_dataset.py:1764 先例）
- variables/out_variables 为数据集级常量（dataset 属性保留）；historical 拼接 load_x_y、滑窗跳段、test 末 21 样本等逻辑原样平移（deep-dive-climax-side.md §6）
- Normalize 优先复用 ppsci 既有实现（先查 `ppsci/data/process/` 有无等价 transform；无则把 normalize.py 36 行并入 dataset 文件，Q7 复用优先）
- 注册：`ppsci/data/dataset/__init__.py` 无条件 import + `__all__`
- 源：PaddleCFD feat/climax 的 `examples/climax/{dataset,datamodule,normalize}.py`
- 注意：历史隐含要求——test 目录的 `inputs/outputs_historical.nc` 软链接由使用者自建（沿用 ppcfd README 口径，文档票承担说明）

## Acceptance criteria

- [x] 与 PaddleCFD feat/climax 版同 index 张量逐位一致（max_abs_diff=0.0，历史范式）
- [x] DataLoader 出批形状 x=(1,10,4,32,64)、lead_times=(1,1)、label=(1,1,32,64)
- [x] 主包 `import ppsci` 不因新文件破坏

## Answer

### 做了什么

- 新增 `/home/lkyu/baidu/PaddleScience/ppsci/data/dataset/climatebench_dataset.py`（仅此一文件 + `__init__.py` 注册，均未 commit）：
  - 模块级原样平移 ppcfd `dataset.py` 的 `load_x_y`（含 historical 拼接、pr*86400、rename/transpose/drop）、`input_for_training`/`output_for_training`（滑窗 + skip_historical/len_historical=165）、`split_train_val`。
  - `ClimateBenchDataset(io.Dataset)` 合并了 datamodule 的装配逻辑：partition="train"/"val" 走 train_val 六仿真 + 前 2 个 skip + 随机 9:1 划分；partition="test" 走 test 目录 + skip_historical=True + 末 21 样本 + `get_rmse_normalization()`。构造签名仿 era5climate_dataset（root_dir + input_keys/label_keys 位置参数）。
  - `__getitem__` 返回三元组：`({"x": (T,V,H,W) float32 ndarray, "lead_times": (1,) float32}, {"tas": (Vout,H,W) ndarray}, None)`（atmospheric_dataset.py:1764 先例）。variables/out_variables/lat/lon 为 dataset 属性；另有 `set_normalize`、`get_lat_lon`、`y_normalization`、`inp_transform.mean/.std` 供下游读取。
  - Normalize 未复用 ppsci transform：`ppsci/data/process/transform/preprocess.py` 的 Normalize 是整 dict 逐 key 平铺广播（(C,) 会错位对齐 W 维，且会误伤 lead_times），不等价 → 按票把 normalize.py 36 行实现并入文件（改为 numpy 数组运算，逐位等价）。
  - 相对 ppcfd 的一处必要新增：`seed: Optional[int]` 参数（split 前 `np.random.seed(seed)`）。原因：合并后 train/val 是独立实例各自重放划分，无 seed 则两次 permutation 不同、train/val 集合不一致；ppcfd 单 datamodule 内一次划分无此问题。seed=None 保持原全局随机态行为。
  - 注册：`ppsci/data/dataset/__init__.py` 按字母序插入无条件 import + `__all__` 尾部追加。
- 已过 black 22 / isort（仓库参数 `--multi-line=7 --sl --profile=black`）。

### 关键证据（命令均真实执行）

- 逐位对照（/tmp 脚本，两侧子进程隔离：ppcfd PYTHONPATH=examples/climax、ppsci PYTHONPATH=PaddleScience 根；ppcfd 侧 np.random.seed(42) 后建 datamodule，ppsci 侧 seed=42）：train(4 索引含首/末)/val(4 索引)/test(4 索引含末) 的 x/out/lead_times 全部 max_abs_diff=0.0，lat/lon/variables/长度全等，`RESULT: ALL BIT-EXACT`。分区长度 train=793 / val=89 / test=21 两侧一致。
- test 分区 `y_normalization`（= ppcfd `get_test_clim()`）两侧同为 2.072781562805176。
- DataLoader（batch_size=1，`collate_fn=ppsci.data.process.batch_transform.default_collate_fn`）：x=(1,10,4,32,64) float32、lead_times=(1,1) float32、tas=(1,1,32,64) float32、weight=None，`BATCH SHAPES OK`。
- `cd PaddleScience && PYTHONPATH=. python -c "import ppsci; from ppsci.data.dataset import ClimateBenchDataset"` 正常，模块路径 `ppsci.data.dataset.climatebench_dataset`。

### 遗留事项（给下游票）

- **collate 关键注意**：paddle 3.4 内置 default collate 遇样本第三元素 None 会在 collate 线程抛 TypeError 且主迭代挂死（已实测最小复现）。example/solver 票建 DataLoader 时必须传 `collate_fn=ppsci.data.process.batch_transform.default_collate_fn`（ppsci/data/__init__.py:142-143 对 graph_grid_mesh 数据集即此做法），否则 YAML 配置默认 collate_fn=None 会踩坑。
- 实施中自修一处初版 bug：partition="val" 曾误取 split 的 train 部分（val_len 793≠89），已改为按分区取 x_val/y_val 并复验逐位一致。
- train/val 语义说明：各 partition 实例独立重放"加载→滑窗→划分"全流程（与 ppcfd datamodule 一致逻辑），多实例场景必须同 seed 才能得到同一划分；test 目录的 inputs/outputs_historical.nc 软链接口径归文档票。
