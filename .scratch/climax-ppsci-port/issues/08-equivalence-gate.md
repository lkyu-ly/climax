# 08: 端到端等价门（TDD 缝）

**Type:** task
**Status:** resolved
**Blocked by:** 02, 03

## What to build

`/home/lkyu/baidu/CLIMAX/tools/compare_ppsci_ppcfd.py`（本地验证资产，**不进上游**）：同源权重下 ppsci 侧与 PaddleCFD 侧的三重对照工具，全过并记录数字。

## 实施要点

（TDD 语义：脚本先行，02/03 实施每步可跑此门；本票完成 = 三对照全过）

- 对照一（数据管线）：两侧 dataset 同 index 张量 max_abs_diff = 0.0（历史范式）
- 对照二（前向）：同源初始权重（climax_initial.pdparams，strict 直载不走清洗路径）eval 前向 mean_abs 目标 < 1e-6（代码平移应近逐位；与 torch 的 2.24e-08 是另一层，不必重测）
- 对照三（单批 loss）：w_mse 单批值与 ppcfd 版一致（06 票完成后纳入）
- 两侧 import 隔离：ppsci 侧 `PYTHONPATH=/home/lkyu/baidu/PaddleScience`，ppcfd 侧 `PYTHONPATH=/home/lkyu/baidu/PaddleCFD`，互不污染（同进程不可行则分子进程）
- 环境：`/home/lkyu/miniconda3/envs/paddletorch/bin/python`；数据与权重路径见 map.md Notes

## Acceptance criteria

- [x] 三对照全过，数字记录于本票 Answer（供 10/15/PR message 引用）
- [x] 工具留在 CLIMAX/tools/（不进 PaddleScience 仓）

## Answer

### 做了什么

新增 `/home/lkyu/baidu/CLIMAX/tools/compare_ppsci_ppcfd.py`（单文件，black 22 / isort 过；不进 PaddleScience 仓）：驱动器内嵌两侧子进程脚本（materialize 到 `--workdir`，默认 /tmp/climax08_gate），两侧 import 隔离——ppsci 侧 `PYTHONPATH=PaddleScience/examples/climax:PaddleScience`，ppcfd 侧 `PYTHONPATH=PaddleCFD/examples/climax:PaddleCFD`，均 `paddle.set_device("cpu")` + `CUDA_VISIBLE_DEVICES=""` 保证逐位可复现，各自 dump JSON 后驱动器对照并写 report.json，全过 exit 0。

- 对照一：ppsci 侧 ClimateBenchDataset(train/val 各自 seed=42，val/test 复用 train 的 transform，镜像 datamodule set_normalize) vs ppcfd 侧 `ClimateBenchDataModule`（np.random.seed(42)）；每分区 4 索引（含首末 [0,1,n-2,n-1]）比 x/y/lead_times。
- 对照二：两侧同参构造 ClimaXClimateBench（climax.yaml MODEL 15 键），`climax_initial.pdparams` 剥 "net." 前缀 strict 直载（不走 load_mae_weights 清洗路径），eval 模式两组输入：RandomState(42) 合成 x=(1,10,4,32,64)/lead=0.0 与真实 test 首样本，比 preds mean_abs/max_abs。
- 对照三：真实前向输出喂两侧 loss 路径——ppsci 侧 `utils.mse_loss` + `utils.make_w_mse_loss(lat)`；ppcfd 侧真 module 路径 `ClimateProjectionModule.training_step`（net.forward + metrics.mse）与 `validation_step`（net.evaluate + lat_weighted_mse_val），identity denorm、clim=test 分区 y_normalization，比 train_mse/w_mse/clim。

### 关键数字（验收实录，`python tools/compare_ppsci_ppcfd.py` 真实执行，RESULT: ALL GATES PASSED，exit 0）

- 对照一：train len 793 vs 793、val 89 vs 89、test 21 vs 21；三分区 4 索引 x/y/lead_times 全部 **max_abs_diff = 0.0**（03 票结论工具化复验通过）
- 对照二：两侧 strict_load 均 ckpt 123 / model 123 键、missing=[] unexpected=[] shape_mismatch=[]；preds[synthetic] 与 preds[real] 均 **mean_abs_diff = 0.0、max_abs_diff = 0.0**（逐位一致，远优于 <1e-6 目标；02 票"平移应近逐位"的预期坐实）
- 对照三：train_mse = 6.516786098480225、w_mse = 4.55972957611084、clim = 2.072781562805176，三项两侧 **abs_diff = 0.0**
- 运行产物：/tmp/climax08_gate/{out_ppsci,out_ppcfd,report}.json + run.log（含两侧完整输出）

### 遗留

- 无。工具可重复执行（每次重新生成侧脚本）；10/15 票与 PR message 直接引用上述数字。
