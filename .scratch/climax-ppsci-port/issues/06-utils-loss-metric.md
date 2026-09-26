# 06: utils.py：FunctionalLoss 与 FunctionalMetric

**Type:** task
**Status:** resolved
**Blocked by:** 03

## What to build

examples/climax/utils.py 提供与 ppcfd 版数值逐字一致的 w_mse loss 与 lat_weighted 指标族（FunctionalLoss/FunctionalMetric 形态）。

## 实施要点

- loss：`ppsci.loss.FunctionalLoss(w_mse_func)`，func 签名 `(output_dict, label_dict) -> Dict[str, Tensor]`（earthformer `earthformer_enso_train.py:61` 先例）；w_mse 公式逐字平移（权重 cos(lat)/mean(cos(lat))）
- metric：`ppsci.metric.FunctionalMetric(expr)`，签名同上（earthformer `enso_metric.py:90-127` 先例）；平移 PaddleCFD `examples/climax/metrics.py` 的 lat_weighted_* 全族（逐批返回 dict）
- lat 取自 dataset 的 `inputs_<simu>.nc` 坐标；test_clim = dataset_test.y_normalization；denorm 本任务恒等——常量在 main.py 装配时以闭包/参数注入（deep-dive-climax-side.md §4-5）
- keep_batch 必须与 Solver `compute_metric_by_batch` 一致否则 raise（solver.py:278-302）；跨批聚合（原版算术平均）在 FunctionalMetric 内对齐
- 内置 `LatitudeWeightedRMSE` 不用：其权重是 linspace 近似非真实 lat（rmse.py:129-134，数值不等价）

## Acceptance criteria

- [x] 单批 metric/loss 输出与 ppcfd 版同值（08 等价门覆盖）
- [x] metric 键名与既往日志形态一致（`w_mse_tas` 等）
- [x] 公式与 ppcfd 版逐行核对（无"顺手优化"）

## Answer

### 做了什么

新增 `/home/lkyu/baidu/PaddleScience/examples/climax/utils.py`（单文件，black 22 / isort 通过）：

- `mse_loss(output_dict, label_dict, *args) -> {"mse": t}`：训练损失，逐字平移 `metrics.mse(...)["loss"]`（`(pred-y)**2` 对 (B,V,H,W) 先 mean 变量维再全均值；`dim=`→`axis=` 别名同一 op）。键名不用 "loss"——ppsci train.py:134 保留键断言。实测与 ppcfd module training_step 路径逐位一致。
- `make_w_mse_loss(lat, log_postfix=None) -> expr`：验证侧损失 `w_mse`（`lat_weighted_mse_val` 的聚合键公式：lat 权重 `cos(lat)/mean(cos(lat))`，`(1,H,1)` 广播到 (B,V,H,W)）。
- `make_lat_weighted_metrics(lat, transform=None(恒等), clim=None, log_postfix=None) -> {"w_mse": M, "w_rmse": M, "w_nrmse": M}`：metric 实例三件套，对应 module val/test 步的 metric 列表（`lat_weighted_mse_val`/`lat_weighted_rmse`/`lat_weighted_nrmse`），可直接传 SupervisedValidator 的 `metric=`。
- `PerBatchFunctionalMetric`（子类 `ppsci.metric.base.Metric`，`keep_batch=True` 固定）：包装 dict 返回的逐批 expr，值 reshape 成长度 1 张量。**不能用 `ppsci.metric.FunctionalMetric`**：keep_batch=True 时它对返回值断言 `metric.ndim`（dict 无 ndim 直接 AttributeError，实测复现），且 0 维标量过不了 `_eval_by_batch` 的 `paddle.concat`（实测报错）；本包装类与 conf/climax.yaml 已设的 `EVAL.compute_metric_by_batch: true` 严格相容（solver XOR 检查通过）。
- 未平移 `lat_weighted_mse`(训练加权变体)/`lat_weighted_acc`/`pearson`/`lat_weighted_mean_bias`：module.py val/test 步均未引用（deep-dive-climax-side.md L24/L220 同结论）。

### 关键数字（验收命令真实输出）

1. **单批对照**（/tmp/climax06，子进程隔离：ppsci 侧真实 ClimateBenchDataset test 分区 + DataLoader(default_collate_fn) 取 2 批 batch_size=2，preds=y*1.05+0.1 确定性合成；ppcfd 侧同批跑原 metrics.py + module loss 路径）：**9 键 × 2 批全部 max_abs_diff = 0.0**，clim（y_normalization）两侧相等 = 2.072781562805176。样值 batch0：train_loss=0.05064719170331955，w_mse=0.04179790988564491，w_rmse=0.20435094833374023，w_nrmses_tas=0.09827501326799393，w_nrmseg_tas=0.09443384408950806，w_nrmse_tas=0.5704442262649536。
2. **metric 键名清单**（与既往 paddle_baseline_train.log L169-175 逐串一致）：`w_mse_tas_None`、`w_mse`、`w_rmse_tas_None`、`w_rmse`、`w_nrmses_tas`、`w_nrmseg_tas`、`w_nrmse_tas`。注意既往真实日志的 mse/rmse 逐 var 键就带 `_None` 后缀（log_postfix=None 被 f-string 拼入，为保公式逐字保留了它）。
3. **逐行核对**：/tmp/climax06/formula_diff.txt 归一化 diff——公式行全部一致，仅四类表面差异：dict 协议取 pred/y 的适配行、`with paddle.no_grad()` 移到包装类装饰器、聚合键 `np.mean([...cpu()])`→`paddle.mean(paddle.stack([...]))`（单 var 下逐位相同，已由对照 1 证实）、`dim=`→`axis=` 别名。
4. **协议检查**：FunctionalLoss(mse_loss) 可反传；FunctionalLoss(w_mse) 无 "loss" 保留键；三 metric 实例 keep_batch=True 且值均 1 维（concat 安全）。

### 跨批聚合语义（票面要求说明）

原版 train.py `_epoch_means` = 逐批 float 的算术平均；ppsci 侧 `compute_metric_by_batch=true` → `_eval_by_batch` 逐批调 metric 后 `paddle.concat + mean` = 批算术平均，**语义一致**（solver.py:282-284 注释所述）。4 批 × 7 键实测：solver 聚合 vs ppcfd 逐批均值 max_rel_diff = 6.4e-8——纯 float32 归约 vs python float64 求和的舍入差，非语义差；远低于 1e-4 量级的既往对照锚点（10 票）。nrmse 族（对 batch 维取均值）**必须**走逐批路径，全量拼接路径（compute_metric_by_batch=false）会改变其数学定义，故 yaml 的 true 是正确配置。validator 损失侧 `_eval_by_batch` 以 batch_size 加权 AverageMeter 平均，batch_size=1/config 下同样等于批算术平均。

### 遗留

- 票 05 装配提示：`loss=ppsci.loss.FunctionalLoss(utils.mse_loss)`（TRAIN）与 `FunctionalLoss(utils.make_w_mse_loss(lat))`（VAL/TEST）；`metric=utils.make_lat_weighted_metrics(lat, transform=denorm, clim=dataset_test.y_normalization)`——transform 传 train.py 装配序列构造的 denormalization（当前配置 out_transform=Normalize(0,1)，denorm 恒等），clim 仅 test 用。metric 日志键形如 `val/w_mse.w_mse_tas_None`（`{validator}/{metric名}.{键}`），display 命名如需再贴近旧日志由 05 决定。
- 对照脚本在 /tmp/climax06（不入库）；08 票等价门对照三可直接复用 side_ppsci.py/side_ppcfd.py 的形态换真实前向。
