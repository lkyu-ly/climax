# PaddleScience(develop, d57561b1)训练/评估管线深挖 —— 面向 ClimaX 接入

调研对象:`/home/lkyu/baidu/PaddleScience`(develop 分支,commit d57561b1,版本 fallback 1.4.0)。
本机 paddle 3.4.0(用于验证 collate 行为)。所有行号均为该仓库当前代码。

---

## Q1 examples/climateformer/main.py 全文结构

文件:`examples/climateformer/main.py`(全文 184 行)。

train(cfg) 装配序列:

1. dataloader cfg(dict 形式,dataset 子 dict + sampler + batch_size/num_workers),`main.py:25-58`:
```python
train_dataloader_cfg = {
    "dataset": {
        "name": "ERA5ClimateDataset",
        "file_path": cfg.TRAIN_FILE_PATH,
        "input_keys": cfg.MODEL.input_keys,
        "label_keys": cfg.MODEL.output_keys,
        "size": (cfg.IMG_H, cfg.IMG_W), "years": cfg.TRAIN_YEARS,
    },
    "sampler": {"name": "BatchSampler", "drop_last": True, "shuffle": True},
    "batch_size": cfg.TRAIN.batch_size, "num_workers": 4,
}
```
2. constraint,`main.py:61-66`:
```python
sup_constraint = ppsci.constraint.SupervisedConstraint(
    train_dataloader_cfg, ppsci.loss.MSELoss(), name="Sup",
)
constraint = {sup_constraint.name: sup_constraint}
```
3. `ITERS_PER_EPOCH = len(sup_constraint.data_loader)`(`main.py:69`)。
4. eval dataloader cfg(同结构,`training: False`,无 sampler)→ validator,`main.py:72-95`:
```python
sup_validator = ppsci.validate.SupervisedValidator(
    eval_dataloader_cfg, ppsci.loss.MSELoss(),
    metric={"MAE": ppsci.metric.MAE(keep_batch=True),
            "MSE": ppsci.metric.MSE(keep_batch=True)},
    name="Sup_Validator",
)
```
5. model = `ppsci.arch.Climateformer(**cfg.MODEL)`(`main.py:98`,只传 input_keys/output_keys 与超参)。
6. lr_scheduler(Cosine,注入 iters_per_epoch)+ Adam(`main.py:101-105`)。
7. Solver(逐参数传入,非 cfg 模式,`main.py:108-121`):
```python
solver = ppsci.solver.Solver(
    model=model, constraint=constraint, output_dir=cfg.output_dir,
    optimizer=optimizer, epochs=cfg.TRAIN.epochs, iters_per_epoch=ITERS_PER_EPOCH,
    log_freq=cfg.log_freq, eval_during_train=cfg.TRAIN.eval_during_train,
    eval_freq=cfg.TRAIN.eval_freq, validator=validator,
    compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
    eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
)
solver.train(); solver.eval()
```
evaluate(cfg):同样的 dataset→validator→model→Solver(只给 `pretrained_model_path=cfg.EVAL.pretrained_model_path`,无 constraint/optimizer)→ `solver.eval()`(`main.py:128-169`)。入口 hydra `@hydra.main(config_path="./conf", config_name="climateformer.yaml")` 按 cfg.mode 分派(`main.py:172-179`)。

## Q2 Constraint 数据流:dataset → model 调用

链路(以 SupervisedConstraint 为例):

1. `SupervisedConstraint.__init__`(`ppsci/constraint/supervised_constraint.py:56-80`):`dataset.build_dataset(dataloader_cfg["dataset"])` 建数据集;`self.input_keys = _dataset.input_keys`;`output_keys = tuple(output_expr.keys()) if output_expr else _dataset.label_keys`;默认 `output_expr = {key: (lambda out, k=key: out[k])}`(恒等透传,`supervised_constraint.py:74-77`);`super().__init__` → `Constraint.__init__`(`ppsci/constraint/base.py:39-49`)只做 `data.build_dataloader(dataset, cfg)`、`self.loss`、`self.name`。
2. dataloader 构建:`ppsci/data/__init__.py:61-207`。普通 dataset 走 `io.DataLoader(dataset, places=device.get_device(), batch_sampler=..., collate_fn=collate_fn, ...)`(`__init__.py:188-199`);`collate_fn` 仅在 pgl/graph_grid_mesh 分支显式设为 ppsci 的 `batch_transform.default_collate_fn`(`__init__.py:129-152`),其余为 None(用 paddle 自带 default_collate_fn)。
3. 训练循环 `ppsci/solver/train.py:84-133`:每个 iter 对每个 constraint `next(_constraint.data_iter)` 解包三元组:
```python
input_dict, label_dict, weight_dict = next(_constraint.data_iter)   # train.py:90
...
losses_all, losses_constraint = solver.forward_helper.train_forward(
    tuple(_constraint.output_expr for ...), input_dicts, solver.model,
    solver.constraint, label_dicts, weight_dicts)                    # train.py:123-133
```
4. 真正调 model 的地方在 `ppsci/utils/expression.py:96`:
```python
output_dict = model(input_dicts[i])          # 整个 input_dict 传给 model.forward
data_dict = {k: v for k, v in input_dicts[i].items()}
data_dict.update(output_dict)
for name, expr in expr_dicts[i].items():
    output_dict[name] = expr(data_dict)      # expression.py:96-102
if "area" in input_dicts[i]:
    output_dict["area"] = input_dicts[i]["area"]   # expression.py:104-106
losses = cst_obj.loss(output_dict, label_dicts[i], weight_dicts[i])  # expression.py:112-116
```
5. 评估同构:`ppsci/solver/eval.py:90-109` → `eval_forward`(`expression.py:146-194`)。

input_dict 的 value 是否强制 Tensor:不强制,但有隐含约束:
- collate 层(paddle 3.4.0 `paddle.io.dataloader.collate.default_collate_fn`,本机源码验证;ppsci 自带副本在 `ppsci/data/process/batch_transform/__init__.py:38-97`,行为一致且多 None/pgl.Graph 分支):`np.ndarray→np.stack`、`paddle.Tensor→paddle.stack`、`numbers.Number→np.array(batch)`、`str/bytes→原样 list(不堆叠)`、Mapping/Sequence 递归。np.ndarray 输入经 `io.DataLoader(places=device.get_device())` 自动变为 place 上的 Tensor(所有 ERA5 系 dataset 均输出 np.ndarray 且 arch 直接调 paddle 方法,为此佐证);str/list[str] 保持 Python 对象。
- 循环层:`train.py:104-106` 与 `eval.py:94-96` 的 `for v in input_dict.values(): if hasattr(v, "stop_gradient"): v.stop_gradient = False` —— 非 Tensor(list/ndarray→已被 DataLoader 转张量,仅 str/list 保留)无该属性,安全跳过。
- `_compute_batch_size`(`train.py:36-56`):取 input_dict 第一个 value,无 `shape` 属性则 fallback `len(sample)`,注释明确 batch_size 只用于计时,不影响正确性。
- 坑:list[str] value 经 collate 变成 `[default_collate_fn(fields) for fields in zip(*batch)]`(Sequence 分支)→ 形状是 `[变量数][batch]`(转置),且元素是 str。机制上能通过管线,但消费侧需自行处理。

## Q3 input_dict 含非张量值的先例

- ppsci 内置 dataset 的 input_item 值全部是 ndarray/Tensor(如 `era5climate_dataset.py:154-168`、`era5_dataset.py:136`、`stafnet_dataset.py:155-158`)。**未发现内置 dataset 在 input_dict 放 list[str]/str/int 的先例。**
- 弹性先例:
  - `weight_dict` 可为 `None`:`atmospheric_dataset.py:1764-1773`(`return ({input}, {label}, None)`);可为 `{}`:`stafnet_dataset.py:161`。collate 对 None 返回 None(`batch_transform/__init__.py:51-52`)。
  - input_dict 可含超出 `input_keys` 的额外键(张量):`ext_moe_enso_dataset.py:393` `input_item = {self.input_keys[0]: in_seq, "sst_target": target_seq}`。
  - 非 dict-of-tensor 对象(GraphGridMesh)走 `build_dataloader` 专用分支 `use_graph_grid_mesh`(`data/__init__.py:139-152`)+ 专用 collate,不具普遍性。
  - examples:`xrdmatch/main.py:231,236` `{"idx_ulb": index, ...}`(int 直接入 dict),但 xrdmatch 用自建训练循环(仅 `ppsci.arch.VGG`,`main.py:576`),**不是 Solver 管线先例**。
- 结论:collate/循环对 list/str/int 容忍(不崩溃),但"元数据经 input_dict 进 Solver 管线并在 model 内消费"在 ppsci 无完整先例;变量级常量进 dataset 是空白地带。

## Q4 output_transform 机制

- 定义:`ppsci/arch/base.py:183-218` `register_output_transform(transform)`,签名 `Callable[[input_dict, output_dict], output_dict]`;另有 `register_input_transform`(`base.py:150-181`)。
- 语义:输出后处理钩子——基类只存 callable,**需要各 Arch 子类在自己的 forward 里显式调用**(如 `physx_transformer.py:405-406`、`afno.py:568-569/698-699`、`climateformer.py:434-436`)。注意 `climateformer.py:435` 调 `self._output_transform(x, y)` 传的是 concat 后的张量 x 而非原始 input dict,与基类文档签名不符(支线发现,疑为笔误)。
- 使用先例(grep examples):`examples/phycrnet/main.py`、`examples/phygeonet/heat_equation.py`、`examples/gpinn/poisson_1d.py`、`examples/hpinns/holography.py`、`examples/ntopo/ntopo.py`、`examples/fpde/fractional_poisson_2d.py`、`examples/bubble/bubble.py` 等(多为 PINN 因果/边界修正)。
- 对 ClimaX:可用 output_transform 做 denorm/通道选择,但它拿到的 input_dict 是整个输入字典,同样绕不开"元数据在哪"的问题。

## Q5 自定义 Constraint 先例

`grep "class.*Constraint"` 全仓库(ppsci + examples + jointContribution):仅 `ppsci/constraint/` 内 7 个 —— `base.Constraint`、`SupervisedConstraint`、`InteriorConstraint`、`BoundaryConstraint`、`InitialConstraint`、`PeriodicConstraint`、`IntegralConstraint`。**examples/jointContribution 无任何自定义 Constraint**。社区惯例是:SupervisedConstraint + 自定义 dataset(内置注册或直接传实例,见 Q3 补充:`build_dataset` 对已是 `io.Dataset` 实例的 cfg 直接返回,`ppsci/data/dataset/__init__.py:122-129`;也可 `@register_to_dataset` 注册,`__init__.py:167-175`)+ 自定义 loss/metric。ClimaX 无需自定义 Constraint。

## Q6 Solver.to_static 与 CINN

- 实现:`ppsci/solver/solver.py:516-527`:
```python
if self.to_static:
    jit.enable_to_static(self.to_static)
    logger.message("Enable jit.to_static for forward pass ...")
    self.forward_helper.train_forward = paddle.jit.to_static(self.forward_helper.train_forward)
    self.forward_helper.eval_forward = paddle.jit.to_static(self.forward_helper.eval_forward)
```
- **包的对象是 `ExpressionSolver.train_forward/eval_forward` 这两个方法,不是 model 本身**;model.forward 在被 trace 的 train_forward 内部执行(`expression.py:96` 的 `model(input_dicts[i])` 一并被动转静)。时机:Solver.__init__ 末段,在 pretrain 加载(344-349)、DataParallel 包装(409-434)之后。
- 来源:`to_static` 构造参数(`solver.py:159`)或 cfg(`self.to_static = cfg.to_static`,`solver.py:1152`)。
- examples 先例:仅 `examples/euler_beam/conf/euler_beam.yaml:31`(`to_static: false`)+ `euler_beam.py:125,194` 透传。无 true 的先例。
- CINN:ppsci 全仓库 0 处 CINN 引用(grep "cinn|CINN" 无 hit)。ppsci 不设置、不清除任何 FLAGS;`FLAGS_use_cinn` 等外部环境变量作用于 paddle 动转静执行引擎选择,与 Solver.to_static 的包装行为分属两层。**从代码逻辑判断可以共存**(to_static 只负责转静,CINN 由 paddle 运行时按 FLAGS 决定);仓库内无先例,未实测 —— 标注"未确认(逻辑推断)"。另一层含义:input_dict 里若有非张量 Python 对象(list[str]),动转静 tracing 支持有限,会威胁 to_static 路径。

## Q7 checkpoint 机制

- 保存 `ppsci/utils/save_load.py:227-309`:`{prefix}.pdparams`(model.state_dict())、`.pdopt`、`.pdstates`(metric dict)、`.pdscaler`、`.pdeqn`、`_ema.pdparams`、`.pdagg`,输出到 `output_dir/checkpoints/`。训练循环写三种 prefix:`best_model`(metric 创新低时,`solver.py:638-650`)、`epoch_{n}`(每 save_freq,`solver.py:706-717`)、`latest`(每 epoch,`solver.py:720-731`)。
- 加载 pretrained:`save_load.py:46-65`:
```python
param_state_dict = paddle.load(f"{path}.pdparams")
model.set_state_dict(param_state_dict)
```
  - **不支持 `{"state_dict": {...}}` 包装结构**(外层 "state_dict" 键会被当成参数名,加载静默失败——`load_pretrain` 不检查 missing/unexpected,`load_checkpoint` 才有 warning,`save_load.py:184-194`);无 strict 参数;**无自定义加载函数钩子**。
- `EVAL.pretrained_model_path`:`_parse_params_from_cfg` 按 mode 取 `TRAIN/EVAL/INFER.pretrained_model_path`(`solver.py:1174-1179`),构造时 `save_load.load_pretrain`(`solver.py:344-349`)。
- 自定义结构权重先例:**走 Solver 管线没有**。examples 自建循环先例:`examples/UTAE/test_panoptic.py:264-268`(`checkpoint["state_dict"]` 手动拆装)、`examples/unetformer/train_supervision.py:291-295`(`state["model_state_dict"]`)。ClimaX 原始权重的可行做法:构造 Solver 之前自行 `paddle.load` → 拆键/去前缀 → `model.set_state_dict`,不走 `pretrained_model_path`。

## Q8 loss 体系

- 清单(`ppsci/loss/__init__.py:43-62`):`MSELoss`、`CausalMSELoss`、`MSELossWithL2Decay`、`PeriodicMSELoss`、`MAELoss`、`L1Loss`/`PeriodicL1Loss`、`L2Loss`/`L2RelLoss`/`PeriodicL2Loss`、`ChamferLoss`、`IntegralLoss`、`KLLoss`、`BCELoss`、`FocalLoss`、`FunctionalLoss`、`mtl`(Sum/AGDA/PCGrad 聚合器)。
- 接口:`forward(output_dict, label_dict, weight_dict=None) -> Dict[str, Tensor]`(`loss/mse.py:82-106`)。weight_dict 参与:逐 label 键 `loss *= weight_dict[key]`(batch 内可广播张量或标量,`mse.py:89-90`);构造时的 `self.weight`(float 或 dict)再乘一层。
- 纬度加权 loss:**没有直接的 lat-weighted Loss,但有现成 "area" 钩子** —— `mse.py:92-93`:
```python
if "area" in output_dict:
    loss *= output_dict["area"]
```
  而 `expression.py:104-106`(train)与 `179-181`(eval)自动把 `input_dict["area"]` 透传进 output_dict。即 dataset 每样本在 input_dict 放 "area"(纬度 cos 权重,形状可广播到输出)即可获得纬度加权 MSE,零 loss 改造。先例:`examples/aneurysm/aneurysm.py:174-175`(PINN 面积缩放)、`examples/ntopo/functions.py`。注意该行为是 MSELoss 等内置 loss 硬编码,FunctionalLoss 需自行实现。
- examples 自定义 loss 先例:`earthformer/earthformer_enso_train.py:61` `ppsci.loss.FunctionalLoss(enso_metric.train_mse_func)`(func 签名 `(output_dict, label_dict, *args)`,返回 dict of Tensor,`loss/func.py:80-100` 有类型断言);`ntopo/functions.py:287` `FunctionalLossBatch(ppsci.loss.base.Loss)`(继承式,额外收 input_dicts);unetformer/geoseg/losses 下多个 `nn.Layer`(自建循环,不经 Solver)。

## Q9 metric 体系

- `LatitudeWeightedRMSE`(`ppsci/metric/rmse.py:73-160`):
  - 构造:`LatitudeWeightedRMSE(num_lat, std=None, keep_batch=False, variable_dict=None, unlog=False, scale=1e-5)`;`std` reshape 成 `(1,-1)` 逐通道缩放;`variable_dict={"u10": 0, ...}` 输出 `"{key}.{var}"` 逐变量指标(`rmse.py:152-156`)。
  - 调用:`forward(output_dict, label_dict) -> Dict[str, Tensor]`(`rmse.py:139-160`)。
  - **权重是 `get_latitude_weight`(`rmse.py:129-134`)按 num_lat 均匀 `linspace(0,1)` 假设生成的 cos 权重,不接收真实 lat 数组**。均匀网格(如 5.625°)下近似成立,但与 ClimaX 用真实 lat 数值算的权重有数值差;精确复现需自定义。
  - `keep_batch` 必须与 Solver `compute_metric_by_batch` 一致否则 raise(`solver.py:278-302`);keep_batch=True 走 `_eval_by_batch`(逐 batch 算再对样本平均,`eval.py:191-301`),False 走 `_eval_by_dataset`(缓存全量输出最后一次算,`eval.py:63-188`)。
  - 使用先例:`examples/fourcastnet/train_pretrain.py:137-146`(num_lat=IMG_H, std=data_std, variable_dict)。
- 自定义 metric 接法:`ppsci.metric.FunctionalMetric(metric_expr)`(`metric/func.py:27-72`),expr 签名 `(output_dict, label_dict) -> Dict[str, Tensor]`。earthformer 先例:`earthformer_enso_train.py:90-92`:
```python
metric={"rmse": ppsci.metric.FunctionalMetric(enso_metric.eval_rmse_func)}
```
  `enso_metric.py:90-127` 的 `eval_rmse_func` 展示了在 metric 里做 sst→nino 派生再算 RMSE/ACC 的完整写法。继承式先例:`examples/LatentNO/utils.py:9` `class RelLpLoss(base.Metric)`。

## Q10 TRAIN.eval_during_train 机制

- Validator 挂载:构造 Solver 时传 `validator={name: SupervisedValidator}`(`main.py:118`)。执行时机:`Solver.train()` 每个 epoch 训练完成后,`solver.py:632-637`:
```python
if (self.eval_during_train and epoch_id % self.eval_freq == 0
        and epoch_id >= self.start_eval_epoch):
    self.cur_metric, metric_dict_group = self.eval(epoch_id)
```
- **eval_freq 语义:以 epoch 为单位**(每 eval_freq 个 epoch 验证一次);`start_eval_epoch` 控制起始。climateformer 配置:`conf/climateformer.yaml:64-65` `eval_during_train: true`、`eval_freq: 5`。
- 输出:`self.eval`(`solver.py:745-772`)→ `eval.eval_func`(`eval.py:304-320`,按 compute_metric_by_batch 分派)→ 遍历所有 validator,逐 batch `eval_forward` + metric;返回 (target_metric, metric_dict_group) —— **第一个 metric 的第一个值被当作 target/best 依据**(`eval.py:180-186`)。best 创新低则存 `best_model` checkpoint(`solver.py:638-650`);全部 metric 经 `logger.scalar` 写 vdl/wandb/tbd(`solver.py:655-662`),日志打 `[Eval][Epoch n][Avg] ...`(`solver.py:766-769`)。metric 名跨 validator 重复会 warning(`solver.py:304-313`)。

---

## 对 ClimaX 接入路径的判断(元数据容纳方式,按证据强度排序)

ClimaX forward 签名 `forward(x, y, lead_times, variables, out_variables, metric, lat)`,其中:
- `variables/out_variables/lat`:数据集级常量(训练全程不变)。
- `lead_times`:每样本一个标量。
- 训练 loss = 纬度加权 MSE。

**方式 1(证据最强):常量元数据全部放 Arch 构造参数,不进 input_dict;lead_times 走 dataset 输出 ndarray。**
- ppsci 惯例是 input_dict 只放张量:所有内置 dataset 的 input_item 全是 ndarray(Q3);Climateformer 只有单一 "input" 键(Q1)。`ppsci.arch.Climateformer(**cfg.MODEL)` 表明变量表/网格常量这类信息历来走 MODEL 配置(`main.py:98`)。
- lead_times:dataset `__getitem__` 里输出 `np.asarray([lead], dtype="float32")` → collate 成 `[B,1]` ndarray → DataLoader 自动转 place 上的 Tensor(ERA5 系 dataset 全按此模式工作,Q2/Q3)。Arch.forward(input_dict) 内部解包 `x/y/lead_times`,零管线改动,且天然兼容 to_static(全张量输入)。
- ClimaX 的封装类(原 torch 里的 ClimaX+loss 封装)映射为 `ppsci.arch.Arch` 子类:forward 收 dict、内部转调原 forward,loss 移出模型交给 constraint(Q8)。

**方式 2(证据强):纬度加权 MSE 用 "area" 钩子,或 FunctionalLoss。**
- "area":dataset 每样本把 cos(lat) 权重(广播形状)放进 `input_dict["area"]`,`MSELoss` 自动 `loss *= area`(`mse.py:92-93`),管线自动透传(`expression.py:104-106/179-181`),有 aneurysm/ntopo 先例。缺点:area 也会被 collate 成 batch 张量(每样本重复,略费带宽),且语义是"面积"。
- `FunctionalLoss(lat_weighted_mse_func)`(earthformer 先例,`earthformer_enso_train.py:61`):lat 权重做成常量张量在函数内构造,最贴近 ClimaX 原实现、可读性最好;返回 dict of Tensor 即可(`func.py:85-98` 断言)。二选一,FunctionalLoss 更显式。

**方式 3(机制可行、无先例,不推荐):list[str] 元数据直接进 input_dict。**
- collate 不崩溃(paddle 3.4.0:str 保留、number→np.array,Q2),循环的 stop_gradient 检查安全跳过(`train.py:104-106`);但 (a) 变量表 collate 后是 `[V][B]` 转置 list-of-list(坑);(b) 无任何 Solver 管线先例(Q3);(c) to_static/CINN 路径对非张量输入支持不明(未确认)。仅当 lead_times 想走 str/复杂结构时才考虑。

**方式 4(配套选型):metric 用 LatitudeWeightedRMSE(近似)或 FunctionalMetric(精确)。**
- `ppsci.metric.LatitudeWeightedRMSE(num_lat=32, std=..., variable_dict={...})` 直接可用,但其权重是均匀网格 linspace 近似而非真实 lat 数组(`rmse.py:129-134`)——与 ClimaX 对齐数值会差;精确复现用 `FunctionalMetric`(earthformer 先例,lat 以常量张量闭包进去)。keep_batch 需与 compute_metric_by_batch 匹配(`solver.py:282-302`)。

**方式 5(配套选型):权重加载在 Solver 之前手动完成。**
- ClimaX 原始 checkpoint 若是 `{"state_dict": {...}}` 或键带前缀,`load_pretrain` 不支持(`save_load.py:63-64` 直接 set_state_dict,静默失败,无 strict/无钩子,Q7)。做法:构造 Solver 前自行 `paddle.load` → 拆键 → `model.set_state_dict`(UTAE/unetformer 自建循环同型做法,`test_panoptic.py:264-268`);训练期产物交给 Solver 自动 latest/best。

**方式 6(并行可行):to_static + FLAGS_use_cinn 共存。**
- Solver.to_static 包的是 forward_helper 的 train/eval_forward(`solver.py:519-527`),ppsci 不碰任何 FLAGS(Q6);CINN 开关由外部环境变量在 paddle 运行时生效。逻辑上可共存,但要求 input_dict 全张量 → 与方式 1 自洽。仓库无 to_static=true 先例(euler_beam 显式 false),CINN 组合未实测(未确认)。

**自定义 Constraint:不需要** —— 全仓库无先例,SupervisedConstraint + 自定义 dataset(cfg 传实例即可,`build_dataset` 对 Dataset 实例直接返回,`ppsci/data/dataset/__init__.py:122-129`)+ FunctionalLoss + FunctionalMetric 即可覆盖 ClimaX 全部需求。

主要风险清单:
1. list[str] 经 collate 的转置行为(Q2 坑)。
2. LatitudeWeightedRMSE 的 linspace 权重与 ClimaX 真实 lat 权重数值不一致(Q9)。
3. load_pretrain 对包装结构静默失败(Q7)。
4. to_static=true 无仓库先例,含非张量输入必炸;含 dict 解包的 Arch.forward 需保证分支静态可知(Q6,未实测)。
