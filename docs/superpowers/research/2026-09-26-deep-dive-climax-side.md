# ClimaX 侧接口细节与兼容性盘点（ppcfd feat/climax → ppsci 移植前置调研）

日期：2026-09-26。调研对象：`/home/lkyu/baidu/PaddleCFD` `feat/climax` 分支（经 `git show feat/climax:<path>` 读取，行号即该分支文件行号）；对照 `/home/lkyu/baidu/PaddleScience` 工作区；torch 基线参照 `/home/lkyu/baidu/CLIMAX/climax_torch/`。凡未确证处已标注。

---

## Q1 examples/climax/module.py（ClimateProjectionModule）

文件：`examples/climax/module.py`（feat/climax）。

### training_step（module.py:133-146）
```python
def training_step(self, batch: Any, batch_idx: int):
    x, y, lead_times, variables, out_variables = batch
    loss_dict, _ = self.net.forward(
        x, y, lead_times, variables, out_variables, [mse], lat=self.lat
    )
    loss_dict = loss_dict[0]
    ...
    loss = loss_dict["loss"]
    return loss
```
- 调用形态：`net.forward(x, y, lead_times, variables, out_variables, metric=[mse], lat=self.lat)`，**loss 在 net.forward 内部由 metric 函数计算**（不在 module 里算）；module 只取 `loss_dict[0]`（mse 返回的 dict），逐 var 写 `log_history["train/"+var]`（float 化），返回 `loss_dict["loss"]`（Tensor，供 train.py 反传）。
- 注意：训练用 `mse`（非纬度加权）；`lat` 传入但 mse 不使用（metrics.py:6 `lat=None` 形参）。

### validation_step（module.py:148-173）/ test_step（module.py:175-200）
- 均调 `self.net.evaluate(x, y, lead_times, variables, out_variables, transform=self.denormalization, metrics=[...], lat=self.lat, clim=..., log_postfix=None)`。
- val metrics：`[lat_weighted_mse_val, lat_weighted_rmse]`，`clim=self.val_clim`（装配时为 None）；test metrics：`[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_nrmse]`，`clim=self.test_clim`。
- `evaluate` 返回 dict 列表（每个 metric 一个 dict），module 展平合并后按 `val/`、`test/` 前缀记 log_history，返回合并 dict（无反传需求）。

### configure_optimizers（module.py:202-227）
- 参数组：遍历 `self.net.named_parameters()`；`p.stop_gradient` 的跳过（freeze_encoder 冻结参数不进优化器）；名字含 `var_embed`/`pos_embed`/`time_pos_embed` → no_decay 组，其余 → decay 组。
- AdamW(lr, beta1, beta2, eps=1e-8, parameters=[{"params":decay,"weight_decay":wd},{"params":no_decay,"weight_decay":0.0}])。
- 调度器：`LinearWarmupCosineAnnealingLR(opt, warmup_epochs, max_epochs, warmup_start_lr, eta_min)`，返回 `(opt, sched)` 二元组。

### load_mae_weights（module.py:74-115）逐条清洗
1. `paddle.load(path)` → 取 `checkpoint["state_dict"]`（键带 `net.` 前缀，torch Lightning 格式转 pdparams）。
2. `interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)`（在剥前缀**之前**调用，函数内部找 `net.pos_embed` 键，pos_embed.py:48-74）。
3. 剥 `net.` 前缀：`k[len("net."):]`，只保留以 `net.` 开头的键（module.py:84-88）。
4. parallel_patch_embed 时强制要求 checkpoint 含 `token_embeds.proj_weights`，否则 raise（module.py:90-94）。
5. 第一轮键清洗（module.py:95-102）：键名含 `channel` → 重命名 `channel→var`；键含 `token_embeds` 或 `head` → 删除（embedding/head 从头训练）。
6. 第二轮（module.py:103-109）：键不在 `net.state_dict()` 或 shape 不符 → 删除。
7. `set_state_dict(checkpoint_model)`（paddle 默认宽松加载，等价 strict=False），打印返回的 (missing, unexpected)。

### 其他
- setter：`set_denormalization(mean,std)`→`Normalize`；`set_lat_lon(lat,lon)`；`set_pred_range(r)`；`set_val_clim`/`set_test_clim`（module.py:117-131）。`pred_range` 存了但 training/validation/test_step 均未使用（属其他下游任务的遗留接口）。
- `__init__` 内若 `net.freeze_encoder` 为真，再次对所有非 norm 的 `net.blocks` 参数 `stop_gradient=True`（module.py:63-70；与 climatebench.py:58-64 的 `requires_grad_(False)` 双保险，已验证 paddle 3.x Tensor 存在 `requires_grad_`）。

---

## Q2 ppcfd/models/climax/climatebench.py — ClimaXClimateBench.forward

文件：`ppcfd/models/climax/climatebench.py`。

`__init__`（:8-64）：在 ClimaX 基础上新增 `out_vars`（可为 str）、`time_history`、`freeze_encoder`；`time_pos_embed`（sincos 初始化，:42-56）、`time_agg = MultiheadAttention(embed_dim, num_heads, batch_first=True)`、`time_query`（可学习查询，:48-50）、`head = Linear(embed_dim, H*W)`（:57）。

`forward_encoder`（:66-103）时间聚合链路：
1. `x` 形状 `(B,T,V,H,W)` → `flatten(0,1)` 成 `(B*T,V,H,W)`；
2. 逐变量 PatchEmbed（非 parallel 分支时把 var_ids 重新算成 Python int 以便 to_static 索引 ModuleList，:76-84）→ `stack` 得 `(B*T,V,L,D)`；
3. `+ var_embed`（:85-86）→ `aggregate_variables` 变量池化（继承自 arch.py:173-184，var_query 交叉注意力）→ `(B*T,L,D)`；
4. `+ pos_embed`（:88）→ `unflatten(0,(b,t))` → `+ time_pos_embed`（:89-90）→ `+ lead_time_embed(lead_times)` 广播（:91-93）→ pos_drop → blocks → norm（:95-98）；
5. `unflatten(0,(b,t))` → `x.mean(-2)` 对时间维取均值（:99-100）→ `time_agg(time_query, x, x)` 交叉注意力池化（:101-102）→ 返回 `(B,1,D)`。

`forward`（:105-113）全文结构：
```python
def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
    x = self.forward_encoder(x, lead_times, variables)
    preds = self.head(x)
    preds = preds.reshape(-1, 1, self.img_size[0], self.img_size[1])
    if metric is None:
        loss = None
    else:
        loss = [m(preds, y, out_variables, lat) for m in metric]
    return loss, preds
```
- preds 由单枚全局 token 直接线性回归出整幅 `H*W` 图，`reshape` 硬编码 1 个输出变量（head 输出维度即 H*W，:57）；**out_variables 不参与通道选择**（与基类 ClimaX 不同），只作为 metric 的键名。
- loss 为"逐 metric 函数调用结果"的列表（每个元素是 dict）；metric=None 时 loss=None（evaluate 走这条）。
- `evaluate` 继承自 arch.py:235-254：`forward(metric=None)` 后 `[m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]`。

---

## Q3 ppcfd/models/climax/arch.py — ClimaX 元数据依赖

`forward` 签名（arch.py:212）：`forward(self, x, y, lead_times, variables, out_variables, metric, lat)`。

各元数据的用法：
- **variables**（list/tuple 变量名）：`get_var_ids`（lru_cache，:149-152）查 `var_map` 得 id → 选 token_embeds 分支与 var_embed 切片（:190-199、:154-156）。逐 batch 传入但取值来自数据集常量。
- **out_variables**：`get_var_ids` → `preds[:, out_var_ids]` 选择输出通道（:227-228，仅基类 ClimaX 有此逻辑；ClimateBench 子类不用）。
- **lead_times**：`(B,)` 标量 → `lead_time_embed = Linear(1, embed_dim)`（:75），`lead_times.unsqueeze(-1)` 后加到每个 token（:203-205 / climatebench.py:91-93）。
- **lat**：**网络本体完全不使用**，只透传给 metric 函数（arch.py:232）。移到 ppsci 时可从 arch 接口剔除。
- **y**：网络 forward 不用 y，仅传给 metric。

`__init__` 可固化（不随 batch 变）的量：`default_vars`→`var_map`/`var_embed`（:138-147）、`img_size`/`patch_size`/`num_patches`/`pos_embed`（sincos 初始化，:103-110）、`aggregate_variables` 的 var_query/var_agg、blocks/norm/head（基类 head 输出 `len(default_vars)*patch_size**2`，:93-100）。ClimateBench 子类再固化 `out_vars`、`time_history`、`time_pos_embed`、`head(H*W)`。

---

## Q4 examples/climax/metrics.py 逐函数

| 函数 | 签名（行号） | 外部状态依赖 | 逐批/聚合 |
|---|---|---|---|
| `mse` | `(pred, y, vars, lat=None, mask=None)` metrics.py:6 | 无（lat 不用） | 逐批；返回逐 var + `"loss"`；训练损失 |
| `lat_weighted_mse` | `(pred, y, vars, lat, mask=None)` :29 | lat | 逐批；climate_projection 未使用 |
| `lat_weighted_mse_val` | `(pred, y, transform, vars, lat, clim, log_postfix)` :65 | lat、transform（denorm） | 逐批；clim/log_postfix 形参未用（log_postfix 只拼进键名） |
| `lat_weighted_rmse` | 同上 :90 | lat、transform | 逐批（批内先空间均值再批均值） |
| `lat_weighted_acc` | 同上 :120 | lat、transform、clim（tensor，:137 `.to(device)`） | 逐批；本项目未用 |
| `lat_weighted_nrmses` | 同上 :154 | lat、transform、clim（标量归一） | 逐批；对批内均值场算偏差 |
| `lat_weighted_nrmseg` | 同上 :178 | 同上 | 逐批；全局均值偏差 |
| `lat_weighted_nrmse` | 同上 :208 | = nrmses + 5*nrmseg（:221-223） | 逐批合成 |
| `pearson` :237、`lat_weighted_mean_bias` :259 | 多 log_steps/log_days | scipy.stats | 其他下游任务用 |

- 纬度权重统一公式：`w = cos(deg2rad(lat)) / mean(cos(deg2rad(lat)))`（如 :74-75）。
- 聚合方式：**全部函数逐批返回 dict（值为 tensor/float）**，跨批聚合是 train.py 的 `_epoch_means` 对 log_history 追加值做简单算术平均（train.py:44-53、199-207）——即 nrmse 类指标是"逐批再平均"，非全测试集一次聚合（与论文口径可能有出入，torch 基线同此实现）。
- `log_postfix=None` 被直接 f-string 进键名，实际键形如 `w_mse_tas_None`（metrics.py:85；module.py:161/188 传 None）。

---

## Q5 examples/climax/train.py 装配序列（train.py:141-151）

```python
normalization = datamodule.dataset_train.out_transform
mean_norm, std_norm = normalization.mean, normalization.std
mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
module.set_denormalization(mean_denorm, std_denorm)
module.set_lat_lon(*datamodule.get_lat_lon())
module.set_pred_range(0)
module.set_val_clim(None)
module.set_test_clim(datamodule.get_test_clim())
```
- `set_denormalization`：值来自 **dataset_train.out_transform**。ClimateBench 的 out_transform 是恒等 `Normalize([0.0],[1.0])`（dataset.py:121），故 denorm 实为恒等（mean=-0/1, std=1/1）。
- `set_lat_lon`：来自 `datamodule.get_lat_lon()`（datamodule.py:113-114）→ `load_x_y` 从 `inputs_<第一个simu>.nc` 读 latitude/longitude（dataset.py:55-63）。
- `set_pred_range(0)`：常数 0；后续未使用。
- `set_val_clim(None)`：常数 None。
- `set_test_clim`：`datamodule.get_test_clim()` → `dataset_test.y_normalization`（datamodule.py:119-120）= `get_rmse_normalization()` 算的标量张量：test y 的纬度加权全局均值绝对值（dataset.py:139-144）。
- 装配前还有：`ClimaXClimateBench(**cfg.model.net.init_args)`（train.py:104）；可选 `init_state_path` 全量严格加载（train.py:110-120，绕过 load_mae_weights）；可选 CINN `to_static` 包裹（train.py:125-127，须在权重加载后、module 构建前）。
- 训练循环：`lr_scheduler.step()` 在**每个 batch** 内调用（train.py:177）——与 torch 基线一致（climax_torch/climax/climate_projection/module.py:219 `"interval": "step"`），参数名虽叫 epochs 实为步数单位。

---

## Q6 datamodule.py / dataset.py 数据事实

`__getitem__`（dataset.py:149-153）返回 5 元组：
```python
inp = self.inp_transform(paddle.from_numpy(self.X_train_all[index]))  # (T,V,H,W) 逐样本
out = self.out_transform(paddle.from_numpy(self.Y_train_all[index]))  # (Vout,H,W) 逐样本（out_transform 恒等）
lead_times = paddle.Tensor([0.0]).to(dtype=inp.dtype)                 # 常量 0.0，逐样本恒定
return inp, out, lead_times, self.variables, self.out_variables       # 后两者是数据集级常量 list
```
- 逐样本变化：仅 `inp`、`out`。常量：`lead_times`（恒 0.0，故 lead_time_embed 实际只输出 bias）、`variables`、`out_variables`。
- collate（datamodule.py:9-15）：inp/out `stack`，lead_times `concat`（→`(B,)`），variables/out_variables 取 `batch[0][3]/[4]`。
- **historical 拼接位置**：`load_x_y`（dataset.py:25-42）——非 "hist" simu 的输入用 `xr.open_mfdataset([inputs_historical.nc, inputs_<simu>.nc])`，输出用 `xr.concat([outputs_historical, outputs_<simu>], dim="time")`；随后统一 `pr*86400`、rename lon/lat、transpose（:20-24、43-47）。滑窗与跳过历史：`input_for_training`/`output_for_training` 的 `skip_historical` + `len_historical=165`（dataset.py:67-96；train 侧前两个 simu（ssp126/ssp370）跳过历史段，datamodule.py:57-78 `skip_historical=i<2`；test 侧 `skip_historical=True`，datamodule.py:94-105）。
- train/val 划分：`split_train_val` 随机 permutation + train_ratio 切分（dataset.py:99-104）；val/test 的归一化统计沿用 train（datamodule.py:88-90、109-111）。
- test 分区额外裁剪最后 21 个样本（dataset.py:125-128）并计算 `y_normalization`。

---

## Q7 Python 兼容排查

climax 范围（`examples/climax/*.py` + `ppcfd/models/climax/**/*.py`，共 23 个 py）**逐项 grep 结果：零命中** 3.9+/3.10+ 语法：
- `X|Y` 注解 / `None | T`：无（climax 文件只用 `typing.Any/List/Optional/Tuple/Union/Type/Callable`，见 lr_scheduler.py:2、module.py:1、timm/*.py 头部）；
- `match/case`：无；`functools.cache`：无（arch.py:1 用 `lru_cache`，3.2+）；`dict |` 合并、`removeprefix/removesuffix`、`zip(strict=)`、`itertools.pairwise`、参数化内建泛型（`list[int]` 运行时注解）：无。

feat/climax **其他**目录确有 3.10+ 语法（移植范围外，仅记录）：`examples/prose_fd/utils/datapipe_compat.py:82`（`int | None`）、`examples/prose_fd/utils/rotary_embedding_paddle.py:209` 与 `ppcfd/models/prose_fd/rotary_embedding_paddle.py:209`（`tuple[int | float, ...] | Tensor | None`）、`examples/prose_fd/data_utils/cfdbench/{cavity.py:368, cylinder.py:321, dam.py:330, tube.py:309}`（`dict[str, Any]` 注解）、`examples/darcyflow/ppdeeponet/Utils/PlotFigure.py:20-22,41` 与 `ppcfd/models/ppdeeponet/{FNO1d.py:50, FNO2d.py:60, MultiONets.py:25,149,252,303,367}`（`list[...]` 注解，3.9+）。

对照 PaddleScience：`pyproject.toml:12` `requires-python = ">=3.8"`，classifiers 3.8/3.9/3.10（pyproject.toml:26-28、setup.py:50-52）；CI 仅有占位 workflow（`.github/workflows/github-actions-demo.yml`，无 python 矩阵）。本机 pycache 为 cpython-310（ppsci/optimizer/__pycache__）。

**结论**：ClimaX 相关 23 个文件无需任何降级即可满足 ppsci >=3.8 声明；工作量≈0。

---

## Q8 pos_embed 对比：ppcfd pos_embed.py vs ppsci/arch/cvit.py

| 项 | ppcfd `ppcfd/models/climax/pos_embed.py` | ppsci `ppsci/arch/cvit.py` |
|---|---|---|
| `get_1d_sincos_pos_embed_from_grid` | (:30-45) numpy；`omega=np.arange(D//2,dtype=float)`(float64)；einsum+sin/cos concat；返回 np `(M,D)` | (:31-47) paddle；`paddle.arange(D//2,float32)`；同公式；入参/返回均为 Tensor |
| `get_2d_sincos_pos_embed` | (:5-19) `(embed_dim, grid_size_h, grid_size_w, cls_token=False)`，`np.meshgrid(grid_w, grid_h)`（**xy 序**），返回 np `(N,D)`/`(1+N,D)` | (:54-81) `(embed_dim, grid_size: Tuple)`，`paddle.meshgrid(grid_w, grid_h, indexing="ij")` 后 `reshape([2,1,H,W])`，返回 Tensor `(1,N,D)`，无 cls_token |
| 其余 | `interpolate_pos_embed`(:48-74)、`interpolate_channel_embed`(:77-84) checkpoint 工具 | 无对应物 |

**数值一致性：不一致（已数值验证）**。`np.meshgrid` 默认 xy 序（grid[0][i,j]=w[j]，形状 (H,W)）；paddle ij 序（grid[0][a,b]=w[a]，形状 (W,H)）再 reshape 回 (H,W)，展平顺序不同。用 numpy 复现两实现，(H,W)=(4,4)、(4,8)、(16,32)（即 ClimaX 5.625°+patch2 的 16×32 token 网格）均 `allclose=False`；并已用 paddle 实测 meshgrid ij 语义确认（`[[0,0],[1,1],[2,2]]` vs numpy xy `[[0,1,2],[0,1,2]]`）。即 ppsci cvit 版与 MAE/torch ClimaX 的 token 排布互为转置/错位。**移植时若要 torch checkpoint 的 pos_embed 对齐，必须保留 ppcfd 版本（或修 ppsci 版布局）；1D 公式两版数学等价（仅 float64/float32 差异）。**

---

## Q9 lr_scheduler 对比

ppcfd `examples/climax/lr_scheduler.py` `LinearWarmupCosineAnnealingLR`（:5-110）：
- 接口：`__init__(optimizer, warmup_epochs, max_epochs, warmup_start_lr=0, eta_min=0, last_epoch=-1)`，构造时 `step()` 一次（镜像 torch）；`step()` 每 epoch/步 +1 并写 `param_groups` 的 `learning_rate`+`lr` 双键，单值时同步 `optimizer.set_lr`（:91-106）；`get_last_lr()`。
- 递推（torch `_LRScheduler` 链式语义）：epoch 0 → warmup_start_lr；epoch==warmup → base_lrs；warmup 期 → `prev + (base-start)/(warmup-1)`；cosine 期 → 闭式等价递推 `(1+cos(pi t/T))/(1+cos(pi (t-1)/T))*(prev-eta_min)+eta_min`，`T=max-warmup`；`(e-1-max) % (2T)==0` 的重启特判（:48-89）。

PaddleScience `ppsci/optimizer/lr_scheduler.py` 内置款：`Constant, Linear, ExponentialDecay, Cosine, Step, Piecewise, MultiStepDecay, CosineAnnealingWarmRestarts, CosineWarmRestarts, OneCycleLR, LambdaDecay, ReduceOnPlateau, SchedulerList`（类清单 :123-875）。

**等价款**：`Cosine(epochs, iters_per_epoch, learning_rate, eta_min, warmup_epoch, warmup_start_lr, last_epoch, by_epoch)`（:273-338）= `lr.LinearWarmup`（闭式 `start+(end-start)*k/warmup_steps`，已 inspect paddle 源码确认）包 `lr.CosineAnnealingDecay`（T_max=(epochs-warmup)*iters 或 by_epoch 时 epochs-warmup）。关键事实：**paddle `CosineAnnealingDecay.get_lr` 的递推公式与 climax 调度器 cosine 分支逐字符同构**（同为 `(1+cos(pi t/T))/(1+cos(pi(t-1)/T))*(last-eta_min)+eta_min` + 相同的 `(e-1-T)%2T==0` 重启特判，经 inspect 安装的 paddle 3.x 源码确认）。差异仅剩：(a) warmup 除数 `(W-1)` vs `W` 的 off-by-one（torch 在 e=W 精确到 base；paddle 在 k=W 到 end_lr，终点一致、中途差一步）；(b) climax 版手动写 param_groups（因双参数组），ppsci/paddle 版挂在 optimizer 的全局 lr 上（climax 两组 lr 相同，可合并）。

**平移 vs 对接结论**：对接内置可行——torch 基线 `interval="step"`（climax_torch module.py:219）且 ppcfd train.py:177 每 batch step，ppsci 默认 `by_epoch=False` 按 step 递推，语义同构；用 `ppsci.optimizer.lr_scheduler.Cosine`（把总步数换算进 epochs*iters_per_epoch，或 by_epoch=True 以"epoch==step"口径）即可复现曲线，误差仅在 warmup 中段一步之差。若要 bit 级复刻（含 param_groups 直写），平移那 100 行更省事。两者都成立，属取舍而非硬约束。

---

## Q10 timm 子包 import 面与类名冲突

### 对外 import 面
逐文件核对 `ppcfd/models/climax/timm/`：全部 9 个模块只 import **paddle + 标准库**（typing / math / warnings / enum / functools / itertools / collections.abc），**无 numpy、无任何三方依赖**（attention.py:11-13、block.py:10-14、mlp.py:10-12、patch_embed.py:14-19、drop.py:6、weight_init.py:9-11、helpers.py、format.py:8-9、trace_utils.py 零 import）。`__all__` 导出面（timm/__init__.py:17-30）：`Attention, Block, DropPath, drop_path, Format, to_1tuple, to_2tuple, to_ntuple, Mlp, PatchEmbed, _assert, trunc_normal_`。

### 与 PaddleScience 同名类盘点（ppsci 侧均为各 arch 文件内部类，未在 `ppsci/arch/__init__.py` 公共导出）
| 类名 | ppsci 位置 | 公共导出？ |
|---|---|---|
| `Block` | arch/preformer.py:200、arch/climateformer.py:200、arch/meteoformer.py:200、arch/physx_transformer.py:187、arch/moflow_glow.py:254、arch/afno.py:309 | 否（__init__ 只导出 AFNONet/PrecipNet/Preformer/Meteoformer/Climateformer/PhysformerGPT2 等顶层模型，:22-76） |
| `Attention` | arch/preformer.py:130、arch/climateformer.py:130、arch/meteoformer.py:130 | 否 |
| `Mlp` | arch/preformer.py:102、arch/cvit.py:136、arch/climateformer.py:102、arch/meteoformer.py:102 | 否 |
| `PatchEmbed` | arch/cvit.py:226、arch/afno.py:371 | 否 |
| `DropPath` | arch/afno.py:64 | 否 |
| `trunc_normal_` | **ppsci/utils/initializer.py:156，且在 `__all__` 公共导出**（:38）；签名 `(tensor, mean=0., std=1., a=-2., b=2.)` 与 timm 版逐参一致（timm/weight_init.py:49），算法同为 inverse-CDF trunc normal，均就地写回 | 是 |

冲突影响：只要 ClimaX 的 timm 副本作为独立子模块（如 `ppsci/arch/climax/timm`）放置、不进 `ppsci/arch/__init__.py` 顶层命名空间，与 ppsci 现有内部类零冲突；`trunc_normal_` 可直接改用 `ppsci.utils.initializer.trunc_normal_` 删掉 timm/weight_init.py。

---

## 拆解到 ppsci 形态的自然切面

- **arch（`ppsci/arch/climax.py` 或子包）**：`ClimaX`（arch.py 全部：token_embeds/var_embed/var_agg/pos_embed/lead_time_embed/blocks/unpatchify/aggregate_variables）+ `ClimaXClimateBench`（climatebench.py 的 time_pos_embed/time_agg/time_query/head/forward）+ timm 副本（或瘦身并入）+ pos_embed 的 sincos/interpolate（注意 Q8 布局差异，checkpoint 对齐必须保 ppcfd 版）；`trunc_normal_` 换 `ppsci.utils.initializer`。forward 可剥离 `y/metric/lat`（网络本体不用，Q3），让 arch 只出 `preds`。
- **dataset（`ppsci/data/dataset/`）**：ClimateBenchDataset + load_x_y/input_for_training/output_for_training/split_train_val/get_rmse_normalization（dataset.py 整体）；datamodule 的 collate 与 historical 拼接装配（datamodule.py）对应 ppsci 例行放在 example 侧或 dataset 工厂；Normalize（normalize.py）可并入 dataset 或用 ppsci 现有 transform 习惯。
- **example-main（`examples/climax/`）**：train.py 的装配序列（Q5 五个 setter 的值来源）、init_state_path 严格加载、CINN/to_static 开关、early-stopping/best ckpt 循环、metric 打印表——对应 ppsci 的 conf/main.py + utils.py 模式（参照 examples/climateformer、examples/fourcastnet）；module.py 的 load_mae_weights 清洗 + freeze/参数组 + configure_optimizers 归入 example 的 model 构建与 solver 配置（ppsci `build_optimizer` 支持参数组经由 AdamW 封装，需确认自定义分组入口——**未确认**，ppsci/optimizer/optimizer.py 的 AdamW 封装是否透传 param-groups 待查）。
- **metric（`ppsci/metric/`）**：lat_weighted_mse_val/rmse/nrmse(s/g) 映射为 ppsci Metric 类或 FunctionalMetric（`ppsci.metric` 已有 `LatitudeWeightedRMSE`（rmse.py:73，num_lat 重生成纬权、可选 std 反归一）与 `LatitudeWeightedACC`（anomaly_coef.py），口径与 climax 版不同：climax 用任意 lat 数组按均值归一、nrmse 带 clim 标量与 1:5 加权——大概率需新写 climax 专用 metric 以保数值对齐，不宜直接替换）；`mse` 训练损失可走 ppsci loss（MSELoss 逐通道口径需核对 per-var dict 需求）。
- **loss**：训练 loss 即 metrics.mse（逐 var + 总 loss dict）；在 ppsci 里对应 equation-less 的监督约束（`ppsci.loss.MSELoss` 或 Functional），lat 加权训练变体（lat_weighted_mse）本项目未用到。

## 附：关键未确认项清单
1. ppsci `optimizer.optimizer.AdamW` 封装能否表达 decay/no_decay 双参数组（未读该文件全文）。
2. torch 基线 metric 聚合与 ClimateBench 论文口径的差异（逐批平均 vs 全集聚合）是否影响验收——两侧实现一致，但与论文数值的可比性未验证。
3. paddle `LinearWarmup` 在挂接 `CosineAnnealingDecay` 后 last_epoch 对齐的精确行为（inspect 只看了 get_lr 单体公式，未跑逐步数值对照）。
