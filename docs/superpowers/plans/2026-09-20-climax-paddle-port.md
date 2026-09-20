# ClimaX paddle 侧移植实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 paconvert 转换产物 `climax_paddle/` 手动修复至可训练，完成与 torch 基线（`climax_torch/`，5 epoch 短训锚点已就绪）同款的前向对齐与训练测试。

**Architecture:** 三层并行修复——① timm 1.0.24 最小闭包本地化为 `timm_paddle` 包（键名契约锁定）；② 模型/工具层接线（arch、lr_scheduler、normalize）；③ Lightning 彻底舍弃（用户授权），dataset/datamodule/module/train 改为普通 paddle 类 + 自建训练循环（Poseidon/G-FNO/prose-fd 历史范式）。之后串行：最小跑通 → 权重转换 → 前向对齐 → 5 epoch 短训对照。

**Tech Stack:** paddle 3.4.0 / torch 2.11（对照基线，同在 paddletorch 环境）、paconvert v3.3.1、omegaconf、numpy。

## Global Constraints

- 最小努力原则：能满足验证就不扩大；单个小数据集（已就位的 5.625° ClimateBench）贯穿全部验证
- 敏捷修复：跑通过程中的报错就地修复并记入 `ClimaX迁移实验记录.md`（问题-探索-修复循环，极简风格）；超出敏捷范畴（需大改逻辑/换依赖）→ 停下走疑难询问
- 禁止任何形式的 SHA 标记文件；commit message 可含上游 commit 号
- 用户未要求不主动 push；每个 Task 完成即 commit（本计划已授权）
- 环境：`/home/lkyu/miniconda3/envs/paddletorch/bin/python`（torch+paddle 双框架）；运行 paddle 代码时 `PYTHONPATH=/home/lkyu/baidu/CLIMAX/climax_paddle`
- torch 基线锚点（对照用）：5 epoch 后 val/w_mse=0.2118、test/w_mse=0.1390、test/w_rmse=0.3687；初始（未训练）test/w_mse=4.560
- 训练口径：FP32、单卡 RTX 4060 Ti、batch_size=1、seed 42、max_epochs=5（与 torch 基线同款）
- paddle 侧模型 state_dict 键名必须与 torch 侧完全一致（权重转换直映的前提）

---

### Task 1: timm_paddle 最小闭包包

**Files:**
- Create: `climax_paddle/timm_paddle/{__init__,helpers,weight_init,drop,mlp,attention,format,trace_utils,patch_embed,block}.py`
- Create: `climax_paddle/timm_paddle_torch_src/`（抽取的 torch 源，供 paconvert 输入，转换后此目录保留作溯源）

**Interfaces:**
- Produces（后续 Task 2 依赖，签名必须逐字一致）：
  - `from timm_paddle import Block, PatchEmbed, trunc_normal_, to_2tuple`
  - `PatchEmbed(img_size, patch_size, in_chans, embed_dim)`（位置参数），属性 `.num_patches`、`.proj.weight/.proj.bias`
  - `Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=..., norm_layer=paddle.nn.LayerNorm, proj_drop=...)`
  - `trunc_normal_(tensor, std=0.02)`（in-place 语义）；`to_2tuple(x)`
  - state_dict 键名契约（逐键不可改）：`norm1.weight/bias、attn.qkv.weight/bias、attn.proj.weight/bias、norm2.weight/bias、mlp.fc1.weight/bias、mlp.fc2.weight/bias`（Block 内）；`proj.weight/bias`（PatchEmbed 内）

抽取源（timm 1.0.24，`/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/timm/`）与改写要点：

| 目标文件 | 源 | 行 | 说明 |
|---|---|---|---|
| helpers.py | layers/helpers.py | 10-30 | `_ntuple`,`to_2tuple`，纯标准库 |
| weight_init.py | layers/weight_init.py | 19-51,54-78 | `trunc_normal_`（保留 a=-2,b=2 默认路径） |
| drop.py | layers/drop.py | 158-190 | `drop_path`,`DropPath` |
| mlp.py | layers/mlp.py | 14-54 | `Mlp`；删 `from .grn import GlobalResponseNorm` |
| attention.py | layers/attention.py | 12-111 | `maybe_add_mask`,`Attention`；删 `@torch.fx.wrap`、`register_notrace_function`、`apply_rot_embed_cat` |
| format.py | layers/format.py | 6-11 | `Format` 枚举 |
| trace_utils.py | layers/trace_utils.py | 1-5 | `_assert` |
| patch_embed.py | layers/patch_embed.py | 26-142 | `PatchEmbed`；可精简 `set_input_size/feat_ratio/dynamic_feat_size`；删 `nchw_to`、`torch.jit.Final` 注解 |
| block.py | models/vision_transformer.py | 126-207 | `Block`；`_create_attn` 内联为 `Attention(...)` 直构；`ATTN_LAYERS/DiffAttention/LayerType` 死代码不抽；`init_values=None` → `ls1/ls2 = nn.Identity()`（LayerScale 不抽）；norm_layer 默认值改 `nn.LayerNorm`（timm 自定义 LayerNorm 链不抽） |

三个 paconvert 之后的手工改写点：
1. `trunc_normal_` 的 in-place 链（`uniform_/erfinv_/mul_/add_/clamp_`）→ 函数式重写（`paddle.uniform` + `paddle.erf`/`erfinv` + `paddle.assign` 回写原参数），保持 [-2σ, 2σ] 截断语义与边界
2. Attention forward 保留双分支：fused（`paddle.nn.functional.scaled_dot_product_attention`）与手工 `softmax(q@k.T*scale)@v`；**默认走手工分支**（数值对齐基准），fused 留开关
3. `drop_path` 中 `x.new_empty(shape).bernoulli_(keep_prob)` → `paddle.bernoulli(paddle.full(shape, keep_prob, dtype=x.dtype))`；`div_(keep_prob)` → `x / keep_prob`

- [ ] **Step 1: 按上表从 timm 1.0.24 抽取 torch 源到 `timm_paddle_torch_src/`**（改写内部 import 为本地相对引用，按上表删除死代码）
- [ ] **Step 2: paconvert 转换**：`paconvert -i timm_paddle_torch_src -o timm_paddle --log_dir timm_paddle/paconvert.log`（在 climax_paddle/ 下执行）
- [ ] **Step 3: 手工修复三个改写点 + `__init__.py`**（导出四符号）
- [ ] **Step 4: 键名契约自检**（对照 torch timm）：

```python
# /home/lkyu/miniconda3/envs/paddletorch/bin/python，cwd=climax_paddle，PYTHONPATH=.
import torch, paddle
from timm.models.vision_transformer import Block as TBlock, PatchEmbed as TPatchEmbed
from timm_paddle import Block as PBlock, PatchEmbed as PPatchEmbed
t = TBlock(64, 4, 4.0, qkv_bias=True, drop_path=0.1, norm_layer=torch.nn.LayerNorm, proj_drop=0.1)
p = PBlock(64, 4, 4.0, qkv_bias=True, drop_path=0.1, norm_layer=paddle.nn.LayerNorm, proj_drop=0.1)
assert sorted(t.state_dict().keys()) == sorted(p.state_dict().keys()), (sorted(t.state_dict().keys()), sorted(p.state_dict().keys()))
# 逐键形状对照并记录 Linear 类是否需要转置（torch [out,in] vs paddle 侧实际形状）
for k in t.state_dict():
    print(k, tuple(t.state_dict()[k].shape), tuple(p.state_dict()[k].shape))
# PatchEmbed 同样对照 .num_patches 与 proj 键
```

Expected: 键名完全一致；形状差异仅出现在 Linear 类（记录转置规则，Task 6 使用）
- [ ] **Step 5: 数值自检（eval 模式同权重前向）**：torch/paddle 各建 `Block(64,4,...)`，把 torch 权重按 Step 4 规则转给 paddle，固定输入 `x=torch.randn(2,10,64)` seed 42，对比输出（目标 max_abs_error < 1e-5；不过则修到过）
- [ ] **Step 6: Commit**: `git add climax_paddle/timm_paddle climax_paddle/timm_paddle_torch_src && git commit -m "Add timm_paddle: minimal timm 1.0.24 closure for ClimaX backbone"`

### Task 2: 模型层接线（7 处 timm 标记）

**Files:**
- Modify: `climax_paddle/climax/arch.py:56,78,116,122,129`
- Modify: `climax_paddle/climax/parallelpatchembed.py:39-40`

**Interfaces:**
- Consumes: Task 1 的 `timm_paddle` 四符号
- Produces: `climax.arch.ClimaX` / `climax.climate_projection.arch.ClimaXClimateBench` 可 import、可构造

- [ ] **Step 1: arch.py**——删 `>>>>>>` 标记行，`timm.models.vision_transformer.PatchEmbed(...)` → `PatchEmbed(...)`、`Block(...)` → `Block(...)`、`trunc_normal_(` → `trunc_normal_(`（去前缀），文件头加 `from timm_paddle import Block, PatchEmbed, trunc_normal_`
- [ ] **Step 2: parallelpatchembed.py**——两处 `>>>>>> self.img_size = timm.layers.helpers.to_2tuple(img_size)` → `from timm_paddle import to_2tuple` 后 `self.img_size = to_2tuple(img_size)`（patch_size 同理）
- [ ] **Step 3: 构造冒烟**：

```python
# PYTHONPATH=. python -c：
from climax.climate_projection.arch import ClimaXClimateBench
m = ClimaXClimateBench(default_vars=['CO2','SO2','CH4','BC'], out_vars='tas', img_size=[32,64],
    time_history=10, patch_size=2, embed_dim=1024, depth=8, num_heads=16, mlp_ratio=4,
    drop_path=0.1, drop_rate=0.1, parallel_patch_embed=False, freeze_encoder=True)
print(sum(p.numel() for p in m.parameters()))
# 对照 torch 侧同参数构造的参数量（torch 侧先跑一遍记录数值，两边必须相等）
```

Expected: 参数量与 torch 侧一致（冻结逻辑 freeze_encoder 在 paddle 侧用 `stop_gradient=True` 实现，注意 `named_parameters` 仍含全部参数）
- [ ] **Step 4: Commit**: `git commit -am "Wire climax_paddle backbone to timm_paddle"`

### Task 3: utils 层（lr_scheduler + normalize）

**Files:**
- Modify: `climax_paddle/climax/utils/lr_scheduler.py`
- Create: `climax_paddle/climax/utils/normalize.py`

**Interfaces:**
- Produces:
  - `LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs, max_epochs, warmup_start_lr, eta_min, last_epoch)`，方法 `step()`（每优化步调用，递推更新 param_groups 的 lr）与 `get_last_lr()`
  - `Normalize(mean, std)`：`__call__(t)` 返回 `(t-mean)/std`，属性 `.mean/.std`；mean/std 为 numpy 数组（标量或 per-channel），paddle 张量按 numpy 广播规则自动对齐

- [ ] **Step 1: 重写 lr_scheduler.py**——不继承 `paddle.optimizer.lr.LRScheduler`（避开 base_lrs/last_epoch 机制差异），自写类复刻 torch `_LRScheduler` 递推语义：

```python
class LinearWarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs, self.max_epochs = warmup_epochs, max_epochs
        self.warmup_start_lr, self.eta_min = warmup_start_lr, eta_min
        self.base_lrs = [g.get('learning_rate', self.optimizer._learning_rate) for g in optimizer._param_groups]
        self.last_epoch = last_epoch
        self._last_lr = [warmup_start_lr] * len(self.base_lrs)
        self.step()  # 复刻 torch：构造即 step 一次（last_epoch→0）
    def step(self):
        self.last_epoch += 1
        # 逐分支复刻 torch get_lr()（climax_torch/climax/utils/lr_scheduler.py:50-77 原式照搬，group['lr'] 读上一步值）
        ...算出 values 后：
        for g, v in zip(self.optimizer._param_groups, values): g['learning_rate'] = v; g['lr'] = v
        self._last_lr = values
    def get_last_lr(self): return self._last_lr
```

（注意：paddle param_groups 的 lr 键为 `learning_rate`，为兼容 torch 式读取同时写 `lr` 键；`group["lr"]` 递推读上一步值，两个键都维护）
- [ ] **Step 2: 写 normalize.py**（torchvision.transforms.Normalize 的最小复刻）：

```python
import numpy as np, paddle
class Normalize:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype='float32')   # 保持 numpy（train.py 读 .mean/.std 做反变换参数）
        self.std = np.asarray(std, dtype='float32')
    def __call__(self, t):  # t: paddle.Tensor，任意形状，numpy 广播
        mean = paddle.to_tensor(self.mean, dtype=t.dtype)
        std = paddle.to_tensor(self.std, dtype=t.dtype)
        return (t - mean) / std
```
- [ ] **Step 3: 调度器单测**（对照 torch 版逐点一致，历史范式：调度器必须单测锁定）：

```python
# paddletorch 环境，构造同参数 torch/paddle 两个 AdamW+调度器，步进 0,1,50,60,100,300,600 步
# 对比每步 get_last_lr()，abs_err 全部 < 1e-12
```

- [ ] **Step 4: Commit**: `git commit -am "Add paddle lr_scheduler and normalize utils with parity tests"`

### Task 4: 数据侧（dataset/datamodule/去 Lightning/去 torchvision）

**Files:**
- Modify: `climax_paddle/climax/climate_projection/dataset.py`（19,46 行 xarray 误报恢复；127,141 行 Normalize 替换；`torch.utils.data.Dataset` → `paddle.io.Dataset`；文件头 torch import 清理）
- Modify: `climax_paddle/climax/climate_projection/datamodule.py`（21 行基类去除；`torch.utils.data.DataLoader` → `paddle.io.DataLoader`；collate_fn 适配）
- Create: `climax_paddle/configs/climate_projection.yaml`（从 torch 版精简）
- Modify: `climax_paddle/tests/test_parallel_patch_embed.py`（若引用 torch 侧工具则同步修）

**Interfaces:**
- Consumes: Task 3 的 `Normalize`
- Produces（Task 5 依赖）:
  - `ClimateBenchDataset`（paddle.io.Dataset，`__getitem__` 返回 `(inp, out, lead_times, variables, out_variables)`，inp/out 为 paddle.Tensor）
  - `ClimateBenchDataModule(root_dir, history, list_train_simu, list_test_simu, variables, out_variables, train_ratio, batch_size, num_workers, pin_memory)`——普通类；构造时完成全部数据加载与划分（torch 版行为）；`.dataset_train/.dataset_val/.dataset_test`、`.train_dataloader()/.val_dataloader()/.test_dataloader()`、`.get_lat_lon()`、`.get_test_clim()`
  - yaml 结构：`seed / model{...net.init_args, lr 等, pretrained_path} / data{root_dir...} / train{max_epochs, default_root_dir}`

- [ ] **Step 1: dataset.py 修复**——恢复 2 处 xarray 链式调用（误报被 paconvert 拆成 `output_xr = (...)` 带标记，恢复为原链式写法）；2 处 `torchvision.transforms.transforms.Normalize(mean, std)` → `Normalize(mean, std)`（import 自 `climax.utils.normalize`）；`get_rmse_normalization` 里的 `torch.from_numpy` → `paddle.to_tensor`；Dataset 基类替换；collate 兼容性检查（`__getitem__` 返回值含 list[str]，paddle DataLoader 默认 collate 对非张量字段的处理：改用自定义 collate_fn 保持 tuple 结构）
- [ ] **Step 2: datamodule.py**——`>>>>>>class ClimateBenchDataModule(pytorch_lightning.LightningDataModule):` → `class ClimateBenchDataModule:`（删 `super().__init__()` 与 `save_hyperparameters`，hparams 访问点改直接属性）；collate_fn 里 `torch.stack/cat` → `paddle.stack/concat`；DataLoader 换 paddle.io.DataLoader（参数名核对：`collate_fn/num_workers` 一致）
- [ ] **Step 3: 精简 yaml**（复制 `climax_torch/configs/climate_projection.yaml`，删 `trainer.callbacks/logger/precision/strategy` 等 Lightning 字段，保留 seed_everything→seed、model 全段、data 全段，新增 `train: {max_epochs: 5, default_root_dir: /home/lkyu/baidu/CLIMAX/exps/paddle_baseline}`；`data.root_dir` 指向 `dataset/climatebench/5.625deg`）
- [ ] **Step 4: 数据侧冒烟**：PYTHONPATH=. 构造 DataModule，取一个 batch，断言形状 `(1,10,4,32,64)/(1,1,32,64)/(1,)` 且 dtype float32；对同一 index 对比 torch 侧 dataset 的张量值（np.allclose，容差 1e-6——数据管线一致性，历史范式 H1 步骤）
- [ ] **Step 5: Commit**: `git commit -am "Paddle-ize dataset/datamodule, drop lightning/torchvision deps"`

### Task 5: module 普通类化 + 自建训练循环

**Files:**
- Modify: `climax_paddle/climax/climate_projection/module.py`
- Modify: `climax_paddle/climax/climate_projection/train.py`（整体重写）

**Interfaces:**
- Consumes: Task 2 模型、Task 3 调度器、Task 4 DataModule
- Produces（Task 6/8 依赖）:
  - `ClimateProjectionModule`（普通类）：`__init__(net, pretrained_path, lr, beta_1, beta_2, weight_decay, warmup_epochs, max_epochs, warmup_start_lr, eta_min)`；方法 `training_step(batch,batch_idx)->loss`、`validation_step/test_step(batch,batch_idx)->dict`、`configure_optimizers()->(optimizer, scheduler)`、`set_denormalization/set_lat_lon/set_pred_range/set_val_clim/set_test_clim`、`load_mae_weights(pdparams_path)`
  - `train.py main()`：读 yaml + `--key=value` 覆盖 + 自建训练循环（训练/验证/检查点/早停/测试全流程）

- [ ] **Step 1: module.py 改造**——去 `pytorch_lightning.LightningModule` 基类；`save_hyperparameters(...)` → 直接赋 8 个 self 属性（`self.lr=lr` 等），`self.hparams.lr` 等 6 处访问点同步改；`self.log(...)` → `self.log_history.setdefault(name, []).append(float(value))`（一行实现，验证/测试指标由循环读取）；`load_mae_weights`：删 http 分支与 `paddle.hub`，`paddle.load(path)` 读 pdparams（约定结构 `{"state_dict": {键带 net. 前缀}}`，与转换脚本 Task 7 契约一致）；其余清洗逻辑（interpolate/channel→var/删键/strict=False→`set_state_dict(state, strict=False)` 返回结构处理）保持；`torch.load/map_location` 痕迹清理
- [ ] **Step 2: configure_optimizers paddle 化**——AdamW 双参数组：

```python
def configure_optimizers(self):
    decay, no_decay = [], []
    for name, p in self.net.named_parameters():
        if p.stop_gradient: continue          # 冻结参数不入优化器（freeze_encoder）
        if 'var_embed' in name or 'pos_embed' in name or 'time_pos_embed' in name: no_decay.append(p)
        else: decay.append(p)
    opt = paddle.optimizer.AdamW(
        learning_rate=self.lr, beta1=self.beta_1, beta2=self.beta_2, epsilon=1e-8,
        parameters=[{'params': decay, 'weight_decay': self.weight_decay},
                    {'params': no_decay, 'weight_decay': 0.0}])
    sched = LinearWarmupCosineAnnealingLR(opt, self.warmup_epochs, self.max_epochs,
                                          self.warmup_start_lr, self.eta_min)
    return opt, sched
```

- [ ] **Step 3: train.py 重写**（自建循环，同构 torch 侧 main() 的装配序列）：

```python
# 结构骨架：OmegaConf.load(yaml) + argparse --key=value 深覆盖；paddle.seed(seed)+np.random.seed(seed)+random.seed(seed)
# 装配（复刻 torch main()：denorm 参数、lat_lon、pred_range、val_clim、test_clim 五个 set_*）
# 训练循环：for epoch: net.train(); for i,batch in enumerate(train_dl): loss=module.training_step(...); loss.backward(); opt.step(); opt.clear_grad(); sched.step()
#   每 50 步 print(loss)（对齐 torch 基线日志粒度）
# 验证：net.eval(); with paddle.no_grad(): 聚合 validation_step 返回 dict 的均值 → val/w_mse 等
# 检查点：val/w_mse 最优保存 best.pdparams + 每 epoch 末 last.pdparams（paddle.save(net.state_dict())）
# 早停：patience=5（连续 5 epoch 无改善则停，复刻 yaml EarlyStopping）
# 测试：paddle.load(best) → net.set_state_dict → test_step 循环 → 打印指标表（对齐 torch 输出格式）
```

- [ ] **Step 4: Commit**: `git commit -am "Rewrite module/train as plain paddle classes with custom loop"`

### Task 6: 最小跑通（敏捷修复循环）

**Files:** Modify: 上述各文件（哪里报错修哪里，仅限 climax_paddle 内）

- [ ] **Step 1: 构造级冒烟**（不加载预训练权重，`model.pretrained_path=""`）：`PYTHONPATH=. python climax/climate_projection/train.py --train.max_epochs=1 --train.limit_batches=1`（train.py 实现 `--train.limit_batches=N` 快捷参数：训练/验证/测试各取前 N 批，dryrun 专用）
- [ ] **Step 2: 报错→敏捷修复→重试循环**：每个报错修复后记入实验日志（问题-修复一行式）；判断标准——修复限于 climax_paddle 内代码/接口适配为敏捷；涉及改 paddle 依赖版本/重写模型数学逻辑 → 停止并返回报告走疑难询问
- [ ] **Step 3: 全链路跑通标准**：1 epoch × 1 批的训练+验证+检查点保存+best 恢复+测试，输出指标表；初始 test/w_mse 与 torch 侧初始值（4.560）同量级（随机头，不必相等）
- [ ] **Step 4: Commit**: `git commit -am "Paddle minimal run passes full loop"`

### Task 7: 权重导出与转换

**Files:**
- Create: `climax_paddle/tools/export_torch_initial_state.py`
- Create: `climax_paddle/tools/convert_torch_to_paddle.py`

**Interfaces:**
- Produces: `models/climax_paddle/climax_initial.pdparams`（结构 `{"state_dict": {net.前缀键: paddle 张量}}`）供 Task 8 前向对齐与 Task 9 训练初始化

- [ ] **Step 1: export 脚本**——torch 侧构造 `ClimateProjectionModule(net=ClimaXClimateBench(配置参数), pretrained_path=models/climax_torch/5.625deg.ckpt)`（复刻 torch 基线装配，seed 42），`torch.save(module.state_dict(), models/climax_torch/climax_initial.pt)`；打印总参数量与键数（记录，供日志）
- [ ] **Step 2: 键名形状对照表**——paddle 侧构造同参模型，`state_dict()` 与 torch 侧逐键对比：键名集合 diff（应为空）、形状 diff 清单（预期仅 Linear 类 [out,in] vs [in,out] 差异；记录 compat.nn.Linear/MultiheadAttention 的实际布局——若与 torch 同布局则免转置）
- [ ] **Step 3: convert 脚本**——按 Step 2 实测规则：键名直映；需转置的 Linear 类 `w.T`；`torch.Tensor.numpy()` 中转 `paddle.to_tensor`；输出 `{"state_dict": ...}` 结构 pdparams；打印逐桶（token_embeds/blocks/var_agg/head/time_agg）张量数
- [ ] **Step 4: 回灌验证**——paddle module `load_mae_weights(pdparams)` 后 `set_state_dict` 无 missing/unexpected 键（strict 检查单独跑一次）
- [ ] **Step 5: Commit**: `git add climax_paddle/tools && git commit -m "Add torch->paddle weight export/conversion tools"`

### Task 8: 前向对齐

**Files:**
- Create: `climax_paddle/tools/compare_forward.py`

- [ ] **Step 1: 固定输入**——`np.random.RandomState(42)` 生成 `x=(1,10,4,32,64)`、`y=(1,1,32,64)` 存 npz；lat 用数据集实测 lat
- [ ] **Step 2: 两侧前向**——torch 侧：构造 module（加载 `climax_initial.pt`），`module.net.forward(x,y,lead_times,vars,out_vars,metric=[mse],lat)` eval 模式取 preds 存 `torch_out.npz`；paddle 侧同参构造+加载 pdparams，同输入 eval 取 preds 存 `paddle_out.npz`
- [ ] **Step 3: 五指标对比**（max/mean_abs_error、rmse、max/mean_rel_error，历史范式）；**验收：mean_abs_error < 1e-5 且 mean_rel_error 达 1e-6 量级**；不过则定位（分层打印 encoder 中间量）并修复，敏捷范畴内迭代
- [ ] **Step 4: 结果记入实验日志 + Commit**: `git commit -am "Forward alignment tools pass at 1e-6 scale"`

### Task 9: 5 epoch 短训对照 + 汇报暂停

- [ ] **Step 1: paddle 侧同款短训**——配置 `model.pretrained_path` 指向转换的 pdparams（共享初始主干+头权重），seed 42、FP32、batch_size=1、max_epochs=5 全量批次，nohup 后台（参照 torch 侧命令形态）
- [ ] **Step 2: 对照表**——每 epoch train loss（50 步采样均值）、val/w_mse、最终 test 指标 vs torch 基线锚点（0.145/0.300/0.252/0.212/0.408 末批；val 0.2503→0.2118；test w_mse 0.1390/w_rmse 0.3687）；量级一致（相对偏差 <10%）视为对齐通过；偏差大则记录现象、不擅自扩大调试（是否上受控对齐由用户决定）
- [ ] **Step 3: 全部结果记入实验日志（含命令与数字）+ Commit**
- [ ] **Step 4: 暂停并汇报进度**（按用户要求测完即停）

---

## 执行波次（subagent-driven-development 调度）

- **Wave 1（并行 4 代理）**: Task 1 / Task 2+3 / Task 4 / Task 5（接口契约已在计划锁定，无文件交叉）
- **Wave 2**: Task 6（最小跑通+敏捷修复，单代理，带停止边界）
- **Wave 3**: Task 7 → **Wave 4**: Task 8 → **Wave 5**: Task 9（严格串行）

## Self-Review 结论

- 覆盖检查：16 处手动 API（timm 7+Lightning 3+Normalize 3+xarray 误报 2+_LRScheduler 1）→ T1/T2（timm 7）、T4（误报 2+Normalize 3+LightningDataModule）、T5（LightningModule+LightningCLI）、T3（_LRScheduler）全覆盖；用户方案的四要素（闭包调研/改写方式/最小跑通/同款测试）→ T1/各Task/ T6/T8+9 对应 ✓
- 类型一致性：timm_paddle 四符号、Normalize、调度器、DataModule 接口在 T2/T4/T5 消费点逐一核对一致 ✓
- 已知不确定点（执行时验证）：compat.nn.Linear/MultiheadAttention 的实际权重布局（T7 Step 2 实测）；paddle DataLoader 对非张量 batch 字段的 collate（T4 Step 1 自定义 collate 规避）
