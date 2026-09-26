# 02: climax arch 包 Arch 化与注册

**Type:** task
**Status:** resolved
**Blocked by:** 01

## What to build

`ppsci/arch/climax/` 包内 ClimaX 与 ClimaXClimateBench 成为 `base.Arch` 子类（dict 进 dict 出），完成主包注册；完整构造参数量与已验证值一致。

## 实施要点

- 构造参数含 input_keys/output_keys：input_keys=["x","lead_times"]，output_keys 按 out_variables 派生（如 ["tas"]）；`forward(input_dict) -> output_dict`，内部解包转调原逻辑（deep-dive-solver-pipeline.md"接入路径判断"方式 1）
- 常量元数据（default_vars/out_vars/img_size/time_history/patch_size/embed_dim/depth/num_heads/mlp_ratio/drop_path/drop_rate/freeze_encoder 等）全部 `__init__` 参数（经 cfg.MODEL 传入）
- forward 剥 y/metric/lat：y 进 label_dict（网络不用 y）、lat 进 loss/metric 闭包（网络不用 lat，deep-dive-climax-side.md §3）；climatebench 的 loss 计算移出（Q5，FunctionalLoss 在 example 侧），时间聚合 + head 留在 arch
- lead_times 输入为 [B,1] Tensor（dataset 侧 ndarray 经 collate/DataLoader 自动转）
- pos_embed 保 ppcfd 数值布局：两仓 2D sincos 版实测不一致（meshgrid 序错位，deep-dive-climax-side.md §8），**不得**换用 `ppsci/arch/cvit.py` 版本
- `climax/__init__.py`：Apache 2.0 PaddlePaddle 头（年份 2026）+ 导出 ClimaX/ClimaXClimateBench；`ppsci/arch/__init__.py` 无条件 import 注册 + `__all__`（ppsci 惯例，不用 try-import）
- 源：PaddleCFD feat/climax 的 `ppcfd/models/climax/{arch,climatebench,parallelpatchembed,pos_embed}.py` 与 `examples/climax/configs/climate_projection.yaml`（构造默认值参考）

## Acceptance criteria

- [x] import 链冒烟（`from ppsci.arch.climax import ClimaX, ClimaXClimateBench`）
- [x] 完整构造（time_history=10）参数量 = 111,831,040
- [x] 08 等价门前向对照通过（08 未就绪，待 08 票执行数值对照后勾选）

## Answer

### 做了什么

新增 `PaddleScience/ppsci/arch/climax/` 包（5 个文件，均带 Apache 2.0 PaddlePaddle 头，年份 2026）并注册到主包：

- `arch.py`：`ClimaX(base.Arch)`，dict 进 dict 出。`__init__` 新增 `input_keys`/`output_keys`（ppsci 惯例，先 `super().__init__()` 再赋值，照 climateformer.py），并新增 `out_vars`（None→default_vars，str 归一为 tuple）固化原 `out_variables` 通道切片；`forward(input_dict)` 解包 x/lead_times → `forward_encoder` → head → unpatchify → out_vars 切片 → `{output_keys[0]: preds}`。原 forward 的 loss（metric 列表）与 `evaluate()` 整段移出（归 example 侧）。支持 `_input_transform`/`_output_transform` 钩子（Arch 惯例）。
- `climatebench.py`：`ClimaXClimateBench(ClimaX)`，同构改造；时间聚合 + `head(embed_dim, H*W)` 留在 arch；`freeze_encoder` 的 `requires_grad_(False)`（blocks 非 norm 参数）原样保留。
- `pos_embed.py`：ppcfd 数值布局逐行平移（`meshgrid(grid_w, grid_h)` 序保持，未参考 cvit.py 版本）。
- `parallelpatchembed.py`：平移保留独立文件，`timm` import 改 `ppsci.arch.paddle_timm`。
- `__init__.py`：导出两个类；`ppsci/arch/__init__.py` 无条件 import（isort:skip 格式照相邻条目）+ `__all__` 按字母序插入 ChipDeepONets 之后。

**关键改造决定（lead_times 形状）**：实测（原 ppcfd 模型 + instrumented block）证明 ppcfd 前向只接受 flat `[B]` lead_times——其自定义 collate 对 `[1]` 样本 `paddle.concat` 后仍为 `[1]`；喂 `[1,1]` 时原 ppcfd 代码同样在 Attention 处 `too many values to unpack`。机理：`Linear` 将 `(B,1)` 映射为 `(B,D)`，随后 unsqueeze 出 `(B,1,D)` 与 token 序列正确广播。ppsci 的 default_collate_fn 产出 `[B,1]`，故在两个类的 `forward` 顶部 `lead_times = lead_times.reshape([-1])` 归一（值保持，ClimateBench 任务恒为 0.0，bit-exact；B=1 时与 ppcfd 路径完全同秩同算子）。附带使 batch>1 也跑通（ppcfd 本身仅 bs=1 有效）。

### 关键数字（验收实录，最终状态）

- AC1 import+注册：`from ppsci.arch.climax import ClimaX, ClimaXClimateBench` OK；`ppsci.arch.ClimaXClimateBench is` 导入类 OK
- AC2 参数量：以 conf/climax.yaml MODEL 段全 15 键构造 → `num_params = 111,831,040`（锚点逐位一致）；freeze_encoder=True 下 named_parameters 总量同为 111,831,040（frozen=64 全在 blocks 非 norm，trainable=59，blocks norm 参数保持可训）
- AC3 前向：x=(1,10,4,32,64)、lead_times=(1,1) 随机输入 → `out["tas"].shape = [1, 1, 32, 64]`
- 附加：base ClimaX 小配置前向 OK（注意 base 的 out_vars 须为 default_vars 子集，head 按 default_vars 出通道）；batch=2 对 `[B,1]` 与 `[B]` lead_times 均跑通；两模块 doctest 各 5 项全过（`print(list(shape))` 形式规避 paddle.Size 打印差异）；isort/black 全部通过

### 遗留

- 08 票数值等价门未执行（08 未就绪）：本票只验 shape/参数量；forward 代码路径与 ppcfd 同构（同算子同序），lead_times flatten 值保持，预期 bit-exact，待 08 正式对照
- decoder_depth 未出现在 MODEL yaml，作为默认参数（=2）保留在构造签名中；base ClimaX 的 head 先建后由 ClimateBench 子类覆写（照 ppcfd 原序，最终参数量不含被替换 head）
- 08 已验证：climax_initial.pdparams strict 直载（123/123 键）eval 前向，合成与真实输入 preds max_abs_diff = 0.0（逐位一致）
