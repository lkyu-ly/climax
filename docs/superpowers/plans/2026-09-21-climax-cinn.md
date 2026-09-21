# ClimaX CINN 对齐实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 climax_paddle 接入 CINN 编译加速（环境变量单开关 + to_static 分支），完成第三次前向/训练对齐验证与确定性加速比测量，结果合并记入实验记录。

**Architecture:** 按踩坑手册坑四的成熟模式——单一环境变量 `CLIMAX_USE_CINN` 同时控制三个 FLAGS（`import paddle` 前设置）与 `paddle.jit.to_static(net, full_graph=True)` 包裹（权重加载之后）。`full_graph` 优先全量（True/AST），遇难绕问题降级 False（SOT），仍难绕则暂停走疑难询问。对齐验证与前两次同口径（torch 基线锚点不变），加速比用 per-step 计时 + 2 epoch 稳态窗口。

**Tech Stack:** paddle 3.4.0（to_static 的 `full_graph` 已移至 **kwargs、`backend='CINN'` 已为默认值）、paddletorch 环境、RTX 4060 Ti。

## Global Constraints

- 敏捷开发仅解决提出的问题，禁止过度设计和假想问题；CINN 疑难问题（需大改模型逻辑/改框架依赖）**暂停询问**
- 单开关：`CLIMAX_USE_CINN`，**默认 "0"**（保护既有动态图基线行为，验证命令显式 =1 开启）；开 = FLAGS 三件套 true + `FLAGS_print_ir=false` + to_static 包裹；关 = 全 false + 不包裹（纯动态原样）
- FLAGS 必须在 `import paddle` 之前设置（train.py / compare_forward.py 入口最前）
- to_static 包裹位置：权重加载**之后**（参数对象共享不破坏 optimizer）
- 生效标志：日志出现 `Compiling subgraph with CINN backend`（主判据）；`all prim enabled: True` 为辅
- full_graph 策略：先 True；报错难绕 → 降级 False（SOT）并在报告记录；仍难绕 → BLOCKED 走疑难询问，不得自行大改
- 精度验收：CINN 前向 mean_abs_error < 1e-5（相对动态图 2.09e-08 放宽，留融合重排余量）；CINN 5 epoch 训练最终 test 指标 vs torch 基线相对偏差 <10%（与第二次测试同口径）
- 加速比：确定性数据，主指标 = 第 2 epoch 纯训练段墙钟（剔除首 epoch 的 CINN 一次性编译与两侧 GPU 预热），辅以稳态每步中位数（剔除两侧各前 50 步），报告端到端训练加速百分比
- 全部结果（第三次前向/训练测试 + 加速比）合并记入 `ClimaX迁移实验记录.md`（controller 统一记）
- 不做国产显卡验证（本轮明确排除）
- 已预检的坑位对照（踩坑手册）：坑一（trace 路径无 .view/.size/.repeat——两处 `w.view` 在构造期，不在 trace 路径）✓、坑五（forward 显式参数无 **kwargs）✓、坑六（返回 tuple）✓、坑八（无 einops）✓、坑九（无复数）✓；**待实测点**：`paddle.einsum("bvld->blvd")`（arch.py:178）、`repeat_interleave`（arch.py:180、climate_projection/arch.py:97）、无参 `x.squeeze()`（arch.py:182）、`lru_cache` 装饰的 `get_var_ids`（arch.py:149，坑七变体）、compat MultiheadAttention 训练态 trace
- 基线锚点（不变）：torch 5 epoch test w_mse=0.13899536 / w_rmse=0.36871007；paddle 动态图 5 epoch test w_mse=0.13394081 / w_rmse=0.36180352；动态图训练段约 4:27/epoch

---

### Task 1: CINN 开关与 to_static 接入 + 单批冒烟

**Files:**
- Modify: `climax_paddle/climax/climate_projection/train.py`（入口 FLAGS 块 + to_static 分支）
- Modify: `climax_paddle/tools/compare_forward.py`（同款开关，供 Task 2）

**Interfaces:**
- Produces（Task 2/3 依赖）:
  - 环境变量 `CLIMAX_USE_CINN`（"1" 开 / 其他关），train.py 与 compare_forward.py 行为一致
  - train.py 内 `CLIMAX_USE_CINN=1` 时：net 加载初始状态后 `net = paddle.jit.to_static(net, full_graph=True)`（降级时 full_graph=False，代码里以常量/环境变量 `CLIMAX_FULL_GRAPH` 控制，默认 "1"）

- [ ] **Step 1: train.py 入口（import paddle 之前）插入开关块**（手册坑四模式）：

```python
import os
_USE_CINN = os.environ.get("CLIMAX_USE_CINN", "0") == "1"
if _USE_CINN:
    os.environ["FLAGS_prim_enable_dynamic"] = "true"
    os.environ["FLAGS_prim_all"] = "true"
    os.environ["FLAGS_use_cinn"] = "true"
    os.environ.setdefault("FLAGS_print_ir", "false")
else:
    os.environ["FLAGS_prim_enable_dynamic"] = "false"
    os.environ["FLAGS_prim_all"] = "false"
    os.environ["FLAGS_use_cinn"] = "false"

import paddle  # noqa: E402 （必须在上面之后）
```

（注意现有 import 顺序：paddle 相关 import 全部移到开关块之后）
- [ ] **Step 2: to_static 分支**——net 加载 init_state/pretrained 权重之后：

```python
if os.environ.get("CLIMAX_USE_CINN", "0") == "1":
    full_graph = os.environ.get("CLIMAX_FULL_GRAPH", "1") == "1"
    net = paddle.jit.to_static(net, full_graph=full_graph)
    print(f"[CINN] to_static enabled, full_graph={full_graph}")
```

（module 持有 net 引用——确认 module 内 self.net 与包裹后对象一致：包裹发生在 module 构造前则直接传包裹后对象；若 module 已构造则同步 `module.net = net`）
- [ ] **Step 3: compare_forward.py 同款开关**（paddle 侧脚本入口 + net 构造后包裹）
- [ ] **Step 4: 单批 CINN 冒烟**：

```bash
cd /home/lkyu/baidu/CLIMAX/climax_paddle && CLIMAX_USE_CINN=1 PYTHONPATH=. \
  /home/lkyu/miniconda3/envs/paddletorch/bin/python climax/climate_projection/train.py \
  configs/climate_projection.yaml --train.limit_batches=1 --train.max_epochs=1 \
  --model.pretrained_path="" --model.init_state_path=/home/lkyu/baidu/CLIMAX/models/climax_paddle/climax_initial.pdparams \
  --train.default_root_dir=/home/lkyu/baidu/CLIMAX/exps/cinn_smoke 2>&1 | tee /home/lkyu/baidu/CLIMAX/exps/cinn_smoke.log
```

验收：完整循环 EXIT=0；日志含 `Compiling subgraph with CINN backend`；初始 test/w_mse 与 4.54 基本一致（CINN 数值差异容差内）
- [ ] **Step 5: 报错处理协议**（按手册对照，顺序执行）：
  - 坑一类报错（`'int' object has no attribute '__call__'` / `must be double, but got str`）→ grep trace 路径确认无兼容层方法后按实际报错定位
  - 坑二类（`view_shape_grad`）→ 确认 trace 路径无 `.view` 后按实际定位
  - 坑七类（lru_cache/IRMapping）→ get_var_ids 去缓存或 `paddle.in_dynamic_mode()` 守卫
  - einsum/repeat_interleave/squeeze 报错 → 数值等价的原生替换（reshape+transpose / repeat+reshape / 指定 dim 的 squeeze），改动限于 climax_paddle，逐项记录
  - full_graph=True 难绕 → `CLIMAX_FULL_GRAPH=0` 重跑（SOT）；仍难绕 → 状态 BLOCKED 返回，附最小报错信息，**不自行大改**
- [ ] **Step 6: 对照冒烟**（CLIMAX_USE_CINN=0 同命令）确认开关关闭时行为与原基线一致（无 Compiling 日志、指标同前）
- [ ] **Step 7: Commit**: `git commit -m "Add CINN switch (env var + to_static) with smoke-tested full_graph path"`

### Task 2: CINN 前向对齐（第三次测试·前向）

**Files:** 无新文件（用 Task 1 改造后的 compare_forward.py）

- [ ] **Step 1: CINN 前向对比**：

```bash
CLIMAX_USE_CINN=1 PYTHONPATH=. /home/lkyu/miniconda3/envs/paddletorch/bin/python tools/compare_forward.py all
```

（torch 侧不变；paddle 侧 to_static + eval 前向。注意 eval 模式下 to_static 的 train/eval Program 分离（坑七背景），metric=None 路径无 numpy 交互）
- [ ] **Step 2: 验收**：mean_abs_error < 1e-5 且 mean_rel_error 与动态图结果（3.01e-06）同量级或可解释偏差；超差 → 分层定位（compare_forward 的 --trace）按 Task 1 Step 5 协议处理
- [ ] **Step 3: 结果数字完整留存**（报告文件，供 controller 记实验记录）
- [ ] **Step 4: Commit**（若 compare_forward.py 有适配改动）: `git commit -m "Forward alignment under CINN passes"`

### Task 3: CINN 训练对齐 + 确定性加速比（第三次测试·训练）

**Files:**
- Modify: `climax_paddle/climax/climate_projection/train.py`（per-step 计时输出，轻量）

**Interfaces:**
- Produces: 训练日志含每步耗时（`[timing] epoch=N step=M cum=..s avg=..s` 每 50 步一行，且每 epoch 末输出该 epoch 纯训练段墙钟——数据加载与验证剔除）

- [ ] **Step 1: per-step 计时**——训练循环内 `time.perf_counter()` 包 training_step+backward+opt.step+clear_grad+sched.step（不含数据取批），每 50 步打印累积/均值，epoch 末打印本 epoch 纯训练总时（默认始终开启，动态图侧同样生效——两侧同口径计时）
- [ ] **Step 2: 动态图计时基线**（2 epoch）：

```bash
CLIMAX_USE_CINN=0 ... train.py configs/climate_projection.yaml --train.max_epochs=2 \
  --model.pretrained_path="" --model.init_state_path=.../climax_initial.pdparams \
  --train.default_root_dir=/home/lkyu/baidu/CLIMAX/exps/speed_dynamic > exps/speed_dynamic.log
```

- [ ] **Step 3: CINN 5 epoch 训练**（CLIMAX_USE_CINN=1，同 init_state_path，输出 exps/cinn_baseline/ + exps/cinn_baseline_train.log）——一次运行同时服务计时与精度对照
- [ ] **Step 4: 加速比计算**（确定性数据）：
  - 主指标：两侧**第 2 epoch 纯训练段墙钟**对比 → 端到端训练加速百分比 `(t_dyn - t_cinn) / t_dyn * 100%`
  - 辅指标：稳态每步中位数（两侧各剔除前 50 步）；CINN 首次编译耗时单列（第 1 epoch 与第 2 epoch 的时间差主因）
  - 若加速 ≈0 或为负：如实报告（历史区间 0%~28%，不保证收益），不调参不美化
- [ ] **Step 5: 精度对照**（与第二次测试同口径）：CINN 5 epoch 的 val/w_mse 逐 epoch、最终 test 五项 vs torch 基线（偏差 <10%）及 vs paddle 动态基线；loss 曲线量级趋势一致
- [ ] **Step 6: 全部数字写入报告文件**（第三次测试合并结果 + 加速比，供 controller 记实验记录）
- [ ] **Step 7: Commit**: `git commit -m "Add per-step timing; CINN training alignment and speedup measured"`

---

## 执行波次（subagent-driven-development 调度）

串行：Task 1 → Task 2 → Task 3（CINN 强迭代调试性质，无并行空间；每任务实现+审查+fix loop）

## Self-Review 结论

- 覆盖：用户要求全部映射——开关（T1）、full_graph 优先全量可降级（T1 Step 5 + CLIMAX_FULL_GRAPH）、手册对照（Global Constraints 预检清单 + T1 处理协议）、第三次前向/训练测试合并记录（T2/T3 + controller 记日志）、确定性加速比（T3：epoch 口径 + 剔除编译 + 稳态中位数 + 端到端百分比）、敏捷/最小/疑难暂停（Global Constraints）
- 版本差异已核实：paddle 3.4 to_static 的 full_graph 在 kwargs、backend 默认 CINN、FLAGS 名称不变
- 类型一致性：CLIMAX_USE_CINN / CLIMAX_FULL_GRAPH 两个环境变量在 T1/T2/T3 中的语义一致
