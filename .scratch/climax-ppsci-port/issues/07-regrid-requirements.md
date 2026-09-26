# 07: regrid 脚本与 example requirements

**Type:** task
**Status:** resolved
**Blocked by:** None (can start immediately)

## What to build

examples/climax/ 内数据重采样脚本与 example 级依赖清单自 PaddleCFD 原样平移。

## 实施要点

- `regrid_climatebench.py` 原样平移（源：PaddleCFD feat/climax `examples/climax/regrid_climatebench.py`）
- `requirements.txt` 照 fuxi 五行式（无版本号）：xarray、netCDF4、scipy 等 example 级实际所需（deep-dive-conventions.md §Q5：example 级 requirements 有 11 个先例）
- xesmf 属 conda 栈，安装说明归文档票（11/12）承担，脚本与 requirements 内不重复
- 平移后 grep 确认无 `ppcfd` 字样残留

## Acceptance criteria

- [x] 两文件在位、import 自洽（regrid 脚本仅标准库 + xesmf/xarray）
- [x] requirements.txt 语法有效（`pip install -r` 可解析，不必实际安装）
- [x] 无 ppcfd 残留引用

## Answer

### 做了什么

- 新建 `/home/lkyu/baidu/PaddleScience/examples/climax/`（此前不存在），落入两个文件：
  - `examples/climax/regrid_climatebench.py`：自 PaddleCFD feat/climax `examples/climax/regrid_climatebench.py` 字节级原样平移（`git show` 重定向，sha256 两侧一致：`ba90906e…3c6b4`，96 行），逻辑零改动。
  - `examples/climax/requirements.txt`：fuxi 五行式（无版本号、字母序）：`click / netCDF4 / numpy / scipy / xarray`。按脚本实际 import 头（click、xarray、numpy、xesmf + 标准库 os/glob）确定：xesmf 属 conda 栈按票面排除（安装说明归文档票 11/12），netCDF4/scipy 为 xarray 读写 nc 的后端依赖（票面点名）；click 是源脚本真实 import（原脚本用 `@click.command()` 做 CLI），故收录。

### 验收证据（命令真实执行）

1. 两文件在位、import 自洽：ast 解析 regrid 脚本，imports = `['click', 'glob', 'numpy', 'os', 'xarray', 'xesmf']`，第三方依赖恰为 click/numpy/xarray/xesmf（标准库 glob/os），`compile()` 语法 OK。xesmf 在 paddletorch 环境未装（conda 栈，属预期，不做运行时验证）。
2. requirements.txt 可解析：`python -m pip install --dry-run -r examples/climax/requirements.txt` 退出码 0，5 行全部解析成功（click 8.5.0 / netCDF4 1.7.4 / numpy 1.26.4 / scipy 1.14.1 / xarray 2025.6.1，均 already satisfied，未实际安装）。
3. 无 ppcfd 残留：`grep -in ppcfd` 两文件无匹配（exit 1）。

### 遗留事项

- 票面验收第一条写"仅标准库 + xesmf/xarray"，但源脚本事实上还 import click 与 numpy（原样平移不动逻辑），已据实收录进 requirements.txt；如后续票审视 example 全量 import，可再增删。
- examples/climax/ 其余文件（main.py/conf/datamodule 等）由其他票落位，本票只建了上述两文件。
- 运行产物清理：无 __pycache__/pyc 残留（`find` 计 0），/tmp 无本票脚本；PaddleScience 工作区仅新增 `?? examples/climax/`，未 commit/push。
