# 13: 四处文档注册

**Type:** task
**Status:** resolved
**Blocked by:** 11, 12

## What to build

ClimaX 在 ppsci 文档体系完成登记（Climateformer PR #1211 实际注册集）。

## 实施要点

- `mkdocs.yml` 地球科学分组（:106-130 区段）nav 一行（官方成文要求仅此一处，development.md:912）
- `docs/zh/api/arch.md` members 列表加 `ClimaX` 与 `ClimaXClimateBench`（字母序）
- `docs/zh/api/data/dataset.md` members 加 `ClimateBenchDataset`
- `docs/zh/index.md` 案例表（:174 附近）加一行（问题类型、案例名链接 ./examples/climax.md、数据集 ClimateBench zenodo 链接）
- docs/en 侧：api 两个 md 同步加行；examples 导航由 i18n 插件按目录镜像自动映射，**无需**另注册
- 根 README.md 大表**不做**（非强制：climateformer/meteoformer/preformer 均未登记）

## Acceptance criteria

- [x] 注册行与相邻行格式逐字段一致
- [x] mkdocs nav YAML 语法有效

## Answer

改动（PaddleScience @ feat/climax，未 commit）：

1. `mkdocs.yml:117`（票 12 已注册，本次仅核对）：`- ClimaX: examples/climax.md`，8 空格缩进，与相邻行 `ClimateFormer`(:116)/`GraphCast`(:118) 逐字段一致（单条 dict、显示名 → examples 相对路径），无需修正。
2. `docs/zh/api/arch.md:13-14`：members 列表插入 `- ClimaX`、`- ClimaXClimateBench`，位于 `Climateformer`(:12) 与 `CrystalGraphConvNet`(:15) 之间，符合列表大小写不敏感字母序（climateformer < climax < climaxclimatebench < crystalgraphconvnet），8 空格缩进与相邻行一致。
3. `docs/en/api/arch.md:13-14`：同款两行（en 与 zh 该文件逐行同构）。
4. `docs/zh/api/data/dataset.md:10`：插入 `- ClimateBenchDataset`，位于 `ChipHeatDataset`(:9) 与 `ContinuousNamedArrayDataset`(:11) 之间（chipheat < climatebench < continuous）。
5. `docs/en/api/data/dataset.md:10`：同款一行。
6. `docs/zh/index.md:175`：案例表（地球科学 tab 内）加行 `| 天气预报 | [ClimaX 气候预测](./examples/climax.md) | 数据驱动 | Transformer | 监督学习 | [ClimateBench](https://zenodo.org/record/7064308) | [Paper](https://arxiv.org/abs/2301.10343) |`，插在 Climateformer 行(:174) 后，4 空格缩进、7 列结构与相邻行逐字段一致（问题类型沿用气象区"天气预报"，Paper 链接格式同 Preformer/UTAE 行）。
7. `docs/en/index.md:175`：镜像行 `| Weather Forecasting | [ClimaX Climate Prediction](./examples/climax.md) | Data-driven | Transformer | Supervised Learning | [ClimateBench](https://zenodo.org/record/7064308) | [Paper](https://arxiv.org/abs/2301.10343) |`。票面清单未列此文件，但 en/index.md:174 存在 Climateformer 镜像行（#1261 i18n 拆分后 zh/en index 逐行同构），按"Climateformer 实际注册集"补齐以保持镜像一致。
8. 根 `README.md` 未动（票面明确不做，climateformer/meteoformer/preformer 先例均未登记）。

验证：

- mkdocs nav YAML：PyYAML 自定义 loader（`python/name:` tag 按不透明字符串处理，避免依赖 pymdownx 导入）解析通过；nav 地球科学组 22 条，ClimaX 位于 index 10，prev=ClimateFormer、next=GraphCast。
- 6 个改动文件 grep 检查无 CRLF、无 tab。
- `git diff` 为纯插入 12 行、0 删除。
- `from ppsci.arch import ClimaX, ClimaXClimateBench; from ppsci.data.dataset import ClimateBenchDataset` 导入成功（mkdocstrings members 可解析）。
- 无临时文件产生（校验脚本均内联 python -c）。
