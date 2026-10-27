"""ui.control — 界面精准定位控制层（项目无关，可整体移植）。

三件套 + 移植契约，2026-09-28 用户令立项：前端开发必须能精准控制
各组件元素的位置，且方案要能复用到同类型项目。

模块：
- geometry   位置契约断言 API（元素间关系：贴边/对齐/包含/间距）
- golden     黄金截图回归（capture → baseline diff → 热图 → 显式审批更新）
- inspector  运行时元素检查器（悬停显示 x/y/w/h/margins，⌥I 切换）

移植到其他项目（三步）：
1. 整体拷贝本目录 + scripts/lint_ui_literals.py + ui/tokens.py 骨架
2. 写新项目的 tokens.py 数值 + 契约清单（tests/test_geometry_contracts.py）
3. UI_GOLDEN_UPDATE=1 生成基线 PNG 入库
本目录代码不得 import 任何业务模块——发现即视为移植性破坏。
"""
