"""ui/tokens.py — 全应用布局像素值的**唯一来源**。

设计稿定稿后，数值写在这里。改数值 → ⌘R → theme.reload_tokens() → 全局 QSS
重新生成 + 主窗口 apply_tokens() 重放几何，不重启、肉眼验收。

用法：
    from ui.tokens import T
    T.BAND_H    # 顶带高
    T.CTRL_H    # 控件一档高
"""

# ── 窗口 ──
WIN_DEFAULT_W = 1440
WIN_DEFAULT_H = 900
WIN_MIN_W = 1152
WIN_MIN_H = 720

# ── 导航条 ──
RAIL_W = 60
RAIL_BTN_W = 54
RAIL_BTN_H = 50
RAIL_ICON = 46
RAIL_TOP_PAD = 8
RAIL_GAP = 6

# ── 顶带 ──
BAND_H = 40
BAND_SLOT_H = 36
BAND_STATUS_PAD_R = 8

# ── 树列 ──
TREE_W = 212
TREE_MIN = 180
TREE_MAX = 320
TREE_ROW_H = 28
TREE_TITLE_H = 28
TREE_INDENT_PROJECT = 26
TREE_INDENT_SOURCE = 0

# ── 控件一档制 ──
CTRL_H = 28
CTRL_SM_H = 26
CTRL_MIN_W_2CH = 60
CTRL_RADIUS = 6

# ── 行高 ──
ROW_H_TABLE = 28
ROW_H_TREE = 28
HEADER_H = 32
STAT_CARD_H = 32

# ── 分页栏（卡片底栏） ──
PAGER_H = 30
PAGER_BTN_H = 26

# ── 间距档 ──
SP_4 = 4
SP_8 = 8
SP_12 = 12
SP_16 = 16
SP_24 = 24

# ── 树列呼吸（18§3.8 包裹层左右） ──
TREE_WRAP_X = 0  # 容器级呼吸撤销（用户 2026-09-29：滚动区须完全填满容器；行内呼吸由委托 padding 保证）
TREE_WRAP_BOTTOM = 0  # 树底贴分隔线（用户 2026-09-29：底悬空显多余空隙）
TREE_VIEWPORT_TUCK = -6  # 消树视口顶部内衬（D7/任务页同款负内缩）
TREE_PAD_X = 10          # 树行内边距 L10 R10（委托）
TREE_ROW_INSET = 2       # 行框左右收 2（圆角不贴栏缘）
TREE_IND_W = 2           # 选中指示线宽
TREE_IND_X = 3           # 指示线距行框左缘（框收 2 + 1）
SIDEBAR_HANDLE_W = 1     # 树列拖宽手柄=分隔线宽（用户 2026-09-29：3→1）
TREE_IND_VPAD = 4        # 指示线上下留 4（高=行高-8）
TREE_IND_RADIUS = 1

# ── 工具栏 ──
TOOLBAR_MARGIN = 10
TABLE_CELL_PAD_X = 6
TABLE_CELL_PAD_Y = 2

# ── 素材宫格（19§D4.8a） ──
GRID_MARGIN = 10        # 面板内边距
GRID_GAP = 10           # 卡片间距
GRID_CARD_MIN_W = 190   # 列宽下限（auto-fill 依据 190+10）
DUR_BADGE_R = 4         # 时长角标圆角
DUR_BADGE_FS = 10       # 时长角标字号

# ── 卡片/标签 chips ──
CHIP_FS = 11
CHIP_RADIUS = 4
CHIP_PAD_X = 7
CHIP_GAP = 4
CHIP_MARGIN_TOP = 6
CHIP_PILL_PAD_X = 10    # 顶带标签 chip 水平内距（TB1.1）
CARD_META_LINE_GAP = 2  # 卡片文字区行距

# ── 搜索框（19§D1.3） ──
SEARCH_W = 180
SORT_COMBO_W = 90

# ── 复选列（18§4.10/19§D4.8b） ──
CHECK_COL_W = 36

# ── 字号六档 ──
FONT_HINT = 11
FONT_SECONDARY = 12
FONT_SMALL = 12.5      # 正文小档（QSS 解析为 13px，统一走 token 保一致）
FONT_BODY = 13
FONT_SECTION = 15
FONT_NUMBER = 18
FONT_TITLE = 22

# ── 顶带 sm 钮（19§D1.2） ──
SM_BTN_PAD_X = 12

# ── 宫格卡细节（19§D4.8a） ──
DUR_BADGE_PAD_X = 6    # 角标内距与距边
DUR_BADGE_CORNER = 6
CHECKBOX_OVERLAY = 18  # 封面勾选框边长
CHECKBOX_CORNER = 8    # 勾选框距角
CARD_META_PAD = 9      # 文字区 padding 7 9 9 的水平/底
CARD_META_PAD_TOP = 7

# ── 圆角 ──
RADIUS_PILL = 4
RADIUS_CTRL = 6
RADIUS_CARD = 8
RADIUS_DIALOG = 12

# ── 空态 hero（18§4.14） ──
HERO_ICON = 64
HERO_ICON_RADIUS = 16
HERO_EMOJI_FS = 30
HERO_TITLE_GAP = 8
HERO_DESC_GAP = 6
HERO_BTN_GAP = 10

# ── 底部 ──
BOTTOM_SAFE = 0




# ── 勾选框（LM1.2 全选 14px / 卡片 14px） ──
CHECKBOX_SM = 14


class T:
    """命名空间：from ui.tokens import T; T.BAND_H"""

    pass


# 注册循环必须位于文件最末：历史上 CHECKBOX_SM 追加在循环之后，
# T 缺失该属性 → 打包态（无 reload 兜底）页面构建崩
# （'T' has no attribute 'CHECKBOX_SM'，2026-09-29 打包版实测）
for _k, _v in dict(globals()).items():
    if _k.isupper():
        setattr(T, _k, _v)
        # 小写别名：theme().get('band_h') 与色彩 token 同一查询口，
        # 页面代码不必区分几何/色彩两套 API
        setattr(T, _k.lower(), _v)
