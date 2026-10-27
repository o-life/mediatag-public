"""设置页面 - 全局配置与环境检查"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QFormLayout, QFileDialog,
    QMessageBox, QSpinBox, QFrame, QSizePolicy, QCheckBox, QTextEdit,
    QApplication,
    QRadioButton, QButtonGroup, QGridLayout)
from PySide6.QtCore import Qt, Signal
from .api_client import APIClient
from .theme import theme

try:
    from ..widgets.async_loader import AsyncLoader
except ImportError:  # 脚本直跑
    from widgets.async_loader import AsyncLoader


class SettingsPage(QWidget):
    """设置页：含分析主题管理（列表/新建/删除/切换）。"""

    analysis_theme_changed = Signal(str)  # 切换分析主题后广播（主窗口刷新各页）
    manage_tags_requested = Signal()      # 请求打开标签管理页（维护当前主题标签库）
    """设置页面"""

    # 用户点「环境初始化」→ 主窗口弹出下载勾选对话框并执行下载
    env_init_requested = Signal()

    def __init__(self, api_base: str = "http://localhost:8000", parent=None):
        super().__init__(parent)
        self.api = APIClient(base_url=api_base)
        self._loaders = []  # AsyncLoader 引用，防 GC
        self._build_ui()
        self._load_available_schemes()
        self._load_config()
        self._loading = False  # 首次加载完成，开始接受自动保存
        self._load_analysis_themes()

    def _fetch_async(self, fetch, apply, fail=None):
        """设置页统一后台加载：点击后页面立即出现，数据主线程回填
        （用户定案：先反馈后加载——原先构造时同步拉 5 个接口导致转彩球）。"""
        loader = AsyncLoader(fetch)
        loader.finished.connect(apply)
        if fail is not None:
            loader.error.connect(fail)
        self._loaders.append(loader)
        loader.run()

    def _on_nav_changed(self, cur, _prev):
        """设置分区树切换 → 切换右侧分区页。"""
        if cur is not None:
            name = cur.text(0)
            idx = list(self._pages.keys()).index(name) if name in self._pages else 0
            self._stack.setCurrentIndex(idx)

    def _new_form(self, name):
        """在指定分区页新建统一风格的 QFormLayout（14 号方案三段式）。

        内容贴顶打包（inner + 底部 stretch）——消除行间被拉伸出的大空白
        （用户报告：排版差、大量空白、不像付费精品）。
        同分区重复调用时复用首个表单——否则第二个空壳 form 自带的底部
        stretch 会落在页面中段，把后续行推出一大段空白（分析主题页实测）。
        """
        existing = self._forms.get(name)
        if existing is not None:
            return existing
        v = self._pages[name]
        inner = QWidget()
        inner.setMaximumWidth(960)
        f = QFormLayout(inner)
        f.setContentsMargins(0, 4, 0, 0)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(10)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        v.addWidget(inner)
        v.addStretch(1)  # 剩余空间推到底部，行间不再被拉开
        self._forms[name] = f
        return f

    def _build_ui(self):
        """构建UI——水平三段式：设置分区树 + 单分区详情页（14 号方案 v2）

        过渡策略：顶层改三段骨架后，`root` 重指右侧页布局——既有分区代码
        零改动继续工作；后续逐段把 root 内容迁入七个独立分区页。
        """
        from PySide6.QtWidgets import QTreeWidget

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── 设置分区树（左）──
        t0 = theme()
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderHidden(True)
        self.nav_tree.setFixedWidth(212)  # 与主树列同宽（P4 拍板）
        self.nav_tree.setStyleSheet(
            f"QTreeWidget {{ background: {t0.get('bg_secondary')}; border: none;"
            f" color: {t0.get('text_primary')}; font-size: 13px; }}"
            f" QTreeWidget::item {{ height: 28px; padding: 0px 8px; border-radius: 6px; }}"
            f" QTreeWidget::item:selected {{ background: {t0.get('accent')}; color: #FFFFFF; }}"
        )
        for label in ("分析主题", "性能方案", "分析设置", "播放器",
                      "分析引擎", "其他功能"):
            from PySide6.QtWidgets import QTreeWidgetItem
            QTreeWidgetItem(self.nav_tree, [label])
        self.nav_tree.currentItemChanged.connect(self._on_nav_changed)
        self._loading = True  # _build_ui/_load_config 期间抑制自动保存
        _nav_wrap = QWidget()
        _nav_wrap.setStyleSheet(
            f"background: {t0.get('bg_secondary')}; border-right: 1px solid {t0.get('border')};")
        _nav_v = QVBoxLayout(_nav_wrap)
        _nav_v.setContentsMargins(8, 10, 8, 10)
        _nav_v.addWidget(self.nav_tree)
        outer.addWidget(_nav_wrap)

        # ── 右侧：七分区堆栈 ──
        from PySide6.QtWidgets import QStackedWidget
        self._stack = QStackedWidget()
        outer.addWidget(self._stack, 1)
        self._pages = {}
        self._forms = {}
        for _name in ("分析主题", "性能方案", "分析设置", "播放器",
                      "分析引擎", "其他功能"):
            _pg = QWidget()
            _v = QVBoxLayout(_pg)
            _v.setContentsMargins(16, 12, 16, 16)
            _v.setSpacing(10)
            self._stack.addWidget(_pg)
            self._pages[_name] = _v
        root = self._pages["分析主题"]  # 迁移期：构建顺序中按分区切换 root 落页

        # 设置页面背景色
        t = theme()
        # 背景统一（2026-09-29 用户令）：内容区=bg_card，与其他五页一致
        self.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")

        # ── 分析主题（批8：列表/新建/删除/切换；数据库随主题自动管理）──
        theme_row = QHBoxLayout()
        theme_row.setContentsMargins(0, 0, 0, 0)
        theme_row.setSpacing(8)
        theme_row.addWidget(QLabel("分析主题:"))
        self.analysis_theme_combo = QComboBox()
        self.analysis_theme_combo.setStyleSheet(self._input_style())
        self.analysis_theme_combo.setMinimumWidth(200)
        self.analysis_theme_combo.currentTextChanged.connect(
            self._on_analysis_theme_combo_changed)
        self._theme_loading = False
        self._active_theme = None
        theme_row.addWidget(self.analysis_theme_combo, 1)
        theme_new_btn = QPushButton("＋新建主题")
        theme_new_btn.setStyleSheet(self._btn_style())
        theme_new_btn.clicked.connect(self._new_analysis_theme)
        theme_row.addWidget(theme_new_btn)
        # S13：⚙编辑配置（config.yaml 键值表）移除——不把代码给用户看；
        # 改名以独立轻量入口保留（原藏于键值表的 theme.name 改法）
        theme_rename_btn = QPushButton("✎改名")
        theme_rename_btn.setStyleSheet(self._btn_style())
        theme_rename_btn.setToolTip("重命名当前分析主题（目录与注册表同步改名）")
        theme_rename_btn.clicked.connect(self._rename_analysis_theme_dialog)
        theme_row.addWidget(theme_rename_btn)
        theme_del_btn = QPushButton("🗑删除主题")
        theme_del_btn.setStyleSheet(self._btn_style())
        theme_del_btn.clicked.connect(self._delete_analysis_theme)
        theme_row.addWidget(theme_del_btn)
        root.addLayout(theme_row)
        theme_hint = QLabel(
            "切换主题将自动保存当前设置，并切换该主题的数据库/标签库/提示词；"
            "数据库路径自动管理，无需手动配置（重启应用后完全生效）。"
            "每个主题 = 一个 config 文件，锚定该主题的全部资源。"
        )
        theme_hint.setStyleSheet(f"""
            font-size: {t.get('font_size_xs')}px;
            color: {t.get('text_secondary')};
        """)
        root.addWidget(theme_hint)

        # ── 当前主题资源面板（每主题一个 config 文件锚定全部资源）──
        res_box = QFrame()
        res_box.setObjectName("ThemeResources")
        res_box.setStyleSheet(f"""
            QFrame#ThemeResources {{
                background: {t.get('bg_secondary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
            }}
            QLabel {{ background: transparent; }}
        """)
        res_v = QVBoxLayout(res_box)
        res_v.setContentsMargins(12, 10, 12, 10)
        res_v.setSpacing(6)
        res_head = QLabel("当前主题资源（随主题自动管理）")
        res_head.setStyleSheet(f"font-size: {t.get('font_size_sm')}px; font-weight: 600;")
        res_v.addWidget(res_head)

        def _res_row(title):
            row = QHBoxLayout()
            row.setSpacing(8)
            cap = QLabel(title)
            cap.setFixedWidth(70)
            row.addWidget(cap)
            path_label = QLabel("—")
            path_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: {t.get('font_size_xs')}px;")
            row.addWidget(path_label, 1)
            reveal = QPushButton("在访达中显示")
            reveal.setFixedHeight(28)
            reveal.setStyleSheet(self._btn_style())
            reveal.clicked.connect(lambda: self._reveal_in_finder(path_label))
            row.addWidget(reveal)
            res_v.addLayout(row)
            return path_label, reveal

        self.res_db_label, self.res_db_reveal = _res_row("数据库")
        self.res_faces_label, self.res_faces_reveal = _res_row("人脸目录")
        tags_row = QHBoxLayout()
        tags_row.setSpacing(8)
        tags_cap = QLabel("标签库")
        tags_cap.setFixedWidth(70)
        tags_row.addWidget(tags_cap)
        self.res_tags_label = QLabel("—")
        self.res_tags_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: {t.get('font_size_xs')}px;")
        tags_row.addWidget(self.res_tags_label, 1)
        tags_goto = QPushButton("前往标签管理")
        tags_goto.setFixedHeight(28)
        tags_goto.setStyleSheet(self._btn_style())
        tags_goto.clicked.connect(self.manage_tags_requested.emit)
        tags_row.addWidget(tags_goto)
        res_v.addLayout(tags_row)
        root.addWidget(res_box)

        root = self._pages["分析引擎"]
        # S13 ⑪：原「切换离线识别」独立钮移入本地形态面板（见下方引擎表单），
        # 顶部状态行整块移除。

        # ── 设置表格（按分区拆页：每区独立 QFormLayout） ─────────
        root = self._pages["性能方案"]
        form = self._new_form("性能方案")

        # 性能方案
        # 性能方案（S13 ⑨：三档平铺单选——场景+并发参数+定位说明全亮在选项里；
        # 点选即生效走自动保存，无「应用」钮；「方案准备」保留为一键硬件推荐）
        self.scheme_radios = QButtonGroup(self)
        self.scheme_radios.buttonToggled.connect(self._on_scheme_toggled)
        self._scheme_radio_v = QVBoxLayout()
        self._scheme_radio_v.setSpacing(6)
        form.addRow("性能方案:", self._scheme_radio_v)

        prep_h = QHBoxLayout()
        self.env_prepare_btn = QPushButton("方案准备 · 按本机硬件自动检测并应用推荐")
        self.env_prepare_btn.setStyleSheet(self._btn_style())
        self.env_prepare_btn.setToolTip("按当前硬件资源动态检测并应用推荐性能方案")
        self.env_prepare_btn.clicked.connect(self._prepare_scheme)
        prep_h.addWidget(self.env_prepare_btn)
        prep_h.addStretch()
        form.addRow("", prep_h)
        perf_hint = QLabel(
            "三档按本机硬件自动生成：轻量=最低占用，边工作边后台跑；"
            "性能均衡=平衡性能与资源，日常推荐；性能优先=最大吞吐，专注跑库时使用。\n"
            "切换后自动重启调度器，正在处理的任务不受影响。")
        perf_hint.setWordWrap(True)
        perf_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # P3
        perf_hint.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')};")
        form.addRow("", perf_hint)

        root = self._pages["分析主题"]
        form = self._new_form("分析主题")

        # 人脸存储路径（过渡期保留于此；终态并入分析主题表单⑤）
        face_path_h = QHBoxLayout()
        self.face_storage_path_edit = QLineEdit()
        self.face_storage_path_edit.setReadOnly(True)
        self.face_storage_path_edit.setStyleSheet(self._input_style())
        face_path_h.addWidget(self.face_storage_path_edit, 1)
        self.face_storage_browse_btn = QPushButton("浏览")
        self.face_storage_browse_btn.setStyleSheet(self._btn_style())
        self.face_storage_browse_btn.clicked.connect(self._browse_face_storage)
        face_path_h.addWidget(self.face_storage_browse_btn)
        form.addRow("人脸存储路径:", face_path_h)

        root = self._pages["播放器"]
        form = self._new_form("播放器")

        # 播放器（S13 ⑦⑩：平铺单选四项；安装状态实时探测——未安装置灰不可选；
        # 路径行移除，装进「应用程序」文件夹即自动探测）
        self.player_radios = QButtonGroup(self)
        self._player_radio_v = QVBoxLayout()
        self._player_radio_v.setSpacing(6)
        for key, label, desc in (
            ("系统默认", "系统自带（默认）", "仅能打开视频文件，无跳转功能"),
            ("iina", "iina", "免费 · 支持时间轴跳转"),
            ("mpv", "mpv", "免费 · 支持时间轴跳转"),
            ("vlc", "vlc", "免费 · 支持时间轴跳转"),
        ):
            installed = self._detect_player_installed(key)
            rb = QRadioButton(f"{label} · {desc}"
                              + ("" if installed else " —— 未安装，安装后自动可选"))
            rb.setProperty("player_key", key)
            rb.setStyleSheet(self._input_style())
            rb.setEnabled(installed)
            self.player_radios.addButton(rb)
            self._player_radio_v.addWidget(rb)
        form.addRow("播放器:", self._player_radio_v)
        player_hint = QLabel(
            "iina / mpv / vlc 免费且支持时间轴跳转：点击标签封面即可跳到对应画面，"
            "标签阅览体验更完整，推荐安装选用。安装状态实时探测；"
            "若已选播放器被卸载，播放时自动回退系统默认并提示。")
        player_hint.setWordWrap(True)
        player_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # P3
        player_hint.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')};")
        form.addRow("", player_hint)

        root = self._pages["分析设置"]
        form = self._new_form("分析设置")

        # S13：派发策略/映射抽帧上限/合并pass/NSFW 过滤/NFO 行全部移除——
        # 管线内部参数由产品定死（派发=按导入顺序，映射沿用密度，合并恒开，
        # NSFW 随主题属性，NFO 固定格式固定路径），UI 不再出现。

        # 抽帧密度（14 号定稿：按时长动态三档，帧/分钟）
        dens_h = QHBoxLayout()
        self.density_group = QButtonGroup(self)
        for _label, _val in (("省电 · 2 帧/分钟", 2), ("标准 · 4 帧/分钟", 4),
                             ("精细 · 8 帧/分钟", 8)):
            _rb = QRadioButton(_label)
            _rb.setProperty("density", _val)
            _rb.setStyleSheet(self._input_style())
            self.density_group.addButton(_rb, _val)
            dens_h.addWidget(_rb)
        self.density_group.button(4).setChecked(True)
        form.addRow("抽帧密度:", dens_h)
        dens_hint = QLabel("按视频时长自动计算每场景帧数；越精细标签越细、分析越慢。"
                           "精细档接近实时耗时，建议搭配云厂商。")
        dens_hint.setWordWrap(True)
        dens_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # P3
        dens_hint.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')};")
        form.addRow("", dens_hint)
        # （NFO 行已按 S13 移除：固定 Kodi XML、与视频同目录同名，零设置零提及）

        root = self._pages["其他功能"]
        form = self._new_form("其他功能")

        # 人物自动命名（高阶，默认关；内容管理者场景）
        self.auto_naming_check = QCheckBox("开启人物自动命名（默认关闭）")
        self.auto_naming_check.setToolTip(
            "开启后，聚类完成的人物将自动根据文件名/水印识别结果命名。\n"
            "适合内容管理者（演员跨视频复用）；影楼用户无需开启。")
        form.addRow("", self.auto_naming_check)
        naming_hint = QLabel(
            "把文件名中固定出现的非人名信息（厂牌名、系列名等）填到下方排除列表——"
            "设置好后，这些信息不会干扰人物自动命名。\n"
            "例：XX摄影公司-王小明-2026-4k.mp4 不排除厂牌 → 人物可能被命名为「XX摄影公司」。")
        naming_hint.setWordWrap(True)
        naming_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # P3
        naming_hint.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')};")
        form.addRow("", naming_hint)

        # 文件名解析：忽略关键词（每行一个；人物自动命名·排除词，高阶）
        self.ignore_keywords_edit = QTextEdit()
        self.ignore_keywords_edit.setPlaceholderText(
            "每行一个忽略词。文件名解析时会先把这些词从文件名里剔除，"
            "再识别人名/编号，避免把平台名、清晰度当成名字。\n"
            "示例：bilibili（平台名）、1080p（清晰度）、iPhone（设备名）"
        )
        self.ignore_keywords_edit.setMaximumHeight(110)
        self.ignore_keywords_edit.setStyleSheet(self._input_style())
        self.ignore_keywords_edit.setToolTip(
            "作用：解析文件名时，先删除这些关键词，再用剩下的部分识别人名/编号。\n"
            "适用：视频平台名（bilibili）、清晰度（1080p/4K）、压制组、设备名等无意义词。\n"
            "规则：每行一个；不区分大小写；命中任一忽略词的候选名会被剔除。"
        )
        form.addRow("文件名忽略词:", self.ignore_keywords_edit)

        # ── 帧保留三态（14 号定稿：引用保留模型——证据帧永不删，只清未引用中间帧）
        # （S13：并入首个"分析设置"表单——原独立空壳 form 的标题+stretch 造成两行间大空白）
        form = self._forms["分析设置"]
        cleanup_h = QHBoxLayout()
        cleanup_h.setSpacing(8)
        self.cleanup_mode_combo = QComboBox()
        self.cleanup_mode_combo.addItems(["用完即弃", "保留 N 天", "永不（手动清理）"])
        self.cleanup_mode_combo.setMinimumWidth(150)
        self.cleanup_mode_combo.setStyleSheet(self._input_style())
        cleanup_h.addWidget(self.cleanup_mode_combo)
        self.cleanup_days_spin = QSpinBox()
        self.cleanup_days_spin.setRange(1, 365)
        self.cleanup_days_spin.setValue(7)
        self.cleanup_days_spin.setSuffix(" 天")
        self.cleanup_days_spin.setStyleSheet(self._input_style())
        cleanup_h.addWidget(self.cleanup_days_spin)
        self.cleanup_stats_btn = QPushButton("统计")
        self.cleanup_stats_btn.setStyleSheet(self._btn_style())
        self.cleanup_stats_btn.clicked.connect(self._get_cleanup_stats)
        cleanup_h.addWidget(self.cleanup_stats_btn)
        self.cleanup_execute_btn = QPushButton("立即清理")
        self.cleanup_execute_btn.setStyleSheet(self._btn_style())
        self.cleanup_execute_btn.clicked.connect(self._execute_cleanup)
        self.cleanup_execute_btn.setEnabled(False)
        cleanup_h.addWidget(self.cleanup_execute_btn)
        cleanup_h.addStretch()
        form.addRow("帧保留:", cleanup_h)
        cleanup_hint = QLabel("用完即弃：分析完成即清（封面帧保留）；保留 N 天：到期自动清理；"
                              "永不：仅手动清理。标签证据帧与封面在任何模式下都不会被删除。")
        cleanup_hint.setWordWrap(True)
        cleanup_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # P3
        cleanup_hint.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')};")
        form.addRow("", cleanup_hint)
        self.cleanup_stats_label = QLabel("")
        self.cleanup_stats_label.setStyleSheet(f"font-size: 12px; color: {t.get('text_secondary')};")
        self.cleanup_stats_label.setVisible(False)
        form.addRow("", self.cleanup_stats_label)

        root = self._pages["分析引擎"]
        form = self._new_form("分析引擎")

        # S13 ⑪：识别方式双形态——本地极简（没细节可展示就不硬塞）/
        # 云端富信息（多供应商并存=多套餐；地址/Key/模型 ID 就地可改自动保存）。
        # 原「管理供应商」弹窗与「切换离线识别」独立钮由该形态吸收。
        self.engine_mode_group = QButtonGroup(self)
        mode_v = QVBoxLayout()
        mode_v.setSpacing(6)
        self.rb_engine_local = QRadioButton("本地离线（MiniCPM-V） —— 零成本，数据不出本机")
        self.rb_engine_cloud = QRadioButton("云端 VLM —— 更快更细；多供应商/套餐并存，随时切换")
        for _rb in (self.rb_engine_local, self.rb_engine_cloud):
            _rb.setStyleSheet(self._input_style())
            self.engine_mode_group.addButton(_rb)
            mode_v.addWidget(_rb)
        form.addRow("识别方式:", mode_v)
        self.engine_mode_group.buttonToggled.connect(self._on_engine_mode_changed)

        # 本地形态：极简面板（权重校验/下载沿用 S8 的 env_init 流程）
        self.engine_local_pane = QWidget()
        lp = QVBoxLayout(self.engine_local_pane)
        lp.setContentsMargins(0, 4, 0, 0)
        lp.setSpacing(8)
        local_res = QFrame()
        local_res.setObjectName("EngineLocal")
        local_res.setStyleSheet(
            f"QFrame#EngineLocal {{ background: {t.get('glass')};"
            f" border: 1px solid {t.get('border')}; border-radius: 10px; }}")
        lg = QGridLayout(local_res)
        lg.setContentsMargins(12, 10, 12, 10)
        lg.setHorizontalSpacing(10)
        lg.setVerticalSpacing(7)
        lg.addWidget(QLabel("模型"), 0, 0)
        self.engine_model_label = QLabel("MiniCPM-V · 权重已内置（就绪状态以下方校验为准）")
        self.engine_model_label.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')};")
        lg.addWidget(self.engine_model_label, 0, 1)
        self.env_init_btn = QPushButton("校验本地权重（未下载则引导下载）")
        self.env_init_btn.setToolTip(
            "下载本地模型权重后即可离线识别（数据不出本机）；\n"
            "已下载时点击仅校验本地权重有效性，不重复下载")
        self.env_init_btn.setStyleSheet(self._btn_style())
        self.env_init_btn.setFixedHeight(28)
        self.env_init_btn.clicked.connect(self._run_env_init)
        lg.addWidget(self.env_init_btn, 0, 2)
        lg.addWidget(QLabel("数据安全"), 1, 0)
        _safe = QLabel("分析全程数据不出本机")
        _safe.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')};")
        lg.addWidget(_safe, 1, 1)
        lg.setColumnStretch(1, 1)
        lp.addWidget(local_res)
        form.addRow("", self.engine_local_pane)

        # 云端形态：供应商卡片列表（多套餐并存）
        self.engine_cloud_pane = QWidget()
        cp = QVBoxLayout(self.engine_cloud_pane)
        cp.setContentsMargins(0, 4, 0, 0)
        cp.setSpacing(8)
        self._vendor_cards_v = QVBoxLayout()
        self._vendor_cards_v.setSpacing(8)
        cp.addLayout(self._vendor_cards_v)
        add_h = QHBoxLayout()
        self.vendor_add_btn = QPushButton("＋ 添加供应商")
        self.vendor_add_btn.setStyleSheet(self._btn_style())
        self.vendor_add_btn.clicked.connect(self._add_vendor_card)
        add_h.addWidget(self.vendor_add_btn)
        add_h.addStretch()
        cp.addLayout(add_h)
        cloud_hint = QLabel(
            "点「启用」选择当前生效的供应商（可存多个套餐随时换）；"
            "地址 / API Key / 模型 ID 就地修改、自动保存，改完点「测试连接」即时验证"
            "（测试会发送真实图片，验证协议、视觉与输出格式）。")
        cloud_hint.setWordWrap(True)
        cloud_hint.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')};")
        cp.addWidget(cloud_hint)
        form.addRow("", self.engine_cloud_pane)

        self.rb_engine_local.setChecked(True)
        self.engine_cloud_pane.setVisible(False)
        # 初始化供应商列表（异步，回填卡片并同步识别方式 radio）
        self._refresh_vendors()

        root = self._pages["分析主题"]
        form = self._new_form("分析主题")

        # VLM 提示词（S13 ⑧：常驻展开——主题核心资产直接摆出来；
        # 跟随上方主题选择；「从标签库生成」打底，专家可自由修改）
        self.prompt_container = QWidget()
        prompt_layout = QVBoxLayout(self.prompt_container)
        prompt_layout.setContentsMargins(0, 4, 0, 0)
        prompt_layout.setSpacing(6)
        self._prompt_theme_data = {}  # 主题名 → id（加载/保存用）

        self.vlm_prompt_edit = QTextEdit()
        self.vlm_prompt_edit.setPlaceholderText("点「从标签库生成提示词」打底，或直接输入自定义提示词…")
        self.vlm_prompt_edit.setMinimumHeight(150)
        self.vlm_prompt_edit.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                border: 1px solid {t.get('border_light')};
                border-radius: {t.get('radius_md')}px;
                padding: 8px;
                background: {t.get('bg_card')};
                color: {t.get('text_primary')};
            }}
        """)
        prompt_layout.addWidget(self.vlm_prompt_edit)

        prompt_btn_h = QHBoxLayout()
        self.gen_prompt_btn = QPushButton("从标签库生成提示词")
        self.gen_prompt_btn.setToolTip("以当前主题标签库为素材生成骨架；已有内容会先确认覆盖")
        self.gen_prompt_btn.setStyleSheet(self._btn_style())
        self.gen_prompt_btn.clicked.connect(self._generate_prompt_from_tags)
        prompt_btn_h.addWidget(self.gen_prompt_btn)
        self.save_prompt_btn = QPushButton("保存提示词")
        self.save_prompt_btn.setStyleSheet(self._btn_style())
        self.save_prompt_btn.clicked.connect(self._save_vlm_prompt)
        prompt_btn_h.addWidget(self.save_prompt_btn)
        prompt_btn_h.addStretch()
        self.prompt_status_label = QLabel("")
        self.prompt_status_label.setStyleSheet(f"font-size: 12px; color: {t.get('text_secondary')};")
        prompt_btn_h.addWidget(self.prompt_status_label)
        prompt_layout.addLayout(prompt_btn_h)
        form.addRow("", self.prompt_container)

        # 标签库与数据库（一行合并）
        dict_row = QHBoxLayout()
        dict_row.setSpacing(8)
        self.dict_stats_label = QLabel("标签: 0/0")
        self.dict_stats_label.setStyleSheet(f"font-size: 13px; color: {t.get('text_secondary')};")
        dict_row.addWidget(self.dict_stats_label)
        self.dict_refresh_btn = QPushButton("刷新")
        self.dict_refresh_btn.setStyleSheet(self._btn_style())
        self.dict_refresh_btn.clicked.connect(self._refresh_dict_stats)
        dict_row.addWidget(self.dict_refresh_btn)
        self.dict_export_btn = QPushButton("导出")
        self.dict_export_btn.setStyleSheet(self._btn_style())
        self.dict_export_btn.clicked.connect(self._export_dictionary)
        dict_row.addWidget(self.dict_export_btn)
        self.dict_import_btn = QPushButton("导入")
        self.dict_import_btn.setStyleSheet(self._btn_style())
        self.dict_import_btn.clicked.connect(self._import_dictionary)
        dict_row.addWidget(self.dict_import_btn)
        dict_row.addStretch()
        form.addRow("标签库:", dict_row)

        # 数据库路径不再展示（批8 定稿）：路径由分析主题自动管理
        # （<AppSupport>/MediaTag/data/themes/<主题>/data.db，经 path_manager 解析）。

        root.addSpacing(20)

        # ── 底部状态条（自动保存模式：无保存/重置/关于按钮，关于移菜单栏）──
        self._pending_restart_keys = []
        status_h = QHBoxLayout()
        status_h.setContentsMargins(0, 2, 0, 0)
        self.autosave_label = QLabel("✓ 所有更改已自动保存")
        self.autosave_label.setStyleSheet(
            f"font-size: 11px; color: {t.get('success')};")
        status_h.addWidget(self.autosave_label)
        status_h.addStretch()
        self.restart_hint = QLabel("")
        self.restart_hint.setStyleSheet(
            f"font-size: 11px; color: {t.get('warning')};")
        self.restart_hint.setVisible(False)
        status_h.addWidget(self.restart_hint)
        self.restart_btn = QPushButton("立即重启")
        self.restart_btn.setStyleSheet(self._btn_style())
        self.restart_btn.setVisible(False)
        self.restart_btn.clicked.connect(self._restart_application)
        status_h.addWidget(self.restart_btn)

        # S11（用户裁定 2026-09-27）：主题切换/新建后页面下方常驻提醒——
        # 静默切换不打断调整，但"重启才完全生效"必须有处可看。
        # 不随 _mark_autosaved 复位（继续调整配置时提醒不能消失），重启后自然消失。
        self.theme_restart_banner = QLabel("")
        self.theme_restart_banner.setStyleSheet(
            f"font-size: 11px; color: {t.get('warning')};")
        self.theme_restart_banner.setVisible(False)
        self.theme_restart_btn = QPushButton("立即重启")
        self.theme_restart_btn.setStyleSheet(self._btn_style())
        self.theme_restart_btn.setVisible(False)
        self.theme_restart_btn.clicked.connect(self._restart_application)
        theme_banner_h = QHBoxLayout()
        theme_banner_h.setContentsMargins(0, 2, 0, 0)
        theme_banner_h.addWidget(self.theme_restart_banner)
        theme_banner_h.addStretch()
        theme_banner_h.addWidget(self.theme_restart_btn)
        root.addLayout(theme_banner_h)

        root.addLayout(status_h)

        # ── 自动保存接线：任一控件改动 → 防抖 600ms → 静默全量保存 ──
        from PySide6.QtCore import QTimer
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(600)
        self._autosave_timer.timeout.connect(self._autosave_now)
        self.cleanup_mode_combo.currentTextChanged.connect(
            lambda _t: self._sync_cleanup_days_visible())
        self._sync_cleanup_days_visible()
        for _w, _sig in (
            (self.density_group.button(2), "toggled"),
            (self.density_group.button(4), "toggled"),
            (self.density_group.button(8), "toggled"),
            (self.ignore_keywords_edit, "textChanged"),
            (self.cleanup_days_spin, "valueChanged"),
            (self.cleanup_mode_combo, "currentTextChanged"),
            (self.auto_naming_check, "toggled"),
        ):
            getattr(_w, _sig).connect(lambda *_: self._autosave_timer.start())
        self.player_radios.buttonToggled.connect(self._on_player_toggled)

        # ── 控件一档制兜底（17 号规范）：构建完成后全页钳 28，消除
        # QSS padding 换算残差（实测 29/32 混入）。QTextEdit 多行控件除外。
        from PySide6.QtWidgets import QSpinBox as _QSp
        for _w in (self.findChildren(QPushButton) + self.findChildren(QComboBox)
                   + self.findChildren(QLineEdit) + self.findChildren(_QSp)):
            try:
                _w.setFixedHeight(28)
            except RuntimeError:
                pass  # 已被销毁的重建控件

    def _sync_cleanup_days_visible(self):
        """「保留 N 天」模式才显示天数输入。"""
        if hasattr(self, "cleanup_days_spin"):
            self.cleanup_days_spin.setVisible(
                self.cleanup_mode_combo.currentText().startswith("保留"))

    def _autosave_now(self):
        if getattr(self, "_loading", False):
            return
        self._save_config(silent=True)
        self._mark_autosaved()

        # 授权状态区已按 2026-09-24 定稿移除（授权信息统一在「授权管理」窗口展示）

    def apply_theme(self):
        """换肤：按控件类型全量重放样式，不重建控件、保留页面状态。

        旧版逐控件点名重涂——设置页最小化批新增控件未同步名单，还残留
        已删按钮（reset_btn/about_btn/save_btn）的坏引用，主题切换时抛
        AttributeError 令主窗回退整页重建、状态丢失。改为 findChildren
        扫描后新增控件自动覆盖，无需维护名单。
        """
        from .theme import theme
        t = theme()
        # 背景统一（2026-09-29 用户令）：内容区=bg_card，与其他五页一致
        self.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")
        # 分区树（构造时按主题着色，换肤须重放）
        self.nav_tree.setStyleSheet(
            f"QTreeWidget {{ background: {t.get('bg_secondary')}; border: none;"
            f" color: {t.get('text_primary')}; font-size: 13px; }}"
            f" QTreeWidget::item {{ height: 28px; padding: 0px 8px; border-radius: 6px; }}"
            f" QTreeWidget::item:selected {{ background: {t.get('accent')}; color: #FFFFFF; }}"
        )
        for _b in self.findChildren(QPushButton):
            _b.setStyleSheet(self._btn_style())
        for _e in self.findChildren(QLineEdit):
            _e.setStyleSheet(self._input_style())
        for _c in self.findChildren(QComboBox):
            _c.setStyleSheet(self._input_style())
        for _r in self.findChildren(QRadioButton):
            _r.setStyleSheet(self._input_style())
        for _s in self.findChildren(QSpinBox):
            _s.setStyleSheet(self._input_style())
        for _x in self.findChildren(QTextEdit):
            _x.setStyleSheet(self._input_style())
        # 卡片容器（资源面板/引擎本地面板/供应商卡）
        res_box = self.findChild(QFrame, "ThemeResources")
        if res_box is not None:
            res_box.setStyleSheet(f"""
                QFrame#ThemeResources {{
                    background: {t.get('bg_secondary')};
                    border: 1px solid {t.get('border')};
                    border-radius: {t.get('radius_md')}px;
                }}
                QLabel {{ background: transparent; }}
            """)
        for _name in ("EngineLocal", "VendorCard"):
            for _f in self.findChildren(QFrame, _name):
                _f.setStyleSheet(
                    f"QFrame#{_name} {{ background: {t.get('glass')};"
                    f" border: 1px solid {t.get('border')}; border-radius: 10px; }}")
        self.vlm_prompt_edit.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                border: 1px solid {t.get('border_light')};
                border-radius: {t.get('radius_md')}px;
                padding: 8px;
                background: {t.get('bg_card')};
                color: {t.get('text_primary')};
            }}
        """)

    def _set_info_label(self, label: QLabel, text: str, color: str = None):
        """设置信息 label: 空文本自动隐藏(不占 QFormLayout 行高),
        有内容才显示——历史: 多个空 QLabel 占满一行造成视觉空白行。"""
        if not text:
            label.setVisible(False)
            label.setText("")
            return
        label.setText(text)
        label.setVisible(True)
        if color:
            t = theme()
            label.setStyleSheet(f"font-size: 12px; color: {t.get(color, color)};")

    def _btn_style(self) -> str:
        """统一按钮样式（中性色纯文字；字号与全 app 按钮一致 font_size_md）"""
        t = theme()
        return f"""
            QPushButton {{
                background: {t.get('btn_glass_bg')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 4px 12px;
                font-size: {t.get('font_size_md')}px;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
            }}
        """

    def _input_style(self) -> str:
        """统一输入控件样式"""
        t = theme()
        return f"""
            background: {t.get('bg_secondary')};
            border: 1px solid {t.get('border')};
            padding: 4px 8px;
            border-radius: {t.get('radius_md')}px;
        """
    
    def _load_available_schemes(self):
        """加载性能方案（后台拉硬件信息，主线程回填）。"""
        self._fetch_async(
            lambda: self.api.get("/hardware/info"),
            self._apply_available_schemes,
        )

    def _apply_available_schemes(self, result):
        """三档平铺单选（S13 ⑨）：场景名+并发参数+定位说明亮在选项里。"""
        try:
            presets = result.get("presets", {}) or {}
            default_names = {"economy": "极致节省", "balanced": "性能均衡",
                             "performance": "性能优先"}
            metas = (
                ("economy", "轻量占用，边工作边后台跑"),
                ("balanced", "平衡性能与资源，日常推荐"),
                ("performance", "最大吞吐，专注跑库时使用"),
            )
            # 重建前清旧钮（方案准备/切主题会重入）
            for _rb in list(self.scheme_radios.buttons()):
                self.scheme_radios.removeButton(_rb)
                _rb.deleteLater()
            while self._scheme_radio_v.count():
                _it = self._scheme_radio_v.takeAt(0)
                if _it.widget() is not None:
                    _it.widget().deleteLater()

            wanted = getattr(self, "_loaded_scheme_name", "性能均衡")
            for key, positioning in metas:
                preset = presets.get(key, {}) or {}
                name = preset.get("name") or default_names[key]
                mc = preset.get("max_concurrent", "?")
                vc = preset.get("max_vlm_concurrent", "?")
                rb = QRadioButton(f"{name} · 并发 {mc} · VLM {vc} —— {positioning}")
                rb.setProperty("scheme_key", key)
                rb.setProperty("scheme_name", name)
                rb.setStyleSheet(self._input_style())
                self.scheme_radios.addButton(rb)
                self._scheme_radio_v.addWidget(rb)
                if name == wanted:
                    rb.setChecked(True)
            buttons = self.scheme_radios.buttons()
            if buttons and self.scheme_radios.checkedButton() is None:
                buttons[1].setChecked(True)  # 缺省性能均衡
        except Exception:
            # 拉取失败：给最小可用骨架（名称取本地缺省）
            if not self.scheme_radios.buttons():
                for _name in ("极致节省", "性能均衡", "性能优先"):
                    rb = QRadioButton(_name)
                    rb.setProperty("scheme_name", _name)
                    rb.setStyleSheet(self._input_style())
                    self.scheme_radios.addButton(rb)
                    self._scheme_radio_v.addWidget(rb)
                    if _name == "性能均衡":
                        rb.setChecked(True)

    def _on_scheme_toggled(self, _rb, checked):
        """点选即生效（S13 ⑨）：自动保存模型，无需「应用」。"""
        if getattr(self, "_loading", False) or not checked:
            return
        self._apply_performance_scheme(silent=True)

    def _current_scheme_name(self) -> str:
        rb = self.scheme_radios.checkedButton()
        return rb.property("scheme_name") if rb is not None else "性能均衡"

    def _apply_performance_scheme(self, silent: bool = False):
        """应用性能方案：真正写入调度器并发配置。

        方案 → 从 /hardware/info 取对应预设 → 写入 scheduler.max_concurrent
        等配置 → 重启调度器生效。silent=自动保存模型（行内反馈，不弹窗）。
        """
        scheme = self._current_scheme_name()
        scheme_key = {
            "极致节省": "economy",
            "性能均衡": "balanced",
            "性能优先": "performance",
            "极致性能": "performance",
        }.get(scheme)
        if not scheme_key:
            if not silent:
                QMessageBox.warning(self, "提示", f"未知性能方案: {scheme}")
            return

        try:
            # 1. 获取硬件预设（含各方案并发参数）
            hw = self.api.get("/hardware/info")
            preset = hw.get("presets", {}).get(scheme_key, {})

            # 2. 写入调度器并发配置（真正生效）
            if "max_concurrent" in preset:
                self.api.put("/config/scheduler.max_concurrent", json={"value": preset["max_concurrent"]})
            if "max_vlm_concurrent" in preset:
                self.api.put("/config/scheduler.max_vlm_concurrent", json={"value": preset["max_vlm_concurrent"]})
            if "max_sifting_concurrent" in preset:
                self.api.put("/config/scheduler.max_sifting_concurrent", json={"value": preset["max_sifting_concurrent"]})
            if "max_md5_concurrent" in preset:
                self.api.put("/config/scheduler.max_md5_concurrent", json={"value": preset["max_md5_concurrent"]})

            # 3. 保存方案名
            self.api.put("/config/performance.scheme", json={"value": scheme})

            # 4. 重启调度器以应用新配置
            self.api.post("/scheduler/restart")

            if silent:
                self.autosave_label.setText(f"✓ 已应用性能方案「{scheme}」")
                self.autosave_label.setStyleSheet(
                    f"font-size: 11px; color: {theme().get('success')};")
            else:
                detail = preset.get("max_concurrent", "?")
                vlm = preset.get("max_vlm_concurrent", "?")
                QMessageBox.information(
                    self, "成功",
                    f"已应用性能方案: {scheme}\n"
                    f"总并发 {detail}, VLM 并发 {vlm}\n调度器已重启以应用新配置",
                )
        except Exception as e:
            if silent:
                self.autosave_label.setText(f"✗ 性能方案应用失败：{e}")
                self.autosave_label.setStyleSheet(
                    f"font-size: 11px; color: {theme().get('danger')};")
            else:
                QMessageBox.critical(self, "错误", f"应用性能方案失败: {e}")
    
    def _generate_prompt_from_tags(self):
        """以当前主题标签库为素材生成提示词骨架（S13 ⑧）；已有内容先确认覆盖。

        空标签库不放行——引导先到下方标签库行导入/建立。"""
        try:
            stats = self.api.get("/dictionary/stats")
            total = int(stats.get("total_tags", 0) or 0)
        except Exception:
            total = 0
        if total <= 0:
            self.prompt_status_label.setText(
                "⚠ 标签库为空，暂无法生成——请先在下方导入或建立标签，再来生成。")
            self.prompt_status_label.setStyleSheet(
                f"color: {theme().get('warning')};")
            return
        current = self.vlm_prompt_edit.toPlainText().strip()
        if current and QMessageBox.question(
                self, "覆盖确认",
                "编辑器里已有提示词内容，确定以当前标签库重新生成并覆盖吗？",
                ) != QMessageBox.StandardButton.Yes:
            return
        cats = []
        try:
            entries = self.api.get("/dictionary").get("tags", []) or []
            grouped = {}
            for t in entries:
                cname = t.get("category") or "未分类"
                grouped.setdefault(cname, [])
                if t.get("name"):
                    grouped[cname].append(t["name"])
            for cname, tags in grouped.items():
                cats.append(f"- {cname}: " + ", ".join(tags))
        except Exception:
            pass
        if not cats:
            cats = ["- （未取到标签内容，请检查标签库后重试）"]
        skeleton = (
            "你是专业视频标签分析助手。请仔细观察给出的视频帧，"
            "从下方标签体系中选出所有匹配的标签。\n\n"
            "## 可用标签体系\n" + "\n".join(cats) + "\n\n"
            "## 输出要求\n"
            "- 只输出命中的标签，逗号分隔；不要输出标签体系之外的词。\n"
            "- 不确定的内容宁可不选。\n"
        )
        self.vlm_prompt_edit.setPlainText(skeleton.replace("\\n", "\n"))
        self.prompt_status_label.setText("✓ 已按标签库生成骨架，可自由修改后保存。")
        self.prompt_status_label.setStyleSheet(
            f"color: {theme().get('success')};")

    def _save_vlm_prompt(self):
        """保存提示词到当前生效主题（S13 ⑧：主题资产归主题，无全局选项）。"""
        prompt = self.vlm_prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "警告", "提示词不能为空")
            return
        active = getattr(self, "_active_analysis_theme", "") or ""
        theme_id = self._prompt_theme_data.get(active)
        try:
            if theme_id:
                self.api.put(f"/themes/{theme_id}/prompt", json={"prompt": prompt})
                self.prompt_status_label.setText(f"✅ 已保存到「{active}」")
            else:
                self.api.post("/config/vlm-prompt", json={"prompt": prompt})
                self.prompt_status_label.setText("✅ 已保存（全局）")
            self.prompt_status_label.setStyleSheet(f"color: {theme().get('success')};")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存提示词失败: {e}")

    def _refresh_vendors(self):
        """拉取供应商 → 重建云端卡片；同步识别方式 radio（本地激活=本地形态）。"""
        self._fetch_async(
            lambda: self.api.get("/vlm/vendors"),
            self._apply_vendors,
            lambda msg: self._engine_fail(f"无法加载供应商: {msg}"),
        )

    def _engine_fail(self, msg):
        """引擎区加载失败：行内反馈（S13 ⑪）。

        顶带/页面窄行禁放原始异常（URL/Max retries 数百字符——标签管理页
        同款治理）：只显示一句短提示，细节走 stdout。"""
        print(f"[settings] 引擎区加载失败: {msg}", flush=True)
        brief = ("无法连接服务，请确认后端已启动"
                 if ("Connection" in msg or "Max retries" in msg)
                 else "加载失败，请稍后重试")
        self.autosave_label.setText(f"✗ {brief}")
        self.autosave_label.setStyleSheet(
            f"font-size: 11px; color: {theme().get('danger')};")

    def _apply_vendors(self, data):
        vendors = data.get("vendors", {}) or {}
        active = data.get("active", "")
        self._vendors_data = vendors
        self._active_vendor = active
        self._card_status = {}

        # 清旧卡（移除按钮组里的旧 radio 一并清掉）
        while self._vendor_cards_v.count():
            _it = self._vendor_cards_v.takeAt(0)
            if _it.widget() is not None:
                _it.widget().deleteLater()
        self._vendor_enable_group = QButtonGroup(self)
        for name, cfg in vendors.items():
            self._vendor_cards_v.addWidget(self._make_vendor_card(name, cfg, name == active))

        # 识别方式 radio 同步（blockSignals 防回环触发 activate）
        local_active = bool(active) and self._vendor_is_local(vendors.get(active), active)
        self.engine_mode_group.blockSignals(True)
        (self.rb_engine_local if local_active else self.rb_engine_cloud).setChecked(True)
        self.engine_mode_group.blockSignals(False)
        self.engine_local_pane.setVisible(local_active)
        self.engine_cloud_pane.setVisible(not local_active)

    @staticmethod
    def _vendor_is_local(cfg, name):
        """本地供应商判定：名称/地址含 ollama 或 11434，或 folder 型。"""
        cfg = cfg or {}
        blob = f"{name} {cfg.get('base_url', '')} {cfg.get('vendor', '')}".lower()
        return "ollama" in blob or "11434" in blob or cfg.get("type") == "folder"

    def _on_engine_mode_changed(self, _rb, checked):
        if not checked or getattr(self, "_loading", False):
            return
        if self.rb_engine_cloud.isChecked():
            self.engine_local_pane.setVisible(False)
            self.engine_cloud_pane.setVisible(True)
            return
        self.engine_local_pane.setVisible(True)
        self.engine_cloud_pane.setVisible(False)
        # 切到本地：激活本地供应商（如有），卡片与生效态经 _refresh_vendors 刷新
        for name, cfg in (getattr(self, "_vendors_data", {}) or {}).items():
            if self._vendor_is_local(cfg, name):
                try:
                    self.api.post(f"/vlm/vendors/{name}/activate")
                    self.autosave_label.setText("✓ 已切换到本地离线识别")
                    self.autosave_label.setStyleSheet(
                        f"font-size: 11px; color: {theme().get('success')};")
                    self._refresh_vendors()
                except Exception:
                    pass
                break

    def _make_vendor_card(self, name, cfg, active):
        """单张供应商卡：启用单选 + 地址/API Key/模型 ID 就地编辑自动保存。"""
        cfg = cfg or {}
        card = QFrame()
        card.setObjectName("VendorCard")
        card.setStyleSheet(
            f"QFrame#VendorCard {{ background: {theme().get('glass')};"
            f" border: 1px solid {theme().get('border')}; border-radius: 10px; }}")
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(7)

        head = QHBoxLayout()
        en = QRadioButton("启用")
        en.setChecked(active)
        en.setStyleSheet(self._input_style())
        en.toggled.connect(lambda on, n=name: self._activate_vendor_card(n, on))
        self._vendor_enable_group.addButton(en)
        head.addWidget(en)
        disp = cfg.get("display_name")
        title = QLabel(name if not disp or disp == name else f"{name} · {disp}")
        title.setStyleSheet("font-size: 13px; font-weight: 600;")
        head.addWidget(title)
        head.addStretch()
        status = QLabel("✓ 当前生效" if active else "○ 未验证")
        status.setStyleSheet(f"font-size: 11px; color: {theme().get('success') if active else theme().get('text_secondary')};")
        head.addWidget(status)
        v.addLayout(head)
        self._card_status[name] = status

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        url_edit = QLineEdit(cfg.get("base_url", ""))
        key_edit = QLineEdit(cfg.get("api_key", ""))
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.setPlaceholderText("云端套餐必填；本地 Ollama 可留空")
        model_edit = QLineEdit(cfg.get("model", ""))
        for row, (cap, _e) in enumerate((("地址", url_edit), ("API Key", key_edit), ("模型 ID", model_edit))):
            _e.setStyleSheet(self._input_style())
            grid.addWidget(QLabel(cap), row, 0)
            grid.addWidget(_e, row, 1)
        grid.setColumnStretch(1, 1)
        v.addLayout(grid)

        btns = QHBoxLayout()
        btns.addStretch()
        test_btn = QPushButton("测试连接")
        test_btn.setStyleSheet(self._btn_style())
        test_btn.clicked.connect(lambda _=False, n=name: self._test_vendor_card(n))
        btns.addWidget(test_btn)
        del_btn = QPushButton("移除")
        del_btn.setStyleSheet(self._btn_style())
        del_btn.clicked.connect(lambda _=False, n=name: self._remove_vendor_card(n))
        btns.addWidget(del_btn)
        v.addLayout(btns)

        def _save_field():
            payload = {
                "name": name,
                "type": cfg.get("type", "url"),
                "vendor": cfg.get("vendor") or name,
                "base_url": url_edit.text().strip(),
                "model": model_edit.text().strip(),
                "api_key": key_edit.text().strip(),
            }
            try:
                self.api.post("/vlm/vendors", json=payload)
                cfg.update({"base_url": payload["base_url"],
                            "model": payload["model"],
                            "api_key": payload["api_key"]})
                status.setText("✓ 已保存")
                status.setStyleSheet(f"font-size: 11px; color: {theme().get('success')};")
            except Exception as e:
                status.setText(f"✗ 保存失败：{e}")
                status.setStyleSheet(f"font-size: 11px; color: {theme().get('danger')};")
        for _e in (url_edit, key_edit, model_edit):
            _e.editingFinished.connect(_save_field)
        return card

    def _activate_vendor_card(self, name, on):
        """卡片「启用」单选 → 激活该供应商并刷新全部卡片生效态。"""
        if not on or getattr(self, "_loading", False):
            return
        try:
            self.api.post(f"/vlm/vendors/{name}/activate")
            self._refresh_vendors()
        except Exception as e:
            self._engine_fail(f"切换供应商失败：{e}")

    def _remove_vendor_card(self, name):
        if QMessageBox.question(self, "确认移除", f"移除云端供应商「{name}」？") != QMessageBox.Yes:
            return
        try:
            self.api.delete(f"/vlm/vendors/{name}")
            self._refresh_vendors()
        except Exception as e:
            QMessageBox.warning(self, "移除失败", str(e))

    def _add_vendor_card(self):
        """新增供应商：以未占用占位名创建空卡，刷新后立即可编辑。"""
        base, i = "新供应商", 1
        names = set(getattr(self, "_vendors_data", {}) or {})
        while base in names:
            i += 1
            base = f"新供应商{i}"
        try:
            self.api.post("/vlm/vendors", json={
                "name": base, "type": "url", "vendor": base,
                "base_url": "", "model": "", "api_key": ""})
        except Exception as e:
            QMessageBox.warning(self, "创建失败", str(e))
            return
        self._refresh_vendors()

    def _test_vendor_card(self, name):
        """测试单张卡（S13 ⑪：沿用 /config/test-connection，自动识别 Ollama/云端）。"""
        cfg = (getattr(self, "_vendors_data", {}) or {}).get(name) or {}
        self._fetch_async(
            lambda: self.api.post("/config/test-connection", json={
                "provider": cfg.get("type", "url"),
                "base_url": cfg.get("base_url", ""),
                "model": cfg.get("model", ""),
            }),
            lambda res: self._apply_card_test(name, res),
            lambda msg: self._set_card_status(name, f"❌ {msg}", danger=True),
        )

    def _apply_card_test(self, name, result):
        if result.get("status") == "success":
            self._set_card_status(name, "✅ 连接成功，可用于分析")
        else:
            self._set_card_status(name, f"❌ {result.get('error', '连接失败')}", danger=True)

    def _set_card_status(self, name, text, danger=False):
        lb = getattr(self, "_card_status", {}).get(name)
        if lb is not None:
            lb.setText(text)
            lb.setStyleSheet(f"font-size: 11px; color: "
                             f"{theme().get('danger') if danger else theme().get('success')};")

    def refresh_environment(self):
        """公开刷新环境状态(下载完成后由主窗口调用)。"""
        self._check_environment()

    def _check_environment(self):
        """检查环境（后台拉取 /env-check，主线程回填三个指示器）。"""
        self._fetch_async(
            lambda: self.api.get("/env-check"),
            self._apply_environment,
            self._fail_environment,
        )

    def _fail_environment(self, msg: str):
        """检查失败：写本地形态面板的模型标签（S13 ⑪：状态标签已移除）。"""
        self.engine_model_label.setText(f"MiniCPM-V · ⚠ 本地权重状态检查失败：{msg}")

    def _apply_environment(self, result):
        """环境检查结果 → 写供应商区状态标签（P0-1 修复：FFmpeg/人脸模型已打包
        内置，不再展示；仅呈现 VLM 就绪状态）。"""
        try:
            vlm_data = result.get("vlm", {})
            vlm_status = vlm_data.get("status", "error")
            vlm_msg = vlm_data.get("message", "未检测")
            if vlm_status == "ok":
                self.engine_model_label.setText(f"MiniCPM-V · 就绪 ✓（{vlm_msg}）")
            else:
                self.engine_model_label.setText(f"MiniCPM-V · ⚠ {vlm_msg}")
        except Exception as e:
            self.engine_model_label.setText(f"MiniCPM-V · ⚠ 本地权重状态检查失败：{e}")
    
    def _run_env_init(self):
        """切换离线识别（14 号定稿）：

        - 本地权重未下载 → 触发下载（MainWindow 统一执行，env_init_requested）
        - 已下载 → 不重复下载，仅校验本地权重有效性并回显结果
        """
        try:
            from src.core.dependency_checker import DependencyChecker
            deps = DependencyChecker().check_all()
            model = deps.get("minicpm_model") or {}
            installed = bool(model.get("installed", False))
        except Exception:
            installed = False
        if installed:
            QMessageBox.information(
                self, "离线识别",
                "本地模型权重校验通过，离线识别可正常使用，无需重复下载。")
            return
        self.env_init_requested.emit()

    def _prepare_scheme(self):
        """方案准备：动态硬件检测，按真实资源应用推荐性能方案。

        - 极致性能 / 性能均衡：根据硬件并发上限动态调整
        - 极致节省：固定单进程，无需调整
        """
        try:
            self.env_prepare_btn.setEnabled(False)
            self.env_prepare_btn.setText("检测中...")
            QApplication.processEvents()
            try:
                hw = self.api.get("/hardware/info")
            finally:
                self.env_prepare_btn.setEnabled(True)
                self.env_prepare_btn.setText("方案准备")

            device_class = hw.get("device_class", "unknown")
            presets = hw.get("presets", {})

            # 刷新下拉（重新拉取动态预设）
            self._load_available_schemes()

            # 应用推荐方案（balanced）
            balanced = presets.get("balanced", {})
            if balanced:
                self.api.put("/config/scheduler.max_concurrent", json={"value": balanced.get("max_concurrent", 1)})
                self.api.put("/config/scheduler.max_vlm_concurrent", json={"value": balanced.get("max_vlm_concurrent", 1)})
                self.api.put("/config/scheduler.max_sifting_concurrent", json={"value": balanced.get("max_sifting_concurrent", 1)})
                self.api.put("/config/scheduler.max_md5_concurrent", json={"value": balanced.get("max_md5_concurrent", 1)})
                self.api.put("/config/performance.scheme", json={"value": "性能均衡"})
                self.api.post("/scheduler/restart")

            b = presets.get("balanced", {})
            p = presets.get("performance", {})
            QMessageBox.information(
                self, "方案准备完成",
                f"设备分级: {device_class}\n\n"
                f"性能均衡: 并发 {b.get('max_concurrent', '?')}, VLM {b.get('max_vlm_concurrent', '?')}\n"
                f"极致性能: 并发 {p.get('max_concurrent', '?')}, VLM {p.get('max_vlm_concurrent', '?')}\n"
                f"极致节省: 固定单进程\n\n"
                f"已按「性能均衡」应用并重启调度器。",
            )
            self._load_config()
        except Exception as e:
            self.env_prepare_btn.setEnabled(True)
            self.env_prepare_btn.setText("方案准备")
            QMessageBox.critical(self, "错误", f"方案准备失败: {e}")

    def _refresh_dict_stats(self):
        """刷新标签库统计（后台拉取）。"""
        self._fetch_async(
            lambda: self.api.get("/dictionary/stats"),
            self._apply_dict_stats,
            lambda msg: self.dict_stats_label.setText("统计刷新失败"),
        )

    def _apply_dict_stats(self, stats):
        total = stats.get("total_tags", 0)
        categories = stats.get("category_count", 0)
        self.dict_stats_label.setText(f"分类: {categories} | 标签: {total}")
    
    def _export_dictionary(self):
        """导出标签库"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出标签库", "", "Markdown Files (*.md);;All Files (*)"
        )
        if file_path:
            try:
                result = self.api.get("/dictionary/export")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result.get("content", ""))
                QMessageBox.information(self, "成功", f"标签库已导出到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {e}")
    
    def _import_dictionary(self):
        """导入标签库"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入标签库", "", "Markdown Files (*.md);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                result = self.api.post("/dictionary/import", json={"content": content})
                if result.get("success"):
                    QMessageBox.information(self, "成功", f"成功导入 {result.get('imported', 0)} 个标签")
                    self._refresh_dict_stats()
                else:
                    QMessageBox.warning(self, "警告", f"导入失败: {result.get('error', '未知错误')}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败: {e}")
    
    def _get_cleanup_stats(self):
        """获取帧清理统计"""
        try:
            days = self.cleanup_days_spin.value()
            stats = self.api.get(f"/frames/cleanup/stats?days={days}")
            count = stats.get("cleanable_count", 0)
            size_mb = stats.get("cleanable_size_mb", 0)
            self._set_info_label(self.cleanup_stats_label, f"可清理 {count} 个帧目录，占用 {size_mb:.2f} MB")
            self.cleanup_execute_btn.setEnabled(count > 0)
        except Exception as e:
            self._set_info_label(self.cleanup_stats_label, f"获取统计失败: {e}")
            self.cleanup_execute_btn.setEnabled(False)
    
    def _execute_cleanup(self):
        """执行帧清理"""
        days = self.cleanup_days_spin.value()
        reply = QMessageBox.question(
            self, "确认清理",
            f"确定要删除超过 {days} 天的帧文件吗？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        try:
            result = self.api.post("/frames/cleanup", json={"days": days})
            cleaned = result.get("cleaned_count", 0)
            size_mb = result.get("cleaned_size_mb", 0)
            self._set_info_label(self.cleanup_stats_label, f"已清理 {cleaned} 个目录，释放 {size_mb:.2f} MB")
            self.cleanup_execute_btn.setEnabled(False)
            self._get_cleanup_stats()
        except Exception as e:
            self._set_info_label(self.cleanup_stats_label, f"清理失败: {e}")
    
    # ── 分析主题管理（批8）──────────────────────────────────

    def _theme_modules(self):
        # 桌面运行态 src 不在 sys.path（与全库 core.* 导入口径一致）；
        # API/打包态则相反——两条路径都试。
        try:
            from src.core.theme_registry import get_theme_registry
            from src.core.theme_workspace import get_theme_workspace
            return get_theme_registry(), get_theme_workspace()
        except ImportError:
            pass
        from core.theme_registry import get_theme_registry
        from core.theme_workspace import get_theme_workspace
        return get_theme_registry(), get_theme_workspace()

    def _load_analysis_themes(self):
        """主题清单入下拉；当前项 = 注册表 current。"""
        reg, _ = self._theme_modules()
        names = reg.names()
        cur = reg.current() or reg.effective_current()
        self._theme_loading = True
        self.analysis_theme_combo.blockSignals(True)
        self.analysis_theme_combo.clear()
        self.analysis_theme_combo.addItems(names)
        if cur in names:
            self.analysis_theme_combo.setCurrentText(cur)
        self.analysis_theme_combo.blockSignals(False)
        self._active_theme = cur if cur in names else (names[0] if names else None)
        if self._active_theme:
            self.analysis_theme_combo.setCurrentText(self._active_theme)
        self._theme_loading = False
        self._refresh_theme_resources()

    def _refresh_theme_resources(self):
        """当前主题资源面板：config/数据库/人脸目录/标签统计（随主题刷新）。"""
        name = getattr(self, "_active_theme", None) or self.analysis_theme_combo.currentText()
        if not name:
            for lb in (self.res_db_label,
                       self.res_faces_label, self.res_tags_label):
                lb.setText("—")
            return
        reg, ws = self._theme_modules()

        # 前端不展示原始路径（既不必要也有风险）：行内只显示友好描述，
        # 真实路径收进悬停提示；「在访达中显示」按钮按需定位。
        # S13 方案 B + G7 分离管理：数据库=App 管理的业务命脉；人脸=用户根目录下的可再生数据
        # effective_db_path：legacy 主题实际生效的是共享库而非档内 data.db，悬停须如实
        db_p = ws.effective_db_path(name)
        self.res_db_label.setText("data.db · App 管理（AppSupport，随主题存储）")
        self.res_db_label.setToolTip(str(db_p))
        faces = ws.effective_faces_dir(name)
        self.res_faces_label.setText(f"{faces.name}/ · 用户根目录下按主题分层（可再生）")
        self.res_faces_label.setToolTip(str(faces))
        # 异步取统计（原为同步 GET——后端断连重试冻结主线程 4 秒，
        # 用户感知「点设置等几秒才出现页面」，2026-09-29 实测 profile 实锤）
        self._fetch_async(
            lambda: self.api.get("/dictionary/stats"),
            lambda st: self.res_tags_label.setText(
                f"分类 {st.get('category_count', 0)} · "
                f"标签 {st.get('total_tags', 0)}"),
            lambda _msg: self.res_tags_label.setText("—"),
        )

    def _reveal_in_finder(self, label):
        """在访达中显示资源（路径取自对应标签 tooltip；文件不存在则打开所在目录）。"""
        import subprocess
        from pathlib import Path as _P
        p = _P(label.toolTip() or label.text())
        target = p if p.exists() else p.parent
        subprocess.run(["open", "-R", str(target)])

    def _on_analysis_theme_combo_changed(self, text):
        if getattr(self, "_theme_loading", False):
            return
        if text and text != getattr(self, "_active_theme", None):
            self._switch_analysis_theme(text)

    def _switch_analysis_theme(self, name):
        """切换分析主题：先自动保存当前设置，再切注册表并刷新（用户定案）。

        S11（2026-09-27 用户裁定）：全程静默——保存走 silent 自动保存，
        不弹「调度器已重启」/「重启后生效」任何弹窗；切换后表单载入新主题
        参数供用户继续调整（页面本为自动保存模型），反馈只走行内状态条。
        """
        try:
            self._save_config(silent=True)
        except Exception as exc:
            QMessageBox.warning(self, "切换失败", f"保存当前设置失败: {exc}")
            self._load_analysis_themes()
            return
        reg, ws = self._theme_modules()
        reg.set_current(name)
        ws.ensure_theme(name)
        # P0-2 修复：同步写入该主题 config 的 analysis.active_theme
        # （config.py 的解析源——只写注册表会导致库/标签/提示词不跟随切换）
        try:
            import logging
            import yaml
            cfg_path = ws.config_path(name)
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            cfg.setdefault("analysis", {})["active_theme"] = name
            cfg_path.write_text(
                yaml.dump(cfg, allow_unicode=True, default_flow_style=False, sort_keys=True),
                encoding="utf-8")
        except Exception as exc:
            logging.getLogger(__name__).warning(f"同步 active_theme 到主题配置失败: {exc}")
        self._active_theme = name
        self.analysis_theme_changed.emit(name)
        self._load_available_schemes()
        self._load_config()
        self._refresh_theme_resources()
        self.autosave_label.setText(f"✓ 已切换到「{name}」，可继续调整（自动保存）")
        self.autosave_label.setStyleSheet(
            f"font-size: 11px; color: {theme().get('success')};")
        self._show_theme_restart_hint(name)

    def _show_theme_restart_hint(self, name: str, created: bool = False):
        """主题切换/新建后的底部常驻提醒（S11）：重启应用后资源才完全生效。"""
        verb = "已新建" if created else "已切换到"
        self.theme_restart_banner.setText(
            f"⚠ {verb}「{name}」：标签库/提示词已刷新；"
            "数据库等资源在重启应用后完全生效")
        self.theme_restart_banner.setVisible(True)
        self.theme_restart_btn.setVisible(True)

    def _new_analysis_theme(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "新建分析主题", "主题名称（同时作为数据目录名）：")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if "/" in name or "\\" in name or name in (".", ".."):
            QMessageBox.warning(self, "无效名称", "主题名不能包含路径分隔符")
            return
        reg, ws = self._theme_modules()
        if reg.get(name):
            QMessageBox.warning(self, "已存在", f"主题「{name}」已存在")
            return
        reg.upsert(name)
        ws.ensure_theme(name)
        self._load_analysis_themes()
        # 选中新主题（触发自动保存+切换）
        self.analysis_theme_combo.setCurrentText(name)
        # 切换链路横幅措辞为「已切换到」，此处校正为「已新建」
        self._show_theme_restart_hint(name, created=True)

    def _rename_analysis_theme_dialog(self):
        """改名入口（S13：⚙ 键值表移除后，改名以此轻量对话框承接）。"""
        from PySide6.QtWidgets import QInputDialog
        old_name = self.analysis_theme_combo.currentText()
        if not old_name:
            return
        name, ok = QInputDialog.getText(
            self, "重命名分析主题", f"将「{old_name}」重命名为：")
        if not ok:
            return
        self._apply_theme_rename(old_name, name.strip())

    def _apply_theme_rename(self, old_name: str, new_name: str) -> bool:
        """重命名分析主题核心：注册表改名单 + 工作区目录搬移（保留数据）。

        S2 起无独立重命名按钮；S3：⚙编辑配置内改 theme.name 键走这里。
        返回是否实际完成改名。
        """
        new_name = (new_name or "").strip()
        if not new_name or new_name == old_name:
            return False
        if "/" in new_name or "\\" in new_name or new_name in (".", ".."):
            QMessageBox.warning(self, "无效名称", "主题名不能包含路径分隔符")
            return False
        reg, ws = self._theme_modules()
        entry = reg.get(old_name)
        if entry is None:
            QMessageBox.warning(self, "失败", "主题不存在")
            return False
        if reg.get(new_name):
            QMessageBox.warning(self, "已存在", f"主题「{new_name}」已存在")
            return False
        try:
            ws.rename_theme(old_name, new_name)
            # 先注册新名再删旧名（字段为空也必须注册，否则主题凭空消失）
            reg.upsert(new_name, **{k: v for k, v in entry.items() if k != "name"})
            reg.delete(old_name)
        except Exception as exc:
            QMessageBox.warning(self, "重命名失败", str(exc))
            self._load_analysis_themes()
            return False
        was_active = (old_name == getattr(self, "_active_theme", None))
        self._load_analysis_themes()
        if was_active:
            reg.set_current(new_name)  # 当前主题改名后保持其当前地位
            self._active_theme = new_name
            self.analysis_theme_combo.setCurrentText(new_name)
            self.analysis_theme_changed.emit(new_name)
            QMessageBox.information(
                self, "重命名完成",
                "已重命名。数据库将在重启应用后按新名称完全生效。")
        return True

    def _delete_analysis_theme(self):
        name = self.analysis_theme_combo.currentText()
        if not name:
            return
        reg, ws = self._theme_modules()
        if len(reg.names()) <= 1:
            QMessageBox.information(self, "无法删除", "至少保留一个分析主题")
            return
        if name == (reg.current() or reg.effective_current()):
            QMessageBox.information(self, "无法删除", "先切换到其他主题，再删除当前主题")
            return
        if reg.get(name) and reg.get(name).get("is_default"):
            QMessageBox.information(self, "无法删除", "默认主题不可删除，请先切换")
            return
        ret = QMessageBox.question(
            self, "删除主题",
            f"确定删除分析主题「{name}」吗？\n\n该主题的数据库与人脸目录将一并删除，不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes:
            return
        import shutil
        reg.delete(name)
        tdir = ws.theme_dir(name)
        if tdir.exists():
            shutil.rmtree(tdir, ignore_errors=True)
        self._load_analysis_themes()

    def _mark_autosaved(self, need_restart_keys=None):
        """自动保存完成的行内反馈；need_restart_keys 非空时显示条件重启钮。"""
        self.autosave_label.setText("✓ 所有更改已自动保存")
        keys = need_restart_keys or []
        self._pending_restart_keys = keys
        has = bool(keys)
        self.restart_hint.setText(
            f"⚠ {len(keys)} 项更改需重启应用生效：{'、'.join(keys)}")
        self.restart_hint.setVisible(has)
        self.restart_btn.setVisible(has)

    def _restart_application(self):
        """重启应用使"需重启生效"的改动完全生效。"""
        from PySide6.QtWidgets import QApplication
        import os, sys
        QApplication.processEvents()
        os.execl(sys.executable, sys.executable, *sys.argv)

    def _load_config(self):
        """加载配置（依赖链整体在工作线程完成——原先 apply 回调里又串
        /themes、/config、prompt 三个同步请求，后端断连时主线程冻结
        12-18 秒，用户感知为「点设置等几秒才出现」。线程内只做网络与
        计算，回主线程仅回填。）"""
        def _fetch_bundle():
            api = self.api
            bundle = {"config": api.get("/config") or {}}
            # 提示词链（S13 ⑧）：跟随当前生效主题
            try:
                themes = api.get("/themes").get("themes", [])
            except Exception:
                themes = []
            try:
                cfg_active = ((bundle["config"].get("analysis") or {})
                              .get("active_theme") or "").strip()
            except Exception:
                cfg_active = ""
            prompt_map = {th.get("name", ""): th.get("id") for th in themes}
            default_name = next((th.get("name") for th in themes
                                 if th.get("is_default")), "")
            target = cfg_active if cfg_active in prompt_map else default_name
            prompt_text, global_prompt = "", ""
            theme_id = prompt_map.get(target)
            try:
                if theme_id:
                    prompt_text = api.get(
                        f"/themes/{theme_id}/prompt").get("prompt", "")
            except Exception:
                prompt_text = ""
            try:
                global_prompt = (api.get("/config").get("vlm")
                                 or {}).get("prompt", "")
            except Exception:
                global_prompt = ""
            if not prompt_text or prompt_text in ("（未设置）", "(empty)"):
                prompt_text = global_prompt
            bundle.update({
                "prompt_map": prompt_map,
                "active_analysis_theme": cfg_active,
                "prompt_text": prompt_text,
                "default_theme": default_name,
            })
            # 标签库统计（供 _refresh_dict_stats 回填，免去主线程请求）
            try:
                bundle["dict_stats"] = api.get("/dictionary/stats")
                bundle["dict_tags"] = api.get("/dictionary").get("tags", []) or []
            except Exception:
                bundle["dict_stats"] = {}
                bundle["dict_tags"] = []
            return bundle

        self._fetch_async(_fetch_bundle, self._apply_config)

    def _apply_config(self, bundle):
        try:
            config = bundle.get("config", {})
            self._refresh_theme_resources()

            # 提示词回填（数据已在线程内取好——主线程零请求）
            self._active_analysis_theme = bundle.get("active_analysis_theme", "")
            self._prompt_theme_data = bundle.get("prompt_map", {})
            if hasattr(self, "prompt_edit"):
                self.prompt_edit.setPlainText(bundle.get("prompt_text", ""))
            
            # 性能方案（S13 ⑨：平铺单选回填）
            perf = config.get("performance", {})
            scheme = perf.get("scheme", "性能均衡")
            self._loaded_scheme_name = scheme
            for _rb in self.scheme_radios.buttons():
                if _rb.property("scheme_name") == scheme:
                    _rb.setChecked(True)
                    break
            
            # 抽帧密度（14 号：三档帧/分钟，缺省标准 4）
            video = config.get("video", {})
            _dens = int(video.get("frame_density", 4) or 4)
            _dens_btn = self.density_group.button(_dens)
            if _dens_btn is not None:
                _dens_btn.setChecked(True)

            # 文件名解析忽略词（每行一个）
            fp = config.get("filename_parsing", {}) or {}
            kws = fp.get("ignore_keywords", []) or []
            self.ignore_keywords_edit.setPlainText("\n".join(str(k) for k in kws))
            
            # NFO 零设置（内置 Kodi XML，与视频同目录同名）
            
            # 帧保留三态（从 frame_cleanup.enabled/keep_days 反推）
            cleanup = config.get("frame_cleanup", {})
            _enabled = bool(cleanup.get("enabled", False))
            _kd = int(cleanup.get("keep_days", 0) or 0)
            if not _enabled:
                self.cleanup_mode_combo.setCurrentText("永不（手动清理）")
            elif _kd == 0:
                self.cleanup_mode_combo.setCurrentText("用完即弃")
            else:
                self.cleanup_mode_combo.setCurrentText("保留 N 天")
                self.cleanup_days_spin.setValue(_kd)
            self._sync_cleanup_days_visible()

            # 派发策略（新键优先，旧键迁移）
            # S13：派发策略/映射抽帧上限/合并pass/NSFW 已移除 UI，
            # 配置值保留在主题 config 中不动（管线按现值运行）。
            ana = config.get("analysis", {}) or {}
            self.auto_naming_check.setChecked(bool(ana.get("auto_naming", False)))
            
            # 数据库路径不再展示（随分析主题自动管理）

            # 播放器（S13 ⑦：radio 回填；历史值带「（推荐）」等后缀，归一化。
            # 存量选中项未安装 → 回落系统默认，等用户重新选择）
            player = config.get("player", {})
            _ptype = str(player.get("type", "系统默认")).split("（")[0].strip() or "系统默认"
            self._set_player_radio(_ptype)
            
            # 人脸存储路径
            face_recognition = config.get("face_recognition", {})
            faces_dir = face_recognition.get("faces_dir", "data/actor_faces")
            self.face_storage_path_edit.setText(faces_dir)
            
            # 刷新标签库统计
            self._refresh_dict_stats()
            
        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载配置失败: {e}")
    
    @staticmethod
    def _detect_player_installed(key: str) -> bool:
        """播放器安装探测（S13 ⑩）：检查标准安装位置的 .app。"""
        if key == "系统默认":
            return True
        app_names = {"iina": "IINA.app", "mpv": "mpv.app", "vlc": "VLC.app"}
        name = app_names.get(key)
        if not name:
            return False
        for base in (Path("/Applications"), Path.home() / "Applications"):
            if (base / name).exists():
                return True
        return False

    def _set_player_radio(self, key: str):
        """按 config 值选中播放器 radio；未安装/未知值回落系统默认。"""
        target = None
        fallback = None
        for rb in self.player_radios.buttons():
            if rb.property("player_key") == "系统默认":
                fallback = rb
            if rb.property("player_key") == key and rb.isEnabled():
                target = rb
                break
        (target or fallback).setChecked(True)

    def _on_player_toggled(self, _rb, checked):
        """播放器点选即生效（S13 ⑦：自动保存模型）。"""
        if getattr(self, "_loading", False) or not checked:
            return
        self._save_config(silent=True)
        self._mark_autosaved()

    def _browse_face_storage(self):
        """浏览并设置人脸存储路径"""
        current_path = self.face_storage_path_edit.text()
        
        # 如果是相对路径，转换为绝对路径用于对话框
        if not Path(current_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            current_path = str(project_root / current_path)
        
        # 打开文件夹选择对话框
        new_path = QFileDialog.getExistingDirectory(
            self,
            "选择人脸存储路径",
            current_path
        )
        
        if not new_path:
            return
        
        # 保存配置
        try:
            self.api.put("/config/face_recognition.faces_dir", json={"value": new_path})
            self.face_storage_path_edit.setText(new_path)
            QMessageBox.information(
                self, 
                "成功", 
                f"人脸存储路径已更新为:\n{new_path}\n\n注意：已有人脸数据需要手动迁移到新路径。"
            )
        except Exception as e:
            QMessageBox.warning(self, "警告", f"保存配置失败: {e}")
    
    def _save_config(self, silent: bool = False):
        """保存配置（silent=True：自动保存，成功不弹窗、失败写状态条）"""
        try:
            # VLM 供应商由「VLM 供应商管理」区负责（新增/切换/保存）
            # 这里只保存其他配置
            # 保存性能方案（只写干净的方案名，不含下拉里的括号参数）
            self.api.put("/config/performance.scheme", json={"value": self._current_scheme_name()})
            
            # 保存抽帧密度（三档帧/分钟）
            self.api.put("/config/video.frame_density",
                         json={"value": self.density_group.checkedId()})

            # 保存文件名解析忽略词（每行一个，去掉空行）
            _kws = [
                ln.strip()
                for ln in self.ignore_keywords_edit.toPlainText().splitlines()
                if ln.strip()
            ]
            self.api.put("/config/filename_parsing.ignore_keywords", json={"value": _kws})
            
            # 帧保留三态（映射到 frame_cleanup.enabled/keep_days）
            _mode = self.cleanup_mode_combo.currentText()
            if _mode == "用完即弃":
                self.api.put("/config/frame_cleanup.enabled", json={"value": True})
                self.api.put("/config/frame_cleanup.keep_days", json={"value": 0})
            elif _mode.startswith("保留"):
                self.api.put("/config/frame_cleanup.enabled", json={"value": True})
                self.api.put("/config/frame_cleanup.keep_days",
                             json={"value": self.cleanup_days_spin.value()})
            else:  # 永不（手动清理）
                self.api.put("/config/frame_cleanup.enabled", json={"value": False})

            # S13：派发策略/映射抽帧上限/合并pass/NSFW 不再从 UI 写入，
            # 主题 config 中的现值原样保留（管线按现值运行）。
            self.api.put("/config/analysis.auto_naming",
                         json={"value": self.auto_naming_check.isChecked()})

            # 数据库路径随分析主题自动管理，不再保存

            # 保存播放器配置
            # 播放器（写归一化后的 key；路径由运行时自动探测，不再存储）
            _ptype = self.player_radios.checkedButton()
            _ptype = _ptype.property("player_key") if _ptype is not None else "系统默认"
            self.api.put("/config/player.type", json={"value": _ptype})
            
            # 重启调度器以应用新的性能方案
            self.api.post("/scheduler/restart")
            
            if not silent:
                QMessageBox.information(self, "成功", "配置已保存，调度器已重启以应用新配置")
        except Exception as e:
            if silent:
                self.autosave_label.setText(f"✗ 自动保存失败：{e}")
                self.autosave_label.setStyleSheet(
                    "font-size: 11px; color: #FF453A;")
            else:
                QMessageBox.critical(self, "错误", f"保存配置失败: {e}")
