"""首装引导（14 号方案 P2）：模态门槛——引导完成前主窗体不创建。

三步流程：欢迎 → 分析主题（六项表单）→ 引擎就绪。
完成条件 = 主题已创建 + 所选分析方式引擎就绪确认；
完成后发 completed 信号（携带主题名），由 desktop_app 创建主窗体。

主题创建复用与设置页同一条服务链：registry.upsert + workspace.ensure_theme，
并把 表单六项 写入主题 config.yaml（数据库名/人脸路径/提示词/分析方式）。
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QStackedWidget, QWidget, QTextEdit, QFileDialog,
    QButtonGroup, QRadioButton, QMessageBox, QLayout,
)

from .theme import theme


def _theme_modules():
    """与设置页同口径的双路径导入。"""
    try:
        from src.core.theme_registry import get_theme_registry
        from src.core.theme_workspace import get_theme_workspace
        return get_theme_registry(), get_theme_workspace()
    except ImportError:
        from core.theme_registry import get_theme_registry
        from core.theme_workspace import get_theme_workspace
        return get_theme_registry(), get_theme_workspace()


def _needs_onboarding() -> bool:
    """无任何分析主题 ⇒ 需要引导（与"删光主题强制引导"定案一致）。"""
    try:
        reg, _ws = _theme_modules()
        return len(reg.names()) == 0
    except Exception:
        return False


# 内置提示词骨架（活动影像体系——人生全周期泛化）
_PROMPT_SKELETON = (
    "## 任务\n"
    "你是专业活动影像的剪辑助理。以下是某场活动（可能是婚礼、寿宴、"
    "宝宝宴、年会等）某段流程的连续视频帧，请分析画面内容并结构化输出。\n\n"
    "## 输出要求\n"
    "严格按 JSON 输出，取值只能从候选列表选择，不确定的维度输出空数组：\n"
    '{ "事件类型": [], "人物角色": [], "服装造型": [], "场地场景": [], '
    '"镜头语言": [], "情绪氛围": [], "画面质量": [], "交付建议": [] }\n\n'
    "## 候选列表\n"
    "（由系统从当前主题标签库实时生成）"
)


class OnboardingDialog(QDialog):
    """首装引导：完成分析主题创建与引擎确认后，才进入主界面。"""

    completed = Signal(str)  # 携带创建的主题名

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MediaTag 初始设置")
        self.setFixedSize(760, 560)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.created_theme = ""
        t = theme()
        self.setStyleSheet(
            f"background: {t.get('bg_primary')}; color: {t.get('text_primary')};")

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 18)
        root.setSpacing(12)

        self.title_lb = QLabel()
        self.title_lb.setStyleSheet("font-size: 18px; font-weight: 700;")
        root.addWidget(self.title_lb)
        self.sub_lb = QLabel()
        self.sub_lb.setWordWrap(True)
        self.sub_lb.setStyleSheet("font-size: 12px; color: #98989D;")
        root.addWidget(self.sub_lb)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self._build_page_welcome()
        self._build_page_theme()
        self._build_page_engine()

        nav = QHBoxLayout()
        self.back_btn = QPushButton("上一步")
        self.back_btn.clicked.connect(lambda: self._goto(self.stack.currentIndex() - 1))
        nav.addWidget(self.back_btn)
        nav.addStretch()
        self.skip_hint = QLabel()
        self.skip_hint.setStyleSheet("font-size: 11px; color: #636366;")
        nav.addWidget(self.skip_hint)
        self.next_btn = QPushButton("下一步")
        self.next_btn.setStyleSheet(
            "background: #0A84FF; color: #FFFFFF; font-weight: 600; padding: 6px 22px;")
        self.next_btn.clicked.connect(self._on_next)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)

        self._goto(0)

    # ── 页面构建 ──
    def _build_page_welcome(self):
        pg = QWidget()
        v = QVBoxLayout(pg)
        v.addStretch()
        hero = QLabel("MediaTag")
        hero.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero.setStyleSheet("font-size: 30px; font-weight: 800;")
        v.addWidget(hero)
        sub = QLabel("AI 视频元数据管理 —— 标签、人脸、封面，一步到位")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("font-size: 13px; color: #98989D; margin-top: 8px;")
        v.addWidget(sub)
        note = QLabel("首次使用需要完成初始设置（约 1 分钟）。\n"
                      "设置过程中不会进入主界面，完成后自动进入。")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet("font-size: 12px; color: #636366; margin-top: 16px;")
        v.addWidget(note)
        v.addStretch()
        self.stack.addWidget(pg)

    def _build_page_theme(self):
        pg = QWidget()
        v = QVBoxLayout(pg)
        v.setSpacing(10)

        def row(label, widget, hint=""):
            h = QHBoxLayout()
            lb = QLabel(label)
            lb.setFixedWidth(110)
            h.addWidget(lb)
            if isinstance(widget, QLayout):
                _wrap = QWidget()
                _wrap.setLayout(widget)
                h.addWidget(_wrap, 1)
            else:
                h.addWidget(widget, 1)
            v.addLayout(h)
            if hint:
                hp = QLabel(hint)
                hp.setWordWrap(True)
                hp.setStyleSheet("font-size: 11px; color: #636366; margin-left: 114px;")
                v.addWidget(hp)

        self.name_edit = QLineEdit("我的工作室")
        row("主题名称", self.name_edit)

        self.db_edit = QLineEdit("我的工作室")
        row("数据库名称", self.db_edit, "默认与主题名称相同；保存于默认位置，无需记忆路径")

        self.taglib_combo = QComboBox()
        self.taglib_combo.addItem("活动影像体系（内置模板）", "activity")
        self.taglib_combo.addItem("简单通用模板", "simple")
        row("标签库", self.taglib_combo, "标签本身由 AI 分析自动产出，模板只定分类骨架")

        self.prompt_edit = QTextEdit(_PROMPT_SKELETON)
        self.prompt_edit.setMinimumHeight(110)
        self.prompt_edit.setStyleSheet(
            "QTextEdit { font-family: 'Consolas','Monaco',monospace; font-size: 12px; }")
        row("提示词", self.prompt_edit, "已预填活动影像骨架，可按需修改；之后随时在设置页调整")

        face_h = QHBoxLayout()
        self.faces_edit = QLineEdit()
        self.faces_edit.setPlaceholderText("默认位置（自动管理），点击「浏览」可换到更大容量磁盘")
        face_h.addWidget(self.faces_edit, 1)
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._pick_faces_dir)
        face_h.addWidget(browse)
        face_wrap = QWidget()
        face_wrap.setLayout(face_h)
        row("人脸存放位置", face_wrap)

        mode_h = QHBoxLayout()
        self.mode_offline = QRadioButton("离线识别（推荐，隐私安全）")
        self.mode_cloud = QRadioButton("云端识别（更快，需配置云厂商）")
        grp = QButtonGroup(self)
        grp.addButton(self.mode_offline)
        grp.addButton(self.mode_cloud)
        self.mode_offline.setChecked(True)
        mode_h.addWidget(self.mode_offline)
        mode_h.addWidget(self.mode_cloud)
        mode_h.addStretch()
        row("分析方式", mode_h)
        self.stack.addWidget(pg)

    def _build_page_engine(self):
        pg = QWidget()
        v = QVBoxLayout(pg)
        v.addStretch()
        self.engine_status = QLabel("检测中…")
        self.engine_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.engine_status.setStyleSheet("font-size: 15px; font-weight: 600;")
        v.addWidget(self.engine_status)
        self.engine_detail = QLabel("")
        self.engine_detail.setWordWrap(True)
        self.engine_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.engine_detail.setStyleSheet("font-size: 12px; color: #98989D; margin-top: 6px;")
        v.addWidget(self.engine_detail)
        recheck_h = QHBoxLayout()
        recheck_h.addStretch()
        recheck = QPushButton("重新检测")
        recheck.setFixedWidth(120)
        recheck.clicked.connect(self._check_engine)
        recheck_h.addStretch()
        v.addLayout(recheck_h)
        note = QLabel("离线识别：需要本地推理引擎（当前版本检测 Ollama 服务，内嵌组件在后续版本提供）\n"
                      "云端识别：需要已配置云厂商供应商")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 11px; color: #636366; margin-top: 10px;")
        v.addWidget(note)
        v.addStretch()
        self.stack.addWidget(pg)

    def _pick_faces_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择人脸数据存放文件夹")
        if d:
            self.faces_edit.setText(d)
            self.faces_edit.setPlaceholderText("")

    # ── 页面切换 ──
    _TITLES = [
        ("欢迎使用 MediaTag", "三步完成初始设置"),
        ("创建你的分析主题", "主题 = 数据库 + 标签库 + 提示词 + 人脸库，一套配置一种业务"),
        ("分析引擎确认", "确认所选分析方式已就绪"),
    ]

    def _goto(self, idx):
        idx = max(0, min(idx, self.stack.count() - 1))
        self.stack.setCurrentIndex(idx)
        title, sub = self._TITLES[idx]
        self.title_lb.setText(title)
        self.sub_lb.setText(sub)
        self.back_btn.setVisible(idx > 0)
        self.next_btn.setText("完成" if idx == self.stack.count() - 1 else "下一步")

    def _on_next(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            self._goto(1)
            return
        if idx == 1:
            err = self._validate_theme()
            if err:
                QMessageBox.warning(self, "请检查", err)
                return
            self._goto(2)
            self._check_engine()
            return
        self._finish()

    # ── 校验 / 引擎检测 / 完成 ──
    def _validate_theme(self) -> str:
        name = self.name_edit.text().strip()
        if not name:
            return "请填写主题名称"
        if "/" in name or "\\" in name or name in (".", ".."):
            return "主题名称不能包含路径分隔符"
        reg, _ws = _theme_modules()
        if reg.get(name):
            return f"主题「{name}」已存在，请换一个名称"
        return ""

    def _check_engine(self):
        """引擎就绪检测（现状：本地 Ollama 服务 + 模型；云端：供应商已配置）。"""
        offline = self.mode_offline.isChecked()
        if offline:
            ok, detail = self._probe_ollama()
            self.engine_status.setText("✓ 本地引擎就绪" if ok else "⚠ 未检测到本地引擎")
            self.engine_status.setStyleSheet(
                "font-size: 15px; font-weight: 600; color: " +
                ("#30D158;" if ok else "#FF9F0A;"))
            self.engine_detail.setText(
                detail + "\n过渡方案：安装并启动 Ollama 后点「重新检测」；"
                "或返回上一步选择云端识别。内嵌引擎组件将在后续版本提供。")
            self._engine_ready = ok
        else:
            n = self._probe_cloud_vendors()
            self.engine_status.setText(
                f"✓ 已配置 {n} 个云厂商供应商" if n else "⚠ 尚未配置云厂商供应商")
            self.engine_status.setStyleSheet(
                "font-size: 15px; font-weight: 600; color: " +
                ("#30D158;" if n else "#FF9F0A;"))
            self.engine_detail.setText(
                "完成引导后，可在「设置 → 分析引擎 → 管理供应商」中补充配置。")
            self._engine_ready = True
        self.next_btn.setEnabled(True)
        self.next_btn.setText("完成并进入 MediaTag")

    def _probe_ollama(self):
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            models = [m.get("name", "") for m in r.json().get("models", [])]
            return True, (f"Ollama 运行中，模型 {len(models)} 个" if models else "Ollama 运行中，尚未拉取模型")
        except Exception:
            return False, "Ollama 未运行"

    def _probe_cloud_vendors(self):
        """已配置的云厂商供应商数量（读全局 config vlm.vendors）。"""
        try:
            from src.core.config import config_manager
            vendors = config_manager.get("vlm.vendors", []) or []
            return len(vendors)
        except Exception:
            return 0

    def _pending_name(self):
        return self.name_edit.text().strip() or "我的工作室"

    def _finish(self):
        name = self._pending_name()
        if self._validate_theme():
            self._goto(1)
            return
        reg, ws = _theme_modules()
        import yaml
        reg.upsert(name)
        ws.ensure_theme(name)
        try:
            reg.set_current(name)
        except Exception:
            pass

        # 表单六项写入主题 config.yaml
        cfg_path = ws.config_path(name)
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
        cfg.setdefault("theme", {})["name"] = name
        db_name = self.db_edit.text().strip() or name
        cfg.setdefault("database", {})["name"] = db_name  # 逻辑名（库文件沿用主题默认路径）
        faces = self.faces_edit.text().strip()
        if faces:
            cfg.setdefault("face_recognition", {})["faces_dir"] = faces if faces.endswith("/") else faces + "/"
        cfg.setdefault("analysis", {})["active_theme"] = name
        cfg["vlm_prompt"] = self.prompt_edit.toPlainText()
        try:
            cfg_path.write_text(
                yaml.dump(cfg, allow_unicode=True, default_flow_style=False, sort_keys=True),
                encoding="utf-8")
        except Exception:
            pass

        self.created_theme = name
        self.completed.emit(name)
        self.accept()
