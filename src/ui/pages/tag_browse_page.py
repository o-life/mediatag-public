"""TagBrowsePage — 标签浏览主页面。

左侧: 标签树形列表 (TagTreeWidget)
右侧: 视频宫格展示 (VideoGridWidget)
顶部: 操作按钮（全选、反选、修改标签、删除标签）
底部: 分页器（由 VideoGridWidget 内建）
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from ..components.api_client import APIClient
from ..tokens import T
from ..components.theme import (
    theme, button_secondary_style, button_danger_style,
)
from ..widgets.tag_tree_widget import TagTreeWidget
from ..widgets.video_grid_widget import VideoGridWidget


class TagBrowsePage(QWidget):
    """
    标签浏览主页面。

    布局：
    ┌──────────────┬─────────────────────────────────┐
    │              │  [全选] [反选] [修改标签] [删除]  │
    │  TagTree     │                                 │
    │  Widget      │  VideoGridWidget                │
    │              │                                 │
    │              │  [分页器]                        │
    └──────────────┴─────────────────────────────────┘

    信号：
    - video_play_requested(str, int): 请求播放视频，传递 file_path 和 frame_number
    """

    video_play_requested = Signal(str, int)

    def __init__(
        self,
        api_client: Optional[APIClient] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.api_client = api_client or APIClient()

        # 
        self.current_tag: Optional[str] = "All"

        self._setup_ui()
        self._connect_signals()

    # ── UI ────────────────────────────────────────────────────────

    def _setup_ui(self):
        t = theme()
        # 对象名作用域（防级联）：裸 background 声明会涂满所有后代，
        # 网格区被页面底色盖住卡色（2026-09-29 实测）
        self.setObjectName("TagBrowsePage")
        self.setStyleSheet(
            f"QWidget#TagBrowsePage {{ background: {t.get('bg_primary')};"
            f" color: {t.get('text_primary')}; }}")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # （内容区工具栏整条撤除——用户裁定 2026-09-27：本页无多选场景，
        # 「已选中」筛选与「当前：全部」提示均无意义；操作钮都在顶带）

        # 顶带操作面板（B2）：无父级构建，主窗口经 take_tag_bar() 取走
        self._tag_bar = self._build_tag_bar(t)

        # 内容：封面网格（TB4.1 底 bg_card 直角满铺——组件内含分页底栏）
        self.video_grid = VideoGridWidget(api_client=self.api_client)
        self.video_grid.setStyleSheet(
            f"VideoGridWidget {{ background-color: {t.get('bg_card')}; }}")
        self.video_grid.reload_requested.connect(self._reload_current_view)
        main_layout.addWidget(self.video_grid, 1)

        # TB4.4/4.5 空态两分支（未选标签引导 / 标签无文件）——覆盖层
        self._empty_overlay = QLabel("")
        self._empty_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_overlay.setTextFormat(Qt.TextFormat.RichText)
        self._empty_overlay.setStyleSheet(
            f"background: {t.get('bg_card')};")
        self._empty_overlay.hide()
        main_layout.addWidget(self._empty_overlay, 1)

        # 标签树不进本页布局——MainWindow 经 take_tree_panel() 取走并 reparent
        # 到主窗口统一侧栏（10 号三栏，与任务页来源树同机制）。
        # 无父级构建，取走前不 show（否则闪成独立窗口）
        self.tag_tree = TagTreeWidget(api_client=self.api_client)

    def take_tree_panel(self):
        """把标签树打包成独立面板，交给主窗口统一侧栏（10 号三栏）。

        D8 教训：面板必须实色垫底（bg_secondary）——透明透窗背玻璃发黑。"""
        from ..components.theme import sidebar_panel_style
        panel = QWidget()
        panel.setStyleSheet(sidebar_panel_style())
        panel_v = QVBoxLayout(panel)
        panel_v.setContentsMargins(0, 0, 0, 0)
        panel_v.setSpacing(0)
        panel_v.addWidget(self.tag_tree, 1)
        return panel

    def apply_theme(self):
        """换肤：重放页面样式；子组件有 apply_theme 则联动。"""
        # 注意：本模块是 ui.pages 包成员，theme 在 ui.components——
        # 用 `.theme` 会落到不存在的 ui.pages.theme（换肤回退整页重建）
        from ..components.theme import theme
        t = theme()
        self.setStyleSheet(
            f"QWidget#TagBrowsePage {{ background: {t.get('bg_primary')};"
            f" color: {t.get('text_primary')}; }}")
        for child in (self.tag_tree, self.video_grid):
            try:
                if hasattr(child, "apply_theme"):
                    child.apply_theme()
            except Exception:
                pass
        self.restyle_band_bar()

    def _build_tag_bar(self, t) -> QWidget:
        """顶带操作面板（B2/TB1）：chip+修改/删除+刷新+计数。

        必须持久构建、不随工具栏换肤重建——换肤会 replaceWidget 整条
        工具栏，重建出的新钮会让顶带持有的引用失效。
        """
        from ..components.theme import sm_button_style, rgba_str
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(T.SP_8)

        # TB1.1 当前标签 chip：26 高 accent.14 底；未选隐藏
        self.tag_chip = QLabel("")
        self.tag_chip.setStyleSheet(
            f"QLabel {{ padding: 1px {T.CHIP_PILL_PAD_X}px; border-radius: {t.get('radius_ctrl')}px;"
            f" background: {rgba_str(t.get('accent'), 0.14)};"
            f" border: 1px solid {rgba_str(t.get('accent'), 0.4)};"
            f" font-size: {t.get('font_small')}px; font-weight: 600;"
            f" color: {t.get('accent')}; }}")
        self.tag_chip.hide()
        h.addWidget(self.tag_chip)

        # TB1.2 修改/删除：sm 26；未选 disabled（启停在 _sync_band_for_tag）
        self.btn_modify_tag = QPushButton("修改标签")
        self.btn_modify_tag.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_modify_tag.setFixedHeight(T.CTRL_SM_H)
        self.btn_modify_tag.setStyleSheet(sm_button_style())
        h.addWidget(self.btn_modify_tag)

        self.btn_delete_tag = QPushButton("删除标签")
        self.btn_delete_tag.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_tag.setFixedHeight(T.CTRL_SM_H)
        self.btn_delete_tag.setStyleSheet(sm_button_style())
        h.addWidget(self.btn_delete_tag)

        # TB1.3 刷新：sm 钮
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh.setFixedHeight(T.CTRL_SM_H)
        self.btn_refresh.setStyleSheet(sm_button_style())
        h.addWidget(self.btn_refresh)
        h.addStretch()  # 无 stretch 时 band_stack 会把宽度均摊给按钮（实测拉成宽条）

        # TB1.4 文件计数：spring 后；未选=「—」
        self.count_label = QLabel("—")
        self.count_label.setStyleSheet(
            f"font-size: {t.get('font_secondary')}px;"
            f" color: {t.get('text_secondary')}; background: transparent;")
        h.addWidget(self.count_label)

        self.btn_modify_tag.setEnabled(False)
        self.btn_delete_tag.setEnabled(False)

        return bar

    def _enforce_one_tier(self):
        """一档制终检（17 号规范）：显示与样式重放后钳 28（chip 计数除外）。"""
        for _w in (self.findChildren(QPushButton) + self.findChildren(QComboBox)):
            try:
                _w.setFixedHeight(T.CTRL_H)
            except RuntimeError:
                pass

    def showEvent(self, e):
        super().showEvent(e)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._enforce_one_tier)


    def restyle_band_bar(self):
        """顶带换肤钩子：面板已迁出页面，页面级样式表够不到它。"""
        if getattr(self, "_tag_bar", None) is None:
            return
        t = theme()
        self.btn_modify_tag.setStyleSheet(self._btn_style(t, variant="primary"))
        self.btn_delete_tag.setStyleSheet(self._btn_style(t, variant="danger"))
        self.btn_refresh.setStyleSheet(self._btn_style(t, variant="secondary"))
        self._enforce_one_tier()

    def take_tag_bar(self):
        """把顶带操作面板交给主窗口（重复调用返回 None）。"""
        bar, self._tag_bar = self._tag_bar, None
        return bar

    @staticmethod
    def _btn_style(t, variant: str = "secondary") -> str:
        """统一按钮样式：直接复用主题按钮函数（与文件详情工具栏同款玻璃底；
        danger 变体文字为危险色——用户定案：删除只红文字）。"""
        if variant == "danger":
            return button_danger_style()
        return button_secondary_style()

    # ── 信号连接 ──────────────────────────────────────────────────

    def _connect_signals(self):
        # 标签选择
        self.tag_tree.tag_selected.connect(self._on_tag_selected)
        self.tag_tree.all_selected.connect(self._on_all_selected)

        # 顶带三钮（面板持久、不随换肤重建，接一次即可）
        self.btn_modify_tag.clicked.connect(self._on_modify_tag)
        self.btn_delete_tag.clicked.connect(self._on_delete_tag)
        self.btn_refresh.clicked.connect(self._on_refresh)

        # 视频播放
        self.video_grid.play_requested.connect(self._on_play_video)

        # TB1.4 文件计数：数据到达回填「N 个文件」
        self.video_grid.data_loaded.connect(self._on_grid_count)

        # 分页
        self.video_grid.page_changed.connect(self._on_page_changed)

    # ── 公开方法 ──────────────────────────────────────────────────

    def initialize(self):
        """初始化页面，加载标签树。"""
        self.tag_tree.load_tags()
        # 默认加载全部视频
        self.video_grid.load_all()

    # ── 内部方法 ──────────────────────────────────────────────────

    def _on_tag_selected(self, tag: str):
        """标签树选中具体标签（TB3.4：再点已选=取消过滤回空态引导）。"""
        if getattr(self, "current_tag", None) == tag:
            # 再点已选标签 → 取消过滤（与 P9 导航 toggle 同语义）
            self.current_tag = None
            self.tag_tree.clear_selection()
            self._show_empty_overlay("guide")
            self._sync_band_for_tag()
            self.video_grid.hide()
            return
        self.current_tag = tag
        self._show_empty_overlay(None)  # 清空态，等数据决定是否「无文件」
        self.video_grid.setVisible(True)
        self.video_grid.load_by_tag(tag, page=1)
        self._sync_band_for_tag()

    def _on_all_selected(self):
        """「全部」节点。"""
        self.current_tag = None
        self._show_empty_overlay("guide")
        self.video_grid.hide()
        self._sync_band_for_tag()

    # ── TB1.1/1.4/4.4/4.5：顶带 chip/计数与空态两分支 ──

    def _sync_band_for_tag(self):
        """顶带 chip 显隐+文案、修改/删除启停、计数占位。"""
        tag = getattr(self, "current_tag", None)
        has = bool(tag) and tag != "All"
        self.tag_chip.setVisible(has)
        if has:
            self.tag_chip.setText(f"标签：{tag}")
        self.btn_modify_tag.setEnabled(has)
        self.btn_delete_tag.setEnabled(has)
        self.count_label.setText("—")  # 真实计数随 data_loaded 回填（TB1.4）

    def _show_empty_overlay(self, branch: Optional[str], tag: str = ""):
        """空态覆盖层：guide=未选引导；none_tag=标签无文件；None=隐藏。

        轻规格（TB4.4）：眼睛图标容器 64 + 标题 15/600 + 副文案 12/txt2，
        无操作按钮。"""
        t = theme()
        if branch is None:
            self._empty_overlay.hide()
            return
        if branch == "guide":
            title, sub = "在左侧选择一个标签", (
                "点选标签后显示该标签的所有文件；点击封面从标签画面开始播放")
            icon = "👁"
        else:
            title, sub = f"“{tag}”下暂无文件", ""
            icon = "🏷"
        html = (
            f"<div style='text-align:center;'>"
            f"<span style='font-size:{T.HERO_EMOJI_FS}px;'>{icon}</span><br>"
            f"<span style='font-size:{t.get('font_section')}px; font-weight:600;"
            f" color:{t.get('text_primary')};'>{title}</span>")
        if sub:
            html += (f"<br><span style='font-size:{t.get('font_secondary')}px;"
                     f" color:{t.get('text_secondary')};'>{sub}</span>")
        html += "</div>"
        self._empty_overlay.setText(html)
        self._empty_overlay.show()

    def _on_grid_count(self, total: int):
        """TB1.4：「N 个文件」；未选标签时保持「—」。"""
        if getattr(self, "current_tag", None):
            self.count_label.setText(f"{total} 个文件")

    def _on_page_changed(self, page: int):
        """分页器切换页码。"""
        if self.current_tag:
            self.video_grid.load_by_tag(self.current_tag, page=page)
        else:
            self.video_grid.load_all(page=page)

    def _reload_current_view(self):
        """删除等操作后刷新当前视图(保持当前标签/页码)。"""
        page = self.video_grid.current_page or 1
        tag = getattr(self, "current_tag", None)
        if tag and tag != "All":
            self.video_grid.load_by_tag(tag, page=page)
        else:
            self.video_grid.load_all(page=page)

    def _on_play_video(self, file_path: str, frame_number: int = 0):
        """处理视频播放请求 - 支持自定义播放器和进度跳转"""
        
        import logging
        logger = logging.getLogger(__name__)
        
        self.video_play_requested.emit(file_path, frame_number)
        path = Path(file_path)
        if not path.exists():
            QMessageBox.warning(self, "播放失败", f"文件不存在:\n{file_path}")
            return
        
        try:
            import sys
            # 确保项目根目录在 sys.path 中
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from src.core.video_player import VideoPlayerService
            from src.core.config import ConfigManager
            from src.db.models import get_db_engine, get_session, TagFrameMapping
            
            # 读取播放器配置
            config_manager = ConfigManager()
            player_config = config_manager.get("player", {})
            player_type = player_config.get("type", "系统默认")
            player_path = player_config.get("path", "")
            
            logger.info(f"播放器配置: type={player_type}, path={player_path}")
            
            # 创建播放器服务
            player_service = VideoPlayerService(player_type, player_path)
            
            # 从数据库查询时间戳
            timestamp = None
            if frame_number > 0:
                engine = get_db_engine()
                db = get_session(engine)
                try:
                    # 查询该帧的时间戳（同时匹配 frame_number 和 entity_id）
                    from src.db.models import FileLocation
                    # 根据视频文件路径找到 entity_id（通过 file_locations 表）
                    location = db.query(FileLocation).filter(
                        FileLocation.file_path == str(path)
                    ).first()
                    
                    
                    if location:
                        mapping = db.query(TagFrameMapping).filter(
                            TagFrameMapping.video_frame_number == frame_number,
                            TagFrameMapping.entity_id == location.entity_id
                        ).first()
                        
                        
                        if mapping and mapping.timestamp:
                            timestamp = mapping.timestamp
                            logger.info(f"找到时间戳: {timestamp} (frame={frame_number}, entity={location.entity_id})")
                        else:
                            logger.warning(f"未找到时间戳 (frame={frame_number}, entity={location.entity_id})")
                    else:
                        logger.warning(f"未找到视频实体: {path}")
                finally:
                    db.close()
            
            
            logger.info(f"调用播放器: player_type={player_service.player_type}, path={player_service.player_path}")
            # 播放视频
            player_service.play(str(path), timestamp)
            # S13 ⑩：选中播放器未安装已自动回退系统默认——把原因告诉用户
            notice = getattr(player_service, "fallback_notice", "")
            if notice:
                QMessageBox.information(self, "播放器提示", notice)
            
            logger.info("播放调用完成")
            
        except Exception as e:
            # 出错时回退到系统默认播放器
            logger.error(f"播放失败，回退到系统默认: {e}", exc_info=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _on_modify_tag(self):
        """修改标签按钮：为选中视频批量添加标签。"""
        selected = self.video_grid.get_selected_ids()
        if not selected:
            QMessageBox.information(self, "提示", "请先选择要修改标签的视频")
            return

        tag, ok = QInputDialog.getText(
            self,
            "修改标签",
            f"为选中的 {len(selected)} 个视频添加标签:",
        )
        if not ok or not tag.strip():
            return
        tag = tag.strip()

        try:
            self.api_client.post(
                "/api/videos/batch-update-tags",
                json={"video_ids": selected, "tag_name": tag},
            )
            QMessageBox.information(self, "成功", f"已为 {len(selected)} 个视频添加标签「{tag}」")
            self.video_grid.clear_selection()
        except Exception as exc:
            QMessageBox.critical(self, "修改失败", str(exc))

    def _on_delete_tag(self):
        """"""
        selected = self.video_grid.get_selected_ids()
        if not selected:
            QMessageBox.information(self, "提示", "请先勾选要删除标签的作品。")
            return

        # "All" 
        delete_tag = self.current_tag if self.current_tag and self.current_tag != "All" else None
        if not delete_tag:
            QMessageBox.information(
                self, "提示",
                "请先在左侧选择一个具体标签，再删除该标签的关联（「全部」下无法删除）。",
            )
            return

        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定从 {len(selected)} 个视频中删除标签「{self.current_tag}」？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self.api_client.post(
                "/api/videos/batch-delete-tags",
                json={"video_ids": selected, "tag_name": self.current_tag},
            )
            QMessageBox.information(self, "成功", f"已从 {len(selected)} 个视频删除标签「{self.current_tag}」")
            self.video_grid.clear_selection()
            # 刷新当前标签的视频列表
            self.video_grid.load_by_tag(self.current_tag, page=1)
        except Exception as exc:
            QMessageBox.critical(self, "删除失败", str(exc))

    def _on_refresh(self):
        """刷新标签树和视频列表。"""
        # 刷新标签树
        self.tag_tree.load_tags()
        # 刷新视频列表
        if self.current_tag:
            self.video_grid.load_by_tag(self.current_tag, page=1)
        else:
            self.video_grid.load_all(page=1)
