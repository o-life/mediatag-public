"""按厂牌自动归集 UI（影楼 S4.5）—— 预览-确认对话框 + 执行器。

解析纯逻辑在 core/brand_classifier.py。产品语义：
- 只对**未归集**内容做分组建议；预览中每组的**组名可编辑**（启发式命名
  不保证完美）、**勾选可取消**（可疑组不建）；
- 确认后：每组 find-or-create 客户（同名复用，否则以厂牌名新建）+
  项目（同名复用）→ 清单挂接（task_ids，服务端解析内容哈希）；
- 全程走 studio API，不做任何磁盘操作。
"""

from typing import List

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QScrollArea, QVBoxLayout,
    QWidget,
)

from src.ui.tokens import T

DEFAULT_CHECK_THRESHOLD = 5  # ≥5 个内容的组默认勾选，长尾默认不建


class BrandAutoclassDialog(QDialog):
    """分组预览：组名可编辑、勾选可取消；确认返回勾选组。"""

    def __init__(self, parent, groups: dict, unclassified: int):
        super().__init__(parent)
        self.setWindowTitle("按厂牌自动归集 — 预览")
        self.setMinimumSize(560, 560)
        self._rows: List[dict] = []

        layout = QVBoxLayout(self)
        info = QLabel(
            f"按文件名厂牌解析出 {len(groups)} 个候选组。"
            "组名可修改；取消勾选的组不会创建。"
            f"无法识别的 {unclassified} 个（相机/录屏/哈希/时间戳名）保持未归集。")
        info.setWordWrap(True)
        layout.addWidget(info)

        # 滚动区：组行（勾选 + 组名 + 计数）
        inner = QWidget()
        rows_v = QVBoxLayout(inner)
        rows_v.setContentsMargins(4, 4, 4, 4)
        rows_v.setSpacing(T.SP_4)

        named = {k: v for k, v in groups.items() if k != "__singletons__"}
        for name, grp in sorted(named.items(), key=lambda kv: -kv[1]["count"]):
            row = QHBoxLayout()
            cb = QCheckBox()
            cb.setChecked(grp["count"] >= DEFAULT_CHECK_THRESHOLD)
            name_edit = QLineEdit(grp["name"])
            name_edit.setFixedHeight(T.CTRL_SM_H)
            count_lbl = QLabel(f"{grp['count']} 个")
            count_lbl.setStyleSheet("color: #98989D;")
            count_lbl.setFixedWidth(64)
            count_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(cb)
            row.addWidget(name_edit, 1)
            row.addWidget(count_lbl)
            rows_v.addLayout(row)
            self._rows.append({"cb": cb, "edit": name_edit,
                               "task_ids": [it.get("task_id")
                                            for it in grp["items"]]})

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }"
                             " QScrollArea > QWidget > QWidget { background: transparent; }")
        layout.addWidget(scroll, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("归集勾选组")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_groups(self) -> List[dict]:
        """勾选组 → [{name, task_ids}]（组名去空白，空名跳过）。"""
        out = []
        for row in self._rows:
            if row["cb"].isChecked():
                name = row["edit"].text().strip()
                if name:
                    out.append({"name": name,
                                "task_ids": [t for t in row["task_ids"]
                                             if t is not None]})
        return out


class _AutoClassWorker(QThread):
    """后台执行：逐组建客户/项目（同名复用）+ 清单挂接。"""

    progress = Signal(int, int, str)
    note = Signal(str)
    done = Signal(int, int, int, int)  # 组数, 新建项目, 复用项目, 归集内容

    def __init__(self, api, groups: List[dict], parent=None):
        super().__init__(parent)
        self._api = api
        self._groups = groups
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _fetch_all(self, path: str, key: str) -> list:
        out, page = [], 1
        while True:
            resp = self._api.get(path, params={"size": 100, "page": page},
                                 timeout=10)
            batch = resp.get(key, [])
            out.extend(batch)
            if not batch or page * 100 >= resp.get("total", 0):
                return out
            page += 1

    def run(self):
        clients = self._fetch_all("/api/studio/clients", "clients")
        client_by_name = {c.get("name"): c.get("id") for c in clients}
        projects = self._fetch_all("/api/studio/projects", "projects")
        project_by_name = {p.get("name"): p.get("id") for p in projects}

        created_projects = reused_projects = assigned_total = 0
        done = 0
        for grp in self._groups:
            if self._cancelled:
                break
            name = grp["name"]
            self.progress.emit(done, len(self._groups), name)
            try:
                if name in project_by_name:
                    pid = project_by_name[name]
                    reused_projects += 1
                else:
                    cid = client_by_name.get(name)
                    if cid is None:
                        r = self._api.post("/api/studio/clients",
                                           json={"name": name}, timeout=10)
                        cid = r.get("id")
                        client_by_name[name] = cid
                    r = self._api.post("/api/studio/projects",
                                       json={"client_id": cid, "name": name},
                                       timeout=10)
                    pid = r.get("id")
                    created_projects += 1
                    project_by_name[name] = pid

                r = self._api.post(
                    f"/api/studio/projects/{pid}/assign",
                    json={"task_ids": grp["task_ids"]}, timeout=30)
                assigned_total += r.get("assigned", 0)
            except Exception as exc:  # 单组失败记因不中断整批
                self.note.emit(f"组「{name}」失败: {exc}")
            done += 1

        self.done.emit(done, created_projects, reused_projects,
                       assigned_total)


def execute_autoclass(parent, api, groups: List[dict]) -> None:
    """确认后的执行入口：模态进度 + 取消 + 结果汇总。groups 来自对话框。"""
    if not groups:
        return
    worker = _AutoClassWorker(api, groups)

    prog = QDialog(parent)
    prog.setWindowTitle("自动归集中")
    prog.setMinimumWidth(420)
    v = QVBoxLayout(prog)
    lbl = QLabel("准备中…")
    bar = QProgressBar()
    bar.setRange(0, len(groups))
    v.addWidget(lbl)
    v.addWidget(bar)

    state = {"created": 0, "reused": 0, "assigned": 0, "notes": []}

    def on_progress(done, total, name):
        bar.setValue(done)
        lbl.setText(f"{done}/{total}　{name}")

    def on_note(msg):
        state["notes"].append(msg)

    def on_done(groups_done, created, reused, assigned):
        state.update({"created": created, "reused": reused,
                      "assigned": assigned})
        prog.accept()

    worker.progress.connect(on_progress)
    worker.note.connect(on_note)
    worker.done.connect(on_done)
    prog.rejected.connect(worker.cancel)
    worker.start()
    prog.exec()
    worker.wait()

    msg = (f"处理 {len(groups)} 组：新建项目 {state['created']}，"
           f"复用 {state['reused']}，归集内容 {state['assigned']} 个")
    if state["notes"]:
        msg += "\n" + "\n".join(state["notes"][:8])
    QMessageBox.information(parent, "自动归集完成", msg)
