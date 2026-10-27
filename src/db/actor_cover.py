"""人物封面（`actors.cover_face` / `actor_faces.is_cover`）的**单一事实来源**。

历史 bug：create / update / rebuild / 合并 / 拆分 / 人工设封面 各自**内联**维护这两个
字段，增量路径会让两者脱节（`cover_face` 指向 `face_202` 而 `is_cover` 停在 `face_188`），
且 `cover_face` 可能指向已删除文件 → UI 无封面/错封面。

**所有**写封面处一律调用 `apply_actor_cover()`：两字段原子一致，且只选磁盘真实存在的脸。
（`FaceRecognizer._apply_actor_cover` 亦委托到本函数，保持既有调用点不变。）
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve(face_path, faces_dir: Path) -> Optional[Path]:
    """把 face_path（相对 faces_dir 或绝对）解析为磁盘上真实存在的路径；不存在返回 None。"""
    if not face_path:
        return None
    p = Path(str(face_path)).expanduser()
    if p.is_absolute():
        return p if p.exists() else None
    cand = faces_dir / p
    return cand if cand.exists() else None


def apply_actor_cover(
    db,
    actor_id: int,
    faces_dir,
    preferred_path: Optional[str] = None,
) -> Optional[str]:
    """原子地统一 `actor.cover_face` 与 `actor_faces.is_cover`。

    规则：
    1. 清空该 actor 全部 `is_cover` 标记；
    2. `preferred_path` 有效（磁盘文件存在）→ 选它；
    3. 否则当前 `cover_face` 仍有效则保留（历史脸可能优于本次候选）；
    4. 否则回退 `quality_score` 最高且磁盘文件存在的脸；
    5. 同步 `actor.cover_face`（无有效脸 → None）。

    Returns: 最终封面的相对路径（相对 faces_dir），无有效脸返回 None。
    """
    from src.db.models import Actor, ActorFace

    faces_dir = Path(faces_dir).expanduser()
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if actor is None:
            return None

        db.query(ActorFace).filter(ActorFace.actor_id == actor_id).update(
            {"is_cover": 0}, synchronize_session=False
        )

        chosen = _resolve(preferred_path, faces_dir) if preferred_path else None
        if chosen is None and actor.cover_face:
            chosen = _resolve(actor.cover_face, faces_dir)
        if chosen is None:
            rows = (
                db.query(ActorFace)
                .filter(ActorFace.actor_id == actor_id)
                .order_by(ActorFace.quality_score.desc())
                .all()
            )
            for af in rows:
                r = _resolve(af.face_path, faces_dir)
                if r is not None:
                    chosen = r
                    break

        if chosen is None:
            actor.cover_face = None
            db.flush()
            return None

        try:
            rel = str(chosen.relative_to(faces_dir))
        except ValueError:
            rel = chosen.name

        for af in db.query(ActorFace).filter(ActorFace.actor_id == actor_id).all():
            r = _resolve(af.face_path, faces_dir)
            if r is not None and r.resolve() == chosen.resolve():
                af.is_cover = 1
                break

        actor.cover_face = rel
        db.flush()
        return rel
    except Exception as e:  # noqa: BLE001
        logger.warning("应用封面失败 actor_id=%s: %s", actor_id, e)
        return None
