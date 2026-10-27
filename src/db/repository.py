"""Repository (CRUD) layer for all 5 SQLite tables.

All functions accept an optional `path` to override the default DB location.
"""

import json
import sqlite3
from typing import Optional, List, Dict, Any
from src.db.schema import get_connection, init_db


# ────────────────────────────────────────────────────────────────
# tasks
# ────────────────────────────────────────────────────────────────

def task_exists_by_md5(db: sqlite3.Connection, file_md5: str) -> bool:
    row = db.execute("SELECT id FROM tasks WHERE file_md5=? LIMIT 1", (file_md5,)).fetchone()
    return row is not None


def get_task_by_md5(db: sqlite3.Connection, file_md5: str) -> Optional[Dict[str, Any]]:
    """Get a task by its MD5 hash. Returns None if not found."""
    row = db.execute("SELECT * FROM tasks WHERE file_md5=?", (file_md5,)).fetchone()
    return dict(row) if row else None


def delete_task_by_md5(db: sqlite3.Connection, file_md5: str) -> int:
    """Delete a task and all its associated data by MD5. Returns number of tasks deleted."""
    task = get_task_by_md5(db, file_md5)
    if not task:
        return 0
    
    task_id = task["id"]
    # Delete associated data
    db.execute("DELETE FROM face_embeddings WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM face_detections WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM face_dense_frames WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM face_segments WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM montage_units WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM nfo_writes WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM audit_queue WHERE source_task_id=?", (task_id,))
    db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    db.commit()
    return 1


def upsert_task(db: sqlite3.Connection, file_path: str, file_name: str,
                file_size: int, file_md5: Optional[str] = None, status: str = "scheduled") -> Dict[str, Any]:
    # Check by md5 first (unique), then by path (may have dupes)
    existing = db.execute("SELECT id FROM tasks WHERE file_md5=? LIMIT 1", (file_md5,)).fetchone() if file_md5 else None
    if not existing:
        existing = db.execute("SELECT id FROM tasks WHERE file_path=? LIMIT 1", (file_path,)).fetchone()

    is_new = False
    if existing:
        if file_md5:
            db.execute("""
                UPDATE tasks SET file_path=?, file_name=?, file_size=?, status=?, 
                               updated_at=datetime('now'), current_unit=0, total_units=0, 
                               frames_analyzed=0, audit_progress='{}', completed_at=NULL, 
                               file_md5=? WHERE id=?
            """, (file_path, file_name, file_size, status, file_md5, existing[0]))
        else:
            db.execute("""
                UPDATE tasks SET file_path=?, file_name=?, file_size=?, status=?, 
                               updated_at=datetime('now'), current_unit=0, total_units=0, 
                               frames_analyzed=0, audit_progress='{}', completed_at=NULL 
                WHERE id=?
            """, (file_path, file_name, file_size, status, existing[0]))
        db.commit()
        row = db.execute("SELECT * FROM tasks WHERE id=?", (existing[0],)).fetchone()
    else:
        cur = db.execute("""
            INSERT INTO tasks (file_path, file_name, file_size, file_md5, status)
            VALUES (?, ?, ?, ?, ?)
        """, (file_path, file_name, file_size, file_md5, status))
        db.commit()
        is_new = True
        row = db.execute("SELECT * FROM tasks WHERE id=?", (cur.lastrowid,)).fetchone()
    return {**dict(row), "is_new": is_new}


def get_task(db: sqlite3.Connection, task_id: int) -> Optional[Dict[str, Any]]:
    row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return dict(row) if row else None


def update_task_status(db: sqlite3.Connection, task_id: int, status: str) -> None:
    db.execute("""
        UPDATE tasks SET status=?, updated_at=datetime('now'),
            current_unit=CASE WHEN ? IN ('scanning','pending') THEN 0 ELSE current_unit END
        WHERE id=?
    """, (status, status, task_id))
    db.commit()


def update_task_progress(db: sqlite3.Connection, task_id: int,
                         current_unit: int, total_units: int,
                         frames_analyzed: int) -> None:
    db.execute("""
        UPDATE tasks SET current_unit=?, total_units=?, frames_analyzed=?,
            updated_at=datetime('now') WHERE id=?
    """, (current_unit, total_units, frames_analyzed, task_id))
    db.commit()


def complete_task(db: sqlite3.Connection, task_id: int) -> None:
    db.execute("""
        UPDATE tasks SET status='complete', completed_at=datetime('now'),
            updated_at=datetime('now') WHERE id=?
    """, (task_id,))
    db.commit()


def list_tasks(db: sqlite3.Connection, status: Optional[str] = None) -> List[Dict[str, Any]]:
    if status:
        rows = db.execute("SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC", (status,)).fetchall()
    else:
        rows = db.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


# ────────────────────────────────────────────────────────────────
# montage_units
# ────────────────────────────────────────────────────────────────

def get_montage_units(db: sqlite3.Connection, task_id: int) -> List[Dict[str, Any]]:
    rows = db.execute(
        "SELECT * FROM montage_units WHERE task_id=? ORDER BY unit_index", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_pending_unit(db: sqlite3.Connection, task_id: int) -> Optional[Dict[str, Any]]:
    row = db.execute(
        "SELECT * FROM montage_units WHERE task_id=? AND status='pending' ORDER BY unit_index LIMIT 1",
        (task_id,),
    ).fetchone()
    return dict(row) if row else None


def get_analyzed_unit_count(db: sqlite3.Connection, task_id: int) -> int:
    return db.execute(
        "SELECT COUNT(*) as cnt FROM montage_units WHERE task_id=? AND status IN ('analyzed','reviewed')",
        (task_id,),
    ).fetchone()["cnt"]


def create_montage_units(db: sqlite3.Connection, task_id: int,
                         units: List[Dict[str, Any]]) -> None:
    """Insert montage units in one transaction.

    `units` is a list of dicts with keys: unit_index, start_frame, end_frame, duration_seconds.
    """
    db.executemany("""
        INSERT OR IGNORE INTO montage_units (task_id, unit_index, start_frame, end_frame, duration_seconds, status)
        VALUES (:task_id, :unit_index, :start_frame, :end_frame, :duration_seconds, 'pending')
    """, [{"task_id": task_id, **u} for u in units])
    db.commit()


def update_montage_unit(db: sqlite3.Connection, unit_id: int, **fields) -> None:
    allowed = {"status", "frame_paths", "raw_tags", "start_frame", "end_frame", "duration_seconds"}
    set_parts = []
    params = []
    for k, v in fields.items():
        if k in allowed:
            set_parts.append(f"{k}=?")
            params.append(v if k not in ("frame_paths", "raw_tags") else json.dumps(v))
    if not set_parts:
        return
    db.execute(f"UPDATE montage_units SET {', '.join(set_parts)} WHERE id={unit_id}", params)
    db.commit()


# ────────────────────────────────────────────────────────────────
# dictionary
# ────────────────────────────────────────────────────────────────

def get_dictionary_entries(db: sqlite3.Connection, category: Optional[str] = None) -> List[Dict[str, Any]]:
    if category:
        rows = db.execute(
            "SELECT * FROM dictionary WHERE category=? ORDER BY standard_tag", (category,)
        ).fetchall()
    else:
        rows = db.execute("SELECT * FROM dictionary ORDER BY category, standard_tag").fetchall()
    result = []
    for r in rows:
        row = dict(r)
        row["aliases"] = json.loads(row["aliases"]) if row["aliases"] else []
        result.append(row)
    return result


def upsert_dictionary(db: sqlite3.Connection, category: str, standard_tag: str,
                      aliases: Optional[List[str]] = None, description: str = "") -> Dict[str, Any]:
    aliases_json = json.dumps(aliases or [])
    existing = db.execute(
        "SELECT id FROM dictionary WHERE standard_tag=? AND category=?",
        (standard_tag, category),
    ).fetchone()
    if existing:
        db.execute("""
            UPDATE dictionary SET aliases=?, description=?, updated_at=datetime('now')
            WHERE id=?
        """, (aliases_json, description, existing["id"]))
        db.commit()
        row = db.execute("SELECT * FROM dictionary WHERE id=?", (existing["id"],)).fetchone()
    else:
        cur = db.execute("""
            INSERT INTO dictionary (category, standard_tag, aliases, description)
            VALUES (?, ?, ?, ?)
        """, (category, standard_tag, aliases_json, description))
        db.commit()
        row = db.execute("SELECT * FROM dictionary WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)


def delete_dictionary_entry(db: sqlite3.Connection, standard_tag: str, category: str) -> int:
    cur = db.execute("DELETE FROM dictionary WHERE standard_tag=? AND category=?", (standard_tag, category))
    db.commit()
    return cur.rowcount


def get_dictionary_categories(db: sqlite3.Connection) -> List[str]:
    rows = db.execute("SELECT DISTINCT category FROM dictionary ORDER BY category").fetchall()
    return [r["category"] for r in rows]


def get_dictionary_stats(db: sqlite3.Connection) -> Dict[str, Any]:
    """Get dictionary statistics: total tags and category count."""
    total = db.execute("SELECT COUNT(*) as cnt FROM dictionary").fetchone()["cnt"]
    categories = db.execute("SELECT COUNT(DISTINCT category) as cnt FROM dictionary").fetchone()["cnt"]
    return {"total_tags": total, "category_count": categories}


def export_dictionary(db: sqlite3.Connection) -> str:
    """Export dictionary entries as Markdown format.

    Format:
        # 标准标签库

        ## 分类名
        - 标签名 (别名1, 别名2)
        - 标签名2
    """
    entries = get_dictionary_entries(db)
    if not entries:
        return "# 标准标签库\n\n（空）\n"

    # Group by category
    by_category: Dict[str, List[Dict]] = {}
    for entry in entries:
        cat = entry["category"] or "其他"
        by_category.setdefault(cat, []).append(entry)

    lines = ["# 标准标签库", ""]
    for category in sorted(by_category.keys()):
        lines.append(f"## {category}")
        for entry in by_category[category]:
            tag_name = entry["standard_tag"]
            aliases = entry.get("aliases", [])
            if aliases:
                alias_str = ", ".join(aliases)
                lines.append(f"- {tag_name} ({alias_str})")
            else:
                lines.append(f"- {tag_name}")
        lines.append("")

    return "\n".join(lines)


def import_dictionary(db: sqlite3.Connection, content: str, mode: str = "merge") -> Dict[str, Any]:
    """Import dictionary entries from Markdown content.

    Args:
        db: Database connection
        content: Markdown text with ## category headers and - tag (aliases) lines
        mode: "replace" to clear existing first, "merge" to skip duplicates

    Returns:
        dict with imported count, skipped count
    """
    import re

    imported = 0
    skipped = 0
    current_category = None

    # Parse markdown: ## headers for categories, - lines for tags
    for line in content.split("\n"):
        line = line.rstrip()

        # Category header: ## 分类名
        cat_match = re.match(r"^##\s+(.+)$", line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue

        # Tag line: - 标签名 or - 标签名 (别名1, 别名2)
        tag_match = re.match(r"^-\s+(.+)$", line)
        if tag_match and current_category:
            tag_content = tag_match.group(1).strip()

            # Parse tag name and aliases: "标签名 (别名1, 别名2)"
            alias_match = re.match(r"^(.+?)\s*\((.+)\)\s*$", tag_content)
            if alias_match:
                tag_name = alias_match.group(1).strip()
                aliases_str = alias_match.group(2).strip()
                aliases = [a.strip() for a in re.split(r"[,，]", aliases_str) if a.strip()]
            else:
                tag_name = tag_content
                aliases = []

            # Check if tag already exists
            existing = db.execute(
                "SELECT id FROM dictionary WHERE standard_tag=?",
                (tag_name,),
            ).fetchone()

            if existing:
                if mode == "replace":
                    # Update existing entry
                    aliases_json = json.dumps(aliases)
                    db.execute(
                        "UPDATE dictionary SET category=?, aliases=?, updated_at=datetime('now') WHERE id=?",
                        (current_category, aliases_json, existing["id"]),
                    )
                    imported += 1
                else:
                    # merge mode: skip existing
                    skipped += 1
            else:
                # Insert new entry
                aliases_json = json.dumps(aliases)
                db.execute(
                    "INSERT INTO dictionary (category, standard_tag, aliases) VALUES (?, ?, ?)",
                    (current_category, tag_name, aliases_json),
                )
                imported += 1

    if mode == "replace":
        # For replace mode, we updated existing + inserted new
        # No need to delete anything that wasn't in the import
        pass

    db.commit()
    return {"imported": imported, "skipped": skipped}


def clear_dictionary(db: sqlite3.Connection) -> int:
    """Delete all dictionary entries. Returns count deleted."""
    cur = db.execute("DELETE FROM dictionary")
    db.commit()
    return cur.rowcount


# ────────────────────────────────────────────────────────────────
# audit_queue
# ────────────────────────────────────────────────────────────────

def get_audit_queue(db: sqlite3.Connection, task_id: Optional[int] = None,
                    status: Optional[str] = None) -> List[Dict[str, Any]]:
    clauses = []
    params = []
    if task_id is not None:
        clauses.append("source_task_id=?")
        params.append(task_id)
    if status is not None:
        clauses.append("status=?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = db.execute(f"SELECT * FROM audit_queue {where} ORDER BY created_at DESC", params).fetchall()
    return [dict(r) for r in rows]


def get_pending_audit(db: sqlite3.Connection) -> List[Dict[str, Any]]:
    return get_audit_queue(db, status="pending")


def update_audit_status(db: sqlite3.Connection, audit_id: int, status: str,
                        mapped_to: Optional[str] = None) -> None:
    db.execute("""
        UPDATE audit_queue SET status=?, mapped_to=?, reviewed_at=datetime('now')
        WHERE id=?
    """, (status, mapped_to, audit_id))
    db.commit()


def delete_audit_entry(db: sqlite3.Connection, audit_id: int) -> int:
    """Delete an audit_queue entry by id. Returns the number of rows deleted."""
    cur = db.execute("DELETE FROM audit_queue WHERE id=?", (audit_id,))
    db.commit()
    return cur.rowcount


def add_audit_entry(db: sqlite3.Connection, category: str, raw_label: str,
                    source_task_id: int = None) -> Dict[str, Any]:
    """Insert a raw label for audit. If an identical pending entry already exists, return it."""
    existing = db.execute(
        "SELECT id FROM audit_queue WHERE raw_label=? AND category=? AND status='pending' LIMIT 1",
        (raw_label, category),
    ).fetchone()
    if existing:
        return dict(db.execute("SELECT * FROM audit_queue WHERE id=?", (existing["id"],)).fetchone())
    cur = db.execute("""
        INSERT INTO audit_queue (category, raw_label, source_task_id)
        VALUES (?, ?, ?)
    """, (category, raw_label, source_task_id))
    db.commit()
    return dict(db.execute("SELECT * FROM audit_queue WHERE id=?", (cur.lastrowid,)).fetchone())


# ────────────────────────────────────────────────────────────────
# nfo_writes
# ────────────────────────────────────────────────────────────────

def add_nfo_write(db: sqlite3.Connection, task_id: int, nfo_path: str,
                  written_tags: List[str]) -> Dict[str, Any]:
    cur = db.execute("""
        INSERT INTO nfo_writes (task_id, nfo_path, written_tags)
        VALUES (?, ?, ?)
    """, (task_id, nfo_path, json.dumps(written_tags)))
    db.commit()
    return dict(db.execute("SELECT * FROM nfo_writes WHERE id=?", (cur.lastrowid,)).fetchone())


def get_nfo_writes_for_task(db: sqlite3.Connection, task_id: int) -> List[Dict[str, Any]]:
    rows = db.execute(
        "SELECT * FROM nfo_writes WHERE task_id=? ORDER BY wrote_at DESC", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ────────────────────────────────────────────────────────────────
# System status
# ────────────────────────────────────────────────────────────────

def get_system_status(db: sqlite3.Connection) -> Dict[str, Any]:
    total = db.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
    scheduled = db.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status='scheduled'").fetchone()["cnt"]
    scanning = db.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status='scanning'").fetchone()["cnt"]
    analyzing = db.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status='analyzing'").fetchone()["cnt"]
    mapping = db.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status='mapping'").fetchone()["cnt"]
    audit_pending = db.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status='audit_pending'").fetchone()["cnt"]
    audit_done = db.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status='audit_done'").fetchone()["cnt"]
    nfo_writing = db.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status='nfo_writing'").fetchone()["cnt"]
    complete = db.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status='complete'").fetchone()["cnt"]
    pending_audit = db.execute("SELECT COUNT(*) as cnt FROM audit_queue WHERE status='pending'").fetchone()["cnt"]

    return {
        "total_tasks": total,
        "status_breakdown": {
            "scheduled": scheduled,
            "scanning": scanning,
            "analyzing": analyzing,
            "mapping": mapping,
            "audit_pending": audit_pending,
            "audit_done": audit_done,
            "nfo_writing": nfo_writing,
            "complete": complete,
        },
        "pending_audit_labels": pending_audit,
    }



# ────────────────────────────────────────────────────────────────
# face_segments
# ────────────────────────────────────────────────────────────────

def add_face_segment(db: sqlite3.Connection, task_id: int, unit_index: int,
                     start_time: float, end_time: float,
                     face_count_prescan: int = 0,
                     prescan_status: str = "pending",
                     dense_status: str = "pending") -> Dict[str, Any]:
    cur = db.execute(
        """INSERT INTO face_segments
           (task_id, unit_index, start_time, end_time, face_count_prescan,
            prescan_status, dense_status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (task_id, unit_index, start_time, end_time, face_count_prescan,
         prescan_status, dense_status),
    )
    db.commit()
    row = db.execute("SELECT * FROM face_segments WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_face_segments_for_task(db: sqlite3.Connection, task_id: int) -> List[Dict[str, Any]]:
    rows = db.execute(
        "SELECT * FROM face_segments WHERE task_id = ? ORDER BY unit_index, start_time",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_face_segment_status(db: sqlite3.Connection, segment_id: int,
                               prescan_status: Optional[str] = None,
                               dense_status: Optional[str] = None) -> None:
    fields, values = [], []
    if prescan_status is not None:
        fields.append("prescan_status = ?")
        values.append(prescan_status)
    if dense_status is not None:
        fields.append("dense_status = ?")
        values.append(dense_status)
    if not fields:
        return
    values.append(segment_id)
    db.execute(f"UPDATE face_segments SET {', '.join(fields)} WHERE id = ?", values)
    db.commit()


# ────────────────────────────────────────────────────────────────
# face_dense_frames
# ────────────────────────────────────────────────────────────────

def add_face_dense_frame(db: sqlite3.Connection, task_id: int, segment_id: int,
                         frame_index: int, timestamp: float,
                         frame_path: str = "",
                         detection_status: str = "pending") -> Dict[str, Any]:
    cur = db.execute(
        """INSERT INTO face_dense_frames
           (task_id, segment_id, frame_index, timestamp, frame_path, detection_status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (task_id, segment_id, frame_index, timestamp, frame_path, detection_status),
    )
    db.commit()
    row = db.execute("SELECT * FROM face_dense_frames WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_dense_frames_for_segment(db: sqlite3.Connection, segment_id: int) -> List[Dict[str, Any]]:
    rows = db.execute(
        "SELECT * FROM face_dense_frames WHERE segment_id = ? ORDER BY frame_index",
        (segment_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_dense_frames_for_task(db: sqlite3.Connection, task_id: int) -> List[Dict[str, Any]]:
    rows = db.execute(
        "SELECT * FROM face_dense_frames WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_dense_frame_status(db: sqlite3.Connection, frame_id: int,
                              status: str, error_msg: Optional[str] = None) -> None:
    db.execute(
        "UPDATE face_dense_frames SET detection_status = ?, error_message = ? WHERE id = ?",
        (status, error_msg, frame_id),
    )
    db.commit()


# ────────────────────────────────────────────────────────────────
# face_detections
# ────────────────────────────────────────────────────────────────

def add_face_detection(db: sqlite3.Connection, dense_frame_id: int, task_id: int,
                       face_index: int, confidence: float,
                       bbox: List[int], landmarks: Dict,
                       quality_clarity: float = 0.0, quality_pose: float = 0.0,
                       quality_brightness: float = 0.0,
                       quality_overall: float = 0.0) -> Dict[str, Any]:
    cur = db.execute(
        """INSERT INTO face_detections
           (dense_frame_id, task_id, face_index, confidence,
            bbox, landmarks, quality_clarity, quality_pose,
            quality_brightness, quality_overall)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (dense_frame_id, task_id, face_index, confidence,
         json.dumps(bbox), json.dumps(landmarks),
         quality_clarity, quality_pose, quality_brightness, quality_overall),
    )
    db.commit()
    row = db.execute("SELECT * FROM face_detections WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_detections_for_frame(db: sqlite3.Connection, dense_frame_id: int) -> List[Dict[str, Any]]:
    rows = db.execute(
        "SELECT * FROM face_detections WHERE dense_frame_id = ? ORDER BY face_index",
        (dense_frame_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_detections_for_task(db: sqlite3.Connection, task_id: int) -> List[Dict[str, Any]]:
    rows = db.execute(
        "SELECT * FROM face_detections WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def count_faces_for_task(db: sqlite3.Connection, task_id: int) -> int:
    row = db.execute(
        "SELECT COUNT(*) as cnt FROM face_detections WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    return row["cnt"] if row else 0


# ────────────────────────────────────────────────────────────────
# face_embeddings
# ────────────────────────────────────────────────────────────────

def add_face_embedding(db: sqlite3.Connection, face_detection_id: int, task_id: int,
                       embedding: bytes, embedding_dim: int = 512,
                       quality: float = 0.0, face_image_path: str = "",
                       recognition_status: str = "completed") -> Dict[str, Any]:
    cur = db.execute(
        """INSERT INTO face_embeddings
           (face_detection_id, task_id, embedding, embedding_dim,
            quality, face_image_path, recognition_status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (face_detection_id, task_id, embedding, embedding_dim,
         quality, face_image_path, recognition_status),
    )
    db.commit()
    row = db.execute("SELECT * FROM face_embeddings WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_embeddings_for_task(db: sqlite3.Connection, task_id: int) -> List[Dict[str, Any]]:
    rows = db.execute(
        "SELECT * FROM face_embeddings WHERE task_id = ? AND recognition_status = 'completed' ORDER BY id",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_unprocessed_face_ids(db: sqlite3.Connection, task_id: int) -> List[int]:
    """Get face detection IDs that don't have embeddings yet."""
    rows = db.execute(
        """SELECT fd.id FROM face_detections fd
           LEFT JOIN face_embeddings fe ON fd.id = fe.face_detection_id
           WHERE fd.task_id = ? AND fe.id IS NULL
           ORDER BY fd.id""",
        (task_id,),
    ).fetchall()
    return [r["id"] for r in rows]


def delete_face_data_for_task(db: sqlite3.Connection, task_id: int) -> int:
    """Delete all face-related data for a task. Returns number of tasks affected."""
    db.execute("DELETE FROM face_embeddings WHERE task_id = ?", (task_id,))
    db.execute("DELETE FROM face_detections WHERE task_id = ?", (task_id,))
    db.execute("DELETE FROM face_dense_frames WHERE task_id = ?", (task_id,))
    db.execute("DELETE FROM face_segments WHERE task_id = ?", (task_id,))
    db.commit()
    return 1


# ── delete_task (needed by state_machine.py) ─────────────────

def delete_task(db: sqlite3.Connection, task_id: int) -> None:
    """Delete a task and all its associated data."""
    db.execute("DELETE FROM face_embeddings WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM face_detections WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM face_dense_frames WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM face_segments WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM montage_units WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM nfo_writes WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM audit_queue WHERE source_task_id=?", (task_id,))
    db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    db.commit()


# ────────────────────────────────────────────────────────────────
# Convenience — create or open DB
# ────────────────────────────────────────────────────────────────

def get_db(path: Optional[str] = None) -> sqlite3.Connection:
    """Open (or init + open) the database."""
    init_db(path)
    return get_connection(path)
