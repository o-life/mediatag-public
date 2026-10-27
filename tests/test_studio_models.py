"""影楼基础包 S3.1 数据模型测试：新表建表 / 词表种子 / 迁移守卫 / 素材挂接链路。"""

import sys

import pytest
from sqlalchemy import create_engine, inspect, text

sys.path.insert(0, "src")

from src.db.models import Base  # noqa: E402
from src.db import models as db_models  # noqa: E402


@pytest.fixture()
def engine():
    """全新库：create_all 一次成型（含 S3.1 全部新表与列）。"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def conn(engine):
    with engine.connect() as c:
        yield c


def test_studio_tables_created(engine):
    """新库 create_all 后：clients/event_types/projects 存在，tasks 含 project_id。"""
    insp = inspect(engine)
    for table in ("clients", "event_types", "projects"):
        assert insp.has_table(table), f"缺表: {table}"
    task_cols = [c["name"] for c in insp.get_columns("tasks")]
    assert "project_id" in task_cols


def test_seed_event_types_idempotent(conn):
    """词表种子：首次写入 13 类预置；重复调用幂等。"""
    assert db_models._seed_event_types(conn) == 13
    rows = conn.execute(
        text("SELECT name, is_builtin FROM event_types ORDER BY sort_order")
    ).fetchall()
    assert len(rows) == 13
    assert rows[0][0] == "婚礼"
    assert all(r[1] == 1 for r in rows)  # 预置类型标记
    assert db_models._seed_event_types(conn) == 0  # 幂等：非空不重播


def test_seed_presets_cover_life_events(conn):
    """婚丧嫁娶宝宝宴等人生事件类型必须在预置词表中。"""
    db_models._seed_event_types(conn)
    names = {r[0] for r in conn.execute(text("SELECT name FROM event_types")).fetchall()}
    for required in ("婚礼", "婚庆庆典", "宝宝宴·满月", "生日宴", "寿宴", "升学宴", "丧葬礼仪"):
        assert required in names


def test_client_project_task_chain(conn):
    """三层挂接链路：客户 → 活动（带类型/日期）→ 素材（tasks.project_id）。"""
    db_models._seed_event_types(conn)
    conn.execute(text("INSERT INTO clients (name) VALUES ('张三&李四')"))
    client_id = conn.execute(text("SELECT id FROM clients")).scalar()
    conn.execute(
        text(
            "INSERT INTO projects (client_id, event_type_id, name, event_date, status) "
            "VALUES (:cid, (SELECT id FROM event_types WHERE name='婚礼'), "
            "'张三李四婚礼', '2026-10-01', '拍摄中')"
        ),
        {"cid": client_id},
    )
    project_id = conn.execute(text("SELECT id FROM projects")).scalar()
    conn.execute(
        text(
            "INSERT INTO tasks (file_path, status, project_id) "
            "VALUES ('/v/IMG_0016.CR3', 'COMPLETE', :pid)"
        ),
        {"pid": project_id},
    )
    # 关联查询：活动 → 素材，且类型/客户可回溯
    row = conn.execute(
        text(
            "SELECT p.name, et.name, c.name, t.file_path FROM projects p "
            "JOIN event_types et ON et.id = p.event_type_id "
            "JOIN clients c ON c.id = p.client_id "
            "JOIN tasks t ON t.project_id = p.id"
        )
    ).fetchone()
    assert row == ("张三李四婚礼", "婚礼", "张三&李四", "/v/IMG_0016.CR3")


def test_migration_guard_adds_project_id(tmp_path):
    """老库（tasks 无 project_id）→ create_all 补新表 + 迁移守卫补列 + 种子。

    镜像生产顺序：init_db 先 create_all（新表齐了、旧表跳过）再跑迁移守卫。
    """
    engine = create_engine(f"sqlite:///{tmp_path}/old.db")
    with engine.connect() as c:
        # 模拟旧库：tasks 无 project_id，客单相关表不存在
        c.execute(
            text(
                "CREATE TABLE tasks ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "file_path TEXT NOT NULL, "
                "file_name TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'scheduled')"
            )
        )
        c.commit()

    # 生产顺序：create_all（旧 tasks 已存在则跳过，新表全建）
    Base.metadata.create_all(engine)

    conn = engine.connect()
    insp = inspect(engine)
    db_models._migrate_studio_tables(conn, insp)

    # 断言用迁移后的新 inspector（旧实例缓存迁移前的反射结果）
    cols = [c["name"] for c in inspect(engine).get_columns("tasks")]
    assert "project_id" in cols
    # 守卫同时确保词表种子就位
    names = conn.execute(text("SELECT name FROM event_types")).fetchall()
    assert len(names) == 13


def test_task_project_id_nullable_by_default(engine, conn):
    """老数据语义：不指定 project_id 即 NULL（未归类收件箱）。"""
    Base.metadata.create_all(engine)
    conn.execute(
        text("INSERT INTO tasks (file_path, status) VALUES ('/v/legacy.mp4', 'COMPLETE')")
    )
    row = conn.execute(
        text("SELECT project_id FROM tasks WHERE file_path='/v/legacy.mp4'")
    ).fetchone()
    assert row[0] is None


def test_project_custom_fields_column(engine, conn):
    """S3.4+：projects.custom_fields 列存在且默认空对象。"""
    from sqlalchemy import inspect as _inspect
    Base.metadata.create_all(engine)
    cols = [c['name'] for c in _inspect(engine).get_columns('projects')]
    assert 'custom_fields' in cols
    conn.execute(text(
        "INSERT INTO clients (name) VALUES ('字段客户')"
    ))
    cid = conn.execute(text("SELECT id FROM clients WHERE name='字段客户'")).fetchone()[0]
    conn.execute(
        text("INSERT INTO projects (client_id, name) VALUES (:cid, '字段项目')"),
        {"cid": cid},
    )
    row = conn.execute(
        text("SELECT custom_fields FROM projects WHERE name='字段项目'")
    ).fetchone()
    assert row[0] in (None, '{}')


def test_migration_guard_adds_custom_fields(tmp_path):
    """老库 projects 缺 custom_fields → 迁移守卫 ALTER 补列（幂等）。"""
    engine = create_engine(f"sqlite:///{tmp_path}/old2.db")
    with engine.connect() as c:
        c.execute(
            text(
                "CREATE TABLE tasks ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "file_path TEXT NOT NULL, "
                "file_name TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'scheduled', "
                "project_id INTEGER REFERENCES projects(id))"
            )
        )
        c.execute(
            text(
                "CREATE TABLE projects ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "client_id INTEGER NOT NULL, "
                "event_type_id INTEGER, "
                "name TEXT NOT NULL, "
                "status TEXT DEFAULT '拍摄中')"
            )
        )
        c.execute(
            text(
                "CREATE TABLE clients ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "name TEXT NOT NULL)"
            )
        )
        c.commit()

    Base.metadata.create_all(engine)
    conn = engine.connect()
    insp = inspect(engine)
    db_models._migrate_studio_tables(conn, insp)

    cols = [c["name"] for c in inspect(engine).get_columns("projects")]
    assert "custom_fields" in cols
