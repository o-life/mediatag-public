"""Shared test fixtures and utilities for ai-video-factory tests."""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# 测试不得建主题工作区、不得改动仓库 data/ 下的运行时状态：
# 关闭启动链路的自动建档，让解析回退到旧布局（行为与引入主题工作区前一致）。
os.environ.setdefault("VIDEO_TAG_THEME_AUTOCREATE", "0")

# 主题注册表默认落在 <base>/themes/registry.json（开发态即 <repo>/data/）——
# 测试里改到临时目录，避免往仓库写运行时文件。
os.environ.setdefault(
    "VIDEO_TAG_THEME_REGISTRY_PATH",
    os.path.join(tempfile.mkdtemp(prefix="vt_theme_registry_"), "registry.json"),
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.models import (
    Base, get_db_engine, get_session,
    Task, Actor, FileEntity,
    FileLocation
)


@pytest.fixture
def db_engine():
    """Create a temporary in-memory database."""
    engine = get_db_engine(":memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create a database session."""
    session = get_session(db_engine)
    yield session
    session.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a fake video file for testing."""
    video_path = temp_dir / "test_video.mp4"
    video_path.touch()
    return str(video_path)


@pytest.fixture
def sample_actor(db_session):
    """Create a sample actor."""
    actor = Actor(
        code="ACTOR_001",
        name="Test Actor",
        primary_name="John Doe",
        aliases='["J. Doe", "Johnny"]'
    )
    db_session.add(actor)
    db_session.commit()
    return actor


def create_test_entity(db_session, file_path, md5=None, file_size=1000, duration=120.0):
    """Helper to create FileEntity and FileLocation."""
    if md5 is None:
        import hashlib
        md5 = hashlib.md5(file_path.encode()).hexdigest()

    entity = FileEntity(
        file_md5=md5,
        file_size=file_size,
        duration=duration
    )
    db_session.add(entity)
    db_session.flush()

    location = FileLocation(
        entity_id=entity.id,
        file_path=file_path,
        file_name=Path(file_path).name,
        is_primary=1
    )
    db_session.add(location)
    db_session.commit()
    return entity


def create_test_task(db_session, entity, status="PENDING", progress=0):
    """Helper to create a Task."""
    primary_path = None
    for loc in entity.locations:
        if loc.is_primary == 1:
            primary_path = loc.file_path
            break
    if not primary_path and entity.locations:
        primary_path = entity.locations[0].file_path

    task = Task(
        entity_id=entity.id,
        file_path=primary_path,
        file_md5=entity.file_md5,
        file_size=entity.file_size,
        duration=entity.duration,
        status=status,
        progress=progress
    )
    db_session.add(task)
    db_session.commit()
    return task


def create_test_actor(db_session, code=None, name=None):
    """Helper to create an Actor."""
    import uuid
    code = code or f"ACTOR_{uuid.uuid4().hex[:6].upper()}"
    actor = Actor(
        code=code,
        name=name or f"Test Actor {code}"
    )
    db_session.add(actor)
    db_session.commit()
    return actor


# ── smoke 分层 ────────────────────────────────────────────────────
# gate.sh 用 `pytest -m smoke` 跑快速子集（<3 分钟，实际 ~5s）。
# 入选标准：纯逻辑、不依赖模型下载/重 IO（配置分层、sqlite 迁移、门控、
# 编解码、路径解析等）。全集仍每日/合并前跑（scripts/test-full.sh）。
# 新增 smoke 文件：往 SMOKE_FILES 加一行即可。
SMOKE_FILES = frozenset({
    "test_actor_code_alignment.py",
    "test_asset_export.py",
    "test_actor_code_display.py",
    "test_actor_split.py",
    "test_batch_unqueue.py",
    "test_embedded_inference.py",
    "test_perf_tiers.py",
    "test_file_health.py",
    "test_error_taxonomy.py",
    "test_config_layering.py",
    "test_consolidate_workspaces.py",
    "test_dictionary_export.py",
    "test_dictionary_listing.py",
    "test_embedding_codec.py",
    "test_faces_dir_resolution.py",
    "test_lazy_duration.py",
    "test_media_sniff.py",
    "test_photos_importer.py",
    "test_licensing_gate.py",
    "test_parse_tags_from_vlm.py",
    "test_quick_filter_logic.py",
    "test_status_filter.py",
    "test_studio_api.py",
    "test_project_roster.py",
    "test_studio_models.py",
    "test_tag_quality_gates.py",
    "test_theme_registry.py",
    "test_track_grouping.py",
    "test_uniform_sampling_target.py",
    "test_vlm_readiness_gate.py",
    "test_write_gate.py",
    "test_split_faces_dialog.py",
    "test_pagination_widget.py",
    "test_state_machine.py",
})


def pytest_collection_modifyitems(config, items):
    import os
    smoke = pytest.mark.smoke
    for item in items:
        if os.path.basename(item.fspath.strpath) in SMOKE_FILES:
            item.add_marker(smoke)
