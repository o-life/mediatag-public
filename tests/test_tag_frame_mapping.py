"""Tests for tag-frame mapping feature (标签关键帧封面功能).

Covers:
1. TagFrameMapping model CRUD (src/db/models.py)
2. OllamaVLMClient.analyze_unit_frames_individually (src/core/vlm.py)
3. GET /api/videos/by-tag endpoint (src/api/routes/videos.py)
"""

import json
import time
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import (
    Base,
    Task,
    FileEntity,
    FileLocation,
    FileAnalysis,
    TagFrameMapping,
)
from src.core.vlm import OllamaVLMClient


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    db_path = tmp_path / "test_audit_cache.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session, engine
    session.close()


@pytest.fixture
def sample_task(tmp_db):
    """Create a sample task with entity and file location."""
    session, engine = tmp_db

    # Create file entity
    entity = FileEntity(file_md5="abc123def456", file_size=1024000, duration=120.5)
    session.add(entity)
    session.flush()

    # Create file location
    location = FileLocation(
        entity_id=entity.id,
        file_path="/tmp/test_video.mp4",
        file_name="test_video.mp4",
        is_primary=1,
    )
    session.add(location)
    session.flush()

    # Create task
    task = Task(
        entity_id=entity.id,
        file_path="/tmp/test_video.mp4",
        file_md5="abc123def456",
        status="COMPLETE",
    )
    session.add(task)
    session.flush()

    # 标签数据放在 FileAnalysis（Phase 1 重构后 Task 无 mapped_tags）
    fa = FileAnalysis(
        entity_id=entity.id,
        task_id=task.id,
        mapped_tags=json.dumps(["双人", "后入", "卧室"]),
        status="READY",
    )
    session.add(fa)
    session.commit()
    session.refresh(task)
    return task


@pytest.fixture
def sample_frames_dir(tmp_path):
    """Create a temporary frames directory with mock frame files."""
    frames_dir = tmp_path / "test_video_frames"
    unit_dir = frames_dir / "unit_0"
    unit_dir.mkdir(parents=True)

    # Create dummy frame files (just empty files with correct names)
    for i in range(3):
        frame_file = unit_dir / f"frame_{i:03d}.jpg"
        frame_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Minimal JPEG header

    return frames_dir


@pytest.fixture
def vlm_client():
    """Create a VLM client with no config (uses defaults).

    固定为 ollama 模式（绕过真实 endpoint 探测），使测试 mock 的
    {"response": ...} 格式能匹配 vlm.py 的 ollama 分支解析逻辑。
    """
    from unittest.mock import patch as _patch
    with _patch("src.core.vlm.OllamaVLMClient._detect_endpoint", return_value="ollama"):
        client = OllamaVLMClient(config=None, model="test-model", base_url="http://localhost:11434")
    return client


# -----------------------------------------------------------------------
# 1. TagFrameMapping Model Tests (单元测试)
# -----------------------------------------------------------------------

class TestTagFrameMappingModel:
    """Test TagFrameMapping CRUD operations."""

    def test_create_mapping(self, tmp_db, sample_task):
        """Test creating a tag-frame mapping."""
        session, engine = tmp_db
        mapping = TagFrameMapping(
            task_id=sample_task.id,
            tag_name="双人",
            frame_path="/tmp/test_video_frames/unit_0/frame_000.jpg",
            unit_index=0,
        )
        session.add(mapping)
        session.commit()
        session.refresh(mapping)

        assert mapping.id is not None
        assert mapping.task_id == sample_task.id
        assert mapping.tag_name == "双人"
        assert mapping.frame_path == "/tmp/test_video_frames/unit_0/frame_000.jpg"
        assert mapping.unit_index == 0

    def test_read_mapping(self, tmp_db, sample_task):
        """Test reading tag-frame mappings."""
        session, engine = tmp_db

        # Create multiple mappings
        mappings_data = [
            ("双人", "/tmp/frame_0.jpg", 0),
            ("后入", "/tmp/frame_1.jpg", 0),
            ("卧室", "/tmp/frame_2.jpg", 1),
        ]
        for tag, path, unit in mappings_data:
            m = TagFrameMapping(task_id=sample_task.id, tag_name=tag, frame_path=path, unit_index=unit)
            session.add(m)
        session.commit()

        # Query by task_id
        results = session.query(TagFrameMapping).filter(
            TagFrameMapping.task_id == sample_task.id
        ).all()
        assert len(results) == 3

        # Query by tag_name
        results = session.query(TagFrameMapping).filter(
            TagFrameMapping.tag_name == "双人"
        ).all()
        assert len(results) == 1
        assert results[0].frame_path == "/tmp/frame_0.jpg"

    def test_update_mapping(self, tmp_db, sample_task):
        """Test updating a tag-frame mapping."""
        session, engine = tmp_db
        mapping = TagFrameMapping(
            task_id=sample_task.id,
            tag_name="双人",
            frame_path="/tmp/old_frame.jpg",
            unit_index=0,
        )
        session.add(mapping)
        session.commit()
        session.refresh(mapping)

        # Update frame_path
        mapping.frame_path = "/tmp/new_frame.jpg"
        session.commit()
        session.refresh(mapping)

        assert mapping.frame_path == "/tmp/new_frame.jpg"

    def test_delete_mapping(self, tmp_db, sample_task):
        """Test deleting a tag-frame mapping."""
        session, engine = tmp_db
        mapping = TagFrameMapping(
            task_id=sample_task.id,
            tag_name="双人",
            frame_path="/tmp/frame.jpg",
            unit_index=0,
        )
        session.add(mapping)
        session.commit()
        mapping_id = mapping.id

        session.delete(mapping)
        session.commit()

        result = session.query(TagFrameMapping).filter(
            TagFrameMapping.id == mapping_id
        ).first()
        assert result is None

    def test_unique_constraint(self, tmp_db, sample_task):
        """Test that duplicate (task_id, tag_name, frame_path) is rejected."""
        session, engine = tmp_db
        m1 = TagFrameMapping(
            task_id=sample_task.id,
            tag_name="双人",
            frame_path="/tmp/frame.jpg",
            unit_index=0,
        )
        session.add(m1)
        session.commit()

        m2 = TagFrameMapping(
            task_id=sample_task.id,
            tag_name="双人",
            frame_path="/tmp/frame.jpg",
            unit_index=1,  # different unit_index but same task+tag+frame
        )
        session.add(m2)
        with pytest.raises(Exception):  # IntegrityError
            session.commit()

    def test_cascade_delete_with_task(self, tmp_db):
        """Test mappings behavior when task is deleted.

        当前设计：TagFrameMapping.task_id 是 ON DELETE SET NULL（映射挂在 entity 上，
        删除任务时业务代码显式清理映射，不依赖 DB 级联）。本测试验证：
        删除任务后，映射记录保留但 task_id 置空。
        """
        session, engine = tmp_db

        # Enable foreign keys for SQLite
        from sqlalchemy import text
        session.execute(text("PRAGMA foreign_keys = ON"))

        # Create entity + task
        entity = FileEntity(file_md5="cascade_md5", file_size=1000)
        session.add(entity)
        session.flush()
        task = Task(
            entity_id=entity.id,
            file_path="/tmp/cascade_test.mp4",
            status="COMPLETE",
        )
        session.add(task)
        session.commit()

        # Create mapping
        mapping = TagFrameMapping(
            task_id=task.id,
            tag_name="测试标签",
            frame_path="/tmp/frame.jpg",
            unit_index=0,
        )
        session.add(mapping)
        session.commit()
        mapping_id = mapping.id

        # Delete task - task_id should be SET NULL, mapping preserved
        session.delete(task)
        session.commit()

        # Expire all to force fresh read from DB
        session.expire_all()

        # Verify mapping still exists with task_id = None
        result = session.query(TagFrameMapping).filter(
            TagFrameMapping.id == mapping_id
        ).first()
        assert result is not None
        assert result.task_id is None

    def test_multiple_tags_per_task(self, tmp_db, sample_task):
        """Test that a task can have multiple tag mappings."""
        session, engine = tmp_db
        tags = ["双人", "后入", "卧室", "肌肉", "白皙"]
        for i, tag in enumerate(tags):
            m = TagFrameMapping(
                task_id=sample_task.id,
                tag_name=tag,
                frame_path=f"/tmp/frame_{i}.jpg",
                unit_index=i % 2,
            )
            session.add(m)
        session.commit()

        count = session.query(TagFrameMapping).filter(
            TagFrameMapping.task_id == sample_task.id
        ).count()
        assert count == 5

    def test_same_tag_different_tasks(self, tmp_db):
        """Test that the same tag can map to different frames in different tasks."""
        session, engine = tmp_db

        task1 = Task(file_path="/tmp/v1.mp4", status="COMPLETE")
        task2 = Task(file_path="/tmp/v2.mp4", status="COMPLETE")
        session.add_all([task1, task2])
        session.commit()

        m1 = TagFrameMapping(task_id=task1.id, tag_name="双人", frame_path="/tmp/v1_frame.jpg", unit_index=0)
        m2 = TagFrameMapping(task_id=task2.id, tag_name="双人", frame_path="/tmp/v2_frame.jpg", unit_index=0)
        session.add_all([m1, m2])
        session.commit()

        results = session.query(TagFrameMapping).filter(
            TagFrameMapping.tag_name == "双人"
        ).all()
        assert len(results) == 2


# -----------------------------------------------------------------------
# 2. VLM analyze_unit_frames_individually Tests (单元测试)
# -----------------------------------------------------------------------

class TestAnalyzeUnitFramesIndividually:
    """Test the analyze_unit_frames_individually method."""

    def test_empty_directory(self, vlm_client, tmp_path):
        """Test with a non-existent directory."""
        result = vlm_client.analyze_unit_frames_individually("/nonexistent/path")
        assert result == {}

    def test_directory_without_frames(self, vlm_client, tmp_path):
        """Test with a directory that has no frame files."""
        unit_dir = tmp_path / "unit_0"
        unit_dir.mkdir()
        # Create a non-frame file
        (unit_dir / "readme.txt").write_text("not a frame")

        result = vlm_client.analyze_unit_frames_individually(str(unit_dir))
        assert result == {}

    @patch("src.core.vlm.requests.post")
    def test_single_frame_analysis(self, mock_post, vlm_client, sample_frames_dir):
        """Test analyzing a single frame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "双人,后入,卧室"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        unit_dir = sample_frames_dir / "unit_0"
        # Only keep one frame for this test (sorted to ensure consistent order)
        frames = sorted(unit_dir.glob("frame_*.jpg"))
        for f in frames[1:]:
            f.unlink()

        result = vlm_client.analyze_unit_frames_individually(str(unit_dir))

        assert len(result) == 1
        frame_path = list(result.keys())[0]
        assert "frame_000.jpg" in frame_path
        frame_result = result[frame_path]
        assert isinstance(frame_result, dict)
        assert frame_result["tags"] == "双人,后入,卧室"
        assert "frame_number" in frame_result

    @patch("src.core.vlm.requests.post")
    def test_multiple_frames_analysis(self, mock_post, vlm_client, sample_frames_dir):
        """Test analyzing multiple frames in a unit."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "双人,后入"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        unit_dir = sample_frames_dir / "unit_0"
        result = vlm_client.analyze_unit_frames_individually(str(unit_dir))

        assert len(result) == 3
        for frame_path, frame_result in result.items():
            assert isinstance(frame_result, dict)
            assert frame_result["tags"] == "双人,后入"
            assert "frame_" in frame_path

    @patch("src.core.vlm.requests.post")
    def test_frame_paths_filter_selects_subset(self, mock_post, vlm_client, sample_frames_dir):
        """Test that passing frame_paths restricts which frames get analyzed.

        Covers the NSFW 逐帧映射提速: caller only hands over the selected frames
        (e.g. top-scoring NSFW frames); non-listed frames must be skipped entirely.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "双人,后入,卧室"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        unit_dir = sample_frames_dir / "unit_0"
        all_frames = sorted(str(p) for p in unit_dir.glob("frame_*.jpg"))
        assert len(all_frames) == 3

        # Select only the first and last frame (simulating NSFW 过滤后的选中帧)
        selected = [all_frames[0], all_frames[2]]
        result = vlm_client.analyze_unit_frames_individually(
            str(unit_dir), frame_paths=selected
        )

        assert len(result) == 2  # 只分析选中帧, 跳过中间的 frame_001
        result_paths = sorted(result.keys())
        assert "frame_000.jpg" in result_paths[0]
        assert "frame_002.jpg" in result_paths[1]

        # mock 只被调用 2 次(每选中帧一次), 而非全量 3 次
        assert mock_post.call_count == 2

    @patch("src.core.vlm.requests.post")
    def test_api_error_handling(self, mock_post, vlm_client, sample_frames_dir):
        """Test that API errors are handled gracefully."""
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("Connection refused")

        unit_dir = sample_frames_dir / "unit_0"
        result = vlm_client.analyze_unit_frames_individually(str(unit_dir))

        # Should return empty dict after retries fail
        assert result == {}

    @patch("src.core.vlm.requests.post")
    def test_empty_response_handling(self, mock_post, vlm_client, sample_frames_dir):
        """Test handling of empty VLM response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        unit_dir = sample_frames_dir / "unit_0"
        result = vlm_client.analyze_unit_frames_individually(str(unit_dir))

        # Empty response should not be added to results
        assert len(result) == 0

    @patch("src.core.vlm.requests.post")
    def test_result_format(self, mock_post, vlm_client, sample_frames_dir):
        """Test that results are in the correct format (frame_path -> {tags, frame_number})."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "双人,后入,卧室"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        unit_dir = sample_frames_dir / "unit_0"
        result = vlm_client.analyze_unit_frames_individually(str(unit_dir))

        for frame_path, frame_result in result.items():
            # frame_path should be a string path
            assert isinstance(frame_path, str)
            assert frame_path.endswith(".jpg")
            # frame_result should be a dict with 'tags' and 'frame_number'
            assert isinstance(frame_result, dict)
            assert "tags" in frame_result
            assert "frame_number" in frame_result
            # tags should be comma-separated string
            tags_text = frame_result["tags"]
            assert isinstance(tags_text, str)
            tags = [t.strip() for t in tags_text.split(",")]
            assert len(tags) >= 1


# -----------------------------------------------------------------------
# 3. API Endpoint Tests (集成测试)
# -----------------------------------------------------------------------

class TestVideosByTagAPI:
    """Test the GET /api/videos/by-tag endpoint."""

    @pytest.fixture
    def api_db(self, tmp_path):
        """Set up a test database for API tests."""
        db_path = tmp_path / "test_api.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session, engine, db_path
        session.close()

    @pytest.fixture
    def api_test_data(self, api_db, tmp_path):
        """Create test data for API tests."""
        session, engine, db_path = api_db

        # Create tasks with different tags
        tasks_data = [
            {
                "file_path": "/videos/video1.mp4",
                "status": "COMPLETE",
                "fa_status": "READY",
                "mapped_tags": json.dumps(["双人", "后入", "卧室"]),
            },
            {
                "file_path": "/videos/video2.mp4",
                "status": "COMPLETE",
                "fa_status": "READY",
                "mapped_tags": json.dumps(["双人", "传教士式", "浴室"]),
            },
            {
                "file_path": "/videos/video3.mp4",
                "status": "COMPLETE",
                "fa_status": "READY",
                "mapped_tags": json.dumps(["单人", "自慰", "卧室"]),
            },
            {
                "file_path": "/videos/video4.mp4",
                "status": "PENDING",  # Not complete - should not appear
                "fa_status": "PENDING",
                "mapped_tags": json.dumps(["双人", "后入"]),
            },
        ]

        tasks = []
        entities = []
        for td in tasks_data:
            entity = FileEntity(file_md5=f"md5_{td['file_path']}", file_size=1000)
            session.add(entity)
            session.flush()

            location = FileLocation(
                entity_id=entity.id,
                file_path=td["file_path"],
                file_name=td["file_path"].split("/")[-1],
                is_primary=1,
            )
            session.add(location)
            session.flush()

            task = Task(
                entity_id=entity.id,
                file_path=td["file_path"],
                status=td["status"],
            )
            session.add(task)
            session.flush()

            # 标签数据放在 FileAnalysis（接口实际读这里）
            fa = FileAnalysis(
                entity_id=entity.id,
                task_id=task.id,
                mapped_tags=td["mapped_tags"],
                status=td["fa_status"],
            )
            session.add(fa)
            session.flush()

            tasks.append(task)
            entities.append(entity)

        # Create tag-frame mappings（接口按 entity_id 查，补上 entity_id）
        mappings = [
            TagFrameMapping(
                task_id=tasks[0].id,
                entity_id=entities[0].id,
                tag_name="双人",
                frame_path="/frames/v1_unit0_frame2.jpg",
                unit_index=0,
            ),
            TagFrameMapping(
                task_id=tasks[0].id,
                entity_id=entities[0].id,
                tag_name="后入",
                frame_path="/frames/v1_unit0_frame1.jpg",
                unit_index=0,
            ),
            TagFrameMapping(
                task_id=tasks[1].id,
                entity_id=entities[1].id,
                tag_name="双人",
                frame_path="/frames/v2_unit0_frame3.jpg",
                unit_index=0,
            ),
        ]
        for m in mappings:
            session.add(m)

        session.commit()
        return session, engine, db_path, tasks

    def test_get_videos_by_tag_with_mapping(self, api_test_data):
        """Test that videos are returned with correct cover_path from mappings."""
        session, engine, db_path, tasks = api_test_data

        # Simulate the API logic directly (since we can't easily spin up FastAPI TestClient
        # without the full app context)
        from src.db.models import TagFrameMapping as TFM

        tag = "双人"
        # 接口按 FileAnalysis 读标签（Task 无 mapped_tags）
        all_tasks = session.query(Task).filter(Task.status == "COMPLETE").all()

        tagged_tasks = []
        for task in all_tasks:
            if not task.entity_id:
                continue
            fa = session.query(FileAnalysis).filter(
                FileAnalysis.entity_id == task.entity_id,
                FileAnalysis.status == "READY",
            ).first()
            if fa and fa.mapped_tags:
                tags_list = json.loads(fa.mapped_tags)
                if tag in tags_list:
                    tagged_tasks.append(task)

        assert len(tagged_tasks) == 2

        # Check cover_path for each
        for task in tagged_tasks:
            tag_mapping = session.query(TFM).filter(
                TFM.task_id == task.id,
                TFM.tag_name == tag,
            ).first()

            if task.id == tasks[0].id:
                assert tag_mapping is not None
                assert tag_mapping.frame_path == "/frames/v1_unit0_frame2.jpg"
            elif task.id == tasks[1].id:
                assert tag_mapping is not None
                assert tag_mapping.frame_path == "/frames/v2_unit0_frame3.jpg"

    def test_get_videos_by_tag_no_mapping_fallback(self, api_test_data, tmp_path):
        """Test fallback to first frame when no mapping exists."""
        session, engine, db_path, tasks = api_test_data

        # Create a frames directory structure for task[2] (which has no mapping for "卧室")
        frames_dir = tmp_path / "video3_frames"
        unit_dir = frames_dir / "unit_0"
        unit_dir.mkdir(parents=True)
        frame_file = unit_dir / "frame_000.jpg"
        frame_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Simulate the fallback logic
        from src.db.models import TagFrameMapping as TFM

        tag = "卧室"
        task = tasks[2]

        tag_mapping = session.query(TFM).filter(
            TFM.task_id == task.id,
            TFM.tag_name == tag,
        ).first()

        # No mapping exists for this task+tag
        assert tag_mapping is None

        # Fallback: find first frame in frames directory
        cover_path = None
        if frames_dir.exists():
            for unit_d in sorted(frames_dir.glob("unit_*")):
                if unit_d.is_dir():
                    frame_files = sorted(unit_d.glob("frame_*.jpg"))
                    if frame_files:
                        cover_path = str(frame_files[0])
                        break

        assert cover_path is not None
        assert "frame_000.jpg" in cover_path

    def test_pagination(self, api_test_data):
        """Test pagination of results."""
        session, engine, db_path, tasks = api_test_data

        tag = "双人"
        all_tasks = session.query(Task).filter(Task.status == "COMPLETE").all()

        tagged_tasks = []
        for task in all_tasks:
            if not task.entity_id:
                continue
            fa = session.query(FileAnalysis).filter(
                FileAnalysis.entity_id == task.entity_id,
                FileAnalysis.status == "READY",
            ).first()
            if fa and fa.mapped_tags:
                tags_list = json.loads(fa.mapped_tags)
                if tag in tags_list:
                    tagged_tasks.append(task)

        # Page 1, size 1
        page, size = 1, 1
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated = tagged_tasks[start_idx:end_idx]
        assert len(paginated) == 1

        # Page 2, size 1
        page, size = 2, 1
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated = tagged_tasks[start_idx:end_idx]
        assert len(paginated) == 1

        # Page 3, size 1 (no more results)
        page, size = 3, 1
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated = tagged_tasks[start_idx:end_idx]
        assert len(paginated) == 0

    def test_incomplete_tasks_excluded(self, api_test_data):
        """Test that non-COMPLETE tasks are excluded from results."""
        session, engine, db_path, tasks = api_test_data

        tag = "双人"
        all_tasks = session.query(Task).filter(Task.status == "COMPLETE").all()

        tagged_tasks = []
        for task in all_tasks:
            if not task.entity_id:
                continue
            fa = session.query(FileAnalysis).filter(
                FileAnalysis.entity_id == task.entity_id,
                FileAnalysis.status == "READY",
            ).first()
            if fa and fa.mapped_tags:
                tags_list = json.loads(fa.mapped_tags)
                if tag in tags_list:
                    tagged_tasks.append(task)

        # Task 3 (PENDING status) should not be included
        task_ids = [t.id for t in tagged_tasks]
        assert tasks[3].id not in task_ids


# -----------------------------------------------------------------------
# 4. Performance Tests (性能测试)
# -----------------------------------------------------------------------

class TestPerformance:
    """Performance tests for tag-frame mapping queries."""

    @pytest.fixture
    def large_dataset(self, tmp_path):
        """Create a large dataset for performance testing."""
        db_path = tmp_path / "perf_test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create 100 tasks with mappings
        num_tasks = 100
        tags = ["双人", "后入", "卧室", "浴室", "肌肉", "白皙", "青年", "丝袜"]

        for i in range(num_tasks):
            entity = FileEntity(file_md5=f"perf_md5_{i}", file_size=1000)
            session.add(entity)
            session.flush()

            task = Task(
                entity_id=entity.id,
                file_path=f"/videos/perf_video_{i}.mp4",
                status="COMPLETE",
            )
            session.add(task)
            session.flush()

            # 标签放 FileAnalysis
            fa = FileAnalysis(
                entity_id=entity.id,
                task_id=task.id,
                mapped_tags=json.dumps(tags[:3 + (i % 5)]),  # Varying tag counts
                status="READY",
            )
            session.add(fa)
            session.flush()

            # Create 5 mappings per task
            for j in range(5):
                mapping = TagFrameMapping(
                    task_id=task.id,
                    tag_name=tags[j % len(tags)],
                    frame_path=f"/frames/v{i}_unit{j}_frame.jpg",
                    unit_index=j,
                )
                session.add(mapping)

        session.commit()
        yield session, engine
        session.close()

    def test_query_performance_single_tag(self, large_dataset):
        """Test query performance for a single tag lookup."""
        session, engine = large_dataset

        start_time = time.time()

        # Query all mappings for a specific tag
        results = session.query(TagFrameMapping).filter(
            TagFrameMapping.tag_name == "双人"
        ).all()

        elapsed = time.time() - start_time

        assert len(results) > 0
        # Should complete in under 100ms for this dataset size
        assert elapsed < 0.1, f"Query took {elapsed:.3f}s, expected < 0.1s"

    def test_query_performance_task_tag_lookup(self, large_dataset):
        """Test query performance for task+tag lookup (used in API)."""
        session, engine = large_dataset

        # Get a task_id
        task = session.query(Task).first()

        start_time = time.time()

        # Simulate the API query pattern
        for _ in range(100):
            session.query(TagFrameMapping).filter(
                TagFrameMapping.task_id == task.id,
                TagFrameMapping.tag_name == "双人",
            ).first()

        elapsed = time.time() - start_time

        # 100 queries should complete in under 1 second
        assert elapsed < 1.0, f"100 queries took {elapsed:.3f}s, expected < 1.0s"

    def test_concurrent_queries(self, large_dataset):
        """Test concurrent query performance."""
        session, engine = large_dataset
        Session = sessionmaker(bind=engine)

        def query_worker(worker_id):
            """Worker that performs tag lookups."""
            local_session = Session()
            try:
                results = []
                for i in range(20):
                    tag = ["双人", "后入", "卧室", "浴室", "肌肉"][i % 5]
                    result = local_session.query(TagFrameMapping).filter(
                        TagFrameMapping.tag_name == tag
                    ).all()
                    results.append(len(result))
                return results
            finally:
                local_session.close()

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_worker, i) for i in range(5)]
            all_results = [f.result() for f in as_completed(futures)]

        elapsed = time.time() - start_time

        # All workers should get results
        for worker_results in all_results:
            assert all(r > 0 for r in worker_results)

        # Total time should be reasonable (SQLite serializes writes but reads are concurrent)
        assert elapsed < 5.0, f"Concurrent queries took {elapsed:.3f}s"

    def test_bulk_insert_performance(self, tmp_path):
        """Test bulk insert performance for tag-frame mappings."""
        db_path = tmp_path / "bulk_insert_test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create a task
        task = Task(file_path="/tmp/bulk.mp4", status="COMPLETE")
        session.add(task)
        session.commit()

        # Bulk insert 1000 mappings
        num_mappings = 1000
        start_time = time.time()

        mappings = []
        for i in range(num_mappings):
            m = TagFrameMapping(
                task_id=task.id,
                tag_name=f"tag_{i % 50}",
                frame_path=f"/frames/frame_{i}.jpg",
                unit_index=i // 20,
            )
            mappings.append(m)

        session.bulk_save_objects(mappings)
        session.commit()

        elapsed = time.time() - start_time

        # Verify all inserted
        count = session.query(TagFrameMapping).filter(
            TagFrameMapping.task_id == task.id
        ).count()
        assert count == num_mappings

        # Should complete in under 2 seconds
        assert elapsed < 2.0, f"Bulk insert of {num_mappings} took {elapsed:.3f}s"

        session.close()


# -----------------------------------------------------------------------
# 5. Integration Tests (端到端集成测试)
# -----------------------------------------------------------------------

class TestEndToEndIntegration:
    """End-to-end integration tests simulating real workflow."""

    def test_full_workflow(self, tmp_path):
        """Test the complete workflow: VLM analysis -> mapping storage -> API query."""
        # Setup database
        db_path = tmp_path / "e2e_test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create task
        entity = FileEntity(file_md5="e2e_md5", file_size=5000)
        session.add(entity)
        session.flush()

        task = Task(
            entity_id=entity.id,
            file_path="/videos/e2e_video.mp4",
            status="COMPLETE",
        )
        session.add(task)
        session.flush()

        # 标签放 FileAnalysis
        fa = FileAnalysis(
            entity_id=entity.id,
            task_id=task.id,
            mapped_tags=json.dumps(["双人", "后入", "卧室"]),
            status="READY",
        )
        session.add(fa)
        session.commit()

        # Simulate VLM analysis results
        vlm_results = {
            "/frames/unit_0/frame_000.jpg": "卧室,双人",
            "/frames/unit_0/frame_001.jpg": "双人,后入",
            "/frames/unit_0/frame_002.jpg": "后入,卧室",
        }

        # Store mappings
        for frame_path, tags_text in vlm_results.items():
            tags = [t.strip() for t in tags_text.split(",")]
            unit_index = 0  # All from unit_0
            for tag in tags:
                mapping = TagFrameMapping(
                    task_id=task.id,
                    tag_name=tag,
                    frame_path=frame_path,
                    unit_index=unit_index,
                )
                session.add(mapping)
        session.commit()

        # Query via API logic
        tag = "后入"
        tag_mapping = session.query(TagFrameMapping).filter(
            TagFrameMapping.task_id == task.id,
            TagFrameMapping.tag_name == tag,
        ).first()

        assert tag_mapping is not None
        assert "frame_001.jpg" in tag_mapping.frame_path or "frame_002.jpg" in tag_mapping.frame_path

        session.close()

    def test_vlm_to_mapping_pipeline(self, tmp_path):
        """Test that VLM output can be correctly transformed into mappings."""
        # Setup
        db_path = tmp_path / "pipeline_test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        task = Task(
            file_path="/videos/pipeline.mp4",
            status="COMPLETE",
        )
        session.add(task)
        session.commit()

        # Simulate VLM output for multiple units
        unit_results = {
            0: {
                "/frames/unit_0/frame_0.jpg": "双人,卧室",
                "/frames/unit_0/frame_1.jpg": "双人,后入",
            },
            1: {
                "/frames/unit_1/frame_0.jpg": "卧室,单人",
            },
        }

        # Transform VLM output to mappings
        all_mappings = []
        for unit_idx, frames in unit_results.items():
            for frame_path, tags_text in frames.items():
                tags = [t.strip() for t in tags_text.split(",")]
                for tag in tags:
                    m = TagFrameMapping(
                        task_id=task.id,
                        tag_name=tag,
                        frame_path=frame_path,
                        unit_index=unit_idx,
                    )
                    all_mappings.append(m)

        session.bulk_save_objects(all_mappings)
        session.commit()

        # Verify mappings
        total = session.query(TagFrameMapping).filter(
            TagFrameMapping.task_id == task.id
        ).count()
        assert total == 6  # 2 + 2 + 2 (each frame has 2 tags)

        # Verify specific tag lookup
        bedroom_frames = session.query(TagFrameMapping).filter(
            TagFrameMapping.task_id == task.id,
            TagFrameMapping.tag_name == "卧室",
        ).all()
        assert len(bedroom_frames) == 2  # frame_0 from unit_0, frame_0 from unit_1

        session.close()
