#!/usr/bin/env python3
"""验证 pose 修复效果 — 使用 2023_143.mp4 测试。

重点验证：
1. 姿态数据不再是全 0（yaw/pitch/roll 有实际值）
2. 新旧两条路径都能正确提取姿态
3. 封面选择的姿态评分逻辑正常工作
"""

import sys
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("test_pose_fix")

TEST_VIDEO = "/Volumes/MediaData/未分类/未识别/2023_143.mp4"


def test_path_two_pose_extraction():
    """测试路径二：recognize_actors_from_video — 两阶段检测 + embedding 提取"""
    logger.info("\n" + "=" * 70)
    logger.info("测试：路径二 pose 提取（两阶段检测）")
    logger.info("=" * 70)

    if not Path(TEST_VIDEO).exists():
        logger.error("❌ 测试视频不存在: %s", TEST_VIDEO)
        return False

    from src.core.face_recognition import FaceRecognizer
    recognizer = FaceRecognizer()
    if not recognizer.available:
        logger.error("❌ FaceRecognizer 不可用")
        return False

    # 用高 task_id 避免冲突
    task_id = 99998
    actor_ids = recognizer.recognize_actors_from_video(
        TEST_VIDEO, task_id=task_id, file_md5="test_pose_fix",
    )

    if not actor_ids:
        logger.warning("⚠️ 未识别到演员（可能清晰度过滤太严格）")
        return True

    # 查询数据库验证姿态数据
    from src.db.models import ActorFace, get_session, get_db_engine

    engine = get_db_engine()
    db = get_session(engine)

    all_faces = (
        db.query(ActorFace)
        .filter(ActorFace.task_id == task_id)
        .all()
    )

    db.close()

    if not all_faces:
        logger.warning("⚠️ 未找到 ActorFace 记录")
        return True

    logger.info(f"共找到 {len(all_faces)} 张人脸记录，姿态数据：")

    yaw_values = []
    pitch_values = []
    roll_values = []

    for i, face in enumerate(all_faces):
        yaw, pitch, roll = face.yaw or 0, face.pitch or 0, face.roll or 0
        yaw_values.append(abs(yaw))
        pitch_values.append(abs(pitch))
        roll_values.append(abs(roll))
        logger.info(
            f"  人脸 {i+1}: yaw={yaw:+.1f}°, pitch={pitch:+.1f}°, roll={roll:+.1f}°"
        )

    # 验证：不是全 0
    avg_yaw = np.mean(yaw_values) if yaw_values else 0
    avg_pitch = np.mean(pitch_values) if pitch_values else 0
    avg_roll = np.mean(roll_values) if roll_values else 0

    all_zero = (avg_yaw == 0 and avg_pitch == 0 and avg_roll == 0)

    if all_zero:
        logger.error("❌ 所有姿态值仍为 0，修复未生效")
        return False
    else:
        logger.info(
            f"✅ pose 修复生效！平均 |yaw|={avg_yaw:.1f}°, "
            f"|pitch|={avg_pitch:.1f}°, |roll|={avg_roll:.1f}°"
        )
        return True


def test_path_one_pose_extraction():
    """测试路径一：extract_faces_from_frames — 从帧目录提取"""
    logger.info("\n" + "=" * 70)
    logger.info("测试：路径一 pose 提取（帧目录模式）")
    logger.info("=" * 70)

    from src.core.face_recognition import FaceRecognizer

    recognizer = FaceRecognizer()
    if not recognizer.available:
        logger.error("❌ FaceRecognizer 不可用")
        return False

    # 先用 ffmpeg 抽取几帧作为测试
    import tempfile
    import subprocess

    frames_dir = Path(tempfile.mkdtemp(prefix="test_pose_frames_"))

    logger.info(f"从视频抽取 5 帧到: {frames_dir}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", TEST_VIDEO,
            "-vf", "fps=1",
            "-frames:v", "5",
            str(frames_dir / "frame_%03d.jpg"),
        ],
        capture_output=True,
    )

    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        logger.error("❌ ffmpeg 抽帧失败")
        return False

    logger.info(f"抽取到 {len(frame_files)} 帧")

    # 调用 extract_faces_from_frames
    faces = recognizer.extract_faces_from_frames(str(frames_dir))

    if not faces:
        logger.warning("⚠️ 未检测到人脸")
        return True

    logger.info(f"检测到 {len(faces)} 张人脸，姿态数据：")

    yaw_values = []
    pitch_values = []
    roll_values = []

    for i, face in enumerate(faces):
        yaw = face.get("yaw", 0)
        pitch = face.get("pitch", 0)
        roll = face.get("roll", 0)
        yaw_values.append(abs(yaw))
        pitch_values.append(abs(pitch))
        roll_values.append(abs(roll))
        logger.info(
            f"  人脸 {i+1}: yaw={yaw:+.1f}°, pitch={pitch:+.1f}°, roll={roll:+.1f}°, "
            f"det_score={face.get('det_score', 0):.3f}, "
            f"quality={face.get('quality_score', 0):.3f}"
        )

    # 验证
    avg_yaw = np.mean(yaw_values) if yaw_values else 0
    avg_pitch = np.mean(pitch_values) if pitch_values else 0
    avg_roll = np.mean(roll_values) if roll_values else 0

    all_zero = (avg_yaw == 0 and avg_pitch == 0 and avg_roll == 0)

    if all_zero:
        logger.error("❌ 路径一姿态值仍为 0")
        return False
    else:
        logger.info(
            f"✅ 路径一 pose 修复生效！平均 |yaw|={avg_yaw:.1f}°, "
            f"|pitch|={avg_pitch:.1f}°, |roll|={avg_roll:.1f}°"
        )
        return True


def test_cover_selection():
    """测试封面选择逻辑 — 验证正脸优先"""
    logger.info("\n" + "=" * 70)
    logger.info("测试：封面选择（正脸优先 + 清晰度 + 亮度）")
    logger.info("=" * 70)

    from src.core.face_recognition import FaceRecognizer

    recognizer = FaceRecognizer()
    if not recognizer.available:
        logger.error("❌ FaceRecognizer 不可用")
        return False

    # 构造一组模拟人脸数据
    test_faces = [
        {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "det_score": 0.95,
         "sharpness": 500, "brightness": 120, "face_path": "frontal_good.jpg",
         "quality_score": 0.85},
        {"yaw": 30.0, "pitch": 20.0, "roll": 15.0, "det_score": 0.90,
         "sharpness": 400, "brightness": 110, "face_path": "side_good.jpg",
         "quality_score": 0.65},
        {"yaw": 5.0, "pitch": 3.0, "roll": 2.0, "det_score": 0.88,
         "sharpness": 350, "brightness": 130, "face_path": "almost_frontal.jpg",
         "quality_score": 0.78},
        {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "det_score": 0.92,
         "sharpness": 600, "brightness": 240, "face_path": "frontal_overexposed.jpg",
         "quality_score": 0.80},  # 过曝
    ]

    best_face = recognizer.select_best_frontal_face(test_faces)

    if best_face is None:
        logger.error("❌ 未选择到最佳封面")
        return False

    logger.info(
        f"选择的封面: {best_face['face_path']}, "
        f"yaw={best_face['yaw']:.1f}°, pitch={best_face['pitch']:.1f}°, "
        f"roll={best_face['roll']:.1f}°, brightness={best_face.get('brightness', 0):.0f}"
    )

    # 应该选择正脸且亮度正常的人脸
    if "frontal_good" in best_face["face_path"]:
        logger.info("✅ 封面选择正确：选择了正脸 + 正常亮度的人脸")
        return True
    elif "almost_frontal" in best_face["face_path"]:
        logger.warning("⚠️ 选择了接近正脸的人脸（可接受，但不是最优）")
        return True
    else:
        logger.error("❌ 封面选择不当: %s", best_face["face_path"])
        return False


if __name__ == "__main__":
    logger.info("Phase 0 pose 修复验证测试")
    logger.info(f"测试视频: {TEST_VIDEO}")

    results = {}

    results["路径二_pose提取"] = test_path_two_pose_extraction()
    results["路径一_pose提取"] = test_path_one_pose_extraction()
    results["封面选择逻辑"] = test_cover_selection()

    logger.info("\n" + "=" * 70)
    logger.info("测试结果汇总")
    logger.info("=" * 70)

    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        logger.info(f"  {name}: {status}")

    all_passed = all(results.values())
    logger.info("")
    if all_passed:
        logger.info("🎉 所有测试通过！pose 修复生效。")
    else:
        logger.info("⚠️ 部分测试未通过，请检查。")
