"""人脸检测测试对话框

用于测试新的人脸检测和识别模块（两阶段检测）。
"""

import requests
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QSpinBox, QGroupBox, QProgressBar, QMessageBox,
)
from PySide6.QtCore import QThread, Signal

from ..components.theme import button_primary_style, button_secondary_style


class FaceDetectionTestWorker(QThread):
    """人脸检测测试工作线程"""
    
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, api_base: str, task_id: int, test_type: str):
        super().__init__()
        self.api_base = api_base
        self.task_id = task_id
        self.test_type = test_type  # "detection" or "recognition"
    
    def run(self):
        try:
            if self.test_type == "detection":
                url = f"{self.api_base}/api/actors/test-new-face-detection/{self.task_id}"
            else:
                url = f"{self.api_base}/api/actors/test-new-face-recognition/{self.task_id}"
            
            response = requests.post(url, timeout=300)  # 5分钟超时
            response.raise_for_status()
            result = response.json()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class FaceDetectionTestDialog(QDialog):
    """人脸检测测试对话框"""
    
    def __init__(self, api_base: str, parent=None):
        super().__init__(parent)
        self.api_base = api_base
        self.worker: Optional[FaceDetectionTestWorker] = None
        
        self.setWindowTitle("人脸检测模块测试")
        self.setMinimumSize(600, 500)
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 任务ID输入
        task_group = QGroupBox("测试参数")
        task_layout = QVBoxLayout(task_group)
        
        task_id_layout = QHBoxLayout()
        task_id_layout.addWidget(QLabel("任务ID:"))
        self.task_id_spin = QSpinBox()
        self.task_id_spin.setMinimum(1)
        self.task_id_spin.setMaximum(999999)
        self.task_id_spin.setValue(1)
        task_id_layout.addWidget(self.task_id_spin)
        task_id_layout.addStretch()
        task_layout.addLayout(task_id_layout)
        
        layout.addWidget(task_group)
        
        # 测试按钮
        button_layout = QHBoxLayout()
        
        self.detect_btn = QPushButton("测试人脸检测")
        self.detect_btn.setStyleSheet(button_primary_style())
        self.detect_btn.clicked.connect(self._on_test_detection)
        button_layout.addWidget(self.detect_btn)
        
        self.recognize_btn = QPushButton("测试人脸识别")
        self.recognize_btn.setStyleSheet(button_primary_style())
        self.recognize_btn.clicked.connect(self._on_test_recognition)
        button_layout.addWidget(self.recognize_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 结果显示
        result_group = QGroupBox("测试结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-family: 'Courier New', monospace; font-size: 12px;")
        result_layout.addWidget(self.result_text)
        
        layout.addWidget(result_group)
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet(button_secondary_style())
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def _on_test_detection(self):
        """测试人脸检测"""
        task_id = self.task_id_spin.value()
        self._start_test(task_id, "detection")
    
    def _on_test_recognition(self):
        """测试人脸识别"""
        task_id = self.task_id_spin.value()
        self._start_test(task_id, "recognition")
    
    def _start_test(self, task_id: int, test_type: str):
        """开始测试"""
        self.result_text.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 无限进度条
        self.detect_btn.setEnabled(False)
        self.recognize_btn.setEnabled(False)
        
        test_name = "人脸检测" if test_type == "detection" else "人脸识别"
        self.result_text.append(f"开始测试{test_name}（任务ID: {task_id}）...\n")
        
        self.worker = FaceDetectionTestWorker(self.api_base, task_id, test_type)
        self.worker.finished.connect(self._on_test_finished)
        self.worker.error.connect(self._on_test_error)
        self.worker.start()
    
    def _on_test_finished(self, result: dict):
        """测试完成"""
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.recognize_btn.setEnabled(True)
        
        if result.get("success"):
            self.result_text.append("✅ 测试成功!\n")
            self.result_text.append(f"消息: {result.get('message', '')}\n")
            
            if "face_segments" in result:
                self.result_text.append("\n人脸片段:")
                for i, seg in enumerate(result["face_segments"]):
                    self.result_text.append(
                        f"  {i+1}. {seg['start_time']:.2f}s - {seg['end_time']:.2f}s "
                        f"(人脸数: {seg['face_count']})"
                    )
            
            if "dense_frame_count" in result:
                self.result_text.append(f"\n密集帧数: {result['dense_frame_count']}")
            
            if "total_faces_detected" in result:
                self.result_text.append(f"检测到的人脸总数: {result['total_faces_detected']}")
            
            if "prescan_frame_count" in result:
                self.result_text.append(f"预扫描帧数: {result['prescan_frame_count']}")
            
            if "embeddings_extracted" in result:
                self.result_text.append(f"\n提取的特征向量数: {result['embeddings_extracted']}")
            
            if "recognition_results" in result:
                self.result_text.append("\n识别结果（前10个）:")
                for i, rec in enumerate(result["recognition_results"]):
                    self.result_text.append(
                        f"  {i+1}. 质量分={rec.get('quality_score', 0):.3f}, "
                        f"det={rec.get('det_score', 0):.3f}, "
                        f"yaw={rec.get('yaw', 0):.1f}°, "
                        f"向量维度={rec.get('embedding_dim', 0)}"
                    )
            
            if "best_cover" in result and result["best_cover"]:
                bc = result["best_cover"]
                self.result_text.append("\n🌟 最佳封面选择:")
                self.result_text.append(f"  文件: {bc.get('face_path', 'N/A')}")
                self.result_text.append(f"  姿态评分: {bc.get('pose_score', 0):.1f}")
                self.result_text.append(f"  综合质量分: {bc.get('quality_score', 0):.4f}")
                self.result_text.append(f"  清晰度: {bc.get('sharpness', 0):.1f}")
                self.result_text.append(f"  亮度: {bc.get('brightness', 0):.1f}")
                self.result_text.append(f"  朝向: yaw={bc.get('yaw', 0):.1f}°, pitch={bc.get('pitch', 0):.1f}°, roll={bc.get('roll', 0):.1f}°")
        else:
            self.result_text.append("❌ 测试失败!\n")
            self.result_text.append(f"消息: {result.get('message', '未知错误')}\n")
    
    def _on_test_error(self, error_msg: str):
        """测试出错"""
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.recognize_btn.setEnabled(True)
        
        self.result_text.append("❌ 测试出错!\n")
        self.result_text.append(f"错误: {error_msg}\n")
        
        QMessageBox.critical(self, "测试失败", f"测试出错:\n{error_msg}")
