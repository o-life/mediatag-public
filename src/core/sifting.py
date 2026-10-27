import subprocess
import json
from pathlib import Path

class SiftingPipeline:
    def __init__(self, output_dir='data/frames'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_path: str, num_frames=15):
        """
        Extracts frames based on a hybrid Montage Approach:
        Prioritizes distributed sampling across total duration to ensure plot integrity,
        with an internal structure ready for scene-boundary refinement.
        
        Returns:
            str: frames_folder path
        """
        video_file = Path(video_path)
        # Ensure we handle special characters in filenames by using stem properly
        folder_name = f"{video_file.stem}_frames"
        frames_folder = self.output_dir / folder_name
        frames_folder.mkdir(parents=True, exist_ok=True)

        duration = self._get_duration(video_path)
        if not duration:
            return None

        # 获取视频帧率
        fps = self._get_fps(video_path)
        if not fps:
            fps = 30.0  # 默认帧率

        # Adaptive distributed sampling (Montage Baseline)
        interval = duration / num_frames
        frame_mappings = {}  # 记录帧号映射：frame_path -> {"video_frame_number": int, "timestamp": float}
        
        for i in range(num_frames):
            timestamp = i * interval
            frame_path = frames_folder / f"frame_{i:03d}.jpg"
            
            # 计算对应的视频帧号
            video_frame_number = int(timestamp * fps)
            frame_mappings[str(frame_path)] = {
                "video_frame_number": video_frame_number,
                "timestamp": timestamp
            }
            
            cmd = [
                'ffmpeg', '-y', 
                '-ss', str(timestamp), 
                '-i', video_path, 
                '-vframes', '1', 
                '-q:v', '2', 
                str(frame_path)
            ]
            subprocess.run(cmd, capture_output=True, check=False)

        # 保存帧号映射到 JSON 文件
        mappings_file = frames_folder / "frame_mappings.json"
        with open(mappings_file, 'w', encoding='utf-8') as f:
            json.dump(frame_mappings, f, ensure_ascii=False, indent=2)

        return str(frames_folder)

    def _get_duration(self, video_path):
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting duration for {video_path}: {e}")
            return None

    def _get_fps(self, video_path):
        """获取视频帧率"""
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                   '-show_entries', 'stream=r_frame_rate', 
                   '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # 解析帧率，格式可能是 "30/1" 或 "30000/1001"
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                return float(num) / float(den)
            return float(fps_str)
        except Exception as e:
            print(f"Error getting fps for {video_path}: {e}")
            return None

if __name__ == "__main__":
    pipeline = SiftingPipeline()
    print("SiftingPipeline (Montage-ready) initialized.")
