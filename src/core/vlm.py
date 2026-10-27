import os
import sys

import requests
import base64
import time
import functools
from typing import List, Optional, Callable, Any, Dict
from pathlib import Path
import logging
from src.core.config import ConfigManager
from src.db.models import get_session, get_db_engine, DictionaryEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    backoff_seconds: List[float] = [1.0, 3.0, 5.0],
    retryable_exceptions: tuple = (requests.exceptions.RequestException, ConnectionError, TimeoutError)
):
    """Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_seconds: List of delay seconds for each retry.
                         If fewer delays than retries, last delay is reused.
        retryable_exceptions: Tuple of exception types that trigger a retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay_idx = min(attempt, len(backoff_seconds) - 1)
                        delay = backoff_seconds[delay_idx]
                        logger.warning(
                            f"[Retry {attempt + 1}/{max_retries}] "
                            f"{func.__name__} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[Retry {max_retries}/{max_retries}] "
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


class OllamaVLMClient:
    def __init__(self, config: Optional[ConfigManager] = None, model: str = "minicpm-v4.5", base_url: str = "http://localhost:11434"):
        self.config = config  # Store config reference for prompt retrieval
        if config:
            vlm_config = config.get_vlm_config()
            self.model = vlm_config.get('model', 'minicpm-v4.5')
            self.base_url = vlm_config.get('base_url', 'http://localhost:11434')
            self.timeout = vlm_config.get('timeout', 60)
        else:
            self.model = model
            self.base_url = base_url
            self.timeout = 60
        # 协议自动识别：Ollama 原生（/api/generate）或 OpenAI 兼容（/v1/chat/completions）
        # Ollama 也支持 OpenAI 兼容接口，云端 API 通常只支持 OpenAI 兼容。
        # 探测顺序：OpenAI 兼容优先（云端/新版 Ollama 通用），失败则回退 Ollama 原生。
        self.base_url = (self.base_url or "http://localhost:11434").rstrip("/")
        self._endpoint_mode = self._detect_endpoint()
        self.api_url = (
            f"{self.base_url}/v1/chat/completions"
            if self._endpoint_mode == "openai"
            else f"{self.base_url}/api/generate"
        )

    def _detect_endpoint(self) -> str:
        """探测该 URL 支持哪种接口：'openai'（OpenAI 兼容）或 'ollama'（原生）。"""
        # 先试 OpenAI 兼容接口（Ollama 新版也支持 /v1/chat/completions）
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                logger.info("VLM endpoint %s -> OpenAI 兼容接口 (/v1/chat/completions)", self.base_url)
                return "openai"
        except Exception:
            pass
        # 回退：Ollama 原生接口
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": "ping", "stream": False},
                timeout=8,
            )
            if resp.status_code == 200:
                logger.info("VLM endpoint %s -> Ollama 原生接口 (/api/generate)", self.base_url)
                return "ollama"
        except Exception:
            pass
        logger.warning("VLM endpoint %s 两种接口探测均失败，默认按 OpenAI 兼容处理", self.base_url)
        return "openai"

    @retry_with_backoff(max_retries=6, backoff_seconds=[5, 15, 30, 60, 60, 60])
    def _send_request(self, prompt: str, images_b64: list) -> dict:
        """统一发送请求，自动适配接口协议。

        Returns:
            原始响应 dict（openai 模式: {content, usage}；ollama 模式: {response, prompt_eval_count}）

        重试策略：连接失败/超时按上表退避，总覆盖 ≈4 分钟——内嵌推理
        （embedded_inference）崩溃自愈需模型重载，实测 40-90s，且用户机上
        内存清理类工具会周期性 SIGKILL 大内存进程，请求必须能穿越不可用窗口。
        """
        if self._endpoint_mode == "openai":
            # OpenAI 兼容格式：图片作为 content 中的 image_url（data URI）
            content_parts = [{"type": "text", "text": prompt}]
            for img_b64 in images_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content_parts}],
                "max_tokens": 2000,
                "stream": False,
            }
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            content = ""
            try:
                content = result["choices"][0]["message"]["content"] or ""
            except Exception:
                pass
            usage = result.get("usage", {}) or {}
            return {"content": content, "prompt_eval_count": usage.get("prompt_tokens", 0)}
        else:
            # Ollama 原生格式
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": images_b64,
                "stream": False,
            }
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return {
                "content": result.get("response", "").strip(),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
            }

    def _build_dynamic_prompt(self, is_single_frame: bool = False) -> str:
        """从标准标签库动态构建 prompt
        
        Args:
            is_single_frame: 是否为单帧分析模式
        
        Returns:
            包含标准标签库的完整 prompt
        """
        # 从数据库读取标准标签库
        try:
            engine = get_db_engine()
            db = get_session(engine)
            
            # 按分类分组读取标签
            entries = db.query(DictionaryEntry).order_by(
                DictionaryEntry.category, 
                DictionaryEntry.standard_tag
            ).all()
            
            db.close()
            
            # 构建分类标签列表
            categories = {}
            for entry in entries:
                category = entry.category or "其他"
                if category not in categories:
                    categories[category] = []
                categories[category].append(entry.standard_tag)
            
            # 格式化标签列表
            standard_tags_text = ""
            for category, tags in categories.items():
                standard_tags_text += f"{category}：{', '.join(tags)}\n\n"
            
        except Exception as e:
            logger.warning(f"无法从数据库读取标准标签库，使用默认标签: {e}")
            # 使用默认标签作为 fallback
            standard_tags_text = (
                "场景：卧室, 客厅, 浴室, 酒店, 办公室, 更衣室, 健身房, 按摩店, 公园, 街道, 车内, 野外\n\n"
                "体位/动作：口交, 深喉, 69, 后入, 传教士式, 骑乘, 双龙, 侧卧, 站立抱起, 抚摸, 自慰, 互撸, 颜射, 内射\n\n"
                "人数：单人, 双人, 三人, 多人\n\n"
                "年龄/性向：青年, 少年感, 熟男, 中年, Gay, 双性恋\n\n"
                "身材：纤细, 肌肉, 微胖, 熊系\n\n"
                "颜值：精致, 清纯, 刚毅, 粗犷\n\n"
                "视角：器官特写, 第一视角\n\n"
                "肤色：白皙, 古铜, 黝黑\n\n"
                "服装：西装, 制服, 全裸, 半裸\n\n"
            )
        
        # 构建完整 prompt
        end_phrase = "现在请分析这一帧：" if is_single_frame else "现在请分析这些帧："

        prompt = (
            "你是一个成人内容分类专家。请分析视频帧，输出与性活动相关的标签。\n\n"
            "【标准标签——唯一允许的输出词表】\n\n"
            f"{standard_tags_text}"
            "【严格禁止输出的内容】\n"
            "- 词表之外的任何标签(禁止自创、禁止同义改写)\n"
            "- 通用描述词：明显、直接、处于、聚焦、所有、可以、能够、正在\n"
            "- 抽象概念：场景、画面、内容、情况、状态、问题、结果\n"
            "- 连接词和副词：和、与、或、很、非常、也、都\n"
            "- 无意义的动词：看到、出现、展示、包含、呈现\n"
            "- 视频技术描述：慢动作、快进、回放、特写镜头、远景、近景\n"
            "- 无意义的分类词：人群、为主、居多、偏多、偏少\n\n"
            "【输出规则】\n"
            "1. 只能从上方标准标签中逐字选择；若画面内容不在词表中,直接跳过\n"
            "2. 每个标签2-4个汉字\n"
            "3. 用逗号分隔，不要输出其他任何文字\n\n"
            "【输出示例】\n"
            "双人,后入,卧室,肌肉,白皙,青年\n\n"
            f"{end_phrase}"
        )

        return prompt

    # ── 闸 1: 闭式词表注入(2026-09-30 批3) ────────────────────────────
    # 失守点:主题提示词完全替代动态词表且只含 6 个示例词("卧室、客厅…等"),
    # VLM 实际自由发挥出 31,978 种标签。主题 md 是用户策展内容不直接改,
    # 在此程序化追加完整词典与闭集指令。
    _vocab_cache: dict = {"ts": 0.0, "block": ""}

    @staticmethod
    def _load_vocabulary_block(ttl_seconds: float = 60.0) -> str:
        """读词典生成闭式词表块,60s 缓存(照片循环等重复调用避免查库)。"""
        import time
        now = time.monotonic()
        cache = OllamaVLMClient._vocab_cache
        if cache["block"] and now - cache["ts"] < ttl_seconds:
            return cache["block"]
        block = ""
        try:
            engine = get_db_engine()
            db = get_session(engine)
            entries = db.query(DictionaryEntry).order_by(
                DictionaryEntry.category, DictionaryEntry.standard_tag).all()
            db.close()
            categories: dict = {}
            for entry in entries:
                categories.setdefault(entry.category or "其他", []).append(entry.standard_tag)
            listing = "".join(
                f"{cat}：{', '.join(tags)}\n" for cat, tags in categories.items())
            if listing:
                block = (
                    "【词表强制约束(优先级高于本提示词的任何其他条款)】\n"
                    "下面是本次唯一允许输出的标准标签清单：\n\n"
                    f"{listing}\n"
                    "只能逐字输出清单中的标签；禁止自创、禁止同义改写、"
                    "禁止输出清单外的任何词。若画面内容不在清单中,直接跳过。\n"
                )
        except Exception as e:
            logger.warning(f"闭式词表块生成失败,本次不追加: {e}")
        cache["ts"] = now
        cache["block"] = block
        return block

    @staticmethod
    def append_vocabulary_constraint(prompt: str) -> str:
        """在(主题)提示词后追加闭式词表块;词典不可用时原样返回。

        解析边界的 enforce_vocabulary(闸 2)始终兜底,此处失守只损失
        token 而不损失数据质量。
        """
        if not prompt:
            return prompt
        block = OllamaVLMClient._load_vocabulary_block()
        if not block:
            return prompt
        return f"{prompt.rstrip()}\n\n{block}"

    def _image_to_base64(self, image_path: Path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_frames_once(self, frame_paths: List[str], prompt: str) -> str:
        """对给定帧做"一次调用"(不分批), 返回文本。

        用于合并分析模式: 每个场景选好的帧(≤N)一次性送入, 只调用 VLM 一次。
        与 _send_request 同协议(自动适配 openai / ollama)。
        """
        if not frame_paths:
            return ""
        try:
            images_b64 = [self._image_to_base64(Path(p)) for p in frame_paths]
            result = self._send_request(prompt, images_b64)
            return (result.get("content") or "").strip()
        except Exception as e:
            logger.error(f"analyze_frames_once failed: {e}")
            return ""

    def analyze_frames(self, frames_path: str, prompt: str) -> str:
        """
        Sends frames to the VLM with truly adaptive batching.

        Two-phase strategy:
        1. Warmup: send 3 frames, measure actual prompt_eval_count from Ollama
        2. Adaptive: calculate optimal batch size from real token data

        This solves the resolution variance problem:
        - High-res frames → more tokens → smaller batch (no overflow)
        - Low-res frames → fewer tokens → larger batch (full utilization)
        """
        frames_folder = Path(frames_path)
        image_files = sorted(list(frames_folder.rglob("*.jpg")))
        if not image_files:
            logger.error(f"No images found in {frames_path}")
            return ""

        # Phase 1: Warmup — measure actual token usage from API response
        warmup_count = min(3, len(image_files))
        warmup_batch = image_files[:warmup_count]
        prompt_text_tokens = len(prompt) // 4  # rough estimate for prompt text only

        logger.info(
            f"Warmup: sending {warmup_count} frames to measure actual token usage... "
            f"(prompt text ~{prompt_text_tokens} tokens)"
        )
        warmup_b64 = [self._image_to_base64(img) for img in warmup_batch]

        warmup_result, warmup_prompt_tokens = self._send_batch_with_token_count(warmup_b64, prompt)

        if warmup_result and warmup_prompt_tokens > 0:
            # Calculate actual tokens per image from real API data
            image_tokens_total = warmup_prompt_tokens - prompt_text_tokens
            tokens_per_image = max(image_tokens_total // warmup_count, 100)  # minimum 100 safety floor

            logger.info(
                f"Warmup complete: {warmup_prompt_tokens} prompt tokens total, "
                f"~{prompt_text_tokens} text + ~{image_tokens_total} image "
                f"= ~{tokens_per_image} tokens/image"
            )
        else:
            # Fallback: conservative estimate (typical for MiniCPM-V)
            tokens_per_image = 1500
            logger.warning(
                f"Could not measure actual token usage (got {warmup_prompt_tokens} tokens), "
                f"using fallback: {tokens_per_image} tokens/image"
            )

        # Phase 2: Calculate optimal batch size from real data
        # 上下文口径=每槽(llama-server 按 slots 均分总 ctx):从 /props 取真实值,
        # 硬编码 35000 是 MiniCPM 时代假设——MID 档每槽仅 16384,14 帧批必 400
        # (2026-10-14 0.2 实测)。取不到时回退旧值(Ollama/云端路径)。
        max_context_tokens = 35000
        try:
            props = requests.get(f"{self.base_url}/props", timeout=5).json()
            n_ctx = ((props.get("default_generation_settings") or {}).get("n_ctx")
                     or (props.get("model") or {}).get("n_ctx_train"))
            if n_ctx and int(n_ctx) >= 4096:
                max_context_tokens = int(n_ctx)
                logger.info(f"每槽上下文(来自 /props): {max_context_tokens}")
        except Exception as exc:  # noqa: BLE001 - 探测失败走保守回退
            logger.warning(f"读取 /props 失败,按回退上下文 {max_context_tokens} 计算批大小: {exc}")
        reserved_tokens = 5000
        available_tokens = max_context_tokens - reserved_tokens
        optimal_batch_size = max(1, (available_tokens - prompt_text_tokens) // tokens_per_image)
        optimal_batch_size = min(20, max(3, optimal_batch_size))

        total_frames = len(image_files)
        estimated_batches = (total_frames - warmup_count + optimal_batch_size - 1) // optimal_batch_size + 1
        logger.info(
            f"Optimal batch size: {optimal_batch_size} frames "
            f"(total={total_frames}, estimated {estimated_batches} batches, "
            f"tokens_per_image={tokens_per_image})"
        )

        # Phase 3: Process remaining frames at optimal batch size
        all_results = []
        if warmup_result:
            all_results.append(warmup_result)

        remaining = image_files[warmup_count:]

        for i in range(0, len(remaining), optimal_batch_size):
            batch = remaining[i:i + optimal_batch_size]
            batch_num = i // optimal_batch_size + 2  # warmup was batch 1
            total_batches = (len(remaining) + optimal_batch_size - 1) // optimal_batch_size + 1

            batch_b64 = [self._image_to_base64(img) for img in batch]

            try:
                result, _ = self._send_batch_with_token_count(batch_b64, prompt)
                if result:
                    all_results.append(result)
                    logger.info(
                        f"Batch {batch_num}/{total_batches} done: "
                        f"{len(batch)} frames, result length: {len(result)}"
                    )
            except Exception as e:
                logger.error(f"VLM batch {batch_num} failed: {e}")
                # Continue with remaining batches instead of failing entirely

        # Merge all batch results
        if all_results:
            return "\n".join(all_results)
        return ""

    def _send_batch_with_token_count(self, images_b64: list, prompt: str) -> tuple:
        """Send a single batch to VLM and return (response_text, prompt_eval_count).

        统一走 _send_request（自动适配 Ollama 原生 / OpenAI 兼容接口）。
        prompt_eval_count 用于计算最优 batch 大小。
        """
        result = self._send_request(prompt, images_b64)

        return result["content"], result["prompt_eval_count"]

    def get_analysis(self, frames_path: str, theme_name: Optional[str] = None) -> str:
        """
        Default analysis prompt focused on adult content classification.
        Retries automatically via analyze_frames decorator.
        Reads prompt from config if available, otherwise uses default.
        theme_name: 文件所属主题名，用于主题↔提示词联动（有则用该主题 prompt）。
        """
        # Try to get prompt from config（按主题联动）
        prompt = None
        if self.config:
            if theme_name:
                prompt = self.config.get_vlm_prompt_for_theme(theme_name)
            else:
                prompt = self.config.get_vlm_prompt()
        
        # Fallback to dynamic prompt if not configured
        if not prompt:
            prompt = self._build_dynamic_prompt(is_single_frame=False)
        else:
            # 闸 1: 配置/主题提示词缺词表,程序化追加闭式词表块
            prompt = self.append_vocabulary_constraint(prompt)

        try:
            return self.analyze_frames(frames_path, prompt)
        except Exception as e:
            logger.error(f"VLM get_analysis failed after all retries: {e}")
            return ""

    def analyze_scenes_with_progress(
        self,
        frames_base_path: str,
        scene_indices: List[int],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[int, str]:
        """Analyze each scene individually with progress tracking.

        Args:
            frames_base_path: Base directory containing unit_X subdirectories
            scene_indices: List of scene indices to analyze
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            Dict mapping scene_index → VLM analysis text
        """
        results = {}
        total = len(scene_indices)
        base_dir = Path(frames_base_path)

        for i, scene_idx in enumerate(scene_indices):
            scene_dir = base_dir / f"unit_{scene_idx}"
            if not scene_dir.exists():
                logger.warning(f"Scene directory not found: {scene_dir}")
                continue

            # Report progress
            if progress_callback:
                progress_callback(
                    i,
                    total,
                    f"分析中 ({i + 1}/{total} 场景)"
                )

            try:
                analysis = self.get_analysis(str(scene_dir))
                if analysis:
                    results[scene_idx] = analysis
            except Exception as e:
                logger.error(f"Failed to analyze scene {scene_idx}: {e}")
                # Continue with other scenes

        # Final progress update
        if progress_callback:
            progress_callback(total, total, f"分析完成 ({total}/{total} 场景)")

        return results

    def analyze_unit_frames_individually(
        self,
        unit_dir: str,
        prompt: Optional[str] = None,
        frame_paths: Optional[List[str]] = None
    ) -> Dict[str, dict]:
        """分析单个场景的每一帧，返回标签与帧的映射关系。

        Args:
            unit_dir: 场景目录路径（如 /path/to/frames/unit_0）

        Returns:
            Dict mapping frame_path -> {"tags": tags_text, "frame_number": video_frame_number, "timestamp": float}
            例如: {"/path/frame_0.jpg": {"tags": "双人,后入,卧室", "frame_number": 0, "timestamp": 0.0}, ...}
        """
        results = {}
        unit_path = Path(unit_dir)
        
        if not unit_path.exists():
            logger.warning(f"Unit directory not found: {unit_dir}")
            return results

        # 获取该场景的所有帧
        frame_files = sorted(unit_path.glob("frame_*.jpg"))
        # 调用方可指定要分析的帧(如仅 NSFW 帧), 未指定则分析目录内全部帧
        if frame_paths:
            frame_files = [Path(p) for p in frame_paths]
        if not frame_files:
            logger.warning(f"No frames found in {unit_dir}")
            return results

        # 读取帧号映射文件
        frame_mappings = {}
        mappings_file = unit_path.parent / "frame_mappings.json"
        if mappings_file.exists():
            try:
                import json
                with open(mappings_file, 'r', encoding='utf-8') as f:
                    frame_mappings = json.load(f)
                logger.info(f"Loaded frame mappings from {mappings_file}")
            except Exception as e:
                logger.warning(f"Failed to load frame mappings: {e}")

        # 获取 prompt: 优先使用调用方按主题动态传入的提示词
        if not prompt and self.config:
            prompt = self.config.get_vlm_prompt()

        # Fallback to dynamic prompt if not configured
        if not prompt:
            prompt = self._build_dynamic_prompt(is_single_frame=True)
        else:
            # 闸 1: 配置/主题提示词缺词表,程序化追加闭式词表块
            prompt = self.append_vocabulary_constraint(prompt)

        # 逐帧分析
        for frame_file in frame_files:
            try:
                # 将单帧转换为 base64
                image_b64 = self._image_to_base64(frame_file)

                result = self._send_request(prompt, [image_b64])
                tags_text = result["content"].strip()
                
                if tags_text:
                    # 解析 JSON 格式的标签并转换为逗号分隔的字符串
                    import json
                    try:
                        tags_dict = json.loads(tags_text)
                        # 提取所有标签值并合并
                        all_tags = []
                        for key, value in tags_dict.items():
                            if isinstance(value, list):
                                all_tags.extend([tag for tag in value if tag])
                            elif isinstance(value, str) and value:
                                all_tags.append(value)
                        # 去重并转换为逗号分隔的字符串
                        tags_text = ",".join(sorted(set(all_tags)))
                    except json.JSONDecodeError:
                        # 如果不是 JSON，保持原样（已经是逗号分隔格式）
                        pass
                    
                    # 获取帧号和时间戳
                    frame_info = frame_mappings.get(str(frame_file), {})
                    if isinstance(frame_info, dict):
                        frame_number = frame_info.get("video_frame_number", 0)
                        timestamp = frame_info.get("timestamp", 0.0)
                    else:
                        # 兼容旧格式（只有帧号）
                        frame_number = frame_info
                        timestamp = 0.0
                    
                    results[str(frame_file)] = {
                        "tags": tags_text,
                        "frame_number": frame_number,
                        "timestamp": timestamp
                    }
                    logger.info(f"Frame {frame_file.name} (frame #{frame_number}, t={timestamp:.2f}s): {tags_text}")
                    
            except Exception as e:
                logger.error(f"Failed to analyze frame {frame_file}: {e}")
                continue

        return results


class MLXVLMClient:
    """MLX-based VLM client using mlx_vlm library.

    Provides the same interface as OllamaVLMClient but runs inference
    locally via MLX (Apple Silicon optimized).
    """

    def __init__(self, config: Optional[ConfigManager] = None, model: str = "", base_url: str = ""):
        self.config = config
        if config:
            vlm_config = config.get_vlm_config()
            self.model_path = vlm_config.get('model', '')
            self.timeout = vlm_config.get('timeout', 60)
        else:
            self.model_path = model
            self.timeout = 60

        if not self.model_path:
            raise ValueError("MLX VLM model path must be specified in config or as model parameter")

        logger.info(f"Loading MLX VLM model from {self.model_path} ...")
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self._mlx_load = load
        self._mlx_generate = generate
        self._apply_chat_template = apply_chat_template

        self.model, self.processor = load(self.model_path)
        logger.info("MLX VLM model loaded successfully.")

    def _build_dynamic_prompt(self, is_single_frame: bool = False) -> str:
        try:
            engine = get_db_engine()
            db = get_session(engine)

            entries = db.query(DictionaryEntry).order_by(
                DictionaryEntry.category,
                DictionaryEntry.standard_tag
            ).all()

            db.close()

            categories = {}
            for entry in entries:
                category = entry.category or "其他"
                if category not in categories:
                    categories[category] = []
                categories[category].append(entry.standard_tag)

            standard_tags_text = ""
            for category, tags in categories.items():
                standard_tags_text += f"{category}：{', '.join(tags)}\n\n"

        except Exception as e:
            logger.warning(f"无法从数据库读取标准标签库，使用默认标签: {e}")
            standard_tags_text = (
                "场景：卧室, 客厅, 浴室, 酒店, 办公室, 更衣室, 健身房, 按摩店, 公园, 街道, 车内, 野外\n\n"
                "体位/动作：口交, 深喉, 69, 后入, 传教士式, 骑乘, 双龙, 侧卧, 站立抱起, 抚摸, 自慰, 互撸, 颜射, 内射\n\n"
                "人数：单人, 双人, 三人, 多人\n\n"
                "年龄/性向：青年, 少年感, 熟男, 中年, Gay, 双性恋\n\n"
                "身材：纤细, 肌肉, 微胖, 熊系\n\n"
                "颜值：精致, 清纯, 刚毅, 粗犷\n\n"
                "视角：器官特写, 第一视角\n\n"
                "肤色：白皙, 古铜, 黝黑\n\n"
                "服装：西装, 制服, 全裸, 半裸\n\n"
            )

        end_phrase = "现在请分析这一帧：" if is_single_frame else "现在请分析这些帧："

        prompt = (
            "你是一个成人内容分类专家。请分析视频帧，输出与性活动相关的标签。\n\n"
            "【标准标签——唯一允许的输出词表】\n\n"
            f"{standard_tags_text}"
            "【严格禁止输出的内容】\n"
            "- 词表之外的任何标签(禁止自创、禁止同义改写)\n"
            "- 通用描述词：明显、直接、处于、聚焦、所有、可以、能够、正在\n"
            "- 抽象概念：场景、画面、内容、情况、状态、问题、结果\n"
            "- 连接词和副词：和、与、或、很、非常、也、都\n"
            "- 无意义的动词：看到、出现、展示、包含、呈现\n"
            "- 视频技术描述：慢动作、快进、回放、特写镜头、远景、近景\n"
            "- 无意义的分类词：人群、为主、居多、偏多、偏少\n\n"
            "【输出规则】\n"
            "1. 只能从上方标准标签中逐字选择；若画面内容不在词表中,直接跳过\n"
            "2. 每个标签2-4个汉字\n"
            "3. 用逗号分隔，不要输出其他任何文字\n\n"
            "【输出示例】\n"
            "双人,后入,卧室,肌肉,白皙,青年\n\n"
            f"{end_phrase}"
        )

        return prompt

    def _generate_single(self, image_path: str, prompt: str, max_tokens: int = 500) -> str:
        """Run inference on a single image.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text stripped of whitespace.
        """
        formatted_prompt = self._apply_chat_template(
            self.processor, self.model.config, prompt, num_images=1
        )
        output = self._mlx_generate(
            self.model,
            self.processor,
            formatted_prompt,
            image=[image_path],
            max_tokens=max_tokens,
            verbose=False,
        )
        result_text = output.text if hasattr(output, 'text') else str(output)
        return result_text.strip()

    def analyze_frames(self, frames_path: str, prompt: str) -> str:
        """Analyze all frames in a directory with per-frame MLX inference.

        MLX does not support batched image inference, so each frame is
        processed individually and results are joined with newlines
        (matching OllamaVLMClient's multi-batch output format).

        Args:
            frames_path: Path to directory containing .jpg frames.
            prompt: The text prompt to use for each frame.

        Returns:
            Joined analysis text from all frames.
        """
        frames_folder = Path(frames_path)
        image_files = sorted(list(frames_folder.rglob("*.jpg")))
        if not image_files:
            logger.error(f"No images found in {frames_path}")
            return ""

        total_frames = len(image_files)
        logger.info(f"MLX VLM: analyzing {total_frames} frames individually (no batching)")

        all_results = []
        for i, img_path in enumerate(image_files):
            try:
                result_text = self._generate_single(str(img_path), prompt)
                if result_text:
                    all_results.append(result_text)
                logger.info(f"Frame {i + 1}/{total_frames} done: {len(result_text)} chars")
            except Exception as e:
                logger.error(f"MLX VLM frame {i + 1}/{total_frames} failed: {e}")

        if all_results:
            return "\n".join(all_results)
        return ""

    def analyze_frames_once(self, frame_paths: List[str], prompt: str) -> str:
        """对给定帧做分析, 返回合并文本。

        MLX 不支持多图批推理(OllamaVLMClient.analyze_frames_once 是把多帧塞进
        一次请求), 因此这里逐帧本地生成后按行合并 —— 语义一致(每场景一次
        "分析动作"), 但底层是本地多次前向, 无网络往返。
        """
        if not frame_paths:
            return ""
        all_results = []
        for p in frame_paths:
            try:
                txt = self._generate_single(str(p), prompt)
                if txt:
                    all_results.append(txt)
            except Exception as e:
                logger.error(f"MLX analyze_frames_once frame failed: {e}")
        return "\n".join(all_results)

    def get_analysis(self, frames_path: str, theme_name: Optional[str] = None) -> str:
        """Default analysis prompt focused on adult content classification.

        Reads prompt from config if available, otherwise builds a dynamic one.
        theme_name: 文件所属主题名，用于主题↔提示词联动。

        Args:
            frames_path: Path to directory containing frame images.

        Returns:
            Analysis text string (empty on failure).
        """
        prompt = None
        if self.config:
            if theme_name:
                prompt = self.config.get_vlm_prompt_for_theme(theme_name)
            else:
                prompt = self.config.get_vlm_prompt()

        if not prompt:
            prompt = self._build_dynamic_prompt(is_single_frame=False)
        else:
            # 闸 1: 配置/主题提示词缺词表,程序化追加闭式词表块
            prompt = OllamaVLMClient.append_vocabulary_constraint(prompt)

        try:
            return self.analyze_frames(frames_path, prompt)
        except Exception as e:
            logger.error(f"MLX VLM get_analysis failed: {e}")
            return ""

    def analyze_scenes_with_progress(
        self,
        frames_base_path: str,
        scene_indices: List[int],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[int, str]:
        """Analyze each scene individually with progress tracking.

        Args:
            frames_base_path: Base directory containing unit_X subdirectories.
            scene_indices: List of scene indices to analyze.
            progress_callback: Optional callback(current, total, message).

        Returns:
            Dict mapping scene_index -> VLM analysis text.
        """
        results = {}
        total = len(scene_indices)
        base_dir = Path(frames_base_path)

        for i, scene_idx in enumerate(scene_indices):
            scene_dir = base_dir / f"unit_{scene_idx}"
            if not scene_dir.exists():
                logger.warning(f"Scene directory not found: {scene_dir}")
                continue

            if progress_callback:
                progress_callback(
                    i,
                    total,
                    f"分析中 ({i + 1}/{total} 场景)"
                )

            try:
                analysis = self.get_analysis(str(scene_dir))
                if analysis:
                    results[scene_idx] = analysis
            except Exception as e:
                logger.error(f"Failed to analyze scene {scene_idx}: {e}")

        if progress_callback:
            progress_callback(total, total, f"分析完成 ({total}/{total} 场景)")

        return results

    def analyze_unit_frames_individually(
        self,
        unit_dir: str,
        prompt: Optional[str] = None,
        frame_paths: Optional[List[str]] = None
    ) -> Dict[str, dict]:
        """Analyze each frame of a single scene individually, returning tag-to-frame mappings.

        Args:
            unit_dir: Path to the scene directory (e.g. /path/to/frames/unit_0).

        Returns:
            Dict mapping frame_path -> {
                "tags": tags_text,
                "frame_number": video_frame_number,
                "timestamp": float
            }
        """
        results = {}
        unit_path = Path(unit_dir)

        if not unit_path.exists():
            logger.warning(f"Unit directory not found: {unit_dir}")
            return results

        frame_files = sorted(unit_path.glob("frame_*.jpg"))
        # 调用方可指定要分析的帧(如仅 NSFW 帧), 未指定则分析目录内全部帧
        if frame_paths:
            frame_files = [Path(p) for p in frame_paths]
        if not frame_files:
            logger.warning(f"No frames found in {unit_dir}")
            return results

        frame_mappings = {}
        mappings_file = unit_path.parent / "frame_mappings.json"
        if mappings_file.exists():
            try:
                import json
                with open(mappings_file, 'r', encoding='utf-8') as f:
                    frame_mappings = json.load(f)
                logger.info(f"Loaded frame mappings from {mappings_file}")
            except Exception as e:
                logger.warning(f"Failed to load frame mappings: {e}")

        # 获取 prompt: 优先使用调用方按主题动态传入的提示词
        if not prompt and self.config:
            prompt = self.config.get_vlm_prompt()

        if not prompt:
            prompt = self._build_dynamic_prompt(is_single_frame=True)
        else:
            # 闸 1: 配置/主题提示词缺词表,程序化追加闭式词表块
            prompt = OllamaVLMClient.append_vocabulary_constraint(prompt)

        for frame_file in frame_files:
            try:
                tags_text = self._generate_single(str(frame_file), prompt)

                if tags_text:
                    import json
                    try:
                        tags_dict = json.loads(tags_text)
                        all_tags = []
                        for key, value in tags_dict.items():
                            if isinstance(value, list):
                                all_tags.extend([tag for tag in value if tag])
                            elif isinstance(value, str) and value:
                                all_tags.append(value)
                        tags_text = ",".join(sorted(set(all_tags)))
                    except json.JSONDecodeError:
                        pass

                    frame_info = frame_mappings.get(str(frame_file), {})
                    if isinstance(frame_info, dict):
                        frame_number = frame_info.get("video_frame_number", 0)
                        timestamp = frame_info.get("timestamp", 0.0)
                    else:
                        frame_number = frame_info
                        timestamp = 0.0

                    results[str(frame_file)] = {
                        "tags": tags_text,
                        "frame_number": frame_number,
                        "timestamp": timestamp
                    }
                    logger.info(f"Frame {frame_file.name} (frame #{frame_number}, t={timestamp:.2f}s): {tags_text}")

            except Exception as e:
                logger.error(f"Failed to analyze frame {frame_file}: {e}")
                continue

        return results


def create_vlm_client(config: Optional[ConfigManager] = None) -> Any:
    """Factory function to create a VLM client based on config provider.

    Args:
        config: Optional ConfigManager instance. If None, defaults to
                OllamaVLMClient with default settings.

    Returns:
        VLM client instance (MLXVLMClient or OllamaVLMClient).

    Provider selection:
        - 'mlx' -> MLXVLMClient (local MLX inference)
        - 'ollama' or anything else -> OllamaVLMClient (Ollama API)

    内嵌推理（add-public-sale-readiness 任务 0.1）：顶层配置 `embedded_inference.
    enabled: true` 时，先拉起内嵌 llama-server（单例，幂等）再以显式
    base_url/model 构造 OllamaVLMClient——其 OpenAI 兼容协议自动识别与
    llama-server 对接，客户端零改动。
    """
    if config is None:
        return OllamaVLMClient()

    embedded_cfg = {}
    try:
        embedded_cfg = config.get("embedded_inference", {}) or {}
    except Exception:  # noqa: BLE001 - 配置缺失走默认路径
        embedded_cfg = {}

    # 打包态端点隔离（任务 1.2）：frozen 恒走内嵌；开发联调需显式
    # MEDIA_DEV_ENDPOINTS 解除。MEDIA_FORCE_TIER 与本判定无关。
    frozen = getattr(sys, "frozen", False)
    dev_override = bool(os.environ.get("MEDIA_DEV_ENDPOINTS"))
    if frozen and not dev_override:
        embedded_cfg = dict(embedded_cfg, enabled=True)

    if embedded_cfg.get("enabled"):
        from src.core.embedded_inference import get_embedded_server

        server = get_embedded_server(config)
        base_url = server.ensure_running()
        vlm_config = config.get_vlm_config()
        client = OllamaVLMClient(model=server.model_name, base_url=base_url)
        # 构造签名无 timeout 形参(显式模式默认 60s),内嵌大模型按 vlm.timeout 放宽
        client.timeout = int(vlm_config.get("timeout") or 300)
        return client

    vlm_config = config.get_vlm_config()
    provider = vlm_config.get('provider', 'ollama').lower()

    if provider in ('mlx', 'folder'):
        # 本地文件夹：用户下载的模型文件路径，走 MLX 本地加载
        return MLXVLMClient(config=config)
    else:
        # url / ollama / api / 默认：URL API（Ollama 与云端通用，协议自动识别）
        return OllamaVLMClient(config=config)
