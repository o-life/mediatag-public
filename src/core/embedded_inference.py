"""内嵌推理运行时（llama-server 子进程管理）——openspec add-public-sale-readiness 任务 0.1。

职责：把 llama-server 作为 App 内嵌子进程管理——随机可用端口、健康等待、
崩溃监视（自动重启一次）、进程退出清理；供 create_vlm_client 在打包/开发态
零外部端点运行（OllamaVLMClient 的 OpenAI 兼容协议自动识别，客户端零改动）。

边界：本模块只管子进程生命周期；模型资源打包布局与 SHA256 校验属任务 1.1；
协议细节在 vlm.py。测试用假二进制（tests/test_embedded_inference.py），
不依赖真模型。
"""

import atexit
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_RESTART_BUDGET = 3      # 崩溃自动重启次数预算(用户机上内存清理类工具会周期性 SIGKILL 大内存进程)
_HEALTH_TIMEOUT = 240    # 模型加载等待上限（冷加载 ~30-90s，留余量）
_PROC_TIMEOUT = 10
_SHA_CHUNK = 1 << 22     # SHA256 分块 4MB


class EmbeddedInferenceError(RuntimeError):
    """推理生命周期错误——str() 即用户可读文案(进任务 error_message)。

    code: INF_MISSING_ASSET / INF_ASSET_CORRUPT / INF_START_FAILED / INF_CRASHED
    """

    def __init__(self, code: str, user_message: str, suggestion: str, detail: str = ""):
        self.code = code
        self.user_message = user_message
        self.suggestion = suggestion
        msg = f"[{code}] {user_message}（建议：{suggestion}）"
        if detail:
            msg += f" [{detail}]"
        super().__init__(msg)


def _alloc_port() -> int:
    """抓一个当前空闲的高位端口（经典 port-0 技法，存在微小竞态，由
    启动失败重试兜底）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EmbeddedLlamaServer:
    """llama-server 子进程管理器。线程安全；同一实例只服务一个模型。"""

    def __init__(self, cfg: dict, tier: str = "HIGH"):
        cfg = cfg or {}
        home = Path.home()
        self.binary = cfg.get("binary") or "/opt/homebrew/bin/llama-server"
        self.model_path = cfg.get("model") or str(home / "models/Qwen3VL-8B-Instruct-Q4_K_M.gguf")
        self.mmproj_path = cfg.get("mmproj") or str(home / "models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf")
        self.model_name = Path(self.model_path).stem
        self.tier = tier
        # 并发/上下文:显式配置 > 档位表(perf_tiers) > 旧默认
        from src.core.perf_tiers import tier_params
        tp = tier_params(tier)
        self.ctx = int(cfg.get("ctx") or tp["ctx"])
        self.slots = int(cfg.get("slots") or tp["slots"])
        # 默认日志路径冻结感知:打包态 Finder 启动时 CWD=/,相对路径写不进
        self.log_path = Path(cfg.get("log_path") or (_data_root() / "logs" / "embedded_llama.log"))
        self.expected_sha = str(cfg.get("model_sha256") or "")
        self.port: int | None = None
        self.base_url: str | None = None
        self.proc: subprocess.Popen | None = None
        self.failed_reason: str | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._watchdog: threading.Thread | None = None
        self._restarts = 0

    # ── 生命周期 ──────────────────────────────────────────────
    def ensure_running(self, timeout: float = _HEALTH_TIMEOUT) -> str:
        """确保服务健康并返回 base_url；已健康时幂等快速返回。

        看护线程重启时持锁完成全流程,本调用最多阻塞到其结束——调用方
        语义就是"要一个可用端点",阻塞到可用是正确的。"""
        with self._lock:
            if self._healthy():
                assert self.base_url
                return self.base_url
            if self.failed_reason:
                raise RuntimeError(self.failed_reason)
            if self.proc is None or self.proc.poll() is not None:
                return self._start_locked(timeout)
        # 子进程存活但未健康:极小概率的并发窗口,锁外短暂兜底等待
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.failed_reason:
                raise RuntimeError(self.failed_reason)
            if self._healthy():
                return self.base_url
            time.sleep(1.0)
        raise RuntimeError("健康等待超时")

    def _reap_stale(self):
        """回收上次异常残留的实例（App 崩溃时 atexit 不保证执行）。"""
        pidfile = self.log_path.parent / "embedded_llama.pid"
        try:
            if not pidfile.exists():
                return
            old = int(pidfile.read_text().strip())
            cmdline = subprocess.run(
                ["ps", "-p", str(old), "-o", "command="],
                capture_output=True, text=True, timeout=5).stdout.strip()
            if "llama-server" not in cmdline:
                return  # pid 已被复用为无关进程，不动
            os.kill(old, signal.SIGTERM)
            for _ in range(20):
                time.sleep(0.25)
                try:
                    os.kill(old, 0)
                except ProcessLookupError:
                    break
            else:
                os.kill(old, signal.SIGKILL)
            logger.warning("回收残留内嵌实例 pid=%s", old)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        except Exception as e:  # noqa: BLE001 - 回收失败不阻塞启动
            logger.warning("回收残留实例失败(忽略): %s", e)
        finally:
            try:
                pidfile.parent.mkdir(parents=True, exist_ok=True)
                pidfile.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    def _start_locked(self, timeout: float) -> str:
        self._stop.clear()
        self.failed_reason = None
        self._reap_stale()
        # 首启 SHA256 校验（任务 1.1）：缓存命中秒回；清单哈希由打包写入配置
        verify_model_asset(self.model_path, self.log_path.parent, self.expected_sha)
        for attempt in range(3):
            # 重启复用原端口：客户端持有的 base_url 在重启后必须仍然有效
            port = self.port if self.port else _alloc_port()
            args = [
                self.binary, "--host", "127.0.0.1", "--port", str(port),
                "-m", self.model_path,
                # ctx 语义=每槽上下文：llama.cpp 按 slots 均分总上下文,而管线的
                # 多帧合并调用实测需 ~11.4k/次(2026-10-14,0.2 实测),8k/槽必溢出
                # (400 Bad Request)。总上下文=ctx×slots,HIGH=128k/MID=32k/LOW=8k。
                "-c", str(self.ctx * self.slots), "-np", str(self.slots),
                # KV 量化 q8_0:KV 占用减半(实测 HIGH 档 4 槽各 ~4GB 纯 KV),
                # 16GB 机 MID 档才能落在 <14GB 验收线内;质量影响可忽略
                "-ctk", "q8_0", "-ctv", "q8_0",
                # prompt cache 封顶 1GB(默认 8192MiB):0.2 实测它是峰值内存大头
                # (MID 档 llama-server 独占 21.4GB);2GB 档实测仍差 0.32GB,
                # 1GB 档批处理场景几乎无损(逐视频帧不同,前缀复用收益本就小)
                "-cram", "1024",
                "-ngl", "999", "--no-webui",
            ]
            if self.mmproj_path and Path(self.mmproj_path).exists():
                args += ["--mmproj", self.mmproj_path]
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = self.log_path.open("ab")
            logf.write(f"\n[{time.strftime('%F %T')}] start attempt {attempt + 1} port={port}\n".encode())
            logf.flush()
            self.proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT)
            self.port = port
            self.base_url = f"http://127.0.0.1:{port}"
            time.sleep(1.0)
            if self.proc.poll() is not None:
                logger.warning("embedded llama-server 启动即退出(code=%s),换端口重试",
                               self.proc.returncode)
                continue
            try:
                self._wait_healthy(timeout)
            except RuntimeError:
                if self.proc.poll() is None:
                    self._terminate_proc()
                logger.warning("embedded llama-server 健康等待超时,重试")
                continue
            # 注意：_restarts 不在此重置——重启预算按规格只给一次,
            # 重置会让"崩溃→重启→再崩溃"无限循环
            self._spawn_watchdog()
            try:
                (self.log_path.parent / "embedded_llama.pid").write_text(str(self.proc.pid))
            except Exception:  # noqa: BLE001 - pidfile 失败不影响服务
                pass
            logger.info("embedded llama-server 就绪 %s (model=%s)", self.base_url, self.model_name)
            return self.base_url
        self.failed_reason = "连续 3 次启动失败(见日志)"
        raise RuntimeError(self.failed_reason)

    def _wait_healthy(self, timeout: float) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop.is_set():
                raise RuntimeError("服务已停止")
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"推理进程退出(code={self.proc.returncode})")
            try:
                r = requests.get(f"{self.base_url}/health", timeout=3)
                if r.status_code == 200 and "ok" in r.text:
                    return self.base_url
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise RuntimeError("健康等待超时")

    def _spawn_watchdog(self):
        self._watchdog = threading.Thread(target=self._watch_loop, daemon=True,
                                          name="embedded-llama-watchdog")
        self._watchdog.start()

    def _watch_loop(self):
        """崩溃监视：非主动停止的退出自动重启一次；预算耗尽置失败态。"""
        proc = self.proc
        if proc is None:
            return
        proc.wait()
        if self._stop.is_set():
            return
        with self._lock:
            if self._restarts >= _RESTART_BUDGET:
                self.failed_reason = f"推理进程崩溃且重启预算耗尽(code={proc.returncode})"
                logger.error(self.failed_reason)
                return
            self._restarts += 1
            logger.warning("embedded llama-server 意外退出(code=%s),自动重启(%d/%d)",
                           proc.returncode, self._restarts, _RESTART_BUDGET)
            try:
                self._start_locked(_HEALTH_TIMEOUT)
            except Exception as e:  # noqa: BLE001 - 看护线程吞异常置失败态
                self.failed_reason = f"自动重启失败: {e}"
                logger.error(self.failed_reason)

    def _healthy(self) -> bool:
        if self.proc is None or self.proc.poll() is not None or not self.base_url:
            return False
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            return r.status_code == 200 and "ok" in r.text
        except requests.RequestException:
            return False

    def _terminate_proc(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=_PROC_TIMEOUT)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=_PROC_TIMEOUT)

    def stop(self):
        with self._lock:
            self._stop.set()
            self._terminate_proc()
            self.proc = None
            self.port = None
            self.base_url = None

    def is_healthy(self) -> bool:
        with self._lock:
            return self._healthy()


_server: EmbeddedLlamaServer | None = None
_server_lock = threading.Lock()


# ── 资源解析与校验（任务 1.1） ────────────────────────────────────────
def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _bundle_root() -> Path | None:
    """PyInstaller 单目录资源根；开发态返回 None。"""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else None


def _data_root() -> Path:
    """数据根：优先 PathManager（主题工作区），失败回退仓库 data/。"""
    try:
        from src.core.path_manager import get_path_manager
        return Path(get_path_manager().base_dir)
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parent.parent / "data"


def resolve_runtime(cfg: dict) -> dict:
    """解析内嵌运行时三件套路径（binary/model/mmproj）。

    顺序：显式配置 > 打包资源（frozen）> 数据目录 models/ > 开发默认
    （brew + ~/models）。打包态若解析结果落到开发默认即视为缺失
    （端点/资源隔离，任务 1.2）。
    """
    cfg = cfg or {}
    home = Path.home()
    frozen = _is_frozen()
    bundle = _bundle_root()
    base = _data_root()

    model_name = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    candidates_model, candidates_bin = [], []

    if not frozen:
        candidates_model += [home / "models" / model_name]
        candidates_bin += ["/opt/homebrew/bin/llama-server"]
    if bundle:
        candidates_bin.insert(0, bundle / "bundled_bin" / "llama_cpp" / "bin" / "llama-server")
        candidates_model.insert(0, bundle / "models" / "qwen3-vl" / model_name)
    candidates_model.insert(0, base / "models" / "qwen3-vl" / model_name)

    def pick(cands, what):
        for c in cands:
            if c and Path(c).exists():
                return Path(c)
        raise EmbeddedInferenceError(
            "INF_MISSING_ASSET",
            f"内嵌推理{what}缺失",
            "请重新安装 MediaTag（模型资源不完整）",
            detail=f"已尝试: {[str(c) for c in cands]}",
        )

    binary = cfg.get("binary") or str(pick(candidates_bin, "运行时"))
    model = cfg.get("model") or str(pick(candidates_model, "模型文件"))
    # PyInstaller datas 不保证可执行位——内嵌二进制解析后补 +x
    try:
        _bp = Path(binary)
        if _bp.exists() and not os.access(_bp, os.X_OK):
            _bp.chmod(_bp.stat().st_mode | 0o111)
    except PermissionError as e:
        raise EmbeddedInferenceError(
            "INF_START_FAILED", "推理运行时无执行权限", "请重新安装 MediaTag",
            detail=str(e))
    mmproj_name = "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
    mmproj = cfg.get("mmproj") or str(model and Path(model).parent / mmproj_name)
    if not Path(mmproj).exists():
        mmproj = ""
    return {"binary": binary, "model": model, "mmproj": mmproj}


def _sha_cache_path(log_dir: Path) -> Path:
    return log_dir / "model_asset_hashes.json"


def verify_model_asset(model_path: str, log_dir: Path,
                       expected_sha256: str = "") -> str:
    """SHA256 首启校验；结果按 (size, mtime) 缓存避免每次全量重算。

    expected_sha256 来自配置（打包清单写入）；为空时记录基线仅告警不阻断
    （开发态模型无清单）。返回实际哈希。
    """
    p = Path(model_path)
    if not p.exists():
        raise EmbeddedInferenceError(
            "INF_MISSING_ASSET", "模型文件缺失", "请重新安装 MediaTag", detail=model_path)
    cache_file = _sha_cache_path(log_dir)
    cache = {}
    try:
        cache = json.loads(cache_file.read_text())
    except Exception:  # noqa: BLE001 - 缓存损坏即重算
        cache = {}
    stat = p.stat()
    key = str(p)
    ent = cache.get(key)
    if ent and ent.get("size") == stat.st_size and ent.get("mtime") == stat.st_mtime_ns:
        return ent["sha"]
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(_SHA_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    sha = h.hexdigest()
    cache[key] = {"size": stat.st_size, "mtime": stat.st_mtime_ns, "sha": sha}
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))
    except Exception:  # noqa: BLE001 - 缓存写失败不影响校验结论
        pass
    if expected_sha256 and sha.lower() != expected_sha256.lower():
        raise EmbeddedInferenceError(
            "INF_ASSET_CORRUPT", "模型资源校验失败（文件损坏或不完整）",
            "请重新安装 MediaTag；反复出现请联系支持",
            detail=f"expected={expected_sha256[:12]}… got={sha[:12]}…")
    logger.info("模型资源校验通过 %s (%.1fGB)", p.name, stat.st_size / 2**30)
    return sha


def get_embedded_server(config=None) -> EmbeddedLlamaServer:
    """进程级单例；配置键 = config.yaml 顶层 `embedded_inference`（缺省用
    开发机默认值，打包态由任务 1.1 落资源布局）。"""
    global _server
    with _server_lock:
        if _server is None:
            cfg = {}
            tier = "HIGH"
            if config is not None:
                try:
                    cfg = config.get("embedded_inference", {}) or {}
                except Exception:  # noqa: BLE001 - 配置缺失走默认
                    cfg = {}
            from src.core.perf_tiers import active as tier_active
            tier = tier_active(config)["tier"]
            if os.environ.get("MEDIA_FORCE_TIER"):
                # 低配模拟:档位压过显式 ctx/slots(否则模拟失去意义)
                cfg = {k: v for k, v in cfg.items() if k not in ("ctx", "slots")}
            cfg = {**cfg, **resolve_runtime(cfg)}  # 解析三件套(显式配置在 resolve 内最高优先)
            _server = EmbeddedLlamaServer(cfg, tier=tier)
            atexit.register(_server.stop)
        return _server
