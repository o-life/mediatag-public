"""系统相册 PhotoKit 只读导入器。

把 scripts/experiments/photos_spike.py 已实证的机制正式化：
- 只读访问：经 PhotoKit 取资产与资源写盘到 MediaTag 收件箱，
  绝不写 .photoslibrary bundle（Apple 受控结构，直写必坏）；
- 已编辑版本优先（决策点 2）：.photo/.video 资源即当前版本（用户编辑过
  则为编辑版，原件由系统保留为 alternate 资源）；
- Live Photo 成对（决策点 3）：主图 + pairedVideo 同名成对落盘，不拆散；
- 云端素材：允许联网取回，失败按条目报错（决策点 4 的 v0 简化）；
- TCC：裸 python 进程会被系统静默拒绝，必须打包态 .app 壳（spike 实证）。
  授权非 authorized/limited 时由调用方降级到普通文件对话框。
"""

import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

try:
    import Photos  # noqa: F401
    from Foundation import NSURL, NSSortDescriptor

    HAS_PHOTOS = True
except Exception:  # pragma: no cover - 非 macOS/缺依赖时降级
    Photos = None
    NSURL = None
    NSSortDescriptor = None
    HAS_PHOTOS = False

# PHAssetMediaType
_MEDIA_IMAGE = 1
_MEDIA_VIDEO = 2

# PHAuthorizationStatus
_AUTH_NOT_DETERMINED = 0
_AUTH_DENIED = 1
_AUTH_AUTHORIZED = 2
_AUTH_LIMITED = 3

#: UTI → 扩展名兜底映射（资源文件名无后缀时使用；PHAssetResource
#: 在某些 iCloud/历史素材上 filename() 可能为空或 UUID）
_UTI_EXT = {
    "public.jpeg": ".jpg",
    "public.heic": ".heic",
    "public.heif": ".heif",
    "public.png": ".png",
    "public.gif": ".gif",
    "public.webp": ".webp",
    "public.tiff": ".tif",
    "public.mpeg-4": ".mp4",
    "com.apple.quicktime-movie": ".mov",
    "com.adobe.raw-image": ".dng",
    "public.camera-raw-image": ".raw",
}


def is_available() -> bool:
    """Photos 框架是否可导入（非 macOS/缺依赖时 False）。"""
    return HAS_PHOTOS


def authorization_status() -> int:
    """当前照片库授权状态：0 未决定 / 1 拒绝 / 2 授权 / 3 受限。"""
    if not HAS_PHOTOS:
        return _AUTH_DENIED
    # 新 API: +authorizationStatusForAccessLevel:（macOS 10.16+），
    # 旧类方法 authorizationStatus 已废弃但仍可用——双路兜底保证有返回。
    for call in (
        lambda: int(
            Photos.PHPhotoLibrary.authorizationStatusForAccessLevel_(
                Photos.PHAccessLevel.PHAccessLevelReadWrite
            )
        ),
        lambda: int(Photos.PHPhotoLibrary.authorizationStatus()),
    ):
        try:
            return call()
        except Exception:
            continue
    return _AUTH_DENIED


def request_authorize(timeout: float = 30.0) -> bool:
    """请求授权并在主线程 run loop 上等结果（需在主线程调用）。"""
    if not HAS_PHOTOS:
        return False
    if authorization_status() in (_AUTH_AUTHORIZED, _AUTH_LIMITED):
        return True
    done = threading.Event()
    result = {"granted": False}

    def handler(status):
        result["granted"] = status in (_AUTH_AUTHORIZED, _AUTH_LIMITED)
        done.set()

    Photos.PHPhotoLibrary.requestAuthorization_(handler)
    from Foundation import NSDate, NSRunLoop

    rl = NSRunLoop.currentRunLoop()
    deadline = time.time() + timeout
    while not done.is_set() and time.time() < deadline:
        rl.runMode_beforeDate_(
            "kCFRunLoopDefaultMode", NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )
    return result["granted"]


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name or "").strip()
    return name or "asset"


def _unique_path(dest_dir: Path, filename: str) -> Path:
    """目标同名已存在（或与此前导出重名）时追加序号。"""
    base, ext = Path(filename).stem, Path(filename).suffix
    candidate = dest_dir / f"{base}{ext}"
    n = 1
    while candidate.exists():
        candidate = dest_dir / f"{base}_{n}{ext}"
        n += 1
    return candidate


def _ext_for(resource) -> str:
    """资源落盘扩展名：优先文件名后缀，其次 UTI 映射，最后二进制兜底。"""
    fn = (resource.filename() or "").strip()
    suffix = Path(fn).suffix.lower() if fn else ""
    if suffix:
        return suffix
    return _UTI_EXT.get(str(resource.uniformTypeIdentifier()), ".bin")


def _pick_resource(resources, types) -> Optional[object]:
    """按优先级挑资源类型（决策点 2：.photo/.video 即当前已编辑版本）。"""
    by_type = {}
    for r in resources:
        by_type.setdefault(int(r.type()), r)
    for t in types:
        if t in by_type:
            return by_type[t]
    return None


def list_assets() -> List[dict]:
    """枚举照片库资产（按拍摄时间倒序）。阻塞调用，放工作线程。

    返回字段：id / filename / media_type(照片|视频) / is_live / is_favorite /
    date(iso 或 "") / duration(秒，视频)。
    """
    if not HAS_PHOTOS:
        return []
    opts = Photos.PHFetchOptions.alloc().init()
    desc = NSSortDescriptor.sortDescriptorWithKey_ascending_("creationDate", False)
    opts.setSortDescriptors_([desc])
    result = Photos.PHAsset.fetchAssetsWithOptions_(opts)
    assets: List[dict] = []
    for i in range(result.count()):
        a = result.objectAtIndex_(i)
        mt = int(a.mediaType())
        if mt == _MEDIA_IMAGE:
            media_type = "照片"
        elif mt == _MEDIA_VIDEO:
            media_type = "视频"
        else:
            continue
        is_live = bool(a.mediaSubtypes() & Photos.PHAssetMediaSubtypePhotoLive)
        d = a.creationDate()
        date_iso = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(d.timeIntervalSince1970()))
            if d
            else ""
        )
        assets.append(
            {
                "id": str(a.localIdentifier()),
                "filename": str(a.filename() or ""),
                "media_type": media_type,
                "is_live": is_live,
                "is_favorite": bool(a.isFavorite()),
                "date": date_iso,
                "duration": round(float(a.duration() or 0), 1),
            }
        )
    return assets


def export_assets(
    ids: List[str],
    dest_dir: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    network_allowed: bool = True,
) -> tuple:
    """导出资产到收件箱目录。返回 (成功路径列表, 错误字符串列表)。

    - 决策点 2：默认导出当前版本（已编辑则编辑版；原件不随行）；
    - 决策点 3：Live Photo 主图 + pairedVideo 同名成对；
    - 决策点 4：network_allowed=True 时云端素材联网取回，失败计入 errors。
    """
    if not HAS_PHOTOS:
        return [], ["Photos 框架不可用"]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    fetched = Photos.PHAsset.fetchAssetsWithLocalIdentifiers_options_(ids, None)
    by_id = {}
    for i in range(fetched.count()):
        a = fetched.objectAtIndex_(i)
        by_id[str(a.localIdentifier())] = a

    exported: List[Path] = []
    errors: List[str] = []
    manager = Photos.PHAssetResourceManager.defaultManager()
    total = len(ids)

    for n, asset_id in enumerate(ids, start=1):
        asset = by_id.get(asset_id)
        if asset is None:
            errors.append(f"{asset_id[:12]}…: 资产不存在")
            continue
        try:
            resources = list(Photos.PHAssetResource.assetResourcesForAsset_(asset))
            is_video = int(asset.mediaType()) == _MEDIA_VIDEO
            is_live = bool(asset.mediaSubtypes() & Photos.PHAssetMediaSubtypePhotoLive)

            if is_video:
                primary = _pick_resource(resources, (2,))  # PHAssetResourceTypeVideo
            else:
                primary = _pick_resource(resources, (1,))  # PHAssetResourceTypePhoto
            if primary is None:
                errors.append(f"{asset.filename() or asset_id[:12]}: 无可导出资源")
                continue

            pair_resource = None
            if is_live:
                pair_resource = _pick_resource(
                    resources, (7,)  # PHAssetResourceTypePairedVideo
                )

            stem = Path(_sanitize_filename(primary.filename() or "")).stem
            stem = stem or asset_id[:8].replace("-", "")

            # 主资源
            out = _unique_path(dest_dir, f"{stem}{_ext_for(primary)}")
            err = _write_resource(manager, primary, out, network_allowed)
            if err is not None:
                errors.append(f"{stem}: {err}")
                continue
            exported.append(out)

            # Live Photo 伴生视频（同名成对）
            if pair_resource is not None:
                pair_out = _unique_path(dest_dir, f"{stem}{_ext_for(pair_resource)}")
                err = _write_resource(manager, pair_resource, pair_out, network_allowed)
                if err is not None:
                    errors.append(f"{stem} Live 视频: {err}")
                else:
                    exported.append(pair_out)
        except Exception as exc:  # 单条失败不中断整批
            logger.exception("导出资产失败 %s", asset_id)
            errors.append(f"{asset_id[:12]}…: {exc}")
        finally:
            if progress_cb is not None:
                progress_cb(n, total)
    return exported, errors


def _write_resource(manager, resource, out_path: Path, network_allowed: bool) -> Optional[str]:
    """写单个资源到 out_path。成功返回 None，失败返回错误描述。

    completion 在 PhotoKit 内部队列回调，threading.Event 跨线程等待即可。
    """
    opts = Photos.PHAssetResourceRequestOptions.alloc().init()
    opts.setNetworkAccessAllowed_(bool(network_allowed))
    done = threading.Event()
    result = {"err": None}

    def handler(err):
        result["err"] = err
        done.set()

    manager.writeDataForAssetResource_toFile_options_completionHandler_(
        resource,
        NSURL.fileURLWithPath_(str(out_path)),
        opts,
        handler,
    )
    if not done.wait(timeout=180):
        return "导出超时（180s）"
    err = result["err"]
    if err is not None:
        return f"PhotoKit 错误 {int(err.code)}: {err.localizedDescription}"
    if not out_path.exists() or out_path.stat().st_size == 0:
        return "导出结果为空"
    return None
