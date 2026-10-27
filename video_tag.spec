# -*- mode: python ; coding: utf-8 -*-

# MediaTag PyInstaller 打包配置

block_cipher = None


# OpenCV wheel 的许可文本须随包分发：其内嵌 FFmpeg 为 LGPL-2.1（非 GPL 构建），
# 商用分发的声明义务由随包携带这两份文本履行。详见 docs/架构/FFmpeg许可处置-定案.md
import os
import glob as _glob

_OPENCV_LICENSE_DATAS = []
for _d in sorted(_glob.glob('.venv/lib/python3.11/site-packages/opencv_python-*.dist-info')):
    for _f in ('LICENSE.txt', 'LICENSE-3RD-PARTY.txt'):
        _p = os.path.join(_d, _f)
        if os.path.exists(_p):
            _OPENCV_LICENSE_DATAS.append((_p, 'third_party_licenses/opencv'))

# VLM 内置形态（add-public-sale-readiness 任务 1.1，build.sh VLM_BUNDLE=1）：
# 自包含 llama-server 运行时(bundled_bin/llama_cpp, MIT) + Qwen3-VL 模型(≈5.4GB)。
# llama_cpp 的二进制已做 install_name 重写(@loader_path),按 datas 整目录落位
# 即保持 bin/../lib 相对布局,无需 PyInstaller 依赖分析。
_VLM_DATAS = []
if os.environ.get('VLM_BUNDLE') == '1':
    for _item in (
        ('bundled_bin/llama_cpp', 'bundled_bin/llama_cpp'),
        ('models/qwen3-vl', 'models/qwen3-vl'),
        ('third_party_licenses/qwen3vl', 'third_party_licenses/qwen3vl'),  # Apache-2.0 随包义务
        ('third_party_licenses/llamacpp', 'third_party_licenses/llamacpp'),  # MIT 随包义务
    ):
        if not os.path.exists(_item[0]):
            raise SystemExit(f'VLM_BUNDLE=1 但缺少 {_item[0]}')
        _VLM_DATAS.append(_item)

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),  # 图标等资源
        ('src', 'src'),  # 源代码
        ('config.yaml', '.'),  # 默认配置
        ('bundled_bin/ollama', 'bundled_bin'),  # 内置 Ollama(MIT)
        ('bundled_bin/ffmpeg_lgpl', 'bundled_bin/ffmpeg_lgpl'),  # 内置 FFmpeg(LGPL-2.1+ 构建，不含 x264/x265 等 GPL 组件)
        *_OPENCV_LICENSE_DATAS,  # OpenCV wheel 许可文本(内嵌 FFmpeg 为 LGPL-2.1)
        ('resources', 'resources'),  # 预置分析主题素材(resources/themes) + 模板(resources/templates)
        ('.venv/lib/python3.11/site-packages/opennsfw_onnx/open-nsfw.onnx', 'opennsfw_onnx'),  # NSFW 预筛模型(23.6MB, 可商用)
        ('models/auraface', 'models/auraface'),  # AuraFace 人脸模型(Apache-2.0 可商用)
        # PySide6 Qt 插件
        ('.venv/lib/python3.11/site-packages/PySide6/Qt/plugins/platforms', 'PySide6/Qt/plugins/platforms'),
        ('.venv/lib/python3.11/site-packages/PySide6/Qt/plugins/imageformats', 'PySide6/Qt/plugins/imageformats'),
        ('.venv/lib/python3.11/site-packages/PySide6/Qt/plugins/iconengines', 'PySide6/Qt/plugins/iconengines'),
        ('.venv/lib/python3.11/site-packages/PySide6/Qt/plugins/styles', 'PySide6/Qt/plugins/styles'),
        *_VLM_DATAS,  # VLM 内置形态(llama-server 运行时 + Qwen3-VL 模型)
    ],
    hiddenimports=[
        # PySide6 相关
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtMultimedia',
        'PySide6.QtNetwork',

        # macOS PhotoKit 只读导入（从照片导入按钮）
        'Photos',
        'Foundation',
        
        # FastAPI 相关
        'fastapi',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        
        # 数据库相关
        'sqlalchemy',
        'sqlalchemy.dialects.sqlite',
        
        # 视频处理
        'cv2',
        'numpy',
        
        # NSFW 预筛（可商用后端）
        'opennsfw_onnx',
        'opennsfw_onnx.classifier',
        'opennsfw_onnx.model',
        
        # VLM 相关
        'requests',
        'httpx',

        # 授权模块（licensing）：cryptography 用于 Ed25519 响应验签（集成规范 §7）；
        # 不使用 keyring（2026-09-22 用户定案：不访问系统钥匙串，授权存本地文件）
        'cryptography',
        'cryptography.hazmat.primitives.asymmetric.ed25519',
        
        # 其他依赖
        'pydantic',
        'yaml',
        'PIL',
        'insightface',
        'onnxruntime',
        'scipy',
        'scipy.spatial',
        'scipy.spatial.distance',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # ── 排除不需要的模块以减小体积 ──
        # 已确认: 这些库不在主流程(src/main.py → PySide6+FastAPI+人脸识别+TwoStageNSFW+VLM)
        # hybrid_analyzer(RF-DETR/CLIP)仅 ModelManager 惰性注册、从未激活 → torch 链路可排
        # 历史: 误收集 tensorflow 798M + torch 358M + pyarrow 108M + polars 183M
        'tensorflow',
        'torch',
        'pyarrow',
        'polars',
        'transformers',
        'timm',
        # 注意: skimage 是 insightface 人脸对齐(utils/transform.py)的硬依赖, 不可排除
        'mlx',
        'sympy',
        'networkx',
        'streamlit',
        'pydeck',
        'open_clip',
        'matplotlib',
        'pandas',
        'IPython',
        'notebook',
        'pytest',
        # PySide6 用不到的 Qt 模块(按需可再加)
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebEngineCore',
        'PySide6.Qt3DCore',
        'PySide6.QtCharts',
        'PySide6.QtDataVisualization',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MediaTag',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 不显示控制台
    disable_windowed_traceback=False,
    argv_emulation=True,  # macOS 特有：支持文件拖拽
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/AppIcon_from_png.icns',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MediaTag',
)

app = BUNDLE(
    coll,
    name='MediaTag.app',
    icon='assets/AppIcon_from_png.icns',
    bundle_identifier='com.videotag.app',  # 保持不变：同一应用身份，老版本升级原地替换（显示名已更名 MediaTag）
    info_plist={
        'CFBundleName': 'MediaTag',
        'CFBundleDisplayName': 'MediaTag',
        'CFBundleShortVersionString': '1.2.82',
        'CFBundleVersion': '1.2.82',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
        'NSRequiresAquaSystemAppearance': False,
        'NSSupportsAutomaticTermination': True,
        'NSSupportsSuddenTermination': True,
        # PhotoKit 只读导入（从照片导入按钮）：官方要求——缺此声明时 TCC
        # 静默拒绝访问，授权弹窗都不会出现（2026-09-23 实测踩坑）
        'NSPhotoLibraryUsageDescription': 'MediaTag 需要访问照片库，以便将您选择的照片和视频导入本应用进行 AI 分析与整理。',
        # 后续写回档位（相册镜像/元数据写回）所需的"添加到照片库"权限，提前声明
        'NSPhotoLibraryAddUsageDescription': 'MediaTag 需要此权限将分析结果（相册归组/元数据）写回照片库。',
    },
)
