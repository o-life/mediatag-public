"""Model manager for loading and swapping AI models.

Provides a unified interface for model loading with lazy initialization,
supporting multiple backends (ONNX Runtime, InsightFace, MediaPipe, etc.).
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type

logger = logging.getLogger(__name__)

# Registered default model types. Factories are imported lazily (inside
# ``_ensure_default_factories``) so that importing this module never pulls in
# the heavy model stacks.
_MODEL_TYPES = {
    "face_detection": "src.core.models.face_detection_model:AuraFaceDetectionModel",
    "face_recognition": "src.core.models.face_recognition_model:AuraFaceRecognitionModel",
    "object_detection": "src.core.models.rfdetr_model:RFDETRModel",
}


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the model is available and loaded."""
        ...

    @abstractmethod
    def load(self) -> bool:
        """Load the model. Returns True on success."""
        ...

    @abstractmethod
    def unload(self):
        """Unload the model from memory."""
        ...


class ModelManager:
    """Manages model instances with lazy loading.

    Usage:
        manager = ModelManager.get_instance()
        detector = manager.get("face_detection")
        if detector.is_available():
            faces = detector.detect(frame)
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._models: Dict[str, BaseModel] = {}
        self._factories: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Return the process-wide singleton ModelManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_default_factories(self):
        """Lazily register the built-in model factories."""
        if "face_detection" in self._factories and "face_recognition" in self._factories:
            return
        for model_type, dotted_path in _MODEL_TYPES.items():
            if model_type in self._factories:
                continue
            try:
                module_path, class_name = dotted_path.split(":")
                import importlib
                module = importlib.import_module(module_path)
                factory = getattr(module, class_name)
                self.register_factory(model_type, factory)
                logger.info("Registered default model factory: %s", model_type)
            except Exception as e:
                logger.warning("Failed to register model factory %s: %s", model_type, e)

    def register_factory(self, model_type: str, factory: Type[BaseModel]):
        """Register a model factory class."""
        self._factories[model_type] = factory

    def get(self, model_type: str) -> Optional[BaseModel]:
        """Get a model by type, loading it if necessary.

        Returns None if the model type is not registered.
        """
        self._ensure_default_factories()

        if model_type not in self._models:
            if model_type not in self._factories:
                logger.warning("Model type '%s' not registered", model_type)
                return None

            factory = self._factories[model_type]
            model_config = self.config.get(model_type, {})
            model = factory(model_config)

            if model.load():
                self._models[model_type] = model
                logger.info("Loaded model: %s", model_type)
            else:
                logger.error("Failed to load model: %s", model_type)
                return None

        return self._models[model_type]

    def peek(self, model_type: str) -> Optional[BaseModel]:
        """Return an already-loaded model without triggering a load.

        This is a read-only lookup used by wrappers to avoid recursive
        initialization (a wrapper's ``load()`` calls back into the pipeline,
        which peeks here instead of calling ``get``).
        """
        return self._models.get(model_type)

    def unload_all(self):
        """Unload all models from memory."""
        for name, model in self._models.items():
            model.unload()
            logger.info("Unloaded model: %s", name)
        self._models.clear()

    def is_available(self, model_type: str) -> bool:
        """Check if a model is available (loaded and ready)."""
        model = self._models.get(model_type)
        return model is not None and model.is_available()
