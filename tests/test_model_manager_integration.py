"""Tests for the ModelManager integration (singleton, default factories,
and read-only peek used by the face pipeline wrappers).
"""


from src.core.model_manager import ModelManager, BaseModel


class TestModelManagerIntegration:
    """Tests for ModelManager singleton and default registrations."""

    def test_singleton(self):
        """get_instance() should always return the same manager."""
        m1 = ModelManager.get_instance()
        m2 = ModelManager.get_instance()
        assert m1 is m2

    def test_default_factories_registered_lazily(self):
        """Built-in model types should resolve to a factory (may not load)."""
        manager = ModelManager()
        manager._ensure_default_factories()
        assert "face_detection" in manager._factories
        assert "face_recognition" in manager._factories

    def test_peek_does_not_load(self):
        """peek() should return None for never-loaded models (no load triggered)."""
        manager = ModelManager()
        assert manager.peek("face_detection") is None
        # Loading a model type that isn't installed must not raise — get()
        # returns None on load failure, never exceptions.
        result = manager.get("face_detection")
        assert result is None or isinstance(result, BaseModel)

    def test_peek_returns_loaded(self):
        """After get() succeeds, peek() should return the same instance."""
        manager = ModelManager()
        manager.register_factory("dummy", _DummyModel)
        model = manager.get("dummy")
        assert model is not None
        assert manager.peek("dummy") is model

    def test_register_and_get_custom(self):
        """Custom factories work through the standard API."""
        manager = ModelManager()
        manager.register_factory("custom", _DummyModel)
        model = manager.get("custom")
        assert model is not None
        assert model.is_available()
        # Same instance on repeat calls
        assert manager.get("custom") is model


class _DummyModel(BaseModel):
    """A trivial model used to test the manager without real weights."""

    def __init__(self, config=None):
        self.config = config or {}
        self._loaded = False

    def is_available(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        self._loaded = True
        return True

    def unload(self):
        self._loaded = False
