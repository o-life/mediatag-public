"""APIClient — thin wrapper around requests with retry logic."""

import requests


class APIClient:
    """Thin wrapper around requests.Session with retry logic and timeout handling."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    # ── HTTP methods ─────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        last_exc = None
        for attempt in range(self.retries):
            try:
                r = self.session.request(method, url, **kwargs)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < self.retries - 1:
                    import time
                    time.sleep(2)
        raise last_exc  # type: ignore[misc]

    def get(self, path: str, **kwargs) -> dict:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> dict:
        return self._request("POST", path, **kwargs)

    def put(self, path: str, **kwargs) -> dict:
        return self._request("PUT", path, **kwargs)

    def patch(self, path: str, **kwargs) -> dict:
        return self._request("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs) -> dict:
        return self._request("DELETE", path, **kwargs)

    # ── Health check ─────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if the API is reachable."""
        try:
            r = self.session.get(f"{self.base_url}/status", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ── Convenience status check ─────────────────────────────────

    def is_connected(self) -> bool:
        """Alias for health_check (more UI-friendly name)."""
        return self.health_check()
