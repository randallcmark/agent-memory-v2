"""Ollama HTTP client with retry logic for generate and embed endpoints."""

from __future__ import annotations

__all__ = ["OllamaProfile", "OllamaClient"]

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OllamaProfile:
    host: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    raw_prompt: bool = False


def _with_retry[T](
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Retry fn on transient network errors with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError("_with_retry: unreachable")  # pragma: no cover


class OllamaClient:
    def __init__(self, profile: OllamaProfile) -> None:
        self.profile = profile
        self.base_url = profile.host.rstrip("/")
        self.url = self.base_url + "/api/generate"

    def _get(self, path: str) -> requests.Response:
        response = requests.get(self.base_url + path, timeout=self.profile.timeout_seconds)
        response.raise_for_status()
        return response

    def _post(self, path: str, payload: dict[str, Any]) -> requests.Response:
        response = requests.post(
            self.base_url + path,
            json=payload,
            timeout=self.profile.timeout_seconds,
        )
        response.raise_for_status()
        return response

    def version(self) -> str:
        data = self._get("/api/version").json()
        return str(data.get("version") or "").strip()

    def list_models(self) -> list[str]:
        data = self._get("/api/tags").json()
        models = data.get("models") or []
        names: list[str] = []
        for item in models:
            name = (item or {}).get("name")
            if name:
                names.append(str(name))
        return names

    @staticmethod
    def _model_name_matches(expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        return expected.split(":")[0] == actual.split(":")[0]

    def model_exists(self) -> bool:
        return any(self._model_name_matches(self.profile.model, model) for model in self.list_models())

    def healthcheck(self, *, run_generate: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "host": self.profile.host,
            "model": self.profile.model,
            "reachable": False,
            "model_present": False,
            "version": "",
            "models": [],
        }
        version = self.version()
        result["version"] = version
        result["reachable"] = True
        models = self.list_models()
        result["models"] = models
        result["model_present"] = any(
            self._model_name_matches(self.profile.model, model) for model in models
        )
        if run_generate:
            result["generate_ok"] = False
            result["generate_response"] = self.generate("Reply with exactly OK.")
            result["generate_ok"] = True
        return result

    def embedding_healthcheck(self, *, run_embed: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "host": self.profile.host,
            "model": self.profile.model,
            "reachable": False,
            "model_present": False,
            "version": "",
            "models": [],
        }
        version = self.version()
        result["version"] = version
        result["reachable"] = True
        models = self.list_models()
        result["models"] = models
        result["model_present"] = any(
            self._model_name_matches(self.profile.model, model) for model in models
        )
        if run_embed and result["model_present"]:
            result["embed_ok"] = False
            vector = self.embed("oat milk")
            result["embedding_dim"] = len(vector)
            result["embed_ok"] = True
        elif run_embed:
            result["embed_ok"] = False
        return result

    def generate(self, prompt: str) -> str:
        def _call() -> str:
            payload = {
                "model": self.profile.model,
                "prompt": prompt,
                "stream": False,
                "raw": self.profile.raw_prompt,
                "options": {
                    "temperature": self.profile.temperature,
                    "num_predict": self.profile.max_tokens,
                },
            }
            response = self._post("/api/generate", payload)
            data = response.json()
            text = (data.get("response") or "").strip()
            if not text:
                raise RuntimeError("Ollama returned an empty response")
            return text

        return _with_retry(_call)

    def embed(self, text: str) -> list[float]:
        def _call() -> list[float]:
            payload = {
                "model": self.profile.model,
                "input": text,
            }
            response = self._post("/api/embed", payload)
            data = response.json()
            embeddings = data.get("embeddings") or []
            if not embeddings or not isinstance(embeddings[0], list):
                raise RuntimeError("Ollama returned no embeddings")
            vector = embeddings[0]
            if not vector:
                raise RuntimeError("Ollama returned an empty embedding vector")
            return vector

        return _with_retry(_call)
