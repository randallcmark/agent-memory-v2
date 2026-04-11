from agent_memory_v2.ollama import OllamaClient, OllamaProfile


class StubResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def make_client() -> OllamaClient:
    return OllamaClient(
        OllamaProfile(
            host="http://127.0.0.1:11434",
            model="llama3:8b",
            temperature=0.0,
            max_tokens=16,
            timeout_seconds=5,
        )
    )


def test_healthcheck_reports_reachable_model(monkeypatch):
    client = make_client()

    def fake_get(url, timeout):
        if url.endswith("/api/version"):
            return StubResponse({"version": "0.7.0"})
        if url.endswith("/api/tags"):
            return StubResponse({"models": [{"name": "llama3:8b"}]})
        raise AssertionError(url)

    monkeypatch.setattr("agent_memory_v2.ollama.requests.get", fake_get)
    result = client.healthcheck()

    assert result["reachable"] is True
    assert result["model_present"] is True
    assert result["version"] == "0.7.0"


def test_generate_uses_generate_endpoint(monkeypatch):
    client = make_client()

    def fake_post(url, json, timeout):
        assert url.endswith("/api/generate")
        assert json["model"] == "llama3:8b"
        assert json["raw"] is False
        return StubResponse({"response": "OK"})

    monkeypatch.setattr("agent_memory_v2.ollama.requests.post", fake_post)
    assert client.generate("hello") == "OK"


def test_embed_uses_embed_endpoint(monkeypatch):
    client = make_client()

    def fake_post(url, json, timeout):
        assert url.endswith("/api/embed")
        assert json["model"] == "llama3:8b"
        assert json["input"] == "hello"
        return StubResponse({"embeddings": [[0.1, 0.2, 0.3]]})

    monkeypatch.setattr("agent_memory_v2.ollama.requests.post", fake_post)
    assert client.embed("hello") == [0.1, 0.2, 0.3]
