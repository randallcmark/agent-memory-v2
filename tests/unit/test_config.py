from pathlib import Path

from agent_memory_v2.config import load_config


def test_load_config_returns_app_config():
    config = load_config()
    assert config.raw["llm"]["model"] == "llama3:8b"
    assert config.resolve_path("data/memory").is_absolute()
    assert isinstance(config.root_dir, Path)
    assert config.embedding_dim == int(config.embeddings["dimensions"])


def test_current_user_defaults_to_catchall(monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_V2_USER", raising=False)
    config = load_config()
    assert config.current_user == "catchall"


def test_resolve_path_scopes_named_user_under_users_dir(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_V2_USER", "mark")
    config = load_config()
    resolved = config.resolve_path("data/memory/memory.index")
    assert "/data/users/mark/memory/memory.index" in str(resolved)
