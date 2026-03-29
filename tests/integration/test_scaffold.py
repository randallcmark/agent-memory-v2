from agent_memory_v2.config import load_config


def test_project_scaffold_loads_default_config():
    config = load_config()
    assert config.llm["provider"] == "ollama"
    assert config.llm["model"] == "llama3:8b"
