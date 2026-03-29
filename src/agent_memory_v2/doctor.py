from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory_v2.admin import get_store_stats
from agent_memory_v2.config import load_config
from agent_memory_v2.e2e_smoke import main as e2e_main
from agent_memory_v2.ollama import OllamaClient, OllamaProfile


def run_doctor() -> dict:
    config = load_config()
    llm_cfg = config.llm

    client = OllamaClient(
        OllamaProfile(
            host=llm_cfg["host"],
            model=llm_cfg["model"],
            temperature=0.0,
            max_tokens=int(llm_cfg.get("preflight", {}).get("max_tokens", 4)),
            timeout_seconds=int(
                llm_cfg.get("preflight", {}).get("timeout_seconds", llm_cfg["timeout_seconds"])
            ),
        )
    )
    preflight = client.healthcheck(run_generate=True)
    stats = get_store_stats(config)

    with TemporaryDirectory(prefix="agent_memory_v2_doctor_") as tmpdir:
        root = Path(tmpdir)
        raw = deepcopy(config.raw)
        raw["memory"]["index_path"] = "memory/doctor.index"
        raw["memory"]["metadata_path"] = "memory/doctor.json"
        raw["memory"]["interaction_log_path"] = "logs/doctor.jsonl"
        doctor_config = type(config)(root_dir=root, settings_path=config.settings_path, raw=raw)

        doctor_client = OllamaClient(
            OllamaProfile(
                host=llm_cfg["host"],
                model=llm_cfg["model"],
                temperature=float(llm_cfg["temperature"]),
                max_tokens=min(int(llm_cfg["max_tokens"]), 80),
                timeout_seconds=int(llm_cfg["timeout_seconds"]),
            )
        )

        from agent_memory_v2.pipeline import MemoryPipeline
        from agent_memory_v2.models import Message

        pipeline = MemoryPipeline(doctor_config, ollama=doctor_client)
        first_user = Message(role="user", text="Please remember that I prefer oat milk.")
        first_reply = pipeline.respond(first_user)
        first_agent = Message(
            role="agent",
            text=first_reply,
            conversation_id=first_user.conversation_id,
            turn_id=first_user.turn_id,
        )
        pipeline.ingest_turn(first_user, first_agent)
        second_user = Message(role="user", text="What did I say I prefer?")
        recalled = pipeline.recall(second_user)
        second_reply = pipeline.respond(second_user)
        e2e = {
            "ok": bool(recalled) and bool(second_reply.strip()),
            "recalled_count": len(recalled),
            "second_reply": second_reply,
        }

    return {
        "ok": bool(preflight.get("reachable"))
        and bool(preflight.get("model_present"))
        and bool(preflight.get("generate_ok"))
        and bool(e2e["ok"]),
        "preflight": preflight,
        "stats": stats,
        "e2e": e2e,
    }


def main() -> int:
    result = run_doctor()
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
