---
name: agent-memory-v2-ops
description: Operate and maintain the local agent_memory_v2 project. Use when asked to run the system, check health, inspect stores, rebuild state, run maintenance, manage backups, or perform safe admin tasks in /Volumes/Media/Repository/agent_memory_v2.
---

# Agent Memory V2 Ops

Use this skill when the task is about operating or maintaining `agent_memory_v2`.

Project root:

```bash
cd /Volumes/Media/Repository/agent_memory_v2
```

Default health flow:

```bash
make doctor
make maintenance-status
make stats
```

Interactive operation:

```bash
make chat
make chat ARGS="--user mark"
```

Admin and maintenance:

```bash
bash scripts/admin.sh stats
bash scripts/admin.sh list
bash scripts/admin.sh list-sidecar
bash scripts/admin.sh profile
bash scripts/admin.sh aging-report
bash scripts/admin.sh prune-dry-run
bash scripts/admin.sh maintenance-status
bash scripts/admin.sh maintain
bash scripts/admin.sh rebuild --force
bash scripts/admin.sh rebuild-profile --force
```

State safety before destructive work:

```bash
make backup ARGS="--output backups/pre-change.zip"
```

Only run destructive commands such as reset or prune when the user intent is clear.

Prefer this order for recovery or maintenance work:

1. `make doctor`
2. `make backup ...`
3. `make maintenance-status`
4. `make maintain`
5. `make stats`
6. `make list-sidecar`
7. `make profile`

If the task is about evaluation or qualitative review rather than operations, use `agent-memory-v2-quality` instead.
