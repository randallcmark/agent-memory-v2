# Publishing Workflow

Use this flow before pushing `agent_memory_v2` to GitHub.

## Goals

1. remove local runtime state
2. ensure the repo contains only generic, non-sensitive seed data
3. verify the project still works after sanitisation

## Recommended Flow

From the project root:

```bash
make doctor
make sanitize-publish
make seed ARGS="--seed-file seeds/generic_seed.jsonl --user catchall --conversation-id seed"
make doctor
make prompt ARGS="--text 'What do I prefer?'"
git status
```

## What `make sanitize-publish` Does

It removes publish-unsafe runtime state such as:

1. live `data/`
2. local `backups/`
3. local pytest cache

This is intentionally destructive for generated runtime state. Run a backup first if you want to preserve current local data.

## What To Commit

Safe to commit:

1. source code
2. tests
3. docs
4. generic seeds in [seeds/generic_seed.jsonl](/Volumes/Media/Repository/agent_memory_v2/seeds/generic_seed.jsonl)

Do not commit:

1. user-specific runtime state
2. generated memory stores
3. local backups
4. local maintenance state you do not want shared

## GitHub Maintenance Workflow

After sanitising and reseeding:

```bash
git add .
git status
git commit -m "Prepare v2 for GitHub"
git push
```

If you want user-specific demo data in GitHub, create a separate generic seed file first rather than pushing live local state.
