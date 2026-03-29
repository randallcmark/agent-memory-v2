.PHONY: test smoke smoke-generate embedding-smoke install-embedding-model use-ollama-embeddings use-hash-embeddings e2e chat preflight stats list list-sidecar profile aging-report prune-dry-run prune maintain maintenance-status rebuild-profile reset rebuild doctor classify recall prompt generate ingest backup restore seed sanitize-publish

test:
	bash scripts/test.sh

smoke:
	bash scripts/ollama-smoke.sh

smoke-generate:
	bash scripts/ollama-generate-smoke.sh

embedding-smoke:
	bash scripts/embedding-smoke.sh

install-embedding-model:
	bash scripts/install-embedding-model.sh

use-ollama-embeddings:
	bash scripts/admin.sh use-ollama-embeddings $(ARGS)

use-hash-embeddings:
	bash scripts/admin.sh use-hash-embeddings $(ARGS)

e2e:
	bash scripts/e2e-smoke.sh

preflight:
	bash scripts/preflight.sh

chat:
	bash scripts/chat.sh $(ARGS)

ingest:
	bash scripts/ingest.sh $(ARGS)

seed:
	bash scripts/seed.sh $(ARGS)

classify:
	bash scripts/classify.sh $(ARGS)

recall:
	bash scripts/recall.sh $(ARGS)

prompt:
	bash scripts/prompt.sh $(ARGS)

generate:
	bash scripts/generate.sh $(ARGS)

backup:
	bash scripts/admin.sh backup $(ARGS)

restore:
	bash scripts/admin.sh restore $(ARGS)

sanitize-publish:
	bash scripts/sanitise.sh $(ARGS)

stats:
	bash scripts/admin.sh stats $(ARGS)

list:
	bash scripts/admin.sh list $(ARGS)

list-sidecar:
	bash scripts/admin.sh list-sidecar $(ARGS)

profile:
	bash scripts/admin.sh profile $(ARGS)

aging-report:
	bash scripts/admin.sh aging-report $(ARGS)

prune-dry-run:
	bash scripts/admin.sh prune-dry-run $(ARGS)

prune:
	bash scripts/admin.sh prune --force $(ARGS)

maintain:
	bash scripts/admin.sh maintain $(ARGS)

maintenance-status:
	bash scripts/admin.sh maintenance-status $(ARGS)

rebuild-profile:
	bash scripts/admin.sh rebuild-profile --force

reset:
	bash scripts/admin.sh reset --force

rebuild:
	bash scripts/admin.sh rebuild --force

doctor:
	bash scripts/doctor.sh
