.PHONY: test check-schema smoke smoke-generate embedding-smoke install-embedding-model use-ollama-embeddings use-hash-embeddings e2e chat preflight stats list list-sidecar profile aging-report prune-dry-run prune maintain maintenance-status rebuild-profile reset rebuild doctor classify recall prompt generate ingest backup restore seed sanitize-publish eval-classification eval-semantic eval-sentiment eval-profile eval-recall eval-prompt eval-all eval-history eval-compare live-eval-memory live-eval-sentiment live-eval-all live-eval-history live-eval-compare scenario-list scenario-run scenario-show scenario-compare agent-eval-run agent-eval-all agent-eval-history agent-eval-compare exp-list exp-run exp-matrix exp-build-snapshots

test:
	bash scripts/test.sh

check-schema:
	bash scripts/admin.sh check-schema $(ARGS)

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

eval-classification:
	bash scripts/eval.sh classification $(ARGS)

eval-semantic:
	bash scripts/eval.sh semantic $(ARGS)

eval-sentiment:
	bash scripts/eval.sh sentiment $(ARGS)

eval-profile:
	bash scripts/eval.sh profile $(ARGS)

eval-recall:
	bash scripts/eval.sh recall $(ARGS)

eval-prompt:
	bash scripts/eval.sh prompt $(ARGS)

eval-all:
	bash scripts/eval.sh all $(ARGS)

eval-history:
	bash scripts/eval.sh history $(ARGS)

eval-compare:
	bash scripts/eval.sh compare $(ARGS)

live-eval-memory:
	bash scripts/live-eval.sh memory $(ARGS)

live-eval-sentiment:
	bash scripts/live-eval.sh sentiment $(ARGS)

live-eval-all:
	bash scripts/live-eval.sh all $(ARGS)

live-eval-history:
	bash scripts/live-eval.sh history $(ARGS)

live-eval-compare:
	bash scripts/live-eval.sh compare $(ARGS)

scenario-list:
	bash scripts/scenario.sh list $(ARGS)

scenario-run:
	bash scripts/scenario.sh run $(ARGS)

scenario-show:
	bash scripts/scenario.sh show $(ARGS)

scenario-compare:
	bash scripts/scenario.sh compare $(ARGS)

agent-eval-run:
	bash scripts/agent-eval.sh run $(ARGS)

agent-eval-all:
	bash scripts/agent-eval.sh run-all $(ARGS)

agent-eval-history:
	bash scripts/agent-eval.sh history $(ARGS)

agent-eval-compare:
	bash scripts/agent-eval.sh compare $(ARGS)

exp-list:
	bash scripts/experiment.sh list $(ARGS)

exp-run:
	bash scripts/experiment.sh run $(ARGS)

exp-matrix:
	bash scripts/experiment.sh matrix $(ARGS)

exp-build-snapshots:
	bash scripts/experiment.sh build-snapshots $(ARGS)
