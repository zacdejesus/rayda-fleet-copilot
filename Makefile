install:
	uv sync

test:
	uv run pytest -q

eval:
	uv run python evals/run_evals.py $(ARGS)

COMPANY ?= acme-001

run:
	uv run fleet-copilot --company $(COMPANY)
