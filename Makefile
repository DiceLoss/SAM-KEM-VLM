.PHONY: test lint compile protocols quality audit

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest

lint:
	python -m ruff check --select F code/sam_kem code/experiments tests

compile:
	python -m compileall -q code/sam_kem code/experiments tests

protocols:
	@for protocol in code/configs/protocols/*.yaml; do \
		sam-kem check-protocol --protocol "$$protocol" >/dev/null || exit 1; \
	done
	@echo "PASS: all protocol YAML files are valid"

quality: lint compile test protocols

audit:
	sam-kem validate --root . --config code/configs/experiment.yaml
