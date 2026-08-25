# Retired v3 scripts

Every underscore-prefixed script in this directory belongs to the invalid v3
workflow and is retained only for forensic review. They contain machine-specific
paths and known scientific errors. Do not execute or patch them into a new run.

Canonical experiment entry points are under `code/experiments/`; shared tested
implementations are under `code/sam_kem/`. `validate_v3.py` is now only a
compatibility wrapper around the fail-closed canonical audit.
