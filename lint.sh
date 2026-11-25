#!/usr/bin/env bash
set -euo pipefail
uv run stepdown fix py_type_narrower/
uv run ruff check . --fix && uv run ruff format .
uv run ty check
