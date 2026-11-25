# py-type-narrower

A LibCST-powered static analyzer and refactoring tool that detects overly broad Python type annotations (like `str`, `int`, or `T | None`) and suggests or applies narrower types (like `Literal["a", "b"]` or `T`), with optional **soundness verification** backed by your preferred type checker (`ty`, `mypy`, `pyright`, `basedpyright`, or a custom command).

## Rules & Tiers

Rules are split into two tiers by evidence strength:

| Code | Tier | Description | Auto-fix (`fix`) |
| :--- | :---: | :--- | :--- |
| `NT101` | 1 (Local) | Parameter typed `str`/`int` is exhaustively guarded to literals in the function body (`if/elif/else raise`, negative guard + raise, `match`, or constant dict subscript) | Safe (default) |
| `NT102` | 1 (Local) | Return type `T \| None` / `Optional[T]` never returns `None` along any path | Safe (default) |
| `NT103` | 1 (Local) | Return type `str`/`int` only returns constant literals | Safe (default) |
| `NT201` | 2 (Whole-repo) | Parameter typed `str`/`int` only receives literals from a small set across all callers (also flags single-constant parameters) | Requires `--unsafe-fixes` |
| `NT202` | 2 (Whole-repo) | Parameter typed `T \| None` never receives `None` across callers | Requires `--unsafe-fixes` |

Parameter narrowing automatically skips subclass method overrides, `Protocol`s, `ABC`s, and `@abstractmethod`s to avoid breaking Liskov substitution.

## Usage

```bash
# Report all candidates with evidence and confidence scores
py-type-narrower check . -v

# Check only Tier 1 high-confidence rules
py-type-narrower check . --tier 1

# Run in closed-world (application) mode and verify soundness with ty
py-type-narrower check . --closed-world --verify --verifier ty -v --show-rejected

# Preview unified diffs
py-type-narrower diff .

# Apply safe (Tier 1) fixes in place, verified by a type checker
py-type-narrower fix . --verify --verifier ty

# Apply Tier 1 + Tier 2 fixes with multi-pass fixpoint propagation
py-type-narrower fix . --closed-world --unsafe-fixes --max-passes 3 --verify
```

### Configurable Soundness Verifier

Pass `--verify` to validate candidates against a type-checker baseline inside a lightweight symlinked shadow workspace (your working tree is never touched during verification). If a batch introduces new type errors, the verifier bisects to isolate the safe subset and reports the blocking type errors on rejected candidates.

Supported `--verifier` values:
- `auto` (default): detects from `pyproject.toml` (`[tool.ty]`, `[tool.basedpyright]`, `[tool.pyright]`, `[tool.mypy]`) or available binaries in `PATH`
- `ty`
- `mypy`
- `pyright`
- `basedpyright`
- `custom` (paired with `--verify-command "uv run mypy src"`)

### Output Formats

```bash
py-type-narrower check . --format text   # Ruff-style diagnostics (default)
py-type-narrower check . --format json   # Structured JSON with evidence & scores
py-type-narrower check . --format sarif  # SARIF 2.1.0 for editors & GitHub Code Scanning
```

### Inline Ignores

Suppress specific rules or all rules on a line (or the line directly above a definition):

```python
def handle(mode: str) -> int:  # type-narrow: ignore[NT201]
    ...
```

### Configuration (`pyproject.toml`)

```toml
[tool.py-type-narrower]
max-tier = 2
min-confidence = 60
closed-world = true
max-literals = 8
min-call-sites = 2
max-passes = 2
verify = true
verifier = "ty"
select = ["NT101", "NT102", "NT103", "NT201", "NT202"]
ignore = []
```

### Optional Flake8 Plugin

Install with the `flake8` extra to expose Tier 1 (`NT101`, `NT102`, `NT103`) single-file rules directly in `flake8`:

```bash
pip install "py-type-narrower[flake8]"
flake8 your_package/
```

## Development

```bash
uv sync
./lint.sh       # runs stepdown, ruff check/format, and ty check
uv run pytest   # runs the test suite with coverage
```
