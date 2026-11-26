# py-type-narrower

A LibCST-powered static analyzer and refactoring tool that detects overly broad Python type annotations (like `str`, `int`, `bool`, `T | None`, or `A | B | C`) and suggests or applies narrower types (like `Literal["a", "b"]`, `T`, or `A | B`), with optional **soundness verification** backed by your preferred type checker (`ty`, `mypy`, `pyright`, `basedpyright`, or a custom command).

## Installation

```bash
# Run without installing
uvx py-type-narrower check .

# Or install as a CLI tool
uv tool install py-type-narrower

# Or with pip
pip install py-type-narrower
```

Requires Python 3.12+.

## Quick Example

```python
# Before
def format_status(mode: str, code: int) -> str | None:
    if mode not in ("short", "long"):
        raise ValueError(f"Unknown mode: {mode}")
    return f"{mode}:{code}"
```

Running `py-type-narrower fix . --verify` narrows both annotations and adds the `Literal` import:

```python
from typing import Literal


# After
def format_status(mode: Literal["long", "short"], code: int) -> str:
    if mode not in ("short", "long"):
        raise ValueError(f"Unknown mode: {mode}")
    return f"{mode}:{code}"
```

## Design Philosophy & Non-Goals

`py-type-narrower` is built to **complement** static type checkers (`ty`, `mypy`, `pyright`), not replace or duplicate them. Standard type checkers verify *soundness* and deliberately permit annotations that are looser than actual usage.

Out of scope (already handled by type checkers or Ruff):
- Redundant casts or unnecessary `isinstance` checks (`--warn-redundant-casts`, `reportUnnecessaryIsInstance`)
- Duplicate union members (`int | int`)
- Missing annotations (`ruff ANN`, `--disallow-untyped-defs`) or bare `Any` bans (`ANN401`, `--disallow-any-explicit`)

## Rules & Tiers

Rules are split into two tiers by evidence strength:

| Code | Tier | Description | Auto-fix (`fix`) |
| :--- | :---: | :--- | :--- |
| `NT101` | 1 (Local) | Parameter typed `str`/`int` is exhaustively guarded to literals in the function body (`if/elif/else raise`, negative guard + raise, `match`, constant dict subscript, or `StrEnum`/`IntEnum` coercion) | Safe (default) |
| `NT102` | 1 (Local) | Return type `T \| None` / `Optional[T]` never returns `None` along any path | Safe (default) |
| `NT103` | 1 (Local) | Return type `str`/`int` only returns constant literals or `StrEnum`/`IntEnum` members | Safe (default) |
| `NT104` | 1 (Local) | Class attribute narrowed locally (`T \| None` always initialized non-`None`, `str`/`int` only assigned literals, or union with unused branch(es)) | Safe (default) |
| `NT105` | 1 (Local) | Return union `A \| B \| C` has unused branch(es) | Safe (default) |
| `NT106` | 1 (Local) | Variadic `tuple[T, ...]` return always yields a fixed-length tuple literal (`tuple[T1, T2]`) | Safe (default) |
| `NT107` | 1 (Local) | Container or generator return (`list[T \| None]`, `dict[K, V \| None]`, `Iterator[T \| None]`, `Generator[T \| None, ...]`) never contains/yields `None` | Safe (default) |
| `NT108` | 1 (Local) | Return type `Any` or `object` only returns concrete inferable type(s) | Safe (default) |
| `NT109` | 1 (Local) | Structured `dict` return with constant string keys narrows to an existing `TypedDict`, `dict[Literal[...], V]`, or reports a `TypedDict` schema | Safe when fixable |
| `NT201` | 2 (Codebase) | Parameter typed `str`/`int` only receives literals from a small set or `StrEnum`/`IntEnum` members across callers (also flags single-constant parameters) | Requires `--unsafe-fixes` |
| `NT202` | 2 (Codebase) | Parameter typed `T \| None` never receives `None` across callers | Requires `--unsafe-fixes` |
| `NT203` | 2 (Codebase) | Dataclass, `NamedTuple`, `attrs`, Pydantic `BaseModel`, or `TypedDict` field (`str`/`int`, `T \| None`, or `NotRequired[T]`) can be narrowed across instantiations and writes | Requires `--unsafe-fixes` |
| `NT204` | 2 (Codebase) | Parameter union `A \| B \| C` has unused branch(es) across callers | Requires `--unsafe-fixes` |
| `NT205` | 2 (Codebase) | Boolean parameter always receives a single constant (`True` or `False`) | Requires `--unsafe-fixes` |
| `NT206` | 2 (Codebase) | Call-site structured `dict` parameter narrows to an existing `TypedDict`, `dict[Literal[...], V]`, or reports a `TypedDict` schema | Requires `--unsafe-fixes` when fixable |

Narrowing automatically skips `@no_type_check` code, `Protocol`s, `ABC`s, `@abstractmethod`s, subclass overrides, overridden base-class return hooks, PEP 562 module hooks (`__getattr__`, `__dir__`), and fixed Python data-model protocol methods (`__repr__`, `__str__`, `__exit__`, `__getattr__`, etc.) to preserve Liskov substitution and runtime contracts.

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

# Check only git-changed files (or branch diff vs main) while still indexing the full project
py-type-narrower check --changed --closed-world
py-type-narrower check --diff-base main --closed-world --verify
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
min-confidence = 50
closed-world = true
max-literals = 8
min-call-sites = 2
max-passes = 2
verify = true
verifier = "ty"
select = ["NT101", "NT102", "NT103", "NT104", "NT105", "NT106", "NT107", "NT108", "NT109", "NT201", "NT202", "NT203", "NT204", "NT205", "NT206"]
ignore = []
```

### Optional Flake8 Plugin

Install with the `flake8` extra to expose Tier 1 (`NT101`–`NT109`) single-file rules directly in `flake8`:

```bash
pip install "py-type-narrower[flake8]"
flake8 your_package/
```

### Pre-commit

```yaml
repos:
  - repo: https://github.com/didmar/py-type-narrower
    rev: v0.1.0
    hooks:
      - id: py-type-narrower-check
        args: ["--tier", "1"]
```

## Development

```bash
uv sync
./lint.sh       # runs stepdown, ruff check/format, and ty check
uv run pytest   # runs the test suite with coverage
```
