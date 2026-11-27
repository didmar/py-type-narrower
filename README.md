# py-type-narrower

A static analyzer and refactoring tool that detects overly broad Python type annotations and suggests or applies narrower types, optionally backed by your preferred type checker.

```diff
-from typing import Any
+from typing import Any, Literal, TypedDict, NotRequired
+
+
+class MetricsClientBuildEventDict(TypedDict):
+    kind: str
+    level: str
+    unit: NotRequired[str]
+    value: float


 class MetricsClient:
-    endpoint: str | None
-    scheme: str
+    endpoint: str
+    scheme: Literal["http", "https"]

     def __init__(self, endpoint: str, secure: bool = True) -> None:
         self.endpoint = endpoint
         if secure:
             self.scheme = "https"
         else:
             self.scheme = "http"

     def build_event(
-        self, kind: str, level: str, value: float
-    ) -> dict[str, Any]:
+        self, kind: Literal["counter", "gauge"], level: Literal["error", "info", "warn"], value: float
+    ) -> MetricsClientBuildEventDict:
         if level not in ("info", "warn", "error"):
             raise ValueError(f"Unknown level: {level}")
         match kind:
             case "counter":
                 return {"kind": kind, "level": level, "value": value}
             case "gauge":
                 return {"kind": kind, "level": level, "value": value, "unit": "ms"}
             case _:
                 raise ValueError(f"Unknown kind: {kind}")

-    def parse_timeout(self, raw: str) -> int | float | str | None:
+    def parse_timeout(self, raw: str) -> int | float:
         return float(raw) if "." in raw else int(raw)
```

## Installation

```bash
# Run without installing
uvx py-type-narrower check .

# Or install as a CLI tool
uv tool install py-type-narrower

# Or with pip
pip install py-type-narrower
```

Requires Python 3.11+.

## Usage

```bash
# See what can be narrowed
py-type-narrower check .

# Preview the proposed changes as a diff
py-type-narrower diff .

# Apply safe fixes in place and verify them with your type checker
py-type-narrower fix . --verify
```

Other options:
- `--verify` / `--verifier` — Runs your type checker (`auto`, `ty`, `mypy`, `pyright`, `basedpyright`, or `custom` with `--verify-command`) on a shadow copy of the project and automatically drops any change that introduces a type error. Recommended whenever running `fix`.
- `--closed-world` — Use this for applications or internal services where all callers live in the repo (as opposed to a public library). Unlocks narrowing on public functions and methods.
- `--unsafe-fixes` — Also apply Tier 2 fixes inferred from call sites across your codebase (`fix` only applies local Tier 1 fixes by default).
- `--scaffold-typeddict` — Generate `TypedDict` classes for functions returning or accepting fixed-key `dict`s (otherwise reported as advisory suggestions).
- `--changed` / `--diff-base main` — Only check or fix files modified in git or on the current branch, which is useful for gradual adoption or CI.
- `--format` — Output format for `check`: `text` (default), `json`, or `sarif` (for editors and GitHub Code Scanning).
- `-v` / `--show-rejected` — Show why each narrowing was suggested, or why `--verify` rejected one.

## Rules

Rules are split into two tiers: **Tier 1 (Local)** rules prove narrower types from a function or class body alone and are safe to auto-fix; **Tier 2 (Codebase)** rules infer narrower types from call sites across the project and require `--unsafe-fixes`.

| Code | Tier | Description | Auto-fix |
| :--- | :---: | :--- | :---: |
| `NT101` | 1 (Local) | Parameter typed `str`/`int` is exhaustively guarded to literals in the function body (`if/elif/else raise`, negative guard + raise, `match`, constant dict subscript, or `StrEnum`/`IntEnum` coercion) | Safe |
| `NT102` | 1 (Local) | Return type `T \| None` / `Optional[T]` never returns `None` along any path | Safe |
| `NT103` | 1 (Local) | Return type `str`/`int` only returns constant literals or `StrEnum`/`IntEnum` members | Safe |
| `NT104` | 1 (Local) | Class attribute narrowed locally (`T \| None` always initialized non-`None`, `str`/`int` only assigned literals, or union with unused branch(es)) | Safe |
| `NT105` | 1 (Local) | Return union `A \| B \| C` has unused branch(es) | Safe |
| `NT106` | 1 (Local) | Variadic `tuple[T, ...]` return always yields a fixed-length tuple literal (`tuple[T1, T2]`) | Safe |
| `NT107` | 1 (Local) | Container or generator return (`list[T \| None]`, `dict[K, V \| None]`, `Iterator[T \| None]`, `Generator[T \| None, ...]`) never contains/yields `None` | Safe |
| `NT108` | 1 (Local) | Return type `Any` or `object` only returns concrete inferable type(s) | Safe |
| `NT109` | 1 (Local) | Structured `dict` return with constant string keys narrows to an existing `TypedDict`, `dict[Literal[...], V]`, or reports/scaffolds a `TypedDict` (`--scaffold-typeddict`) | Safe* |
| `NT201` | 2 (Codebase) | Parameter typed `str`/`int` only receives literals from a small set or `StrEnum`/`IntEnum` members across callers (also flags single-constant parameters) | Unsafe |
| `NT202` | 2 (Codebase) | Parameter typed `T \| None` never receives `None` across callers | Unsafe |
| `NT203` | 2 (Codebase) | Dataclass, `NamedTuple`, `attrs`, Pydantic `BaseModel`, or `TypedDict` field (`str`/`int`, `T \| None`, or `NotRequired[T]`) can be narrowed across instantiations and writes | Unsafe |
| `NT204` | 2 (Codebase) | Parameter union `A \| B \| C` has unused branch(es) across callers | Unsafe |
| `NT205` | 2 (Codebase) | Boolean parameter always receives a single constant (`True` or `False`) | Unsafe |
| `NT206` | 2 (Codebase) | Call-site structured `dict` parameter narrows to an existing `TypedDict`, `dict[Literal[...], V]`, or reports/scaffolds a `TypedDict` (`--scaffold-typeddict`) | Unsafe* |

*\* `NT109` and `NT206` auto-fix directly when matching an existing `TypedDict` or `dict[Literal[...], V]`; generating a new `TypedDict` is advisory unless `--scaffold-typeddict` is passed.*

To preserve Liskov substitution and runtime contracts, narrowing automatically skips:
- `@no_type_check` code
- `Protocol`s, `ABC`s, and `@abstractmethod`s
- Subclass overrides (`@override` or inherited methods) and public methods on external subclasses when narrowing would break subtyping
- Overridden base-class return hooks
- PEP 562 module hooks (`__getattr__`, `__dir__`) and Python data-model methods (`__repr__`, `__str__`, `__exit__`, `__getattr__`, etc.)

`py-type-narrower` is meant to complement type checkers and linters rather than duplicate them, so the following are intentionally out of scope:
- Redundant casts or unnecessary `isinstance` checks (`--warn-redundant-casts`, `reportUnnecessaryIsInstance`)
- Duplicate union members (`int | int`)
- Missing annotations (`ruff ANN`, `--disallow-untyped-defs`) or bare `Any` bans (`ANN401`, `--disallow-any-explicit`)

## Configuration

Configure defaults in `pyproject.toml`:

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

### Inline Ignores

Suppress specific rules or all rules on a line (or the line directly above a definition):

```python
def handle(mode: str) -> int:  # type-narrow: ignore[NT201]
    ...
```

## Integrations

### Flake8 Plugin

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
