# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Quality Gates

-   `ruff check .`
-   `black .`
-   `pytest -q`

## Pull Requests

-   Keep changes focused.
-   Add/adjust tests where applicable.
-   Update README if behavior changes.
