# Tabular Synth (Synthetic Data Engine)

MVP: Upload a CSV → train a copula-based generator → generate synthetic rows → view fidelity/utility metrics → download CSV.

## Quickstart

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```

# 1) Run with docker:
docker build -t tabular-synth:latest .
docker run -p 8501:8501 -v "$(pwd)/artifacts:/app/artifacts" tabular-synth:latest
```# open http://localhost:8501```

# 2) Run locally
Each run saves outputs to a timestamped folder:
```
artifacts/
└── run-YYYYmmdd-HHMMSS/
    ├── synthetic.csv      # generated dataset
    └── report.html        # fidelity + utility + privacy summary

```
Mounting ./artifacts into the container keeps outputs on your machine.


# 3) CI: run a tiny test on each push (GitHub Actions)

Create **`.github/workflows/ci.yml`**:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt pytest
      - run: pytest -q
```