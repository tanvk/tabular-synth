from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd

def save_run(synth_df: pd.DataFrame, report_html: str, base_dir: str = "artifacts"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "synthetic.csv"
    html_path = run_dir / "report.html"

    synth_df.to_csv(csv_path, index=False)
    html_path.write_text(report_html, encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "csv_path": str(csv_path),
        "html_path": str(html_path),
    }