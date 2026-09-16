from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from threshold_gate.analysis import markdown_report, summarize_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw BAF threshold-transfer Gate outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--effects", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload["rows"])
    metadata = payload["metadata"]
    expected = int(metadata["expected_fit_count"])
    if int(metadata["fit_count"]) != expected:
        raise RuntimeError(f"incomplete run: {metadata['fit_count']}/{expected} fits")
    summary = summarize_gate(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.effects.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    from threshold_gate.analysis import primary_effects
    primary_effects(rows).to_csv(args.effects, index=False)
    args.report.write_text(markdown_report(summary, {**metadata, "python": metadata["environment"]["python"]}), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "primary_median_abs_pe_gap": summary["primary_median_abs_pe_gap"], "bootstrap_ci_95": summary["bootstrap_ci_95"], "ranking_turnover": summary["ranking_turnover"]}, indent=2))


if __name__ == "__main__":
    main()
