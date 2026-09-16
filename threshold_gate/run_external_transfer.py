from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from threshold_gate.config import MODEL_PARAMETERS
from threshold_gate.external_transfer import DOMAIN_PAIRS, STATE_FILES, STATE_HASHES, load_acs_state, run_external_one_model


SEEDS = (42, 314, 2718, 1618, 8675309)
TARGET_FPRS = (0.01, 0.02, 0.05, 0.10)
MODEL_CONFIGS = tuple(MODEL_PARAMETERS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    import folktables
    import lightgbm
    import numpy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "lightgbm": lightgbm.__version__,
        "scikit_learn": sklearn.__version__,
        "folktables": getattr(folktables, "__version__", "unknown"),
    }


def fit_key(source: str, target: str, model_name: str, seed: int) -> str:
    return f"{source}->{target}|{model_name}|{int(seed)}"


def write_checkpoint(path: Path, rows: list[dict], completed: set[str], expected_fit_count: int) -> None:
    payload = {
        "schema": "whyshift-external-threshold-transfer-checkpoint-v1",
        "fit_count": len(completed),
        "expected_fit_count": expected_fit_count,
        "row_count": len(rows),
        "completed_fit_keys": sorted(completed),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_checkpoint(path: Path, expected_fit_count: int) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "whyshift-external-threshold-transfer-checkpoint-v1":
        return [], set()
    completed = set(payload.get("completed_fit_keys", []))
    rows = payload.get("rows", [])
    if payload.get("fit_count") != len(completed) or payload.get("expected_fit_count") != expected_fit_count:
        return [], set()
    return rows, completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen WhyShift external threshold-transfer replication.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "external_transfer_20260916_full")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    expected_fit_count = len(DOMAIN_PAIRS) * len(MODEL_CONFIGS) * len(SEEDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "external_checkpoint.json"
    rows, completed = load_checkpoint(checkpoint_path, expected_fit_count) if args.resume else ([], set())
    state_cache: dict[str, pd.DataFrame] = {}
    state_hashes: dict[str, str] = {}
    started = time.perf_counter()
    cpu_started = time.process_time()
    max_wall_hours = 4.0
    stop_reason = None

    def get_state(name: str) -> pd.DataFrame:
        if name in state_cache:
            return state_cache[name]
        path = args.data_root / STATE_FILES[name]
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        state_hashes[name] = digest
        if digest.upper() != STATE_HASHES[name].upper():
            raise RuntimeError(f"hash mismatch for {name}: {digest}")
        state_cache[name] = load_acs_state(path)
        return state_cache[name]

    for source_name, target_name in DOMAIN_PAIRS:
        source = get_state(source_name)
        target = get_state(target_name)
        for model_name in MODEL_CONFIGS:
            for seed in SEEDS:
                key = fit_key(source_name, target_name, model_name, seed)
                if key in completed:
                    continue
                if time.perf_counter() - started > max_wall_hours * 3600:
                    stop_reason = "max_wall_hours"
                    break
                rows.extend(run_external_one_model(source, target, source_name, target_name, model_name, seed, TARGET_FPRS))
                completed.add(key)
                write_checkpoint(checkpoint_path, rows, completed, expected_fit_count)
            if stop_reason:
                break
        if stop_reason:
            break

    metadata = {
        "protocol": "WhyShift external threshold-transfer replication protocol v0.1",
        "command": " ".join(sys.argv),
        "data_root": str(args.data_root),
        "data_manifest": str(args.data_root.parents[2] / "MANIFEST.md"),
        "state_hashes": state_hashes,
        "state_row_counts_after_filter": {name: int(len(frame)) for name, frame in state_cache.items()},
        "domain_pairs": [f"{source}->{target}" for source, target in DOMAIN_PAIRS],
        "fit_count": len(completed),
        "expected_fit_count": expected_fit_count,
        "row_count": len(rows),
        "stop_reason": stop_reason,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "environment": environment(),
        "seeds": SEEDS,
        "target_fprs": TARGET_FPRS,
        "model_configs": MODEL_CONFIGS,
        "models": MODEL_PARAMETERS,
        "official_filter": "AGEP > 16, PINCP > 100, WKHP > 0, PWGTP >= 1",
        "target_definition": "PINCP > 50000",
        "group_definition": "AGEP > 50 versus AGEP <= 50",
        "survey_weights_used": False,
        "artifact_hashes": {
            "protocol": sha256_file(ROOT.parent / "external_transfer_protocol_v0.1.md"),
            "runner": sha256_file(Path(__file__)),
            "external_module": sha256_file(ROOT / "src" / "threshold_gate" / "external_transfer.py"),
            "analyzer": sha256_file(ROOT / "analyze_external_transfer.py"),
            "figure_generator": sha256_file(ROOT / "make_external_figure.py"),
        },
    }
    output = {"metadata": metadata, "rows": rows}
    (args.output_dir / "external_raw_results.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_dir / "external_raw_results.csv", index=False)
    (args.output_dir / "external_run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"fit_count": len(completed), "expected_fit_count": expected_fit_count, "row_count": len(rows), "stop_reason": stop_reason, "wall_seconds": metadata["wall_seconds"]}, indent=2))
    if stop_reason or len(completed) != expected_fit_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
