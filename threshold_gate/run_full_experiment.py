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

from threshold_gate.config import GateConfig
from threshold_gate.data import load_dataset, temporal_split
from threshold_gate.full_runner import run_full_one_model
from threshold_gate.full_runner import fit_key


EXPECTED_FILES = {
    "Base": "Base.csv",
    "Variant I": "Variant I.csv",
    "Variant II": "Variant II.csv",
    "Variant III": "Variant III.csv",
    "Variant IV": "Variant IV.csv",
    "Variant V": "Variant V.csv",
}
EXPECTED_HASHES = {
    "Base": "7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809",
    "Variant I": "48c637a255d1fa4515da4286ccb99251412a6d905937ec7c108838ddfc1674bd",
    "Variant II": "60bcd971cf28779abb183c3286990cff9a87275bd8b83d9f6054a396c95cc736",
    "Variant III": "64693e549cff7ef802cf043fc7c937839e5929966e33a0852953b3f4c4321339",
    "Variant IV": "c6cddebbad34fa262a278deeb7985e7b669a64308d26e480d8eb85bfd08dac75",
    "Variant V": "470899bbef97c32a100528d63e863e1b1036df9501df6ef8140e7ec13b7365b4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_gib() -> float:
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**3))
    except ImportError:
        return 0.0


def environment() -> dict:
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
    }


def write_fit_checkpoint(path: Path, rows: list[dict], completed: set[str], expected_fit_count: int) -> None:
    payload = {
        "schema": "threshold-transfer-full-checkpoint-v2",
        "fit_count": len(completed),
        "expected_fit_count": expected_fit_count,
        "row_count": len(rows),
        "completed_fit_keys": sorted(completed),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_resume_checkpoint(path: Path, expected_fit_count: int) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "threshold-transfer-full-checkpoint-v2":
        return [], set()
    completed = set(payload.get("completed_fit_keys", []))
    rows = payload.get("rows", [])
    if payload.get("fit_count") != len(completed) or payload.get("expected_fit_count") != expected_fit_count:
        return [], set()
    return rows, completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen BAF threshold-transfer full experiment.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "full_threshold_policy_20260915")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = GateConfig()
    seeds = (42, 314, 2718, 1618, 8675309)
    targets = (0.01, 0.02, 0.05, 0.10)
    severities = (0.0, 0.25, 0.50, 1.00)
    expected_fits = len(cfg.datasets) * len(cfg.model_configs) * len(seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cpu_started = time.process_time()
    peak = 0.0
    checkpoint_path = args.output_dir / "full_checkpoint.json"
    rows, completed = load_resume_checkpoint(checkpoint_path, expected_fits) if args.resume else ([], set())
    data_hashes: dict[str, str] = {}
    fit_count = len(completed)
    stop_reason = None

    for variant in cfg.datasets:
        path = args.data_root / EXPECTED_FILES[variant]
        if not path.exists():
            raise FileNotFoundError(path)
        data_hashes[variant] = sha256_file(path)
        if data_hashes[variant] != EXPECTED_HASHES[variant]:
            raise RuntimeError(f"hash mismatch for {variant}: {data_hashes[variant]}")
        frame = load_dataset(path)
        fit, calibration, test = temporal_split(frame)
        if set(fit.month) & set(calibration.month) or set(fit.month) & set(test.month) or set(calibration.month) & set(test.month):
            raise RuntimeError(f"temporal overlap detected for {variant}")
        for model_name in cfg.model_configs:
            for seed in seeds:
                key = fit_key(variant, model_name, seed)
                if key in completed:
                    continue
                if time.perf_counter() - started > cfg.max_wall_hours * 3600:
                    stop_reason = "max_wall_hours"
                    break
                rows.extend(run_full_one_model(frame, variant, model_name, seed, targets, severities))
                completed.add(key)
                fit_count = len(completed)
                peak = max(peak, rss_gib())
                write_fit_checkpoint(checkpoint_path, rows, completed, expected_fits)
                if time.process_time() - cpu_started > cfg.max_cpu_hours * 3600:
                    stop_reason = "max_cpu_hours"
                    break
                if peak > cfg.max_ram_gb:
                    stop_reason = "max_ram_gb"
                    break
            if stop_reason:
                break
        write_fit_checkpoint(checkpoint_path, rows, completed, expected_fits)
        del frame, fit, calibration, test
        if stop_reason:
            break

    fit_count = len(completed)
    metadata = {
        "protocol": "BAF Threshold-Transfer Full Experiment Protocol v0.1",
        "command": " ".join(sys.argv),
        "data_root": str(args.data_root),
        "data_hashes": data_hashes,
        "fit_count": fit_count,
        "expected_fit_count": expected_fits,
        "row_count": len(rows),
        "stop_reason": stop_reason,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_gib": peak,
        "environment": environment(),
        "seeds": seeds,
        "target_fprs": targets,
        "noise_severities": severities,
        "data_manifest": str(args.data_root / "MANIFEST.md"),
    }
    output = {"metadata": metadata, "rows": rows}
    (args.output_dir / "full_raw_results.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_dir / "full_raw_results.csv", index=False)
    (args.output_dir / "full_run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"fit_count": fit_count, "row_count": len(rows), "stop_reason": stop_reason, "wall_seconds": metadata["wall_seconds"], "peak_rss_gib": peak}, indent=2))
    if stop_reason or fit_count != expected_fits:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
