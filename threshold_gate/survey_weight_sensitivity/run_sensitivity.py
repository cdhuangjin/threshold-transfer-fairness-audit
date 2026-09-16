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
THRESHOLD_GATE_ROOT = ROOT.parent
sys.path.insert(0, str(THRESHOLD_GATE_ROOT / "src"))

from threshold_gate.config import MODEL_PARAMETERS
from threshold_gate.external_transfer import (
    DOMAIN_PAIRS,
    STATE_FILES,
    STATE_HASHES,
    _encode_like_training,
    _fit_model,
    load_acs_state,
    source_calibration_split,
)

from survey_weight_sensitivity import weighted_binary_metrics, weighted_policy_thresholds


SEEDS = (42, 314, 2718, 1618, 8675309)
TARGET_FPRS = (0.01, 0.02, 0.05, 0.10)
MODEL_CONFIGS = tuple(MODEL_PARAMETERS)
CHECKPOINT_SCHEMA = "whyshift-weighted-threshold-transfer-checkpoint-v1"


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


def write_checkpoint(
    path: Path,
    rows: list[dict],
    completed: set[str],
    expected_fit_count: int,
    state_hashes: dict[str, str],
    protocol_hash: str,
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "fit_count": len(completed),
        "expected_fit_count": expected_fit_count,
        "row_count": len(rows),
        "completed_fit_keys": sorted(completed),
        "state_hashes": state_hashes,
        "protocol_hash": protocol_hash,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_checkpoint(
    path: Path,
    expected_fit_count: int,
    state_hashes: dict[str, str],
    protocol_hash: str,
) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        return [], set()
    if payload.get("state_hashes") != state_hashes or payload.get("protocol_hash") != protocol_hash:
        raise RuntimeError("weighted checkpoint does not match the current data or protocol")
    completed = set(payload.get("completed_fit_keys", []))
    rows = payload.get("rows", [])
    if payload.get("fit_count") != len(completed) or payload.get("expected_fit_count") != expected_fit_count:
        return [], set()
    if len(rows) != payload.get("row_count"):
        return [], set()
    return rows, completed


def run_weighted_one_model(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_name: str,
    target_name: str,
    model_name: str,
    seed: int,
    target_fprs: tuple[float, ...],
) -> list[dict]:
    train, calibration = source_calibration_split(source, seed)
    model_started = time.perf_counter()
    model = _fit_model(train, model_name, seed)
    calibration_scores = model.predict_proba(_encode_like_training(calibration, train))[:, 1]
    target_scores = model.predict_proba(_encode_like_training(target, train))[:, 1]
    elapsed = time.perf_counter() - model_started
    calibration_labels = calibration["y"].to_numpy(dtype="int8")
    target_labels = target["y"].to_numpy(dtype="int8")
    calibration_weights = calibration["PWGTP"].to_numpy(dtype=float)
    target_weights = target["PWGTP"].to_numpy(dtype=float)
    target_older = target["older"].to_numpy(dtype=bool)
    rows: list[dict] = []
    for target_fpr in target_fprs:
        thresholds = weighted_policy_thresholds(
            calibration_scores,
            calibration_labels,
            calibration_weights,
            target_scores,
            target_labels,
            target_weights,
            target_fpr,
        )
        for policy, threshold in thresholds.items():
            metric = weighted_binary_metrics(target_labels, target_scores, threshold, target_older, target_weights)
            rows.append(
                {
                    "pair": f"{source_name}->{target_name}",
                    "source": source_name,
                    "target": target_name,
                    "model_config": model_name,
                    "seed": int(seed),
                    "policy": policy,
                    "target_fpr": float(target_fpr),
                    "threshold": float(threshold),
                    "threshold_source": "target_labels" if policy == "target_oracle" else "source_calibration_labels",
                    "source_train_rows": int(len(train)),
                    "source_calibration_rows": int(len(calibration)),
                    "target_rows": int(len(target)),
                    "elapsed_seconds": float(elapsed),
                    "evaluation_weighting": "PWGTP",
                    **metric,
                }
            )
    return rows


def _state_hashes(data_root: Path) -> dict[str, str]:
    hashes = {}
    for state_name, filename in STATE_FILES.items():
        path = data_root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest.upper() != STATE_HASHES[state_name].upper():
            raise RuntimeError(f"hash mismatch for {state_name}: {digest}")
        hashes[state_name] = digest
    return hashes


def _metadata(
    args: argparse.Namespace,
    state_hashes: dict[str, str],
    completed: set[str],
    rows: list[dict],
    started: float,
    cpu_started: float,
    stop_reason: str | None,
    protocol_hash: str,
) -> dict:
    return {
        "protocol": "WhyShift ACS survey-weight sensitivity protocol v0.1",
        "command": " ".join(sys.argv),
        "data_root": str(args.data_root),
        "data_manifest": str(args.data_root.parents[2] / "MANIFEST.md"),
        "state_hashes": state_hashes,
        "domain_pairs": [f"{source}->{target}" for source, target in DOMAIN_PAIRS],
        "fit_count": len(completed),
        "expected_fit_count": len(DOMAIN_PAIRS) * len(MODEL_CONFIGS) * len(SEEDS),
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
        "survey_weights_used": True,
        "weighting_boundary": "unweighted score model and source split; PWGTP for threshold selection and target evaluation",
        "primary_artifact_directory": str(THRESHOLD_GATE_ROOT / "results" / "external_transfer_20260916_full"),
        "artifact_hashes": {
            "protocol": protocol_hash,
            "runner": sha256_file(Path(__file__)),
            "weighted_module": sha256_file(ROOT / "survey_weight_sensitivity.py"),
            "external_module": sha256_file(THRESHOLD_GATE_ROOT / "src" / "threshold_gate" / "external_transfer.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the WhyShift ACS survey-weight sensitivity.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "weighted_external_20260916")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    primary_dir = THRESHOLD_GATE_ROOT / "results" / "external_transfer_20260916_full"
    if args.output_dir.resolve() == primary_dir.resolve():
        raise ValueError("survey-weight sensitivity must use a new output directory")
    expected_fit_count = len(DOMAIN_PAIRS) * len(MODEL_CONFIGS) * len(SEEDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "protocol_v0.1.md"
    protocol_hash = sha256_file(protocol_path)
    state_hashes = _state_hashes(args.data_root)
    checkpoint_path = args.output_dir / "weighted_checkpoint.json"
    rows, completed = load_checkpoint(checkpoint_path, expected_fit_count, state_hashes, protocol_hash) if args.resume else ([], set())
    state_cache = {name: load_acs_state(args.data_root / STATE_FILES[name]) for name in STATE_FILES}
    started = time.perf_counter()
    cpu_started = time.process_time()
    max_wall_hours = 4.0
    stop_reason = None

    for source_name, target_name in DOMAIN_PAIRS:
        source = state_cache[source_name]
        target = state_cache[target_name]
        for model_name in MODEL_CONFIGS:
            for seed in SEEDS:
                key = fit_key(source_name, target_name, model_name, seed)
                if key in completed:
                    continue
                if time.perf_counter() - started > max_wall_hours * 3600:
                    stop_reason = "max_wall_hours"
                    break
                rows.extend(run_weighted_one_model(source, target, source_name, target_name, model_name, seed, TARGET_FPRS))
                completed.add(key)
                write_checkpoint(checkpoint_path, rows, completed, expected_fit_count, state_hashes, protocol_hash)
            if stop_reason:
                break
        if stop_reason:
            break

    metadata = _metadata(args, state_hashes, completed, rows, started, cpu_started, stop_reason, protocol_hash)
    status = {
        "status": "COMPLETE" if stop_reason is None and len(completed) == expected_fit_count else "INCOMPLETE",
        "fit_count": len(completed),
        "expected_fit_count": expected_fit_count,
        "row_count": len(rows),
        "stop_reason": stop_reason,
    }
    output = {"metadata": metadata, "rows": rows}
    (args.output_dir / "weighted_raw_results.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_dir / "weighted_raw_results.csv", index=False, float_format="%.15g")
    (args.output_dir / "weighted_run_manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "weighted_status.json").write_text(json.dumps(status, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({**status, "wall_seconds": metadata["wall_seconds"]}, indent=2))
    if status["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
