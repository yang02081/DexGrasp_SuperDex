#!/usr/bin/env python3
"""Run a small deterministic Shadow/SuperDex physics parameter calibration."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = Path(__file__).with_name("eval_shadow_superdex.py")
PYTHON = Path(sys.executable)
OBJECTS = [
    "004_sugar_box", "011_banana", "025_mug", "032_knife",
    "035_power_drill", "040_large_marker", "051_large_clamp",
    "056_tennis_ball", "061_foam_brick", "077_rubiks_cube",
]
CONTROLLER_CONFIGS = {
    "kp100_kd0p2": (50.0, 1.0, 3.0, 20.60526466369629),
    "kp100_kd0p5": (100.0, 0.5, 3.0, 20.60526466369629),
    "kp100_kd1": (100.0, 1.0, 3.0, 20.60526466369629),
    "kp75_kd0p5": (75.0, 0.5, 3.0, 20.60526466369629),
    "kp75_kd1": (75.0, 1.0, 3.0, 20.60526466369629),
    "kp50_kd1": (50.0, 1.0, 3.0, 20.60526466369629),
}
CONTACT_CONFIGS = {
    "mu1_d20p6": (50.0, 1.0, 1.0, 20.60526466369629),
    "mu2_d20p6": (50.0, 1.0, 2.0, 20.60526466369629),
    "mu3_d20p6": (50.0, 1.0, 3.0, 20.60526466369629),
    "mu2_d10": (50.0, 1.0, 2.0, 10.0),
    "mu2_d40": (50.0, 1.0, 2.0, 40.0),
    "mu3_d40": (50.0, 1.0, 3.0, 40.0),
}


def run_one(out_dir: Path, name: str, params: tuple[float, ...], object_id: str):
    kp, kd, friction, normal_damping = params
    destination = out_dir / name / f"{object_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            data = json.loads(destination.read_text())
            if data.get("episodes") == 5:
                return name, object_id, "cached"
        except Exception:
            pass
    cmd = [
        str(PYTHON), str(EVALUATOR),
        "--object-id", object_id,
        "--episodes", "5", "--label-index", "0", "--seed", "1",
        "--steps", "200", "--lift-step", "130",
        "--lift-guidance-height", "0.5", "--physics-substeps", "4",
        "--tracking-stiffness", str(kp), "--tracking-damping", str(kd),
        "--contact-friction", str(friction),
        "--contact-normal-damping", str(normal_damping),
        "--output", str(destination),
    ]
    child_env = os.environ.copy()
    child_env.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    log = destination.with_suffix(".log")
    with log.open("w") as stream:
        proc = subprocess.run(cmd, cwd=ROOT, env=child_env,
                              stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"failed: {name}/{object_id}; inspect {log}")
    return name, object_id, "completed"


def summarize(out_dir: Path, configs: dict[str, tuple[float, ...]]):
    summary = []
    for name, params in configs.items():
        rows = []
        for object_id in OBJECTS:
            rows.extend(json.loads((out_dir / name / f"{object_id}.json").read_text())["results"])
        peaks = sorted(float(row["peak_contact_force"]) for row in rows)
        summary.append({
            "name": name,
            "tracking_stiffness": params[0],
            "tracking_damping": params[1],
            "contact_friction": params[2],
            "contact_normal_damping": params[3],
            "episodes": len(rows),
            "success_rate": sum(bool(row["success_10cm"]) for row in rows) / len(rows),
            "held_success_rate": sum(bool(row["held_success_10cm"]) for row in rows) / len(rows),
            "stable_success_rate": sum(bool(row["stable_success_10cm"]) for row in rows) / len(rows),
            "mean_final_center_error": sum(float(row["final_target_distance"]) for row in rows) / len(rows),
            "mean_peak_contact_force": sum(peaks) / len(peaks),
            "p95_peak_contact_force": peaks[min(len(peaks) - 1, int(0.95 * len(peaks)))],
            "max_peak_contact_force": max(peaks),
            "mean_peak_object_linear_speed": sum(float(row["peak_object_linear_speed"]) for row in rows) / len(rows),
        })
    summary.sort(key=lambda row: (-row["success_rate"], row["p95_peak_contact_force"]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("controller", "contact"), default="controller")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/shadow_superdex/calibration")
    args = parser.parse_args()
    configs = CONTROLLER_CONFIGS if args.stage == "controller" else CONTACT_CONFIGS
    out_dir = args.output_dir / args.stage
    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for name, params in configs.items():
            for object_id in OBJECTS:
                jobs.append(pool.submit(run_one, out_dir, name, params, object_id))
        complete = 0
        for future in as_completed(jobs):
            name, object_id, status = future.result()
            complete += 1
            print(f"[{complete:02d}/{len(jobs)}] {name}/{object_id}: {status}", flush=True)
    summarize(out_dir, configs)


if __name__ == "__main__":
    main()
