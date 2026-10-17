#!/usr/bin/env python3
"""Evaluate Shadow/SuperDex on CrossDex YCB-45 using public-code semantics."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OBJECT_LIST = Path(__file__).with_name("crossdex_ycb_45.txt")
EVALUATOR = Path(__file__).with_name("eval_shadow_superdex.py")
DEFAULT_PYTHON = Path(sys.executable)

STEPS = 200
LIFT_STEP = 130
LIFT_GUIDANCE_HEIGHT = 0.5
PHYSICS_SUBSTEPS = 4
TRACKING_STIFFNESS = 50.0
TRACKING_DAMPING = 1.0
CONTACT_FRICTION = 3.0
CONTACT_NORMAL_DAMPING = 40.0


def valid_result(path: Path, episodes: int) -> bool:
    try:
        data = json.loads(path.read_text())
        protocol = data.get("evaluation_protocol", {})
        interface = data.get("interface_config", {})
        return (
            data.get("episodes") == episodes
            and protocol.get("paper_protocol") is False
            and protocol.get("success_rule")
            == "final_lift_10cm_and_center_error_below_10cm"
            and interface.get("lift_step") == LIFT_STEP
            and interface.get("lift_guidance_height") == LIFT_GUIDANCE_HEIGHT
            and interface.get("physics_substeps") == PHYSICS_SUBSTEPS
            and interface.get("tracking_stiffness") == TRACKING_STIFFNESS
            and interface.get("tracking_damping") == TRACKING_DAMPING
            and interface.get("contact_friction") == CONTACT_FRICTION
            and interface.get("contact_normal_damping") == CONTACT_NORMAL_DAMPING
        )
    except Exception:
        return False


def run_one(python: Path, object_id: str, out_dir: Path, episodes: int):
    output = out_dir / f"{object_id}.json"
    log = out_dir / f"{object_id}.log"
    if valid_result(output, episodes):
        return object_id, "cached"
    cmd = [
        str(python), str(EVALUATOR),
        "--object-id", object_id,
        "--episodes", str(episodes),
        "--label-index", "0",
        "--seed", "1",
        "--steps", str(STEPS),
        "--lift-step", str(LIFT_STEP),
        "--lift-guidance-height", str(LIFT_GUIDANCE_HEIGHT),
        "--physics-substeps", str(PHYSICS_SUBSTEPS),
        "--tracking-stiffness", str(TRACKING_STIFFNESS),
        "--tracking-damping", str(TRACKING_DAMPING),
        "--contact-friction", str(CONTACT_FRICTION),
        "--contact-normal-damping", str(CONTACT_NORMAL_DAMPING),
        "--output", str(output),
    ]
    child_env = os.environ.copy()
    child_env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    with log.open("w") as stream:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=child_env,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0 or not valid_result(output, episodes):
        raise RuntimeError(f"{object_id} failed; inspect {log}")
    return object_id, "completed"


def aggregate(out_dir: Path, objects: list[str], episodes_per_object: int):
    payloads = [json.loads((out_dir / f"{obj}.json").read_text()) for obj in objects]
    rows = []
    per_object = []
    for object_id, data in zip(objects, payloads):
        episodes = data["results"]
        rows.extend(episodes)
        per_object.append({
            "object_id": object_id,
            "episodes": len(episodes),
            "public_code_success_rate": sum(bool(r["success_10cm"]) for r in episodes) / len(episodes),
            "held_success_rate_10cm": sum(bool(r["held_success_10cm"]) for r in episodes) / len(episodes),
            "stable_success_rate_10cm": sum(bool(r["stable_success_10cm"]) for r in episodes) / len(episodes),
            "mean_final_lift_height": sum(float(r["final_lift_height"]) for r in episodes) / len(episodes),
            "mean_final_center_error": sum(float(r["final_target_distance"]) for r in episodes) / len(episodes),
            "mean_peak_contact_force": sum(float(r["peak_contact_force"]) for r in episodes) / len(episodes),
        })
    successes = sum(bool(r["success_10cm"]) for r in rows)
    summary = {
        "backend": "superdex",
        "hand": "shadow",
        "protocol": "dexgrasp_zero_public_code_ycb45",
        "success_rule": "final_lift_over_0.1m_and_final_center_error_below_0.1m",
        "objects": len(objects),
        "episodes_per_object": episodes_per_object,
        "episodes": len(rows),
        "steps": STEPS,
        "lift_step": LIFT_STEP,
        "lift_guidance_height": LIFT_GUIDANCE_HEIGHT,
        "control_dt": 0.01,
        "physics_substeps": PHYSICS_SUBSTEPS,
        "strict_physics_alignment": True,
        "tracking_stiffness": TRACKING_STIFFNESS,
        "tracking_damping": TRACKING_DAMPING,
        "contact_friction": CONTACT_FRICTION,
        "contact_normal_damping": CONTACT_NORMAL_DAMPING,
        "public_code_success_rate": successes / len(rows),
        "successes": successes,
        "held_success_rate_10cm": sum(bool(r["held_success_10cm"]) for r in rows) / len(rows),
        "stable_success_rate_10cm": sum(bool(r["stable_success_10cm"]) for r in rows) / len(rows),
        "final_contact_rate": sum(bool(r["held_contact"]) for r in rows) / len(rows),
        "mean_final_lift_height": sum(float(r["final_lift_height"]) for r in rows) / len(rows),
        "mean_final_center_error": sum(float(r["final_target_distance"]) for r in rows) / len(rows),
        "mean_max_lift_height": sum(float(r["max_lift_height"]) for r in rows) / len(rows),
        "mean_peak_contact_force": sum(float(r["peak_contact_force"]) for r in rows) / len(rows),
        "max_peak_contact_force": max(float(r["peak_contact_force"]) for r in rows),
        "object_ids": objects,
    }
    (out_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "per_object_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_object[0]))
        writer.writeheader()
        writer.writerows(per_object)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/shadow_superdex/public_code_ycb45_25",
    )
    parser.add_argument("--episodes-per-object", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    objects = [line.strip() for line in OBJECT_LIST.read_text().splitlines() if line.strip()]
    if len(objects) != 45 or len(set(objects)) != 45:
        raise RuntimeError("CrossDex object list must contain exactly 45 unique objects")
    missing = [obj for obj in objects if not (ROOT / "rsc/ycb_crossDex" / obj).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing YCB resources: {missing}")
    if args.list_only:
        print("\n".join(objects))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one, args.python, obj, args.output_dir, args.episodes_per_object): obj
            for obj in objects
        }
        done = 0
        for future in as_completed(futures):
            object_id, status = future.result()
            done += 1
            print(f"[{done:02d}/45] {object_id}: {status}", flush=True)
    aggregate(args.output_dir, objects, args.episodes_per_object)


if __name__ == "__main__":
    main()
