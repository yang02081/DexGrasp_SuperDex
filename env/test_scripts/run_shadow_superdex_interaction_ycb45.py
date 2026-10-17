#!/usr/bin/env python3
"""Run YCB-45 Shadow interaction evaluation without the 0.5 m lift phase."""
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


def valid_result(path: Path, episodes: int) -> bool:
    try:
        data = json.loads(path.read_text())
        protocol = data.get("evaluation_protocol", {})
        interface = data.get("interface_config", {})
        return (
            data.get("episodes") == episodes
            and protocol.get("name") == "interaction_no_lift"
            and protocol.get("lift_guidance_enabled") is False
            and interface.get("tracking_stiffness") == 50.0
            and interface.get("tracking_damping") == 1.0
            and interface.get("contact_friction") == 3.0
            and interface.get("contact_normal_damping") == 40.0
            and interface.get("physics_substeps") == 4
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
        "--label-index", "0", "--seed", "1",
        "--steps", "200", "--lift-step", "130",
        "--disable-lift-guidance",
        "--physics-substeps", "4",
        "--tracking-stiffness", "50.0",
        "--tracking-damping", "1.0",
        "--contact-friction", "3.0",
        "--contact-normal-damping", "40.0",
        "--output", str(output),
    ]
    child_env = os.environ.copy()
    child_env.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    with log.open("w") as stream:
        proc = subprocess.run(cmd, cwd=ROOT, env=child_env,
                              stdout=stream, stderr=subprocess.STDOUT)
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
            "centered_rate": sum(bool(r["interaction_centered"]) for r in episodes) / len(episodes),
            "final_contact_rate": sum(bool(r["held_contact"]) for r in episodes) / len(episodes),
            "stable_contact_rate": sum(bool(r["stable_contact"]) for r in episodes) / len(episodes),
            "interaction_held_success_rate": sum(bool(r["interaction_held_success"]) for r in episodes) / len(episodes),
            "interaction_stable_success_rate": sum(bool(r["interaction_stable_success"]) for r in episodes) / len(episodes),
            "mean_final_center_error": sum(float(r["final_target_distance"]) for r in episodes) / len(episodes),
            "mean_peak_contact_force": sum(float(r["peak_contact_force"]) for r in episodes) / len(episodes),
        })
    mean = lambda key: sum(bool(row[key]) for row in rows) / len(rows)
    summary = {
        "backend": "superdex", "hand": "shadow",
        "protocol": "interaction_no_lift_ycb45",
        "objects": len(objects), "episodes_per_object": episodes_per_object,
        "episodes": len(rows), "steps": 200, "control_dt": 0.01,
        "physics_substeps": 4, "root_tracking_enabled": True,
        "lift_guidance_enabled": False,
        "tracking_stiffness": 50.0, "tracking_damping": 1.0,
        "contact_friction": 3.0, "contact_normal_damping": 40.0,
        "interaction_success_rule": "final_center_error_below_0.1m_and_final_contact",
        "interaction_stable_success_rule": "final_center_error_below_0.1m_and_at_least_two_finger_groups",
        "centered_rate": mean("interaction_centered"),
        "final_contact_rate": mean("held_contact"),
        "stable_contact_rate": mean("stable_contact"),
        "interaction_held_success_rate": mean("interaction_held_success"),
        "interaction_stable_success_rate": mean("interaction_stable_success"),
        "mean_final_center_error": sum(float(r["final_target_distance"]) for r in rows) / len(rows),
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
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/shadow_superdex/interaction_no_lift_ycb45_25")
    parser.add_argument("--episodes-per-object", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    args = parser.parse_args()
    objects = [line.strip() for line in OBJECT_LIST.read_text().splitlines() if line.strip()]
    if len(objects) != 45 or len(set(objects)) != 45:
        raise RuntimeError("CrossDex list must contain 45 unique objects")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one, args.python, obj, args.output_dir,
                        args.episodes_per_object): obj
            for obj in objects
        }
        for done, future in enumerate(as_completed(futures), 1):
            object_id, status = future.result()
            print(f"[{done:02d}/45] {object_id}: {status}", flush=True)
    aggregate(args.output_dir, objects, args.episodes_per_object)


if __name__ == "__main__":
    main()
