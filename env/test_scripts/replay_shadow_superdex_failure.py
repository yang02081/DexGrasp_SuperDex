#!/usr/bin/env python3
"""Replay one failed YCB-45 Shadow/SuperDex evaluation case in the GUI."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = Path(__file__).with_name("eval_shadow_superdex.py")
DEFAULT_RESULTS = ROOT / "runs/shadow_superdex/public_code_ycb45_25"
DEFAULT_PYTHON = Path(sys.executable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--label-index", required=True, type=int)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--realtime-factor", type=float, default=0.25)
    parser.add_argument("--allow-success", action="store_true",
                        help="permit replay even if the stored case succeeded")
    args = parser.parse_args()

    source = args.results_dir / f"{args.object_id}.json"
    if not source.is_file():
        raise FileNotFoundError(f"missing evaluation result: {source}")
    data = json.loads(source.read_text())
    matches = [
        row for row in data["results"]
        if int(row["label_index"]) == args.label_index
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one label {args.label_index} in {source}, found {len(matches)}"
        )
    stored = matches[0]
    if bool(stored["success_10cm"]) and not args.allow_success:
        raise RuntimeError(
            "stored case is successful; pass --allow-success only if intentional"
        )

    replay_dir = args.results_dir / "gui_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.object_id}_label{args.label_index}"
    output = replay_dir / f"{stem}.json"
    trace = replay_dir / f"{stem}_trace.json"

    print("Replaying stored case:")
    print(json.dumps({
        "object_id": args.object_id,
        "label_index": args.label_index,
        "stored_success": bool(stored["success_10cm"]),
        "stored_max_lift_height": stored["max_lift_height"],
        "stored_final_lift_height": stored["final_lift_height"],
        "stored_final_center_error": stored["final_target_distance"],
    }, indent=2))
    print("SuperDex GUI will open; close its window to stop the replay.")

    interface = data.get("interface_config", {})
    cmd = [
        str(args.python), str(EVALUATOR),
        "--object-id", args.object_id,
        "--episodes", "1",
        "--label-index", str(args.label_index),
        "--seed", str(args.label_index + 1),
        "--steps", "200",
        "--lift-step", "130",
        "--lift-guidance-height", "0.5",
        "--physics-substeps", str(interface.get("physics_substeps", 4)),
        "--tracking-stiffness", str(interface.get("tracking_stiffness", 50.0)),
        "--tracking-damping", str(interface.get("tracking_damping", 1.0)),
        "--contact-friction", str(interface.get("contact_friction", 3.0)),
        "--contact-normal-damping", str(interface.get("contact_normal_damping", 40.0)),
        "--visualize",
        "--realtime-factor", str(args.realtime_factor),
        "--trace-output", str(trace),
        "--output", str(output),
    ]
    os.execv(str(args.python), cmd)


if __name__ == "__main__":
    main()
