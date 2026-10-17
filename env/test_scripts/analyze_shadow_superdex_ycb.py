#!/usr/bin/env python3
"""Analyze per-object failures and terminal-force outliers in a YCB run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)

    episodes = []
    for path in sorted(args.run_dir.glob("*.json")):
        if path.name == "aggregate_summary.json":
            continue
        data = json.loads(path.read_text())
        if not isinstance(data.get("results"), list):
            continue
        for row in data["results"]:
            item = dict(row)
            item["object_id"] = path.stem
            episodes.append(item)
    if not episodes:
        raise SystemExit(f"no episode JSON files found in {args.run_dir}")

    forces = np.asarray([x["final_contact_force"] for x in episodes], float)
    positive = forces[forces > 0]
    percentiles = {str(p): float(np.percentile(forces, p)) for p in (50, 75, 90, 95, 99, 99.5, 100)}
    robust_cutoff = float(np.percentile(positive, 75) + 3.0 * (
        np.percentile(positive, 75) - np.percentile(positive, 25)
    )) if positive.size else 0.0
    force_cutoff = max(1000.0, robust_cutoff)

    grouped = {}
    failure_rows = []
    force_rows = []
    for row in episodes:
        grouped.setdefault(row["object_id"], []).append(row)
        if not row["paper_success"]:
            failure_rows.append({
                "object_id": row["object_id"], "label_index": row["label_index"],
                "final_lift_height": row["final_lift_height"],
                "max_lift_height": row["max_lift_height"],
                "final_target_distance": row["final_target_distance"],
                "final_finger_groups": row["final_finger_groups"],
                "final_contact_force": row["final_contact_force"],
                "failure_mode": (
                    "never_lifted_10cm" if row["max_lift_height"] < 0.10 else
                    "lifted_then_lost" if row["final_lift_height"] < 0.10 else
                    "center_error" if row["final_target_distance"] >= 0.10 else
                    "other"
                ),
            })
        if row["final_contact_force"] >= force_cutoff:
            force_rows.append({k: row[k] for k in (
                "object_id", "label_index", "paper_success", "final_contact_force",
                "final_lift_height", "final_target_distance", "final_finger_groups"
            )})

    object_rows = []
    for object_id, rows in grouped.items():
        n = len(rows)
        failure_modes = {m: 0 for m in ("never_lifted_10cm", "lifted_then_lost", "center_error", "other")}
        for row in rows:
            if row["paper_success"]:
                continue
            if row["max_lift_height"] < 0.10: failure_modes["never_lifted_10cm"] += 1
            elif row["final_lift_height"] < 0.10: failure_modes["lifted_then_lost"] += 1
            elif row["final_target_distance"] >= 0.10: failure_modes["center_error"] += 1
            else: failure_modes["other"] += 1
        object_rows.append({
            "object_id": object_id, "episodes": n,
            "success_rate": np.mean([r["paper_success"] for r in rows]),
            "failure_rate": 1.0 - np.mean([r["paper_success"] for r in rows]),
            "never_lifted_10cm": failure_modes["never_lifted_10cm"],
            "lifted_then_lost": failure_modes["lifted_then_lost"],
            "center_error": failure_modes["center_error"],
            "mean_final_force": np.mean([r["final_contact_force"] for r in rows]),
            "max_final_force": np.max([r["final_contact_force"] for r in rows]),
            "mean_final_lift": np.mean([r["final_lift_height"] for r in rows]),
        })
    object_rows.sort(key=lambda x: (-x["failure_rate"], -x["max_final_force"], x["object_id"]))
    failure_rows.sort(key=lambda x: (x["object_id"], x["label_index"]))
    force_rows.sort(key=lambda x: -x["final_contact_force"])

    def write_csv(name, rows):
        if not rows:
            return
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)

    write_csv("per_object_failures.csv", object_rows)
    write_csv("failed_labels.csv", failure_rows)
    write_csv("force_outliers.csv", force_rows)
    summary = {
        "objects": len(grouped), "episodes": len(episodes),
        "success_rate": float(np.mean([r["paper_success"] for r in episodes])),
        "force_percentiles": percentiles,
        "force_outlier_cutoff": force_cutoff,
        "force_outlier_count": len(force_rows),
        "worst_objects": object_rows[:15],
        "largest_force_outliers": force_rows[:20],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
