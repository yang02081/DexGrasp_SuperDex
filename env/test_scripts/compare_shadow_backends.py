#!/usr/bin/env python3
"""Step-by-step Shadow_Test parity check between Raisim and SuperDex.

The same official pre-grasp label, checkpoint action and lift switch are replayed
in both backends. Raisim is the reference: its action is applied to both worlds,
while the action SuperDex would have produced is recorded as a diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
import trimesh
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
for path in (ROOT, WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Registers the checkout as ``dexgrasp_zero`` before importing the extension.
from eval_shadow_superdex import build_actor
from dexgrasp_zero.env.bin.cross_hand import Shadow_Test
from dexgrasp_zero.env.superdex.shadow_env import ShadowCrossHandSuperDexEnv
from dexgrasp_zero.helper.initial_pose_final import get_initial_pose_faive_label
from dexgrasp_zero.helper import rotations

from dexgrasp_zero.env.superdex.geometry import (
    nearest_vectors as _nearest_vectors,
    surface_points as _surface_points,
)


def make_reset(object_root: Path, object_id: str, label_index: int):
    object_dir = object_root / object_id
    top = trimesh.load_mesh(object_dir / "top_watertight_tiny.obj", force="mesh")
    bottom = trimesh.load_mesh(object_dir / "bottom_watertight_tiny.obj", force="mesh")
    fake = np.array([0.346408, 0.346408, 0.346408])
    non_aff = None if np.linalg.norm(bottom.centroid - fake) < 0.01 else bottom
    labels = joblib.load(object_root / f"{object_id}_label.pkl")
    key = f"{object_id}_{label_index % len(labels)}"
    label = labels[key]
    rot, pos, bias = get_initial_pose_faive_label(
        top, non_aff, "shadow", label["target_direction"],
        label["rand_offset"], label["target_center"],
    )
    lowest = float((object_dir / "lowest_point_new.txt").read_text())
    obj = np.array([1.0, 0.0, 0.502 - lowest, 1.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    obj_matrix = rotations.quat2mat(obj[3:7])
    wrist_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
    wrist_world = rotations.mat2euler(np.matmul(obj_matrix, rotations.euler2mat(wrist_obj)))[0]
    hand = np.zeros(28, np.float32)
    hand[-4] = 1.7
    hand[:3] = obj[:3] + np.matmul(obj_matrix, pos[0])
    hand[3:6] = wrist_world
    return hand, obj, np.asarray(bias, np.float32), np.asarray(top.centroid, np.float32), top, bottom, non_aff is not None


def extend_raisim_observation(base, global_state, top_surface, bottom_surface, contain_non_aff):
    joints = global_state[69:135].reshape(22, 3)
    object_to_wrist = Rotation.from_euler("XYZ", global_state[:3])
    affordance = object_to_wrist.apply(_nearest_vectors(joints, top_surface)).reshape(-1)
    non_affordance = object_to_wrist.apply(_nearest_vectors(joints, bottom_surface)).reshape(-1)
    if not contain_non_aff:
        non_affordance.fill(0.0)
    result = np.concatenate((base, affordance, non_affordance)).astype(np.float32)
    if result.shape != (268,) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid Raisim policy observation: {result.shape}")
    return result


def create_raisim(object_id: str, object_root: Path):
    cfg = yaml.safe_load((ROOT / "model/multi_hand/shadow_cfg_reg_p2.yaml").read_text())["environment"]
    cfg.update(num_envs=1, num_threads=1, visualize=False, load_set=object_root.name)
    env = Shadow_Test(str(object_root.parent), yaml.safe_dump(cfg))
    env.load_multi_articulated([f"{object_id}/{object_id}.urdf"])
    return env


def reset_raisim(env, hand, obj, goal, object_center):
    q = hand[None].astype(np.float32)
    zero = np.zeros_like(q)
    env.reset_state(q, np.zeros_like(q), zero, zero, obj[None].astype(np.float32))
    one = lambda x: np.asarray(x, np.float32).reshape(1, -1)
    dummy = np.zeros((1, 1), np.float32)
    env.set_goals(one(goal), one(object_center), dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy)


def read_raisim(env, top_surface, bottom_surface, contain_non_aff):
    base = np.zeros((1, 136), np.float32)
    left = np.zeros((1, 1), np.float32)
    gs = np.zeros((1, 179), np.float32)
    env.observe(base, left)
    env.get_global_state(gs)
    return extend_raisim_observation(base[0], gs[0], top_surface, bottom_surface, contain_non_aff), gs[0]


def step_raisim(env, action):
    right = action.reshape(1, 28).astype(np.float32)
    left = np.zeros_like(right)
    reward_r = np.zeros(1, np.float32)
    reward_l = np.zeros(1, np.float32)
    done = np.zeros(1, bool)
    env.step(right, left, reward_r, reward_l, done)
    return float(reward_r[0]), bool(done[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", default="058_golf_ball")
    parser.add_argument("--label-index", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lift-step", type=int, default=130)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/shadow_backend_parity.json")
    args = parser.parse_args()
    object_root = ROOT / "rsc/ycb_crossDex"
    hand, obj, goal, center, top_mesh, bottom_mesh, has_bottom = make_reset(
        object_root, args.object_id, args.label_index
    )
    top_surface = _surface_points(top_mesh)
    bottom_surface = _surface_points(bottom_mesh)
    actor = build_actor(ROOT / "model/multi_hand/full_2700_r.pt",
                        ROOT / "model/multi_hand/model_gnn_cfg_L10_v4.yaml", args.device)
    raisim = create_raisim(args.object_id, object_root)
    rows = []
    try:
        reset_raisim(raisim, hand, obj, goal, center)
        r_obs, r_gs = read_raisim(raisim, top_surface, bottom_surface, has_bottom)
        with ShadowCrossHandSuperDexEnv(object_id=args.object_id, object_root=object_root,
                                        max_steps=args.steps) as superdex:
            superdex.reset_state(hand, obj_pose=obj)
            superdex.set_goals(goal[None], center[None])
            s_obs = superdex.observe(np.array([[float(has_bottom)]], np.float32))[0][0]
            for step in range(args.steps):
                if step == args.lift_step:
                    raisim.switch_root_guidance(True)
                    superdex.switch_root_guidance(True)
                with torch.inference_mode():
                    r_action = actor(torch.from_numpy(r_obs[None]).to(args.device), 2).cpu().numpy()[0]
                    s_action = actor(torch.from_numpy(s_obs[None]).to(args.device), 2).cpu().numpy()[0]
                step_raisim(raisim, r_action)
                s_obs, _, _, _, s_info = superdex.step(r_action)
                r_obs, r_gs = read_raisim(raisim, top_surface, bottom_surface, has_bottom)
                rows.append({
                    "step": step + 1,
                    "base_obs_mae": float(np.mean(np.abs(r_obs[:136] - s_obs[:136]))),
                    "policy_obs_mae": float(np.mean(np.abs(r_obs - s_obs))),
                    "policy_obs_max": float(np.max(np.abs(r_obs - s_obs))),
                    "action_mae": float(np.mean(np.abs(r_action - s_action))),
                    "superdex_lift": float(s_info["lift_height"]),
                    "superdex_groups": int(s_info["finger_groups"]),
                })
    finally:
        raisim.close()
    result = {
        "object_id": args.object_id, "label_index": args.label_index,
        "steps": len(rows), "reference_actions": "raisim",
        "initial": rows[0] if rows else None,
        "final": rows[-1] if rows else None,
        "mean_base_obs_mae": float(np.mean([r["base_obs_mae"] for r in rows])),
        "mean_policy_obs_mae": float(np.mean([r["policy_obs_mae"] for r in rows])),
        "mean_action_mae": float(np.mean([r["action_mae"] for r in rows])),
        "trace": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
