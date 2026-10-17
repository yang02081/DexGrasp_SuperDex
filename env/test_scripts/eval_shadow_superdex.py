#!/usr/bin/env python3
"""Evaluate the unchanged DexGrasp-Zero MAGCN Shadow policy in SuperDex.

This is the SuperDex counterpart of ``env/test_scripts/eval.py --hand shadow``.
It deliberately builds the actor with the repository's original graph parser,
model factory and multi-hand checkpoint; only the physics backend is replaced.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The source checkout is named DexGrasp-Zero-code while internal imports use
# the installed package name dexgrasp_zero.  Register the checkout as that
# package so the evaluator also works before an editable installation.
if "dexgrasp_zero" not in sys.modules:
    package_spec = importlib.util.spec_from_file_location(
        "dexgrasp_zero", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["dexgrasp_zero"] = package
    package_spec.loader.exec_module(package)

from dexgrasp_zero.algo.ppo import model_factory
from dexgrasp_zero.helper.constants import HandType, get_cross_hand_urdf_path, get_data_dict
from dexgrasp_zero.helper.graph_data_builder import GraphDataBuilder
from dexgrasp_zero.helper.initial_pose_final import get_initial_pose_faive_label
from dexgrasp_zero.helper import rotations
from dexgrasp_zero.helper.urdf_parser import URDFGraphParser
from dexgrasp_zero.env.superdex.shadow_env import ShadowCrossHandSuperDexEnv


def _official_model_config():
    """Dependency-free form of model/multi_hand/model_gnn_cfg_L10_v4.yaml."""
    return {
        "model_type": "gnn", "version": "v4", "use_urdf": True,
        "gnn_params": {
            "node_dim": 16, "edge_dim": 1, "global_dim": 24,
            "gcn_layers": [
                {"out_dim": 128, "activation": "relu", "norm": "layer"},
                *[{"out_dim": 256, "activation": "relu", "norm": "layer"} for _ in range(7)],
            ],
            "global_mlp": {"hidden_dims": [256], "activation": "leaky_relu"},
            "rt_head": {"hidden_dims": [128], "activation": "leaky_relu", "out_dim": 6},
            "joint_head": {"hidden_dims": [128], "activation": "leaky_relu", "out_dim": 3},
        },
        "infoB_config": {
            "input_dim": 33, "use_fusion_mlp": True,
            "fusion_mlp": {"embedded_dim": 16, "hidden_dims": [32], "activation": "leaky_relu"},
        },
        "infoC_config": {"input_dim": 3},
        "critic_gnn_params": {
            "value_head": {"hidden_dims": [128], "out_dim": 1, "activation": "relu"},
        },
    }


def build_actor(checkpoint: Path, model_config: Path, device: str):
    if not model_config.is_file():
        raise FileNotFoundError(model_config)
    config = _official_model_config()
    data_dict = get_data_dict(TRAIN_SHADOW=True)
    parser = URDFGraphParser()
    parser.parse_urdf(get_cross_hand_urdf_path(str(ROOT), "shadow"), HandType.SHADOW.value)
    graph_builder = GraphDataBuilder(
        parser, data_dict, data_dict[HandType.SHADOW.value]["act_dim"],
        version=config.get("version", "v4"),
    )
    actor = model_factory.build_model_from_config(
        config, graph_builder, model_role="actor", data_dict=data_dict,
    ).to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(payload["actor_architecture_state_dict"], strict=True)
    return actor


def normalize_observation(obs, mean, variance):
    return np.clip((obs - mean) / np.sqrt(variance + 1e-8), -10.0, 10.0).astype(np.float32)


def official_shadow_reset(env, *, object_root: Path, object_id: str, update: int):
    """Replay the exact labelled pre-grasp construction used by eval.py."""
    object_dir = object_root / object_id
    top_mesh = trimesh.load_mesh(object_dir / "top_watertight_tiny.obj", force="mesh")
    bottom_mesh = trimesh.load_mesh(object_dir / "bottom_watertight_tiny.obj", force="mesh")
    fake_non_aff_center = np.array([0.346408, 0.346408, 0.346408])
    non_aff_mesh = None if np.linalg.norm(bottom_mesh.centroid - fake_non_aff_center) < 0.01 else bottom_mesh

    labels = joblib.load(object_root / f"{object_id}_label.pkl")
    label_key = f"{object_id}_{update % len(labels)}"
    if label_key not in labels:
        raise KeyError(f"missing official pre-grasp label {label_key}")
    label = labels[label_key]
    rot, pos, bias = get_initial_pose_faive_label(
        top_mesh, non_aff_mesh, "shadow",
        label["target_direction"], label["rand_offset"], label["target_center"],
    )

    lowest_point = float((object_dir / "lowest_point_new.txt").read_text())
    object_pose = np.array([1.0, 0.0, 0.502 - lowest_point, 1.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    object_matrix = rotations.quat2mat(object_pose[3:7])
    wrist_pose_object = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
    wrist_matrix = rotations.euler2mat(wrist_pose_object)
    wrist_pose_world = rotations.mat2euler(np.matmul(object_matrix, wrist_matrix))[0]

    hand_pose = np.zeros(28, np.float32)
    hand_pose[-4] = 1.7
    hand_pose[:3] = object_pose[:3] + np.matmul(object_matrix, pos[0])
    hand_pose[3:6] = wrist_pose_world
    env.reset_state(hand_pose, obj_pose=object_pose)
    env.set_goals(np.asarray(bias, np.float32).reshape(1, 3),
                  np.asarray(top_mesh.centroid, np.float32).reshape(1, 3))
    observation, _ = env.observe(np.array([[1.0 if non_aff_mesh is not None else 0.0]], np.float32), shadow=True)
    return observation[0]


def evaluate(args):
    actor = build_actor(args.checkpoint, args.model_config, args.device)
    # Official RaisimGymVecEnvTest is constructed with normalize_ob=False;
    # its evaluator loads these files but passes raw observations to the actor.
    # Keep the artifacts as provenance checks without applying them here.
    if not args.mean.is_file() or not args.variance.is_file():
        raise FileNotFoundError("multi-hand normalization artifacts are missing")

    rows = []
    traces = []
    with ShadowCrossHandSuperDexEnv(
        object_id=args.object_id, object_root=args.object_root, seed=args.seed, max_steps=args.steps,
        root_guided=not args.no_root_guide,
        freeze_wrist_rotation=args.freeze_wrist_rotation,
        tracking_stiffness=args.tracking_stiffness,
        tracking_damping=args.tracking_damping,
        tracking_saturation=args.tracking_saturation,
        contact_friction=args.contact_friction,
        contact_normal_damping=args.contact_normal_damping,
        contact_penalty_coefficient=args.contact_penalty_coefficient,
        contact_smoothing_half_distance=args.contact_smoothing_half_distance,
        contact_max_alignment_normals=args.contact_max_alignment_normals,
        lift_guidance_height=args.lift_guidance_height,
        lift_ramp_steps=args.lift_ramp_steps,
        physics_substeps=args.physics_substeps,
        strict_physics_alignment=not args.legacy_physics_approximation,
        record_substep_diagnostics=args.substep_diagnostics,
        hand_urdf_override=args.hand_urdf,
        geometry_sample_seed=args.geometry_sample_seed,
    ) as env:
        if args.visualize:
            if not env.attach_debugger():
                raise RuntimeError("SuperDex debugger GUI failed to attach")
            print("SuperDex GUI 已连接；关闭 GUI 窗口即可停止可视化。")
        for episode in range(args.episodes):
            env.seed(args.seed + episode)
            obs = official_shadow_reset(
                env, object_root=args.object_root, object_id=args.object_id,
                update=args.label_index + episode,
            )
            max_lift = 0.0
            max_groups = 0
            initial_distance = float(env.last_info["target_distance"])
            min_distance = initial_distance
            final_distance = initial_distance
            min_object_height = float(env.last_info["object_height"])
            max_action_abs = 0.0
            final_lift = 0.0
            final_groups = 0
            final_contact_force = 0.0
            peak_contact_force = 0.0
            peak_object_linear_speed = 0.0
            peak_object_angular_speed = 0.0
            episode_trace = []
            paper_hold_streak = 0
            paper_max_hold_steps = 0
            paper_required_hold_steps = int(round(
                args.paper_hold_seconds / 0.01
            ))
            for step in range(args.steps):
                step_started = time.perf_counter()
                if args.visualize and not env.debugger_is_attached():
                    print("SuperDex GUI 已关闭，停止评估。")
                    return None
                if step == args.lift_step and not args.disable_lift_guidance:
                    env.switch_root_guidance(True)
                with torch.inference_mode():
                    action = actor(
                        torch.from_numpy(obs[None]).to(args.device),
                        HandType.SHADOW.value,
                    ).cpu().numpy()[0]
                if args.freeze_wrist_rotation or args.zero_wrist_rotation_action:
                    action[3:6] = 0.0
                max_action_abs = max(max_action_abs, float(np.max(np.abs(action))))
                obs, _, terminated, truncated, info = env.step(action)
                max_lift = max(max_lift, float(info["lift_height"]))
                max_groups = max(max_groups, int(info["finger_groups"]))
                final_distance = float(info["target_distance"])
                min_distance = min(min_distance, final_distance)
                min_object_height = min(min_object_height, float(info["object_height"]))
                final_lift = float(info["lift_height"])
                final_groups = int(info["finger_groups"])
                final_contact_force = float(info["contact_force"])
                peak_contact_force = max(peak_contact_force, final_contact_force)
                peak_object_linear_speed = max(peak_object_linear_speed, float(info["object_linear_speed"]))
                peak_object_angular_speed = max(peak_object_angular_speed, float(info["object_angular_speed"]))
                if float(info["lift_height"]) >= args.paper_lift_height:
                    paper_hold_streak += 1
                    paper_max_hold_steps = max(
                        paper_max_hold_steps, paper_hold_streak
                    )
                else:
                    paper_hold_streak = 0
                if args.trace_output is not None:
                    episode_trace.append({
                        "step": int(step),
                        "lift_height": float(info["lift_height"]),
                        "target_distance": float(info["target_distance"]),
                        "finger_groups": int(info["finger_groups"]),
                        "contact_force": float(info["contact_force"]),
                        "wrist_linear_speed": float(info["wrist_linear_speed"]),
                        "object_linear_speed": float(info["object_linear_speed"]),
                        "object_vertical_velocity": float(info["object_vertical_velocity"]),
                        "object_angular_speed": float(info["object_angular_speed"]),
                        "control_error_norm": float(info["control_error_norm"]),
                        "root_control_error_norm": float(info["root_control_error_norm"]),
                        "lift_ratio": float(info["lift_ratio"]),
                        "action_max_abs": float(np.max(np.abs(action))),
                        "action": action.tolist(),
                        "hand_state": info["hand_state"].tolist(),
                        "control_target": info["control_target"].tolist(),
                        "control_error": info["control_error"].tolist(),
                        "controller_force": info["controller_force"].tolist(),
                        "wrist_controller_force_norm": float(info["wrist_controller_force_norm"]),
                        "finger_controller_force_norm": float(info["finger_controller_force_norm"]),
                        "frame_surface_distance": info["frame_surface_distance"].tolist(),
                        "finger_surface_distance": info["finger_surface_distance"].tolist(),
                        "physics_substeps": info.get("substeps", []),
                    })
                if terminated or truncated:
                    break
                if args.visualize and args.realtime_factor > 0:
                    frame_seconds = 0.01 / args.realtime_factor
                    time.sleep(max(0.0, frame_seconds - (time.perf_counter() - step_started)))
            # Original DexGrasp-Zero success is evaluated at the final
            # state: object height gain exceeds the threshold and the final
            # target-center error is below 10 cm. Maximum lift remains a
            # diagnostic because a collision can launch an ungrasped object.
            final_success_4cm = final_lift >= 0.04 and final_distance < 0.10
            final_success_10cm = final_lift >= 0.10 and final_distance < 0.10
            held_contact = final_groups >= 1 and final_contact_force >= 0.1
            stable_contact = final_groups >= 2 and final_contact_force >= 0.1
            interaction_centered = final_distance < 0.10
            interaction_held_success = interaction_centered and held_contact
            interaction_stable_success = interaction_centered and stable_contact
            held_success_4cm = final_success_4cm and held_contact
            held_success_10cm = final_success_10cm and held_contact
            stable_success_10cm = final_success_10cm and stable_contact
            protocol_success = (
                paper_max_hold_steps >= paper_required_hold_steps
                if args.paper_protocol else final_success_10cm
            )
            if args.trace_output is not None:
                traces.append({
                    "episode": episode,
                    "seed": args.seed + episode,
                    "label_index": args.label_index + episode,
                    "steps": episode_trace,
                })
            rows.append({
                "episode": episode,
                "seed": args.seed + episode,
                "label_index": args.label_index + episode,
                "steps": step + 1,
                "max_lift_height": max_lift,
                "final_lift_height": final_lift,
                "max_finger_groups": max_groups,
                "final_finger_groups": final_groups,
                "final_contact_force": final_contact_force,
                "peak_contact_force": peak_contact_force,
                "peak_object_linear_speed": peak_object_linear_speed,
                "peak_object_angular_speed": peak_object_angular_speed,
                "initial_target_distance": initial_distance,
                "min_target_distance": min_distance,
                "final_target_distance": final_distance,
                "min_object_height": min_object_height,
                "max_action_abs": max_action_abs,
                "max_lift_success_4cm": bool(max_lift >= 0.04),
                "max_lift_success_10cm": bool(max_lift >= 0.10),
                "success_4cm": bool(final_success_4cm),
                "success_10cm": bool(final_success_10cm),
                "paper_success": bool(protocol_success),
                "paper_protocol_success": bool(protocol_success) if args.paper_protocol else None,
                "paper_lift_height": float(args.paper_lift_height),
                "paper_required_hold_steps": int(paper_required_hold_steps),
                "paper_max_hold_steps": int(paper_max_hold_steps),
                "paper_max_hold_seconds": float(paper_max_hold_steps * 0.01),
                "held_contact": bool(held_contact),
                "stable_contact": bool(stable_contact),
                "interaction_centered": bool(interaction_centered),
                "interaction_held_success": bool(interaction_held_success),
                "interaction_stable_success": bool(interaction_stable_success),
                "held_success_4cm": bool(held_success_4cm),
                "held_success_10cm": bool(held_success_10cm),
                "stable_success_10cm": bool(stable_success_10cm),
            })

    result = {
        "backend": "superdex",
        "hand": "shadow",
        "checkpoint": str(args.checkpoint),
        "observation_dim": 268,
        "action_dim": 28,
        "episodes": len(rows),
        "evaluation_protocol": {
            "name": ("interaction_no_lift" if args.disable_lift_guidance else
                     ("crossdex_ycb45_paper" if args.paper_protocol else "legacy_10cm")),
            "paper_protocol": bool(args.paper_protocol),
            "lift_guidance_enabled": not args.disable_lift_guidance,
            "lift_height_m": float(args.paper_lift_height),
            "hold_seconds": float(args.paper_hold_seconds),
            "hold_control_steps": int(round(args.paper_hold_seconds / 0.01)),
            "control_dt": 0.01,
            "success_rule": ("final_center_error_below_10cm_and_final_contact"
                             if args.disable_lift_guidance else
                             ("lift_height_at_least_0.5m_continuously_for_2s"
                              if args.paper_protocol else "final_lift_10cm_and_center_error_below_10cm")),
        },
        "interface_config": {
            "tracking_stiffness": args.tracking_stiffness,
            "wrist_rotation_frozen": bool(args.freeze_wrist_rotation),
            "wrist_rotation_action_zeroed": bool(args.zero_wrist_rotation_action),
            "tracking_damping": args.tracking_damping,
            "wrist_translation_damping": 0.1,
            "tracking_saturation": args.tracking_saturation,
            "contact_friction": args.contact_friction,
            "contact_normal_damping": args.contact_normal_damping,
            "contact_penalty_coefficient": args.contact_penalty_coefficient,
            "contact_smoothing_half_distance": args.contact_smoothing_half_distance,
            "contact_max_alignment_normals": args.contact_max_alignment_normals,
            "lift_guidance_height": args.lift_guidance_height,
            "lift_ramp_steps": args.lift_ramp_steps,
            "physics_substeps": args.physics_substeps,
            "physics_dt": 0.01 / args.physics_substeps,
            "strict_physics_alignment": not args.legacy_physics_approximation,
            "hand_urdf_override": None if args.hand_urdf is None else str(args.hand_urdf),
            "geometry_sampling": env._geometry_sampling,
            "collision_geometry": ("raisim_box_meshes" if not args.legacy_physics_approximation
                                   else "superdex_convex"),
            "table_support": ("physical_plane_actor_filter" if not args.legacy_physics_approximation
                              else "height_constraint"),
            "contact_impulse": ("contact_point_force_x_0.0025" if not args.legacy_physics_approximation
                                else "net_force_x_0.0025"),
            "friction_falloff_velocity": 0.1,
            "lift_step": args.lift_step,
            "lift_guidance_enabled": not args.disable_lift_guidance,
        },
        "success_rate_4cm": float(np.mean([r["success_4cm"] for r in rows])),
        "success_rate_10cm": float(np.mean([r["success_10cm"] for r in rows])),
        "paper_success_rate": float(np.mean([r["paper_success"] for r in rows])),
        "held_success_rate_4cm": float(np.mean([r["held_success_4cm"] for r in rows])),
        "held_success_rate_10cm": float(np.mean([r["held_success_10cm"] for r in rows])),
        "stable_success_rate_10cm": float(np.mean([r["stable_success_10cm"] for r in rows])),
        "final_contact_rate": float(np.mean([r["held_contact"] for r in rows])),
        "final_stable_contact_rate": float(np.mean([r["stable_contact"] for r in rows])),
        "interaction_centered_rate": float(np.mean([r["interaction_centered"] for r in rows])),
        "interaction_held_success_rate": float(np.mean([r["interaction_held_success"] for r in rows])),
        "interaction_stable_success_rate": float(np.mean([r["interaction_stable_success"] for r in rows])),
        "max_lift_success_rate_4cm": float(np.mean([r["max_lift_success_4cm"] for r in rows])),
        "max_lift_success_rate_10cm": float(np.mean([r["max_lift_success_10cm"] for r in rows])),
        "mean_final_lift_height": float(np.mean([r["final_lift_height"] for r in rows])),
        "mean_final_center_error": float(np.mean([r["final_target_distance"] for r in rows])),
        "mean_max_lift_height": float(np.mean([r["max_lift_height"] for r in rows])),
        "mean_peak_contact_force": float(np.mean([r["peak_contact_force"] for r in rows])),
        "max_peak_contact_force": float(np.max([r["peak_contact_force"] for r in rows])),
        "mean_peak_object_linear_speed": float(np.mean([r["peak_object_linear_speed"] for r in rows])),
        "mean_peak_object_angular_speed": float(np.mean([r["peak_object_angular_speed"] for r in rows])),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps({
            "object_id": args.object_id,
            "lift_step": args.lift_step,
            "lift_ramp_steps": args.lift_ramp_steps,
            "episodes": traces,
        }, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    model_dir = ROOT / "model" / "multi_hand"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=model_dir / "full_2700_r.pt")
    parser.add_argument("--model-config", type=Path, default=model_dir / "model_gnn_cfg_L10_v4.yaml")
    parser.add_argument("--mean", type=Path, default=model_dir / "mean_r2700.csv")
    parser.add_argument("--variance", type=Path, default=model_dir / "var_r2700.csv")
    parser.add_argument("--object-id", default="058_golf_ball")
    parser.add_argument("--object-root", type=Path, default=ROOT / "rsc" / "ycb_crossDex")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None,
                        help="control steps; defaults to 400 for --paper-protocol, otherwise 200")
    parser.add_argument("--lift-step", type=int, default=130)
    parser.add_argument(
        "--disable-lift-guidance", action="store_true",
        help="keep root target/orientation tracking but do not add the lift target",
    )
    parser.add_argument("--paper-protocol", action="store_true",
                        help="use the CrossDex/DexGrasp-Zero paper metric: lift 0.5 m and hold for 2 s")
    parser.add_argument("--paper-lift-height", type=float, default=0.5)
    parser.add_argument("--paper-hold-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label-index", type=int, default=0,
                        help="first official pre-grasp label to replay (default: 0)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-root-guide", action="store_true")
    parser.add_argument(
        "--freeze-wrist-rotation", action="store_true",
        help="diagnostic: hold wrist rotation at its reset orientation",
    )
    parser.add_argument(
        "--zero-wrist-rotation-action", action="store_true",
        help="diagnostic: zero policy action[3:6] but retain root orientation guidance",
    )
    parser.add_argument("--tracking-stiffness", type=float, default=100.0)
    parser.add_argument("--tracking-damping", type=float, default=2.0)
    parser.add_argument("--tracking-saturation", type=float, default=-1.0)
    parser.add_argument("--contact-friction", type=float, default=3.0)
    parser.add_argument("--contact-normal-damping", type=float, default=40.0)
    parser.add_argument("--contact-penalty-coefficient", type=float, default=None)
    parser.add_argument("--contact-smoothing-half-distance", type=float, default=None)
    parser.add_argument("--contact-max-alignment-normals", type=float, default=None)
    parser.add_argument("--lift-guidance-height", type=float, default=None,
                        help="wrist lift command; defaults to 0.5 m for --paper-protocol, otherwise 0.2 m")
    parser.add_argument("--lift-ramp-steps", type=int, default=0,
                        help="smooth lift target over this many control steps; 0 matches Raisim")
    parser.add_argument("--physics-substeps", type=int, default=4,
                        help="physics integrations per 10 ms control step (default: Raisim-compatible 4)")
    parser.add_argument("--geometry-sample-seed", type=int, default=None,
                        help="use trimesh.sample_surface with this deterministic seed")
    parser.add_argument("--hand-urdf", type=Path, default=None,
                        help="optional Shadow URDF override for controlled A/B tests")
    parser.add_argument("--legacy-physics-approximation", action="store_true",
                        help="use the earlier convex geometry and height-constraint table")
    parser.add_argument("--trace-output", type=Path, default=None,
                        help="optional per-control-step trajectory JSON")
    parser.add_argument("--substep-diagnostics", action="store_true",
                        help="record contact points and velocities after every physics substep")
    parser.add_argument("--visualize", action="store_true",
                        help="attach the SuperDex debugger GUI")
    parser.add_argument("--realtime-factor", type=float, default=0.25,
                        help="GUI playback speed; 0 disables pacing (default: 0.25x)")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "shadow_superdex" / "evaluation.json")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 400 if args.paper_protocol else 200
    if args.lift_guidance_height is None:
        args.lift_guidance_height = 0.5 if args.paper_protocol else 0.2
    required = int(round(args.paper_hold_seconds / 0.01))
    if args.paper_protocol and args.steps - args.lift_step < required:
        parser.error(
            "--paper-protocol requires at least lift_step + "
            f"{required} steps to measure the requested hold duration"
        )
    return args


if __name__ == "__main__":
    evaluate(parse_args())
