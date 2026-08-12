#!/usr/bin/python

import argparse
import glob
import os
import time
from typing import Callable, Dict, Tuple, Optional, List

import joblib
import numpy as np
import torch
import yaml
from ruamel.yaml import YAML, dump, RoundTripDumper

from dexgrasp_zero.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from dexgrasp_zero.env.bin.cross_hand import (
    NormalSampler,
    Allegro_Test,
    Shadow_Test,
    Leap_Test,
    Inspire_Test,
    Ability_Test,
    Schunk_Test,
    Brainco_Test,
    Barrett_Test,
)
from dexgrasp_zero.helper import rotations
from dexgrasp_zero.helper.constants import (
    HandType,
    get_cross_hand_spec,
    get_cross_hand_urdf_path,
    list_cross_hand_names,
    get_data_dict,
    observe_kwargs_for_hand_type,
)
from dexgrasp_zero.helper.initial_pose_final import get_initial_pose_faive_label, get_initial_pose_faive
from dexgrasp_zero.helper.raisim_gym_helper import ConfigurationSaver, load_param_for_eval
from dexgrasp_zero.helper.urdf_parser import URDFGraphParser
from dexgrasp_zero.helper.graph_data_builder import GraphDataBuilder
import dexgrasp_zero.algo.ppo.module as ppo_module
import dexgrasp_zero.algo.ppo.ppo as PPO
import dexgrasp_zero.algo.ppo.model_factory as model_factory

from dexgrasp_zero.env.test_scripts.eval_utils import (
    EvalSpec,
    RunningEval,
    compute_eval_metrics,
    compute_lifted_mask,
    print_batch_report,
    print_running_report,
)
from dexgrasp_zero.helper.utils import get_cross_hand_env_cfg



TASK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def _auto_pick_weight(weight_dir: str, weight_id: Optional[str]) -> str:
    weight_files = sorted(glob.glob(os.path.join(weight_dir, "full*_r.pt")))
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {weight_dir}")
    if weight_id is not None:
        for wf in weight_files:
            base = os.path.basename(wf)
            if base.startswith("full_") and base[len("full_") : -len("_r.pt")] == str(weight_id):
                return wf
        raise FileNotFoundError(f"No weight file with id {weight_id} found in {weight_dir}")

    def extract_id(wf: str) -> int:
        base = os.path.basename(wf)
        try:
            return int(base[len("full_") : -len("_r.pt")])
        except Exception:
            return -1

    return max(weight_files, key=extract_id)


def _auto_pick_model_cfg(weight_dir: str) -> str:
    for ext in ["yaml", "yml"]:
        files = sorted(glob.glob(os.path.join(weight_dir, f"model*.{ext}")))
        if files:
            return files[0]
    raise FileNotFoundError(f"No model config yaml (model*.yaml/yml) found in {weight_dir}")


HAND_ENV_CTORS: Dict[str, Callable] = {
    "allegro": Allegro_Test,
    "shadow": Shadow_Test,
    "leap": Leap_Test,
    "inspire": Inspire_Test,
    "ability": Ability_Test,
    "schunk": Schunk_Test,
    "brainco": Brainco_Test,
    "barrett": Barrett_Test,
}


def _build_data_dict(hand_type: int) -> Dict[int, dict]:
    # Use the unified get_data_dict (includes fields required by GraphDataBuilder, e.g., connections).
    flags = dict(
        TRAIN_LEAP=(hand_type == HandType.LEAP.value),
        TRAIN_ALLEGRO=(hand_type == HandType.ALLEGRO.value),
        TRAIN_SHADOW=(hand_type == HandType.SHADOW.value),
        TRAIN_SCHUNK=(hand_type == HandType.SCHUNK.value),
        TRAIN_INSPIRE=(hand_type == HandType.INSPIRE.value),
        TRAIN_ABILITY=(hand_type == HandType.ABILITY.value),
        TRAIN_BRAINCO=(hand_type == HandType.BRAINCO.value),
        TRAIN_BARRETT=(hand_type == HandType.BARRETT.value),
    )
    return get_data_dict(**flags)


def _reset_one_update(
    *,
    hand: str,
    hand_type: int,
    env: VecEnv,
    obj_list,
    update: int,
    directory_path: str,
    label_list: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    """Reset one evaluation update.

    Returns:
        contain_non_aff, obj_pose_reset, obs_new_r, obj_names_in_env, new_labels_to_save
    """
    num_envs = len(obj_list)
    act_dim = env.num_acts

    qpos_reset_r = np.zeros((num_envs, act_dim), dtype="float32")
    qpos_reset_l = np.zeros((num_envs, act_dim), dtype="float32")
    obj_pose_reset = np.zeros((num_envs, 8), dtype="float32")

    target_center = np.zeros_like(env.affordance_center)
    object_center = np.zeros_like(env.affordance_center)
    fake_non_aff_center = [0.346408, 0.346408, 0.346408]
    contain_non_aff = np.zeros((num_envs, 1), dtype="float32")

    obj_names_in_env = env.obj_list

    new_labels_to_save = {}

    for i in range(num_envs):
        # print(f"debug: processing env id :{i}, object: {obj_list[i]}")
        lowest_point = 0.0
        txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
        with open(txt_file_path, "r") as txt_file:
            lowest_point = float(txt_file.read())

        obj_pose_reset[i, :] = [1.0, -0.0, 0.502, 1.0, -0.0, -0.0, 0.0, 0.0]
        obj_pose_reset[i, 2] -= lowest_point

        # non-aff mesh
        if np.linalg.norm(env.non_aff_mesh[i].centroid - fake_non_aff_center) < 0.01:
            non_aff_mesh = None
        else:
            non_aff_mesh = env.non_aff_mesh[i]
            contain_non_aff[i, 0] = 1.0

        obj_idx = obj_list[i] + f"_{update}"
        
        current_label = label_list.get(obj_list[i], {}).get(obj_idx)

        # init qpos for hand
        if hand in ["allegro", "leap", "shadow"]:
            qpos_reset_r[i, -4] = 1.7
        elif hand == "schunk":
            qpos_reset_r[i, 6] = 1.0
        elif hand == "inspire":
            qpos_reset_r[i, 6] = 1.3
        elif hand == "ability":
            qpos_reset_r[i, 6] = -1.9
        elif hand == "brainco":
            qpos_reset_r[i, 6] = 0.9

        if current_label:
            rot, pos, bias = get_initial_pose_faive_label(
                env.aff_mesh[i],
                non_aff_mesh,
                hand,
                current_label["target_direction"],
                current_label["rand_offset"],
                current_label["target_center"],
            )
        else:
            rot, pos, bias, new_label_data = get_initial_pose_faive(
                env.aff_mesh[i],
                non_aff_mesh,
                hand,
                return_label_data=True,
            )
            if obj_list[i] not in new_labels_to_save:
                new_labels_to_save[obj_list[i]] = {}
            new_labels_to_save[obj_list[i]][obj_idx] = new_label_data


        # apply pose to wrist in world
        obj_mat = rotations.quat2mat(obj_pose_reset[i, 3:7])
        wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
        wrist_mat = rotations.euler2mat(wrist_pose_obj)
        wrist_in_world = np.matmul(obj_mat, wrist_mat)
        wrist_pose = rotations.mat2euler(wrist_in_world)
        qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat, pos[0, :])
        qpos_reset_r[i, 3:6] = wrist_pose[0, :]

        if hand_type == HandType.BARRETT.value:
            # Barrett wrist defaults: 0.3m above the object and pointing down.
            qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.array([0.0, 0, 0.3])
            qpos_reset_r[i, 3:6] = np.array([3.14, 0, 0.0])

        target_center[i, :] = bias[:]
        object_center[i, :] = env.affordance_center[i]

    env.reset_state(
        qpos_reset_r,
        qpos_reset_l,
        np.zeros((num_envs, act_dim), "float32"),
        np.zeros((num_envs, act_dim), "float32"),
        obj_pose_reset,
    )
    env.set_goals(target_center, object_center, *[np.zeros((num_envs, 1), "float32") for _ in range(8)])

    obs_new_r, _ = env.observe(contain_non_aff, **observe_kwargs_for_hand_type(hand_type))
    return contain_non_aff, obj_pose_reset, obs_new_r, obj_names_in_env, new_labels_to_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", "-hand", type=str, required=True, choices=list_cross_hand_names())
    parser.add_argument("--weight_dir", type=str, required=True)
    parser.add_argument("--weight_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--obj_num", type=int, default=-1)
    parser.add_argument("--obj_id", type=int, default=-1)
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--visualize", action="store_true", help="enable the RaiSim visualization server")
    parser.add_argument("--realtime", action="store_true", help="pace simulation steps to the control period")
    parser.add_argument(
        "--dataset",
        default="ycb",
        type=str,
        choices=["affordance_level", "ycb"],
        help="evaluation dataset (default: ycb, loaded from rsc/ycb_crossDex)",
    )
    parser.add_argument("--no_root_guide", action="store_true", help="whether to use root guidance during testing")
    args = parser.parse_args()

    spec = get_cross_hand_spec(args.hand)
    env_ctor = HAND_ENV_CTORS[args.hand]

    weight_file = _auto_pick_weight(args.weight_dir, args.weight_id)
    cfg_file = _auto_pick_model_cfg(args.weight_dir)

    print(f"Using weight file: {weight_file}")
    print(f"Using config file: {cfg_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg_file, "r") as f:
        model_config = yaml.safe_load(f)
    use_urdf = bool(model_config.get("use_urdf", False))
    print(f"use_urdf={use_urdf}")

    # paths / objects
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = os.path.abspath(os.path.join(task_path, "../../../../"))
    raisim_gym_torch_root = os.path.abspath(os.path.join(task_path, "../../.."))
    cat_name = args.dataset
    if cat_name == "ycb":
        cat_name = "ycb_crossDex"
    directory_path = os.path.join(home_path, f"rsc/{cat_name}/")
    items = os.listdir(directory_path)
    folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    obj_list = folder_names.copy()
    if args.obj_num > 0:
        if args.obj_id != -1:
            obj_list = [obj_list[args.obj_id]]
            obj_list = obj_list * args.obj_num
        else:
            obj_list = obj_list[: args.obj_num]
    if args.debug_mode:
        obj_list = obj_list[:2]

    num_envs = len(obj_list)

    urdf_path = get_cross_hand_urdf_path(home_path, args.hand)

    # load labels
    # Persist "target_direction", "rand_offset", and "target_center" to make evaluation deterministic.
    label_list = {}
    for obj in obj_list:
        label_pth = os.path.join(directory_path, obj) + "_label.pkl"
        if os.path.exists(label_pth):
            label_list[obj] = joblib.load(label_pth)
        else:
            print(f"No label found for {obj}, will generate poses.")

    overrides = None
    if args.no_root_guide:
        overrides = {"environment": {"root_guided": False}}
        print("Disable root guidance for evaluation.")

    cfg, _used_cfg_path = get_cross_hand_env_cfg(
        hand=args.hand,
        seed=args.seed,
        visualize=args.visualize,
        cat_name=cat_name,
        len_obj=num_envs,
        overrides=overrides,
    )

    # env
    env = VecEnv(obj_list, env_ctor(home_path + "/rsc", dump(cfg["environment"], Dumper=RoundTripDumper)), home_path, cat_name=cat_name)
    obj_path_list = [os.path.join(f"{obj_item}/{obj_item}.urdf") for obj_item in obj_list]
    env.load_multi_articulated(obj_path_list)
    print("env initialization finished")

    # data_dict + graph builder
    data_dict = _build_data_dict(spec.hand_type)
    data_dict = {k: v.copy() for k, v in data_dict.items()}
    max_obs_dim = data_dict[spec.hand_type]["obs_dim"]
    max_act_dim = data_dict[spec.hand_type]["act_dim"]

    if use_urdf:
        urdf_parser = URDFGraphParser()
        urdf_parser.parse_urdf(urdf_path, spec.hand_type)
    else:
        urdf_parser = None
    graph_builder = GraphDataBuilder(urdf_parser, data_dict, max_act_dim, version=model_config.get("version","v0"))

    actor_r = ppo_module.Actor(
        model_factory.build_model_from_config(
            model_config,
            graph_builder,
            model_role="actor",
            data_dict=data_dict
        ),
        ppo_module.MultivariateGaussianDiagonalCovariance(max_act_dim, num_envs, 1.0, NormalSampler(max_act_dim)),
        str(device),
    )


    saver = ConfigurationSaver(log_dir=args.weight_dir + "/eval_log/", save_items=[task_path + "/eval.py"], test_dir=True)


    load_param_for_eval(weight_file, env, actor_r.architecture, saver.data_dir)

    eval_running = RunningEval()
    eval_spec = None

    all_new_labels = {}

    for update in range(args.updates):
        print("update", update)
        start = time.time()

        contain_non_aff, obj_pose_reset, obs_new_r, obj_names_in_env, new_labels = _reset_one_update(
            hand=args.hand,
            hand_type=spec.hand_type,
            env=env,
            obj_list=obj_list,
            update=update,
            directory_path=directory_path,
            label_list=label_list,
        )
        # Update the collected labels
        for obj_name, labels in new_labels.items():
            if obj_name not in all_new_labels:
                all_new_labels[obj_name] = {}
            all_new_labels[obj_name].update(labels)


        for step in range(args.steps):
            obs_r = obs_new_r.astype("float32")
            if step == 130:
                env.switch_root_guidance(True)

            action_r = actor_r.architecture(torch.from_numpy(obs_r).to(device), spec.hand_type)
            action_r = action_r.cpu().detach().numpy()
            action_l = np.zeros_like(action_r)

            
            frame_start = time.time()
            env.step(action_r.astype("float32"), action_l.astype("float32"))

            observe_kwargs = observe_kwargs_for_hand_type(spec.hand_type)
            obs_new_r, _ = env.observe(contain_non_aff, **observe_kwargs)
            obs_new_r = obs_new_r.astype("float32")

            frame_end = time.time()
            wait_time = cfg["environment"]["control_dt"] - (frame_end - frame_start)
            if args.realtime and wait_time > 0.0:
                time.sleep(wait_time)

        obs_r, _ = env.observe(contain_non_aff, **observe_kwargs)
        obs_r = obs_r.astype("float32")

        global_state = env.get_global_state()
        if eval_spec is None:
            extra_dim =  0
            eval_spec = EvalSpec.from_runtime(
                act_dim=data_dict[spec.hand_type]["act_dim"],
                num_bodyparts=data_dict[spec.hand_type]["keypoints_num"],
                obs_dim=obs_r.shape[1],
                extra_dim=extra_dim,
            )

        lifted = compute_lifted_mask(eval_spec, obs_r, global_state, obj_pose_reset, lift_height_thresh=0.1, center_thresh=0.1)
        metrics = compute_eval_metrics(eval_spec, obs_r, global_state)
        print_batch_report(obj_names_in_env, lifted, metrics, global_state, eval_spec)
        eval_running.update_from_batch(lifted, metrics, update=update)
        print_running_report(eval_running)

        print("update wall time", time.time() - start)

    # Save any newly generated labels
    for obj_name, labels in all_new_labels.items():
        if labels: # only save if there are new labels
            save_path = os.path.join(directory_path, f"{obj_name}_label.pkl")
            # In case a file already exists, we should merge, but for now, we just save the new ones.
            # The logic assumes that if a label file doesn't exist at the start, we generate for all updates.
            print(f"Saving {len(labels)} generated labels for {obj_name} to {save_path}")
            joblib.dump(labels, save_path)

    print("done")


if __name__ == "__main__":
    main()
