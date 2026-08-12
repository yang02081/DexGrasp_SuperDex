import numpy as np
from dexgrasp_zero.helper import rotations
import argparse
# import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch
import random
import os
from ruamel.yaml import YAML
from typing import Optional, Dict


from dexgrasp_zero.helper.constants import build_dof_to_finger_id_map,\
        build_joint_semantic_map,\
        build_node_to_joint_sequence,\
    FingerIdType, FLEX, HandType, get_cross_hand_spec, get_finger_node_indices, observe_kwargs_for_hand_type
from dexgrasp_zero.helper.initial_pose_final import get_initial_pose_faive   

try:
    import open3d as o3d
except ImportError:  # Optional visualization dependency.
    o3d = None
import yaml
import glob


def find_weight_and_modelcfg(weight_dir, weight_id=None):
    weight_files = sorted(glob.glob(os.path.join(weight_dir, 'full*_r.pt')))
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {weight_dir}")
    # Parse weight id
    if weight_id is not None:
        # Exact match
        weight_file = None
        for wf in weight_files:
            base = os.path.basename(wf)
            if base.startswith('full_') and base[len('full_'):-len('_r.pt')] == str(weight_id):
                weight_file = wf
                break
        if weight_file is None:
            raise FileNotFoundError(f"No weight file with id {weight_id} found in {weight_dir}")
    else:
        # Pick the largest id
        def extract_id(wf):
            base = os.path.basename(wf)
            try:
                return int(base[len('full_'):-len('_r.pt')])
            except Exception:
                return -1
        weight_file = max(weight_files, key=extract_id)
    print(f"Using weight file: {weight_file}")

    # Find model cfg (only model*.yaml/yml)
    model_cfg_path = None
    for ext in ['yaml', 'yml']:
        files = sorted(glob.glob(os.path.join(weight_dir, f'model*.{ext}')))
        if files:
            model_cfg_path = files[0]
            break
    if not model_cfg_path:
        raise FileNotFoundError(f"No model config yaml (model*.yaml/yml) found in {weight_dir}")
    print(f"Using config file: {model_cfg_path}")
    with open(model_cfg_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    return weight_file, model_config,model_cfg_path

def get_combined_obs(obs_new_r, object_pose_initial, object_names_in_env, pointnet_features_dict, data_dict, hand_order, max_obs_dim):
    combined_obs_list = []
    for hand_id, hand_type in enumerate(hand_order):
        start_idx = sum(data_dict[ht]["num_envs_per_hand"] for ht in hand_order[:hand_id])
        end_idx = start_idx + data_dict[hand_type]["num_envs_per_hand"]

        obs_hand = obs_new_r[start_idx:end_idx]
        object_pose_hand = object_pose_initial[start_idx:end_idx]
        object_names_hand = object_names_in_env[start_idx:end_idx]

        current_hand_combined_obs = []
        for i in range(obs_hand.shape[0]):
            obj_name = object_names_hand[i]
            if obj_name in pointnet_features_dict:
                pointnet_feature = pointnet_features_dict[obj_name][0].cpu().numpy().flatten()
                # Concatenate 512-dim PointNet++ feature and 7-dim object initial pose
                combined_feature = np.concatenate((pointnet_feature, object_pose_hand[i, :7]))
                current_hand_combined_obs.append(np.concatenate((obs_hand[i], combined_feature)))
            else:
                # If feature not found, pad with zeros or handle as an error
                padding = np.zeros(512 + 7)
                current_hand_combined_obs.append(np.concatenate((obs_hand[i], padding)))

        combined_obs_list.extend(current_hand_combined_obs)

    # Pad to max_obs_dim if necessary
    combined_obs_padded = np.array(combined_obs_list, dtype=np.float32)
    if combined_obs_padded.shape[1] < max_obs_dim:
        combined_obs_padded = np.pad(combined_obs_padded, ((0, 0), (0, max_obs_dim - combined_obs_padded.shape[1])), mode='constant', constant_values=0)
    return combined_obs_padded

def get_reward_info(reward_r, env, hand_type, dis_info, obs_new_r, cfg, finger_weights, device, env_mode):
    # dis_info ： [min_dis_aff (N x node_num), min_dis_non_aff (N x node_num)]
    node_num = int(finger_weights.shape[1])
    affordance_reward_r = -np.sum((dis_info[:, :node_num]) * finger_weights, axis=1)
    non_affordance_reward_r = -np.sum((dis_info[:, node_num:node_num*2]) * finger_weights, axis=1)

    direction_loss = np.sum(np.square(obs_new_r[:, 3:6]), axis=1)
    center_loss = np.sum(np.square(obs_new_r[:, :3]), axis=1)

    rewards_r = env.get_reward_info_r()

    for i in range(env.num_envs):
        rewards_r[i].update({
            'affordance_reward': affordance_reward_r[i] * cfg['environment']['reward']['affordance_reward']['coeff'],
            'not_affordance_reward': non_affordance_reward_r[i] * cfg['environment']['reward']['not_affordance_reward']['coeff'],
            'direction_reward': direction_loss[i] * cfg['environment']['reward']['direction_reward']['coeff'],
            'center_reward': center_loss[i] * cfg['environment']['reward']['center_reward']['coeff']
        })
        
        if env_mode == "floating":
            # Add velocity-related reward shaping (only used in the second stage).
            obj_vel_pul = rewards_r[i]['obj_vel_reward_']
            if obj_vel_pul < -0.75:
                obj_vel_pul = (obj_vel_pul + 0.75) / 4.0 - 0.75
            if obj_vel_pul < -1.0:
                obj_vel_pul = -1.0
            obj_qvel_pul = rewards_r[i]['obj_qvel_reward_']
            if obj_qvel_pul < -0.75:
                obj_qvel_pul = (obj_qvel_pul + 0.75) / 4.0 - 0.75
            if obj_qvel_pul < -1.0:
                obj_qvel_pul = -1.0

            wrist_vel_pul = rewards_r[i]['wrist_vel_reward_']
            if wrist_vel_pul < -0.75:
                wrist_vel_pul = (wrist_vel_pul + 0.75) / 4.0 - 0.75
            if wrist_vel_pul < -1.0:
                wrist_vel_pul = -1.0
            wrist_qvel_pul = rewards_r[i]['wrist_qvel_reward_']
            if wrist_qvel_pul < -0.75:
                wrist_qvel_pul = (wrist_qvel_pul + 0.75) / 4.0 - 0.75
            if wrist_qvel_pul < -1.0:
                wrist_qvel_pul = -1.0        
            rewards_r[i]['obj_vel_reward_'] = -rewards_r[i]['obj_vel_reward_'] + obj_vel_pul
            rewards_r[i]['obj_qvel_reward_'] = - rewards_r[i]['obj_qvel_reward_'] + obj_qvel_pul
            rewards_r[i]['wrist_vel_reward_'] = - rewards_r[i]['wrist_vel_reward_'] + wrist_vel_pul
            rewards_r[i]['wrist_qvel_reward_'] = - rewards_r[i]['wrist_qvel_reward_'] + wrist_qvel_pul
            rewards_r[i]['reward_sum'] = sum(rewards_r[i][k] for k in [
                'reward_sum', 
                'affordance_reward',  # distance-to-affordance reward per joint
                'not_affordance_reward',
                'direction_reward',
                'center_reward',
                "obj_vel_reward_",
                "obj_qvel_reward_",
                "wrist_vel_reward_",
                "wrist_qvel_reward_",
                ] if k is not None) 
        elif env_mode=="fixed":
            # Compute total reward
            rewards_r[i]['reward_sum'] = sum(rewards_r[i][k] for k in [
                'reward_sum', 
                'affordance_reward',
                'not_affordance_reward',
                'direction_reward',
                'center_reward',
            ] if k is not None)
        else:
            raise ValueError(f"env_mode :{env_mode} unknown")
        reward_r[i] = rewards_r[i]['reward_sum']

    return reward_r, rewards_r


def parse_observation_right(
    obs_r: np.ndarray,
    *,
    act_dim: int,
    num_contacts: int,
    keypoints_num: int,
):
    """Parse the right-hand observation vector strictly following the SOP (`obDouble_r_ << ...`).

    SOP definition (C++ side `obDouble_r_`):
        target_center_dif            3
        wrist_ori_error              3
        gc_r_.tail(gcDim_-6)         act_dim-6   (finger joints only; excludes wrist 6DoF)
        torque_error                 act_dim
        wrist_vel                    3
        wrist_qvel                   3
        contacts_r_af                num_contacts
        impulses_r_af                num_contacts
        obj_vel_in_wrist             3
        obj_qvel_in_wrist            3
        contacts_r_non_af            num_contacts
        impulses_r_non_af            num_contacts

    Then Python-side `observe()` appends:
        af_vec                       3*keypoints_num
        non_aff_vec_obs              3*keypoints_num

    Returns:
        A dict of per-field slices (numpy views/copies).
    """
    if obs_r.ndim != 2:
        raise ValueError(f"obs_r must be 2D (N,D), got {obs_r.ndim}D")
    n = act_dim
    N = num_contacts
    K = keypoints_num

    base_dim = 12 + 2 * n + 4 * N
    expected_total_min = base_dim + 6 * K
    if obs_r.shape[1] < expected_total_min:
        raise ValueError(
            f"obs_r dim too small: got {obs_r.shape[1]}, expected >= {expected_total_min} (base={base_dim},K={K})"
        )

    off = 0
    target_center_dif = obs_r[:, off:off + 3]; off += 3
    wrist_ori_error = obs_r[:, off:off + 3]; off += 3

    finger_dof = n - 6
    if finger_dof < 0:
        raise ValueError(f"act_dim must be >=6, got {n}")
    finger_joints = obs_r[:, off:off + finger_dof]; off += finger_dof

    torque_error = obs_r[:, off:off + n]; off += n
    wrist_vel = obs_r[:, off:off + 3]; off += 3
    wrist_qvel = obs_r[:, off:off + 3]; off += 3
    contacts_af = obs_r[:, off:off + N]; off += N
    impulses_af = obs_r[:, off:off + N]; off += N
    obj_vel_in_wrist = obs_r[:, off:off + 3]; off += 3
    obj_qvel_in_wrist = obs_r[:, off:off + 3]; off += 3
    contacts_non_af = obs_r[:, off:off + N]; off += N
    impulses_non_af = obs_r[:, off:off + N]; off += N

    # appended vectors
    af_vec = obs_r[:, off:off + 3 * K]; off += 3 * K
    non_aff_vec_obs = obs_r[:, off:off + 3 * K]; off += 3 * K

    return {
        "target_center_dif": target_center_dif,
        "wrist_ori_error": wrist_ori_error,
        "finger_joints": finger_joints,
        "torque_error": torque_error,
        "wrist_vel": wrist_vel,
        "wrist_qvel": wrist_qvel,
        "contacts_af": contacts_af,
        "impulses_af": impulses_af,
        "obj_vel_in_wrist": obj_vel_in_wrist,
        "obj_qvel_in_wrist": obj_qvel_in_wrist,
        "contacts_non_af": contacts_non_af,
        "impulses_non_af": impulses_non_af,
        "af_vec": af_vec,
        "non_aff_vec_obs": non_aff_vec_obs,
        "_base_dim": base_dim,
        "_total_used": off,
    }


def get_reward_info_pinch_pingpong(
    reward_r,
    env,
    hand_type,
    dis_info,
    obs_new_r,
    cfg,
    finger_weights,
    env_mode,
    data_dict_for_hand: dict,
    # --- pinch task hyperparams (safe defaults, can be overridden in cfg) ---
    contact_threshold: float = 0.005,
    near_threshold: float = 0.003,
    other_far_threshold: float = 0.02,
    # tip distance
    ball_diameter: float = 0.04,
    tip_distance_tolerance: float = 0.02,
    # optional extra inputs
    joints: np.ndarray = None,
):
    """Specialized reward for the "pinch ping-pong ball" task.

    Goals:
        1) Encourage thumb + index (prefer fingertip/end nodes) to approach and
           contact the object's affordance surface.
        2) Keep other fingers away from the object surface.
        3) Keep other fingers extended (low FLEX) to reduce accidental contact.
        4) (Optional) Make the distance between thumb tip and index tip approach
           the ball diameter (`ball_diameter`).

    Inputs:
        - dis_info: shape=(N, 2*K), K=keypoints_num.
          The first K are nearest distances to affordance; the last K are nearest
          distances to non-affordance (zeros if contain_non_aff == 0).
        - obs_new_r: observation produced by `observe()`.
        - data_dict_for_hand: entry from `constants.get_data_dict()[hand_type]`.
        - joints: shape=(N, K, 3) keypoints positions (world/object frame) aligned
          with `dis_info`. Prefer passing/caching the `joints` from
          `env/RaisimGymVecEnvOther.observe()`.

    Returns:
        reward_r (updated in-place), rewards_r (list[dict])
    """
    node_dim = int(data_dict_for_hand.get("keypoints_num", finger_weights.shape[1]))
    # --- node indices for thumb/index/others ---
    finger_nodes = get_finger_node_indices(data_dict_for_hand)
    thumb_nodes = finger_nodes["thumb"]
    index_nodes = finger_nodes["index"]
    other_nodes = finger_nodes["middle"] + finger_nodes["ring"] + finger_nodes["pinky"]

    # prefer fingertip nodes if available
    tips = data_dict_for_hand.get("fingertips_id", [])
    thumb_tip = [i for i in thumb_nodes if i in tips]
    index_tip = [i for i in index_nodes if i in tips]

    aff = dis_info[:, :node_dim]
    # --- pinch target reward: encourage thumb/index to be close ---
    thumb_min = np.min(aff[:, thumb_tip], axis=1)
    index_min = np.min(aff[:, index_tip], axis=1)

    # smooth near reward in [0,1]: (near_threshold - d)/near_threshold clipped
    thumb_near = np.clip((near_threshold - thumb_min) / max(near_threshold, 1e-6), 0.0, 1.0)
    index_near = np.clip((near_threshold - index_min) / max(near_threshold, 1e-6), 0.0, 1.0)
    pinch_near_reward = 0.5 * (thumb_near + index_near)

    # contact reward: both in contact -> 1, else 0 (can be softened)
    thumb_contact = (thumb_min < contact_threshold).astype(np.float32)
    index_contact = (index_min < contact_threshold).astype(np.float32)
    pinch_contact_reward = thumb_contact * index_contact

    # --- other fingers: keep away ---
    other_min = np.min(aff[:, other_nodes], axis=1)
    # near penalty: if closer than other_far_threshold, penalize linearly
    other_near_penalty = np.clip((other_far_threshold - other_min) / max(other_far_threshold, 1e-6), 0.0, 1.0)
    other_contact_penalty = (other_min < contact_threshold).astype(np.float32)

    # --- other fingers straightness (FLEX) ---

    straight_penalty = np.zeros(env.num_envs, dtype=np.float32)
    parsed = parse_observation_right(
        obs_new_r,
        act_dim=int(data_dict_for_hand["act_dim"]),
        num_contacts=int(cfg['environment'].get('num_contacts', 0)) if isinstance(cfg, dict) else 0,
        keypoints_num=int(data_dict_for_hand.get('keypoints_num', node_dim)),
    )
    finger_joints = parsed['finger_joints']
    joint_sem_map = build_joint_semantic_map()[hand_type]
    node2seq = build_node_to_joint_sequence()[hand_type]
    dof2finger = build_dof_to_finger_id_map(hand_type, exclude_wrist=True)

    # Flatten finger-DoF joint names (skip the 6 wrist DoFs in node0; align with `dof2finger`/finger_joints).
    flat_joint_names = []
    for node_id, joint_names in enumerate(node2seq):
        if node_id == 0:
            continue
        for jn in joint_names:
            if not isinstance(jn, str) or len(jn) == 0:
                raise ValueError(f"Invalid joint name in node2seq[{node_id}]: {jn!r}")
            flat_joint_names.append(jn)


    # Select FLEX DoF indices for middle/ring/pinky (avoid string-based matching).
    other_flex_indices = []
    for idx, jname in enumerate(flat_joint_names):
        axis_id, _sign = joint_sem_map[jname]
        if axis_id != FLEX:
            continue
        fid = dof2finger[idx]
        if fid in (FingerIdType.MIDDLE, FingerIdType.RING, FingerIdType.PINKY):
            other_flex_indices.append(idx)

    straight_penalty = np.mean(np.abs(finger_joints[:, other_flex_indices]), axis=1).astype(np.float32)

    # --- tip distance to match pingpong diameter (optional) ---
    tip_distance_reward = np.zeros(env.num_envs, dtype=np.float32)
    tip_distance_value = np.zeros(env.num_envs, dtype=np.float32)
    thumb_tip_idx = thumb_tip[0]
    index_tip_idx = index_tip[0]
    if joints is not None:
        delta = joints[:, thumb_tip_idx, :] - joints[:, index_tip_idx, :]
        dist = np.linalg.norm(delta, axis=1).astype(np.float32)
        tip_distance_value = dist
        err = np.abs(dist - float(ball_diameter))
        tip_distance_reward = np.clip(1.0 - err / max(float(tip_distance_tolerance), 1e-6), 0.0, 1.0).astype(np.float32)
    else:
        tip_distance_reward = np.zeros(env.num_envs, dtype=np.float32)
    # --- combine into rewards_r dict ---
    rewards_r = env.get_reward_info_r()
    for i in range(env.num_envs):
        pinch_cfg = cfg.get('environment', {}).get('reward', {})
        w_ti_near = float(pinch_cfg.get('pinch_thumb_index_near')["coeff"])
        w_ti_contact = float(pinch_cfg.get('pinch_thumb_index_contact')["coeff"])
        w_o_near = float(pinch_cfg.get('pinch_other_near_penalty')["coeff"])
        w_o_contact = float(pinch_cfg.get('pinch_other_contact_penalty')["coeff"])
        w_o_flex = float(pinch_cfg.get('pinch_other_flex_penalty')["coeff"])
        _w_tip_dist = float(pinch_cfg.get('pinch_tip_distance')["coeff"])

        rewards_r[i].update({
            'pinch_thumb_index_near': float(pinch_near_reward[i]) * w_ti_near,
            'pinch_thumb_index_contact': float(pinch_contact_reward[i]) * w_ti_contact,
            # penalties are non-negative magnitudes; use coeff sign from cfg (typically negative)
            'pinch_other_near_penalty': float(other_near_penalty[i]) * w_o_near,
            'pinch_other_contact_penalty': float(other_contact_penalty[i]) * w_o_contact,
            'pinch_other_flex_penalty': float(straight_penalty[i]) * w_o_flex,
            'pinch_tip_distance': float(tip_distance_reward[i]) * _w_tip_dist,
            'pinch_tip_distance_raw': float(tip_distance_value[i]),
        })

        direction_loss = float(np.sum(np.square(obs_new_r[i, 3:6])))
        center_loss = float(np.sum(np.square(obs_new_r[i, :3])))
        rewards_r[i]['direction_reward'] = direction_loss * cfg['environment']['reward']['direction_reward']['coeff']
        rewards_r[i]['center_reward'] = center_loss * cfg['environment']['reward']['center_reward']['coeff']

        base_sum = rewards_r[i].get('reward_sum', 0.0)
        total = base_sum + \
            rewards_r[i]['pinch_thumb_index_near'] + \
            rewards_r[i]['pinch_thumb_index_contact'] + \
            rewards_r[i]['pinch_other_near_penalty'] + \
            rewards_r[i]['pinch_other_contact_penalty'] + \
            rewards_r[i]['pinch_other_flex_penalty'] + \
            rewards_r[i]['pinch_tip_distance'] + \
            rewards_r[i]['direction_reward'] + \
            rewards_r[i]['center_reward']

        if env_mode == "floating":
            for key in ["obj_vel_reward_", "obj_qvel_reward_", "wrist_vel_reward_", "wrist_qvel_reward_"]:
                if key in rewards_r[i]:
                    total += float(rewards_r[i][key])

        rewards_r[i]['reward_sum'] = total
        reward_r[i] = total

    return reward_r, rewards_r


def get_cfg(cfg_path,args,cat_name,len_obj):
    cfg = YAML().load(open( cfg_path, 'r'))
    cfg['environment']['visualize'] = args.visualize
    if args.seed != 1:
        cfg['seed'] = args.seed
    cfg['environment']['load_set'] = cat_name
    cfg['environment']['num_envs'] = len_obj
    return cfg


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _default_cross_hand_cfg_path() -> str:
    """Default base cfg for cross_hand env.

    We keep a single base cfg and override per-hand `environment.hand_model_r` in code.
    """
    here = os.path.dirname(os.path.realpath(__file__))
    # dexgrasp_zero/helper/utils.py -> dexgrasp_zero/env/envs/cross_hand/cfgs
    return os.path.abspath(
        os.path.join(here, "..", "env", "envs", "cross_hand", "cfgs", "cross_hand_base_cfg_reg_p2.yaml")
    )


def get_cross_hand_env_cfg(
    *,
    hand: Optional[str] = None,
    base_cfg_path: Optional[str] = None,
    args=None,
    cat_name: Optional[str] = None,
    len_obj: Optional[int] = None,
    seed: Optional[int] = None,
    visualize: Optional[bool] = None,
    urdf_path: Optional[str] = None,
    hand_model_r: Optional[str] = None,
    overrides: Optional[dict] = None,
):
    """Load cross_hand env cfg from ONE base yaml and apply runtime overrides.

    Priority for `hand_model_r`:
    1) explicit `hand_model_r`
    2) basename of `urdf_path`
    3) default mapping by `hand`

    This keeps cfgs/ from exploding with near-duplicate files per hand.
    """
    cfg_path = base_cfg_path or _default_cross_hand_cfg_path()
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Base cfg not found: {cfg_path}. Expected the shared base cfg to exist."
        )

    cfg = YAML().load(open(cfg_path, "r"))
    if "environment" not in cfg:
        raise KeyError(f"Invalid cfg (missing 'environment'): {cfg_path}")

    if seed is None and args is not None and hasattr(args, "seed"):
        seed = int(args.seed)
    if seed is not None and seed != 1:
        cfg["seed"] = seed

    if visualize is None and args is not None and hasattr(args, "visualize"):
        visualize = bool(args.visualize)
    if visualize is not None:
        cfg["environment"]["visualize"] = visualize

    if cat_name is not None:
        cfg["environment"]["load_set"] = cat_name
    if len_obj is not None:
        cfg["environment"]["num_envs"] = int(len_obj)

    resolved_hand_model_r = hand_model_r
    if resolved_hand_model_r is None and urdf_path:
        resolved_hand_model_r = os.path.basename(urdf_path)
    if resolved_hand_model_r is None and hand:
        resolved_hand_model_r = get_cross_hand_spec(hand).hand_model_r
    if resolved_hand_model_r is not None:
        cfg["environment"]["hand_model_r"] = resolved_hand_model_r

    if overrides:
        _deep_update(cfg, overrides)

    return cfg, cfg_path

def process_obj_list(home_path,cat_name,env_mode,obj_num=-1,single_obj_name=None,env_num_per_hand_single_obj=1,obj_repeate_scale=3):
    obj_list = []
    obj_path_list=[]
    directory_path = home_path + f"/rsc/{cat_name}/"
    items = os.listdir(directory_path)

    # # Filter out only the folders (directories) from the list of items
    folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

    if single_obj_name is not None:
        if single_obj_name not in folder_names:
            raise ValueError(f"Object '{single_obj_name}' does not exist under directory '{directory_path}'.")
        obj_ori_list = [single_obj_name]
        obj_repeate_scale = env_num_per_hand_single_obj
    else:
        obj_ori_list = folder_names
        obj_repeate_scale = obj_repeate_scale

    for i in range(obj_repeate_scale):
        for item in obj_ori_list:
            obj_list.append(item)

    if single_obj_name is None and obj_num!=-1:
        obj_list = obj_list[:obj_num]
    obj_full_path_list=[]
    for obj_item in obj_list:
        if env_mode =="fixed":
            obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}_fixed_base.urdf"))
        else:
            obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
        obj_full_path_list.append(os.path.join(directory_path,obj_item))

    return obj_list,obj_path_list,obj_full_path_list

def reset_environment(env,
                      hand_type,
                      param_data,
                      act_dim):
    """
        Environment reset helper.

        Returns:
                contain_non_aff, obs_new_r, rewards_r_sum, handtypes
    """

    (target_center,
     object_center,
     contain_non_aff,
     qpos_reset_r,
     qpos_reset_l,
     obj_pose_reset,
     handtypes) = param_data

    # Step 1: reset environment state
    env.reset_state(
        qpos_reset_r,
        qpos_reset_l,
        np.zeros((env.num_envs, act_dim), 'float32'),
        np.zeros((env.num_envs, act_dim), 'float32'),
        obj_pose_reset,
    )

    # Step 2: set goals
    env.set_goals(
        target_center, object_center,
        *(np.zeros((env.num_envs, 1), 'float32') for _ in range(8))
    )
    # Step 3: get observation and init reward dict
    obs_new_r, _ = env.observe(contain_non_aff, 
                    **observe_kwargs_for_hand_type(hand_type)
                    )
    rewards_r_sum = env.get_reward_info_r()
    for i in range(len(rewards_r_sum)):
        rewards_r_sum[i].update({k:0 for k in rewards_r_sum[i].keys()})

    return  contain_non_aff, obs_new_r, rewards_r_sum, handtypes 


def build_reset_params(env, hand_type, act_dim, env_mode, obj_full_path_list=None):
    assert env_mode in ["fixed","floating"], f"env_mode {env_mode} not found"
    target_center = np.zeros_like(env.affordance_center)
    object_center = np.zeros_like(env.affordance_center)
    fake_non_aff_center = [0.346408, 0.346408, 0.346408]
    contain_non_aff = np.zeros((env.num_envs, 1), dtype='float32')
    qpos_reset_r = np.zeros((env.num_envs, act_dim), dtype='float32')
    qpos_reset_l = np.zeros((env.num_envs, act_dim), dtype='float32')
    obj_pose_reset = np.zeros((env.num_envs, 8), dtype='float32')
    handtypes = np.ones((env.num_envs,1), dtype='int32') * hand_type
    
    # Initialize hand/object poses
    for i in range(env.num_envs):
        # Default initial object pose
        obj_pose_reset[i, :] = [1., -0., 0.55, 1., -0., -0., 0., 0.]
        
        if env_mode=="floating":
            lowest_point = 0.
            txt_file_path = obj_full_path_list[i] + "/lowest_point_new.txt"
            with open(txt_file_path, 'r') as txt_file:
                lowest_point = float(txt_file.read())
            obj_pose_reset[i, 2] -= lowest_point

        # Initialize hand joint angles by hand type
        if hand_type in [ HandType.ALLEGRO.value , HandType.SHADOW.value , HandType.LEAP.value] :
                qpos_reset_r[i, -4] = 1.7
        elif hand_type == HandType.SCHUNK.value:
                qpos_reset_r[i, 6] = 1.0
        elif hand_type == HandType.INSPIRE.value:
                qpos_reset_r[i, 6] = 1.3
        elif hand_type == HandType.ABILITY.value:
                qpos_reset_r[i, 6] = -1.9
        elif hand_type == HandType.BRAINCO.value:
            qpos_reset_r[i, 6] = 0.9
        else:
            raise ValueError("hand type unknown")
        
        # Handle "non-affordance region" existence
        non_aff_mesh = None
        if np.linalg.norm(env.non_aff_mesh[i].centroid - fake_non_aff_center) >= 0.01:
            non_aff_mesh = env.non_aff_mesh[i]
            contain_non_aff[i, 0] = 1.

        # Compute rotation and set wrist pose
        obj_mat = rotations.quat2mat(obj_pose_reset[i, 3:7])
        hand_type_str = {
                HandType.ALLEGRO.value:"allegro",
                HandType.SHADOW.value :"shadow",
                HandType.LEAP.value : "leap",
                HandType.SCHUNK.value : "schunk",
                HandType.INSPIRE.value : "inspire",
                HandType.ABILITY.value : "ability",
                HandType.BRAINCO.value : "brainco",
            }
        if hand_type in hand_type_str.keys():
    
            rot, pos, bias = get_initial_pose_faive(env.aff_mesh[i], non_aff_mesh, hand_type=hand_type_str[hand_type] )
            wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
            wrist_mat = rotations.euler2mat(wrist_pose_obj)
            wrist_in_world = np.matmul(obj_mat, wrist_mat)
            wrist_pose = rotations.mat2euler(wrist_in_world)
            
            qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat, pos[0, :])
            qpos_reset_r[i, 3:6] = wrist_pose[0, :]
            target_center[i, :] = bias[:]
            object_center[i, :] = env.affordance_center[i]
        else:
            raise ValueError(f"hand_type {hand_type} not supported")


    return (target_center, object_center, contain_non_aff,
            qpos_reset_r, qpos_reset_l, obj_pose_reset, handtypes)




def concat_dict(dict):
    ret = {}
    for key in dict.keys():
        for k in dict[key].keys():
            if k in ret.keys():
                ret[k] = np.concatenate((ret[k], dict[key][k]), axis=0)
            else:
                ret[k] = dict[key][k]
    return ret

IDX_TO_OBJ = {
    1: ['002_master_chef_can',0.414, 0, [0.051,0.139,0.0]],
    2: ['003_cracker_box', 0.453, 1, [0.06, 0.158, 0.21]],
    3: ['004_sugar_box', 0.514, 1, [0.038, 0.089, 0.175]],
    4: ['005_tomato_soup_can', 0.349, 0, [0.033, 0.101,0.0]],
    5: ['006_mustard_bottle', 0.431,2, [0.0,0.0,0.0]],
    6: ['007_tuna_fish_can', 0.171, 0, [0.0425, 0.033,0.0]],
    7: ['008_pudding_box', 0.187, 3, [0.21, 0.089, 0.035]],
    8: ['009_gelatin_box', 0.097, 3, [0.028, 0.085, 0.073]],
    9: ['010_potted_meat_can', 0.37, 3, [0.05, 0.097, 0.089]],
    10: ['011_banana', 0.066,2, [0.028, 0.085, 0.073]],
    11: ['019_pitcher_base', 0.178,2, [0.0,0.0,0.0]],
    12: ['021_bleach_cleanser', 0.302,2, [0.0,0.0,0.0]], # not sure about weight here
    13: ['024_bowl', 0.147,2, [0.0,0.0,0.0]],
    14: ['025_mug', 0.118,2, [0.0,0.0,0.0]],
    15: ['035_power_drill', 0.895,2, [0.0,0.0,0.0]],
    16: ['036_wood_block', 0.729, 3, [0.085, 0.085, 0.2]],
    17: ['037_scissors', 0.082,2, [0.0,0.0,0.0]],
    18: ['040_large_marker', 0.01, 3, [0.009,0.121,0.0]],
    19: ['051_large_clamp', 0.125,2, [0.0,0.0,0.0]],
    20: ['052_extra_large_clamp', 0.102,2, [0.0,0.0,0.0]],
    21: ['061_foam_brick', 0.028, 1, [0.05, 0.075, 0.05]],
}

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def get_obj_pcd(path,num_p=100):
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    mesh.remove_duplicated_vertices()
    obj_vertices = mesh.sample_points_uniformly(number_of_points=num_p)
    obj_vertices = np.asarray(obj_vertices.points)

    return obj_vertices

def first_nonzero(arr, axis, invalid_val=-1):
    arr = torch.Tensor(arr)
    mask = arr!=0
    mask = mask.to(torch.uint8)
    return torch.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def dgrasp_to_mano(param, is_right=True):
    bs = param.shape[0]
    eulers = param[:,6:].reshape(bs,-1, 3).copy()

    # # exchange ring finger and little finger's sequence
    # temp = eulers[:,6:9].copy()
    # eulers[:,6:9] = eulers[:,9:12]
    # eulers[:,9:12] = temp

    eulers = eulers.reshape(-1,3)
    # change euler angle to axis angle
    rotvec = R.from_euler('XYZ', eulers, degrees=False)
    rotvec = rotvec.as_rotvec().reshape(bs,-1)
    global_orient = R.from_euler('XYZ', param[:,3:6], degrees=False)
    global_orient = global_orient.as_rotvec()

    # translation minus an offset
    if is_right:
        offset = np.array([[0.09566994, 0.00638343, 0.0061863]])
    else:
        offset = np.array([[-0.09566994, 0.00638343, 0.0061863]])
    mano_param = np.concatenate([global_orient, rotvec, param[:,:3] - offset],axis=1)

    return mano_param

def show_pointcloud_objhand(hand, obj):
    '''
    Draw hand and obj xyz at the same time
    :param hand: [778, 3]
    :param obj: [3000, 3]
    '''


    hand_dim = hand.shape[0]
    obj_dim = obj.shape[0]
    handObj = np.vstack((hand, obj))
    c_hand, c_obj = np.array([[1, 0, 0]]), np.array([[0, 0, 1]]) # RGB
    c_hand = np.repeat(c_hand, repeats=hand_dim, axis=0) # [778,3]
    c_obj = np.repeat(c_obj, repeats=obj_dim, axis=0) # [3000,3]
    c_hanObj = np.vstack((c_hand, c_obj)) # [778+3000, 3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(handObj)
    pc.colors = o3d.utility.Vector3dVector(c_hanObj)
    o3d.visualization.draw_geometries([pc])

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg.yaml')
    parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
    parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
    parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default="grasping")
    parser.add_argument('-w', '--weight', type=str, default='full_400.pt')
    parser.add_argument('-sd', '--storedir', type=str, default='data_all')
    parser.add_argument('-pr', '--prior', action="store_true")
    parser.add_argument('-o', '--obj_id', type=int, default=7)
    parser.add_argument('-t', '--test', action="store_true")
    parser.add_argument('-mc', '--mesh_collision', action="store_true")
    parser.add_argument('-ao', '--all_objects', action="store_true")
    parser.add_argument('-to', '--test_object_set', type=int, default=-1)
    parser.add_argument('-ac', '--all_contact', action="store_true")
    parser.add_argument('-seed', '--seed', type=int, default=1)
    parser.add_argument('-itr', '--num_iterations', type=int, default=3001)
    parser.add_argument('-nr', '--num_repeats', type=int, default=10)
    parser.add_argument('-ev', '--vis_evaluate', action="store_true")
    parser.add_argument('-sv', '--store_video', action="store_true")

    args = parser.parse_args()

    return args

def repeat_label(label_dict, num_repeats):
    ret = {}
    for k,v in label_dict.items():
        if 'type' in k or 'idx' in k or 'name' in k:
            ret[k] = np.repeat(v, num_repeats, 0)
        else :
            ret[k] = np.repeat(v,num_repeats,0).astype('float32')

    return ret

def euler_noise_to_quat(quats, palm_pose, noise):
    eulers_palm_mats = np.array([rotations.euler2mat(pose) for pose in palm_pose]).copy()
    eulers_mats =  np.array([rotations.quat2mat(quat) for quat in quats])

    rotmats_list = np.array([rotations.euler2mat(noise) for noise in noise])

    eulers_new = np.matmul(rotmats_list,eulers_mats)
    eulers_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_new])

    eulers_palm_new = np.matmul(rotmats_list,eulers_palm_mats)
    eulers_palm_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_palm_new])

    quat_list = [rotations.euler2quat(noise) for noise in eulers_rotmated]

    return np.array(quat_list), eulers_new, eulers_palm_rotmated
