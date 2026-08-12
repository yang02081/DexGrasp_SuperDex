from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np


EPS = 1e-7


@dataclass(frozen=True)
class EvalSpec:
    """Index specification for evaluation (avoids hand-specific hard-coded slices).

        Assumptions:
        - data_dict[hand_type]["act_dim"]         == gcDim
        - data_dict[hand_type]["keypoints_num"]   == num_bodyparts
        - data_dict[hand_type]["obs_dim"]         == (C++ obDim_r_) + 6*num_bodyparts
            (because Python appends `af_vec` and `non_aff_vec` to the raw observation)
        """

    act_dim: int
    num_bodyparts: int
    num_contacts: int

    # global_state indices (test env)
    obj_pos_z_idx: int
    direction_slice: Tuple[int, int]
    across_mul_idx: int

    # observation indices (C++ raw observation inside obs_r prefix)
    contacts_af_slice: Tuple[int, int]
    contacts_non_af_slice: Tuple[int, int]

    @staticmethod
    def from_data_dict_entry(d: Dict[str, Any]) -> "EvalSpec":
        act_dim = int(d["act_dim"])
        num_bodyparts = int(d["keypoints_num"])
        obs_dim = int(d["obs_dim"])

        # Infer C++ obDim_r_ (raw observation dimension).
        raw_ob_dim = obs_dim - 6 * num_bodyparts
        if raw_ob_dim <= 0:
            raise ValueError(f"Invalid obs_dim/keypoints_num: obs_dim={obs_dim}, keypoints_num={num_bodyparts}")

        # obDim = 12 + 2*gcDim + 4*num_contacts  => num_contacts = (obDim - (12+2*gcDim))/4
        num_contacts_float = (raw_ob_dim - (12 + 2 * act_dim)) / 4.0
        num_contacts = int(round(num_contacts_float))
        if abs(num_contacts - num_contacts_float) > 1e-6 or num_contacts <= 0:
            raise ValueError(
                f"Cannot infer num_contacts from data_dict: raw_ob_dim={raw_ob_dim}, act_dim={act_dim}, "
                f"num_contacts_float={num_contacts_float}"
            )

        # obs layout (prefix, before af_vec/non_aff_vec):
        # target_center_dif(3) + wrist_ori_error(3) + finger_joints(gcDim-6) + torque_error(gcDim)
        # + wrist_vel(3) + wrist_qvel(3) + contacts_af(N) + impulses_af(N) + obj_vel(3) + obj_qvel(3)
        # + contacts_non_af(N) + impulses_non_af(N)
        contacts_af_start = 2 * act_dim + 6
        contacts_af_end = contacts_af_start + num_contacts
        contacts_non_af_start = 2 * act_dim + 12 + 2 * num_contacts
        contacts_non_af_end = contacts_non_af_start + num_contacts

        # global_state layout (test env):
        # obj_pose_wrist(3) + frame_y_in_obj(3*num_bodyparts) + joint_pos_in_obj(3*num_bodyparts)
        # + Obj_Position(3) + direction_error(3) + across_mul(1) + ...
        obj_pos_z_idx = 3 + 6 * num_bodyparts + 2
        direction_start = 3 + 6 * num_bodyparts + 3
        direction_slice = (direction_start, direction_start + 3)
        across_mul_idx = direction_start + 3

        return EvalSpec(
            act_dim=act_dim,
            num_bodyparts=num_bodyparts,
            num_contacts=num_contacts,
            obj_pos_z_idx=obj_pos_z_idx,
            direction_slice=direction_slice,
            across_mul_idx=across_mul_idx,
            contacts_af_slice=(contacts_af_start, contacts_af_end),
            contacts_non_af_slice=(contacts_non_af_start, contacts_non_af_end),
        )

    @staticmethod
    def from_runtime(
        *,
        act_dim: int,
        num_bodyparts: int,
        obs_dim: int,
        extra_dim: int = 0,
    ) -> "EvalSpec":
        """Build from runtime tensor shapes.

        This is recommended when `data_dict[hand_type]["obs_dim"]` may be stale or when
        additional features (e.g., PointNet) are concatenated to the end of observations.

        Args:
            obs_dim: Actual `obs_r.shape[1]` used at evaluation time.
            extra_dim: Extra feature dimension appended to the tail of obs (e.g., 512+7).
                This is subtracted before inferring contact indices.
        """
        act_dim = int(act_dim)
        num_bodyparts = int(num_bodyparts)
        obs_dim = int(obs_dim)
        extra_dim = int(extra_dim)

        base_dim = obs_dim - extra_dim
        raw_ob_dim = base_dim - 6 * num_bodyparts
        if raw_ob_dim <= 0:
            raise ValueError(
                f"Invalid runtime dims: obs_dim={obs_dim}, extra_dim={extra_dim}, keypoints_num={num_bodyparts}"
            )

        num_contacts_float = (raw_ob_dim - (12 + 2 * act_dim)) / 4.0
        num_contacts = int(round(num_contacts_float))
        if abs(num_contacts - num_contacts_float) > 1e-6 or num_contacts <= 0:
            raise ValueError(
                f"Cannot infer num_contacts from runtime dims: raw_ob_dim={raw_ob_dim}, act_dim={act_dim}, "
                f"num_contacts_float={num_contacts_float}"
            )

        # contact slices in obs prefix
        contacts_af_start = 2 * act_dim + 6
        contacts_af_end = contacts_af_start + num_contacts
        contacts_non_af_start = 2 * act_dim + 12 + 2 * num_contacts
        contacts_non_af_end = contacts_non_af_start + num_contacts

        # global_state indices (test env)
        obj_pos_z_idx = 3 + 6 * num_bodyparts + 2
        direction_start = 3 + 6 * num_bodyparts + 3
        direction_slice = (direction_start, direction_start + 3)
        across_mul_idx = direction_start + 3

        return EvalSpec(
            act_dim=act_dim,
            num_bodyparts=num_bodyparts,
            num_contacts=num_contacts,
            obj_pos_z_idx=obj_pos_z_idx,
            direction_slice=direction_slice,
            across_mul_idx=across_mul_idx,
            contacts_af_slice=(contacts_af_start, contacts_af_end),
            contacts_non_af_slice=(contacts_non_af_start, contacts_non_af_end),
        )


@dataclass
class RunningEval:
    """Matches the original script's accumulation logic.

    - Per-update average over all envs
    - Lifted-subset average accumulated with `lifted_num` as weight
    """

    success_rate: float = 0.0
    center_error: float = 0.0
    suc_center_error: float = 0.0
    direction_error: float = 0.0
    suc_direction_error: float = 0.0
    rotation_error: float = 0.0
    suc_rotation_error: float = 0.0
    contact_ratio: float = 0.0
    suc_contact_ratio: float = 0.0
    lifted_num: float = 0.0
    update: int = -1

    def update_from_batch(self, lifted: np.ndarray, metrics: Dict[str, np.ndarray], update: int) -> None:
        self.update = int(update)
        num_envs = float(lifted.shape[0])
        lifted_count = float(np.sum(lifted))

        cur_success = lifted_count / max(num_envs, 1.0)
        self.success_rate = (update * self.success_rate + cur_success) / (update + 1)

        def mean_all(x: np.ndarray) -> float:
            return float(np.sum(x) / max(num_envs, 1.0))

        def mean_lifted(x: np.ndarray) -> float:
            return float(np.sum(x[lifted]) / max(lifted_count, 1.0))

        cur_center = metrics["center_error"]
        cur_dir = metrics["direction_error"]
        cur_angle = metrics["angle_error"]
        cur_contact = metrics["contact_ratio"]

        self.center_error = (update * self.center_error + mean_all(cur_center)) / (update + 1)
        self.direction_error = (update * self.direction_error + mean_all(cur_dir)) / (update + 1)
        self.rotation_error = (update * self.rotation_error + mean_all(cur_angle)) / (update + 1)
        self.contact_ratio = (update * self.contact_ratio + mean_all(cur_contact)) / (update + 1)

        # Lifted-subset average: accumulate weighted by lifted_num
        self.suc_center_error = (self.lifted_num * self.suc_center_error + np.sum(cur_center[lifted])) / (
            self.lifted_num + lifted_count + EPS
        )
        self.suc_direction_error = (self.lifted_num * self.suc_direction_error + np.sum(cur_dir[lifted])) / (
            self.lifted_num + lifted_count + EPS
        )
        self.suc_rotation_error = (self.lifted_num * self.suc_rotation_error + np.sum(cur_angle[lifted])) / (
            self.lifted_num + lifted_count + EPS
        )
        self.suc_contact_ratio = (self.lifted_num * self.suc_contact_ratio + np.sum(cur_contact[lifted])) / (
            self.lifted_num + lifted_count + EPS
        )

        self.lifted_num += lifted_count


def compute_eval_metrics(
    spec: EvalSpec,
    obs_r: np.ndarray,
    global_state: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return per-env metric arrays (caller decides how to print/aggregate)."""
    center_error = np.linalg.norm(obs_r[:, :3], axis=1)
    ds, de = spec.direction_slice
    direction_error = np.linalg.norm(global_state[:, ds:de], axis=1)
    across = np.clip(global_state[:, spec.across_mul_idx], -1.0, 1.0)
    angle_error = np.arccos(across)

    af_s, af_e = spec.contacts_af_slice
    naf_s, naf_e = spec.contacts_non_af_slice
    contacts_af_sum = np.sum(obs_r[:, af_s:af_e], axis=1)
    contacts_naf_sum = np.sum(obs_r[:, naf_s:naf_e], axis=1)
    contact_ratio = contacts_af_sum / (contacts_af_sum + contacts_naf_sum + EPS)

    return {
        "center_error": center_error,
        "direction_error": direction_error,
        "angle_error": angle_error,
        "contact_ratio": contact_ratio,
    }


def compute_lifted_mask(
    spec: EvalSpec,
    obs_r: np.ndarray,
    global_state: np.ndarray,
    obj_pose_reset: np.ndarray,
    lift_height_thresh: float = 0.15,
    center_thresh: float = 0.15,
) -> np.ndarray:
    center_error = np.linalg.norm(obs_r[:, :3], axis=1)
    lifted = (global_state[:, spec.obj_pos_z_idx] - obj_pose_reset[:, 2] > lift_height_thresh) & (center_error < center_thresh)
    return lifted


def print_batch_report(
    obj_names_in_env,
    lifted: np.ndarray,
    metrics: Dict[str, np.ndarray],
    global_state: np.ndarray,
    spec: EvalSpec,
) -> None:
    num_envs = float(lifted.shape[0])
    lifted_count = float(np.sum(lifted))

    failed_objects = [obj_names_in_env[i] for i, ok in enumerate(lifted) if not ok]
    print("current failed objects:", failed_objects)
    print("current success rate", lifted_count / max(num_envs, 1.0))

    for k, label in [
        ("center_error", "center error"),
        ("direction_error", "direction error"),
        ("angle_error", "angle error"),
        ("contact_ratio", "contact ratio"),
    ]:
        v = metrics[k]
        print(f"current {label}", float(np.sum(v) / max(num_envs, 1.0)))
        print(f"current {label} lifted", float(np.sum(v[lifted]) / max(lifted_count, 1.0 + EPS)))
    print(" ")


def print_running_report(r: RunningEval) -> None:
    print("average success rate", r.success_rate)
    print("average center error", r.center_error)
    print("average center error lifted", r.suc_center_error)
    print("average direction error", r.direction_error)
    print("average direction error lifted", r.suc_direction_error)
    print("average rotation error", r.rotation_error)
    print("average rotation error lifted", r.suc_rotation_error)
    print("average contact ratio", r.contact_ratio)
    print("average contact ratio lifted", r.suc_contact_ratio)
    print(" ")


def write_final_result(path: str, r: RunningEval) -> None:
    with open(path, "w") as f:
        f.write("final success rate " + str(r.success_rate) + "\n")
        f.write("final center error " + str(r.center_error) + "\n")
        f.write("final center error lifted " + str(r.suc_center_error) + "\n")
        f.write("final direction error " + str(r.direction_error) + "\n")
        f.write("final direction error lifted " + str(r.suc_direction_error) + "\n")
        f.write("final rotation error " + str(r.rotation_error) + "\n")
        f.write("final rotation error lifted " + str(r.suc_rotation_error) + "\n")
        f.write("final contact ratio " + str(r.contact_ratio) + "\n")
        f.write("final contact ratio lifted " + str(r.suc_contact_ratio) + "\n")


