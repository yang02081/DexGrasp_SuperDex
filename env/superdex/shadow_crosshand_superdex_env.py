"""DexGrasp-Zero Shadow cross-hand environment backed by SuperDex.

The class mirrors the control-facing part of ``Shadow_Test`` in
``DexGrasp-Zero-code/env/envs/cross_hand/Environment.hpp``.  It intentionally
keeps the official 28-D residual action and 268-D observation contracts so the
unchanged MAGCN actor can be used by the SuperDex evaluator.
"""
from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import superdex.physics as physics

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from .geometry import nearest_vectors as _nearest_vectors, surface_points as _surface_points
from .shadow_superdex_env import (
    DT, SUBSTEPS, SHADOW_URDF, SHADOW_RAISIM_GEOMETRY_URDF,
    ShadowSuperDexEnv, _read_pose, _read_velocity,
)


ACTION_DIM = 28
BASE_OBSERVATION_DIM = 136
OBSERVATION_DIM = 268
GLOBAL_STATE_DIM = 179
HAND_TYPE = 2

# Exact node order from DexGrasp-Zero's Shadow graph specification.
FRAME_NAMES = (
    "palm",
    "ffproximal", "ffmiddle", "ffdistal", "fftip_back",
    "mfproximal", "mfmiddle", "mfdistal", "mftip_back",
    "rfproximal", "rfmiddle", "rfdistal", "rftip_back",
    "lfmetacarpal", "lfproximal", "lfmiddle", "lfdistal", "lftip_back",
    "thproximal", "thmiddle", "thdistal", "thtip_back",
)
CONTACT_LINK_NAMES = (
    "palm",
    "ffproximal", "ffmiddle", "ffdistal",
    "mfproximal", "mfmiddle", "mfdistal",
    "rfproximal", "rfmiddle", "rfdistal",
    "lfmetacarpal", "lfproximal", "lfmiddle", "lfdistal",
    "thproximal", "thmiddle", "thdistal",
)


def _vec3(value) -> np.ndarray:
    return np.asarray([value[0], value[1], value[2]], dtype=np.float32)


def _transforms(actor):
    result = physics.DynamicArrayTransformRT()
    result.resize(len(actor.get_articulated_shape_info().link_names))
    actor.get_articulated_link_transforms(result)
    return result


def _rotation(transform) -> Rotation:
    return Rotation.from_quat(np.asarray(transform.rotation.tolist(), dtype=np.float64))


def _raisim_object_to_superdex(object_state: np.ndarray, scene_origin: np.ndarray) -> np.ndarray:
    """Convert xyz+wxyz+joint into SuperDex xyz+rotation-vector+joint."""
    state = np.asarray(object_state, np.float32)
    if state.shape not in ((7,), (8,)):
        raise ValueError("object pose must have shape (7,) or Raisim-compatible (8,)")
    result = np.zeros(7, np.float32)
    result[:3] = state[:3] - scene_origin
    if state.shape == (8,):
        quat_wxyz = state[3:7]
        norm = float(np.linalg.norm(quat_wxyz))
        if norm < 1e-8:
            raise ValueError("object quaternion must be non-zero")
        quat_wxyz = quat_wxyz / norm
        result[3:6] = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_rotvec()
        result[6] = state[7]
    else:
        result[3:] = state[3:]
    return result


def _superdex_object_to_raisim(object_state: np.ndarray, scene_origin: np.ndarray) -> np.ndarray:
    """Convert SuperDex xyz+rotation-vector+joint into xyz+wxyz+joint."""
    state = np.asarray(object_state, np.float32)
    if state.shape != (7,):
        raise ValueError("SuperDex object pose must have shape (7,)")
    quat_xyzw = Rotation.from_rotvec(state[3:6]).as_quat()
    return np.concatenate((
        state[:3] + scene_origin, quat_xyzw[[3, 0, 1, 2]], state[6:7]
    )).astype(np.float32)


def _raisim_contact_channels(top_force: np.ndarray, bottom_force: np.ndarray):
    """Apply Shadow_Test's affordance-first overlap semantics."""
    top_force = np.asarray(top_force, np.float32)
    bottom_force = np.asarray(bottom_force, np.float32)
    top_contact = (top_force > 0.1).astype(np.float32)
    bottom_contact = (bottom_force > 0.1).astype(np.float32)
    overlap = (top_contact > 0.0) & (bottom_contact > 0.0)
    bottom_contact[overlap] = 0.0
    top_impulse = top_force * DT
    bottom_impulse = bottom_force * DT
    bottom_impulse[overlap] -= top_impulse[overlap]
    return top_contact, top_impulse, bottom_contact, bottom_impulse


class ShadowCrossHandSuperDexEnv:
    """One-environment implementation of the official Shadow cross-hand contract."""

    num_envs = 1
    num_acts = ACTION_DIM
    num_obs_r = OBSERVATION_DIM
    num_obs_l = 1
    num_gs = GLOBAL_STATE_DIM
    hand_type = HAND_TYPE

    def __init__(self, *, object_id: str = "14680ffbf", object_root=None, seed: int = 1,
                 max_steps: int = 400, root_guided: bool = True,
                 freeze_wrist_rotation: bool = False,
                 tracking_stiffness: float = 50.0,
                 tracking_damping: float = 1.0,
                 tracking_saturation: float = -1.0,
                 contact_friction: float = 3.0,
                 contact_normal_damping: float = 40.0,
                 contact_penalty_coefficient: float | None = None,
                 contact_smoothing_half_distance: float | None = None,
                 contact_max_alignment_normals: float | None = None,
                 lift_guidance_height: float = 0.2,
                 lift_ramp_steps: int = 0,
                 physics_substeps: int = SUBSTEPS,
                 strict_physics_alignment: bool = True,
                 record_substep_diagnostics: bool = False,
                 hand_urdf_override=None,
                 geometry_sample_seed: int | None = None):
        if int(physics_substeps) <= 0:
            raise ValueError("physics_substeps must be positive")
        self.backend = ShadowSuperDexEnv(
            object_id=object_id, object_root=object_root,
            seed=seed,
            max_steps=max_steps,
            object_locked=False,
            gravity_z=-9.81,
            task="generic",
            tracking_stiffness=tracking_stiffness,
            tracking_damping=tracking_damping,
            tracking_saturation=tracking_saturation,
            contact_friction=contact_friction,
            contact_normal_damping=contact_normal_damping,
            contact_friction_falloff=0.1,
            contact_penalty_coefficient=contact_penalty_coefficient,
            contact_smoothing_half_distance=contact_smoothing_half_distance,
            contact_max_alignment_normals=contact_max_alignment_normals,
            hand_urdf=(hand_urdf_override if hand_urdf_override is not None else
                       (SHADOW_RAISIM_GEOMETRY_URDF if strict_physics_alignment else SHADOW_URDF)),
            ground_height=(0.5 - 0.35 if strict_physics_alignment else 0.0),
            hand_contact_layer=('hand' if strict_physics_alignment else None),
            object_contact_layer=('object' if strict_physics_alignment else None),
            use_contact_layers=strict_physics_alignment,
        )
        # Raisim filters hand-table collisions while keeping object-table
        # support. SuperDex contact-layer changes on articulated actors are not
        # safe in the current runtime, so table support is reproduced by a
        # unilateral object-height constraint in _enforce_table_support().
        self.max_steps = int(max_steps)
        self._strict_physics_alignment = bool(strict_physics_alignment)
        self._physical_table = bool(strict_physics_alignment)
        self.rng = np.random.default_rng(seed)
        self.root_guided = bool(root_guided)
        # Diagnostic option: the root rotation coordinates are relative to
        # the reset pose, so a zero target preserves the initial orientation.
        self.freeze_wrist_rotation = bool(freeze_wrist_rotation)
        self._root_guidance_requested = bool(root_guided)
        self._steps = 0
        self._lift = False
        self._lift_start_step = None
        self._lift_ramp_steps = max(int(lift_ramp_steps), 0)
        # Preserve the 10 ms policy/control period while optionally reducing
        # the physics integration interval to diagnose contact penetration.
        self._physics_substeps = int(physics_substeps)
        self._physics_dt = DT * SUBSTEPS / self._physics_substeps
        self._record_substep_diagnostics = bool(record_substep_diagnostics)
        # Translate the Raisim world into the legal range of SuperDex's
        # explicit [-0.5, 0.5] wrist prismatic joints. This rigid translation
        # leaves every policy observation and hand-object relative vector
        # unchanged.
        self._scene_origin = np.array([0.65, 0.0, 0.35], np.float32)
        self._goal_center = np.zeros(3, np.float32)
        self._object_center = np.zeros(3, np.float32)
        self._ready = False
        self._contain_non_aff = True

        info = self.backend.hand.get_articulated_shape_info()
        self._link_indices = {name: i for i, name in enumerate(info.link_names)}
        self._link_actors = {}
        for handle in self.backend.hand.get_nested_link_actors():
            actor = self.backend.scene.get_actor(handle)
            if actor is not None:
                self._link_actors[actor.get_name().rsplit("/", 1)[-1]] = actor
        self._palm_actor = self._link_actors["palm"]
        self._top_actor = self.backend.object_links["top"]
        self._use_contact_point_impulses = bool(strict_physics_alignment)
        self._contact_point_query_handles = []
        if self._use_contact_point_impulses:
            for actor in self._link_actors.values():
                if actor.is_query_supported(physics.QueryType.CONTACT_POINTS):
                    self._contact_point_query_handles.append(
                        actor.register_query(physics.QueryType.CONTACT_POINTS)
                    )
        missing = sorted(set(FRAME_NAMES) - set(self._link_indices))
        joint_order = [
            name for name, dof in zip(info.joint_names, info.dof_info)
            if dof.get_size() > 0 and name != "world_joint"
        ]
        if len(joint_order) != ACTION_DIM:
            raise RuntimeError(f"expected 28 controllable Shadow joints, got {joint_order}")
        limits = {}
        for joint in ET.parse(self.backend.hand_urdf).getroot().findall("joint"):
            limit = joint.find("limit")
            if limit is not None and limit.get("lower") is not None:
                limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
        if any(name not in limits for name in joint_order):
            absent = [name for name in joint_order if name not in limits]
            raise RuntimeError(f"missing Shadow joint limits: {absent}")
        self._joint_low = np.array([limits[name][0] for name in joint_order], np.float32)
        self._joint_high = np.array([limits[name][1] for name in joint_order], np.float32)
        if missing:
            self.close()
            raise RuntimeError(f"Shadow SuperDex resource is missing graph frames: {missing}")

        # The earlier single-hand prototype did not expose lfmetacarpal as a
        # contact sensor. Register it here to match the 17 official sensors.
        self._contact_actors = dict(self.backend.hand_links)
        if "lfmetacarpal" not in self._contact_actors:
            extra = self.backend._query_by_name(self.backend.hand, {"lfmetacarpal"})
            if "lfmetacarpal" not in extra:
                self.close()
                raise RuntimeError("lfmetacarpal does not support contact-force queries")
            self._contact_actors.update(extra)
            self.backend.query_handles.append(
                extra["lfmetacarpal"].register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
            )

        if geometry_sample_seed is None:
            self._top_surface = _surface_points(self.backend.top_mesh)
            self._bottom_surface = _surface_points(self.backend.bottom_mesh)
            self._geometry_sampling = "legacy_superdex_fixed"
        else:
            import trimesh
            self._top_surface = trimesh.sample.sample_surface(
                self.backend.top_mesh, 200, seed=int(geometry_sample_seed)
            )[0].astype(np.float32)
            self._bottom_surface = trimesh.sample.sample_surface(
                self.backend.bottom_mesh, 200, seed=int(geometry_sample_seed) + 1
            )[0].astype(np.float32)
            self._geometry_sampling = f"trimesh_seed_{int(geometry_sample_seed)}"
        self._action_scale = np.array([0.005] * 6 + [0.015] * 22, np.float32)
        # PoseController dynamics differ from Raisim PD. A fractional guidance
        # update avoids overshoot while preserving the same target geometry.
        self._root_guidance_gain = 1.0
        self._lift_guidance_height = float(lift_guidance_height)
        self._observation = np.zeros(OBSERVATION_DIM, np.float32)
        self._global_state = np.zeros(GLOBAL_STATE_DIM, np.float32)
        self._last_info: dict = {}
        self._initial_object_height = 0.0
        self._initial_wrist_object_rotation = Rotation.identity()
        self._initial_wrist_world_rotation = Rotation.identity()
        self._initial_hand_center_local = np.zeros(3, np.float32)
        self._initial_grasp_axis_object = np.zeros(3, np.float32)
        self._initial_across_axis_object = np.zeros(3, np.float32)
        self._initial_across_axis_wrist = np.zeros(3, np.float32)
        self._initial_root_position_world = np.zeros(3, np.float32)
        self._initial_root_rotation = Rotation.identity()
        self._last_raisim_target = np.zeros(ACTION_DIM, np.float32)
        # Shadow_Test applies m*g/Kp as a translational PD bias every step.
        # Read the same mass from the resource instead of tuning this offset.
        hand_mass = 0.0
        for link in ET.parse(self.backend.hand_urdf).getroot().findall("link"):
            mass = link.find("inertial/mass")
            if mass is not None:
                hand_mass += float(mass.get("value", "0"))
        self._gravity_compensation = float(hand_mass * 9.81 / tracking_stiffness)

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

    setSeed = seed

    def getNumOfEnvs(self): return 1
    def getRightObDim(self): return OBSERVATION_DIM
    def getLeftObDim(self): return 1
    def getActionDim(self): return ACTION_DIM
    def getGSDim(self): return GLOBAL_STATE_DIM

    def _contact_arrays(self):
        top = np.zeros(17, np.float32)
        bottom = np.zeros(17, np.float32)
        top_actor = self.backend.object_links["top"]
        bottom_actor = self.backend.object_links.get("bottom")
        for i, name in enumerate(CONTACT_LINK_NAMES):
            actor = self._contact_actors[name]
            top[i] = np.linalg.norm(self.backend._force(actor, top_actor))
            if bottom_actor is not None:
                bottom[i] = np.linalg.norm(self.backend._force(actor, bottom_actor))
        return top, bottom

    def _contact_point_channels(self, top_force, bottom_force):
        """Approximate Raisim's per-contact final-step impulse channels.

        Raisim stores one contact impulse per hand body (the final matching
        contact in its list), rather than the norm of the body's net force.
        SuperDex exposes contact-point forces but no native impulse query, so
        use the largest matching point force times the 2.5 ms source step.
        This avoids cancellation in TOTAL_CONTACT_FORCE and avoids summing
        multiple points into a value Raisim never reports.
        """
        if not self._use_contact_point_impulses:
            return _raisim_contact_channels(top_force, bottom_force)
        top_contact = np.zeros(17, np.float32)
        bottom_contact = np.zeros(17, np.float32)
        top_impulse = np.zeros(17, np.float32)
        bottom_impulse = np.zeros(17, np.float32)
        targets = {
            self.backend.object_links['top'].get_handle().get_hash():
                (top_contact, top_impulse),
        }
        bottom_actor = self.backend.object_links.get('bottom')
        if bottom_actor is not None:
            targets[bottom_actor.get_handle().get_hash()] = (
                bottom_contact, bottom_impulse
            )
        for i, name in enumerate(CONTACT_LINK_NAMES):
            actor = self._contact_actors[name]
            for point in actor.get_contact_points_world():
                a = point.actor_a.get_hash()
                b = point.actor_b.get_hash()
                own = actor.get_handle().get_hash()
                other = b if a == own else a
                channels = targets.get(other)
                if channels is None:
                    continue
                contact, impulse = channels
                contact[i] = 1.0
                point_impulse = float(np.linalg.norm(_vec3(point.force)) * DT)
                impulse[i] = max(float(impulse[i]), point_impulse)
        overlap = (top_contact > 0.0) & (bottom_contact > 0.0)
        bottom_contact[overlap] = 0.0
        bottom_impulse[overlap] -= top_impulse[overlap]
        if not self._contain_non_aff:
            bottom_contact.fill(0.0)
            bottom_impulse.fill(0.0)
        return top_contact, top_impulse, bottom_contact, bottom_impulse

    def _kinematics(self):
        hand_tf = _transforms(self.backend.hand)
        obj_tf = _transforms(self.backend.obj)
        wrist_tf = hand_tf[self._link_indices["palm"]]
        top_tf = obj_tf[self.backend.top_index]
        wrist_rot = _rotation(wrist_tf)
        object_rot = _rotation(top_tf)
        wrist_pos = _vec3(wrist_tf.translation)
        object_pos = _vec3(top_tf.translation)
        frame_world = np.stack([
            _vec3(hand_tf[self._link_indices[name]].translation) for name in FRAME_NAMES
        ]).astype(np.float32)
        frame_object = object_rot.inv().apply(frame_world - object_pos).astype(np.float32)
        relative_rotation = wrist_rot.inv() * object_rot
        return (wrist_rot, object_rot, wrist_pos, object_pos, frame_world,
                frame_object, relative_rotation)

    @staticmethod
    def _hand_center(frame_world):
        # FFJ2, MFJ2, RFJ2, LFJ2 and thumb tip in FRAME_NAMES.
        finger_center = frame_world[[2, 6, 10, 15]].mean(axis=0)
        center = 0.5 * (finger_center + frame_world[21])
        return center, finger_center

    def _initialize_reference(self):
        wrist_rot, object_rot, wrist_pos, _, frame_world, _, _ = self._kinematics()
        center, finger_center = self._hand_center(frame_world)
        self._initial_hand_center_local = wrist_rot.inv().apply(center - wrist_pos).astype(np.float32)
        self._initial_wrist_object_rotation = object_rot.inv() * wrist_rot
        self._initial_wrist_world_rotation = wrist_rot
        grasp = center - wrist_pos
        grasp /= max(np.linalg.norm(grasp), 1e-8)
        across = finger_center - center
        across /= max(np.linalg.norm(across), 1e-8)
        self._initial_grasp_axis_object = object_rot.inv().apply(grasp).astype(np.float32)
        self._initial_across_axis_object = object_rot.inv().apply(across).astype(np.float32)
        self._initial_across_axis_wrist = wrist_rot.inv().apply(across).astype(np.float32)

    def _read_raisim_hand_state(self) -> np.ndarray:
        """Return Shadow_Test generalized coordinates from the SuperDex pose."""
        raw = _read_pose(self.backend.hand)[6:]
        result = raw.copy()
        world_translation = raw[:3] + self._scene_origin
        result[:3] = self._initial_root_rotation.inv().apply(
            world_translation - self._initial_root_position_world
        )
        world_rotation = Rotation.from_euler("XYZ", raw[3:6])
        result[3:6] = (
            self._initial_root_rotation.inv() * world_rotation
        ).as_euler("XYZ")
        return result.astype(np.float32)

    def _raisim_target_to_superdex(self, target: np.ndarray) -> np.ndarray:
        """Map a relative Shadow_Test PD target to explicit SuperDex joints."""
        target = np.asarray(target, np.float32)
        result = target.copy()
        world_translation = self._initial_root_position_world + (
            self._initial_root_rotation.apply(target[:3])
        )
        result[:3] = world_translation - self._scene_origin
        world_rotation = self._initial_root_rotation * Rotation.from_euler(
            "XYZ", target[3:6]
        )
        result[3:6] = world_rotation.as_euler("XYZ")
        return result.astype(np.float32)

    def reset_state(self, init_state_r, init_state_l=None, init_vel_r=None,
                    init_vel_l=None, obj_pose=None):
        q = np.asarray(init_state_r, np.float32)
        if q.ndim == 2:
            q = q[0]
        if q.shape != (28,):
            raise ValueError(f"Shadow reset state must have shape (28,), got {q.shape}")
        object_state = np.asarray(obj_pose, np.float32)
        if object_state.ndim == 2:
            object_state = object_state[0]
        if object_state.shape not in ((7,), (8,)):
            raise ValueError("object pose must have shape (7,) or Raisim-compatible (8,)")

        # Shadow_Test installs q[:6] as the articulated-system base transform
        # and starts its six relative root joints at zero.
        self._initial_root_position_world = q[:3].copy()
        self._initial_root_rotation = Rotation.from_euler("XYZ", q[3:6])
        raisim_q = q.copy()
        raisim_q[:6] = 0.0
        hand_pose = np.zeros(34, np.float32)
        hand_pose[6:] = self._raisim_target_to_superdex(raisim_q)
        superdex_object = _raisim_object_to_superdex(object_state, self._scene_origin)
        self.backend.scene.set_gravity([0.0, 0.0, -9.81])
        self.backend.hand.set_articulated_pose_from_joints(hand_pose)
        self.backend.hand.set_articulated_joint_velocities(np.zeros(34, np.float32))
        self.backend.obj.set_articulated_pose_from_joints(superdex_object)
        self.backend.obj.set_articulated_joint_velocities(np.zeros(7, np.float32))
        self.backend.control_target = hand_pose.copy()
        self.backend.hand.reset_articulated_target_pose(hand_pose)
        self._last_raisim_target = raisim_q.copy()
        self._initial_object_height = float(superdex_object[2])
        self._object_support_z = float(superdex_object[2])
        self._object_reset_pose = object_state.copy()
        self._steps = 0
        self._lift = False
        self._lift_start_step = None
        # Match Raisim reset_state: publish transforms/contacts without
        # advancing dynamics. Advancing four warm-up steps here created a
        # large upward wrist velocity before the policy's first action.
        self.backend.scene.step(0.0)
        self.backend.hand.set_articulated_joint_velocities(np.zeros(34, np.float32))
        self.backend.obj.set_articulated_joint_velocities(np.zeros(7, np.float32))
        self._initialize_reference()
        self._ready = True
        self._update_observation()

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.seed(seed)
        q = np.zeros(28, np.float32)
        q[:3] = [0.35, 0.0, 0.65]
        q[6 + 18] = 1.7  # same Shadow thumb pre-shape used by official eval.py
        obj = np.array([0.35, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0, 0.0], np.float32)
        self.reset_state(q, obj_pose=obj)
        return self._observation.copy()

    def set_goals(self, target_center, object_center, *unused):
        target = np.asarray(target_center, np.float32)
        center = np.asarray(object_center, np.float32)
        self._goal_center = target.reshape(-1, 3)[0].copy()
        self._object_center = center.reshape(-1, 3)[0].copy()
        if self._ready:
            self._initialize_reference()
            self._update_observation()

    def switch_root_guidance(self, is_on: bool):
        # Shadow_Test uses this call at step 130 to enter its lift phase.
        self.root_guided = bool(is_on)
        self._lift = True
        if self._lift_start_step is None:
            self._lift_start_step = self._steps

    def switch_root_guidance_after_reset(self, is_on: bool):
        self._root_guidance_requested = bool(is_on)

    def _update_observation(self):
        (wrist_rot, object_rot, wrist_pos, object_pos, frame_world,
         frame_object, relative_rotation) = self._kinematics()
        top_force, bottom_force = self._contact_arrays()
        q = self._read_raisim_hand_state()

        center_world = wrist_pos + wrist_rot.apply(self._initial_hand_center_local)
        target_world = object_pos + object_rot.apply(self._goal_center)
        target_difference = wrist_rot.inv().apply(target_world - center_world)
        desired_wrist = object_rot * self._initial_wrist_object_rotation
        # Shadow_Test observes joint rotation coordinates minus the desired
        # wrist rotation expressed relative to the reset wrist frame.
        desired_from_reset = self._initial_wrist_world_rotation.inv() * desired_wrist
        orientation_difference = q[3:6] - desired_from_reset.as_euler("XYZ")

        if not self._contain_non_aff:
            bottom_force = np.zeros_like(bottom_force)
        # Raisim exposes per-contact impulse from the final physics integrate.
        contacts_top, impulse_top, contacts_bottom, impulse_bottom = (
            self._contact_point_channels(top_force, bottom_force)
        )
        torque_proxy = self._last_raisim_target - q
        # Raisim uses palm-frame link velocities, not generalized joint rates.
        wrist_linear = wrist_rot.inv().apply(_vec3(self._palm_actor.get_linear_velocity()))
        wrist_angular = wrist_rot.inv().apply(_vec3(self._palm_actor.get_angular_velocity()))
        object_linear_world = _vec3(self._top_actor.get_linear_velocity())
        object_angular_world = _vec3(self._top_actor.get_angular_velocity())
        object_linear = wrist_rot.inv().apply(object_linear_world) - wrist_linear
        object_angular = wrist_rot.inv().apply(object_angular_world) - wrist_angular
        base = np.concatenate((
            target_difference, orientation_difference, q[6:], torque_proxy,
            wrist_linear, wrist_angular, contacts_top, impulse_top,
            object_linear, object_angular, contacts_bottom, impulse_bottom,
        )).astype(np.float32)
        if base.shape != (BASE_OBSERVATION_DIM,):
            raise RuntimeError(f"invalid Shadow base observation shape {base.shape}")

        affordance = relative_rotation.apply(
            _nearest_vectors(frame_object, self._top_surface)
        ).astype(np.float32)
        non_affordance = relative_rotation.apply(
            _nearest_vectors(frame_object, self._bottom_surface)
        ).astype(np.float32)
        if not self._contain_non_aff:
            non_affordance.fill(0.0)
        observation = np.concatenate((base, affordance.reshape(-1), non_affordance.reshape(-1)))
        if observation.shape != (OBSERVATION_DIM,) or not np.isfinite(observation).all():
            raise RuntimeError("Shadow cross-hand observation is not finite 268-D")
        self._observation = observation.astype(np.float32)

        frame_y_local = np.tile(np.array([0.0, -1.0, 0.0]), (22, 1))
        # Shadow_Test switches this basis for i > 16 (LF tip and thumb).
        frame_y_local[17:] = np.array([-0.717, 0.0, -0.717])
        frame_y_object = np.empty((22, 3), np.float32)
        hand_tf = _transforms(self.backend.hand)
        for i, name in enumerate(FRAME_NAMES):
            frame_rot = _rotation(hand_tf[self._link_indices[name]])
            frame_y_object[i] = object_rot.inv().apply(frame_rot.apply(frame_y_local[i]))
        obj_pose_wrist = (wrist_rot.inv() * object_rot).as_euler("XYZ")
        # Match Shadow_Test: both axes are the reset-frame vectors rotated by
        # the current wrist, rather than recomputed from deformed finger links.
        grasp_world = wrist_rot.apply(self._initial_hand_center_local)
        grasp_world /= max(np.linalg.norm(grasp_world), 1e-8)
        grasp_delta = object_rot.inv().apply(grasp_world) - self._initial_grasp_axis_object
        across_world = wrist_rot.apply(self._initial_across_axis_wrist)
        across_now = object_rot.inv().apply(across_world)
        quat_xyzw = wrist_rot.as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        object_q = _read_pose(self.backend.obj)
        obj_state = _superdex_object_to_raisim(object_q, self._scene_origin)
        global_state = np.concatenate((
            obj_pose_wrist, frame_y_object.reshape(-1), frame_object.reshape(-1),
            object_pos + self._scene_origin, grasp_delta,
            [float(np.dot(across_now, self._initial_across_axis_object))],
            wrist_pos + self._scene_origin, quat_wxyz, q[6:], obj_state,
        )).astype(np.float32)
        if global_state.shape != (GLOBAL_STATE_DIM,) or not np.isfinite(global_state).all():
            raise RuntimeError(f"invalid Shadow global state shape {global_state.shape}")
        self._global_state = global_state

        group_forces = np.array((
            top_force[1:4].sum(), top_force[4:7].sum(), top_force[7:10].sum(),
            top_force[10:14].sum(), top_force[14:17].sum(),
        ), np.float32)
        frame_surface_distance = np.linalg.norm(
            _nearest_vectors(frame_object, self._top_surface), axis=1
        ).astype(np.float32)
        finger_surface_distance = np.array((
            frame_surface_distance[1:5].min(),
            frame_surface_distance[5:9].min(),
            frame_surface_distance[9:13].min(),
            frame_surface_distance[13:18].min(),
            frame_surface_distance[18:22].min(),
        ), np.float32)
        target_distance = float(np.linalg.norm(target_world - center_world))
        wrist_object_distance = float(np.linalg.norm(object_pos - wrist_pos))
        controller_force = self._controller_force()
        self._last_info = {
            "contact_force": float(top_force.sum() + bottom_force.sum()),
            "controller_force": controller_force,
            "wrist_controller_force_norm": float(np.linalg.norm(controller_force[:6])),
            "finger_controller_force_norm": float(np.linalg.norm(controller_force[6:])),
            "finger_groups": int(np.count_nonzero(group_forces > 0.1)),
            "lift_height": float(object_pos[2] - self._initial_object_height),
            "object_height": float(object_pos[2]),
            "target_distance": target_distance,
            "wrist_object_distance": wrist_object_distance,
            "wrist_position": wrist_pos.copy(),
            "object_position": object_pos.copy(),
            "top_forces": top_force,
            "bottom_forces": bottom_force,
            "wrist_linear_speed": float(np.linalg.norm(wrist_linear)),
            "wrist_angular_speed": float(np.linalg.norm(wrist_angular)),
            "object_linear_speed": float(np.linalg.norm(object_linear_world)),
            "object_vertical_velocity": float(object_linear_world[2]),
            "object_angular_speed": float(np.linalg.norm(object_angular_world)),
            "control_error_norm": float(np.linalg.norm(torque_proxy)),
            "root_control_error_norm": float(np.linalg.norm(torque_proxy[:6])),
            "hand_state": q.copy(),
            "control_target": self._last_raisim_target.copy(),
            "control_error": torque_proxy.copy(),
            "frame_surface_distance": frame_surface_distance,
            "finger_surface_distance": finger_surface_distance,
            "lift_ratio": float(self._current_lift_ratio()),
            "step": self._steps,
        }

    def _current_lift_ratio(self) -> float:
        if not self._lift:
            return 0.0
        if self._lift_ramp_steps <= 0 or self._lift_start_step is None:
            return 1.0
        return min(max(
            (self._steps - self._lift_start_step + 1) / self._lift_ramp_steps,
            0.0,
        ), 1.0)

    def _enforce_table_support(self):
        if self._physical_table:
            return
        pose = _read_pose(self.backend.obj)
        if pose[2] >= self._object_support_z:
            return
        pose[2] = self._object_support_z
        velocity = _read_velocity(self.backend.obj)
        velocity[2] = max(float(velocity[2]), 0.0)
        # Approximate static table friction while supported. Once lifted, the
        # object remains unconstrained and retains its complete velocity.
        velocity[0:2] *= 0.2
        self.backend.obj.set_articulated_pose_from_joints(pose)
        self.backend.obj.set_articulated_joint_velocities(velocity)

    def _controller_force(self):
        try:
            values = list(self.backend.hand.get_articulated_controller_force())
        except Exception:
            # Reset publishes an observation before the first physics step;
            # query results become available after that first integration.
            return np.zeros(ACTION_DIM, dtype=np.float32)
        raw = np.asarray(values, dtype=np.float32)
        if raw.shape != (self.backend.hand.get_num_dofs(),):
            raise RuntimeError(f"unexpected controller force shape {raw.shape}")
        return raw[6:].copy()

    def _capture_substep_diagnostics(self, substep_index: int):
        """Capture contact-point and motion data after one physics integration."""
        target_hashes = {
            actor.get_handle().get_hash(): name
            for name, actor in self.backend.object_links.items()
        }
        contacts = []
        link_force = {}
        for link_name in CONTACT_LINK_NAMES:
            actor = self._contact_actors[link_name]
            own_hash = actor.get_handle().get_hash()
            for point in actor.get_contact_points_world():
                hash_a = point.actor_a.get_hash()
                hash_b = point.actor_b.get_hash()
                other_hash = hash_b if hash_a == own_hash else hash_a
                object_link = target_hashes.get(other_hash)
                if object_link is None:
                    continue
                force = _vec3(point.force)
                force_norm = float(np.linalg.norm(force))
                link_force[link_name] = link_force.get(link_name, 0.0) + force_norm
                velocity_a = _vec3(point.point_velocity_a)
                velocity_b = _vec3(point.point_velocity_b)
                contacts.append({
                    "hand_link": link_name,
                    "object_link": object_link,
                    "distance": float(point.distance),
                    "position_a": _vec3(point.pos_a).tolist(),
                    "position_b": _vec3(point.pos_b).tolist(),
                    "normal": _vec3(point.normal).tolist(),
                    "force": force.tolist(),
                    "force_norm": force_norm,
                    "relative_speed": float(np.linalg.norm(velocity_a - velocity_b)),
                })
        object_linear = _vec3(self._top_actor.get_linear_velocity())
        object_angular = _vec3(self._top_actor.get_angular_velocity())
        controller_force = self._controller_force()
        return {
            "substep": int(substep_index),
            "time_offset": float((substep_index + 1) * self._physics_dt),
            "contact_count": len(contacts),
            "total_point_force": float(sum(x["force_norm"] for x in contacts)),
            "max_point_force": float(max((x["force_norm"] for x in contacts), default=0.0)),
            "link_forces": link_force,
            "object_linear_velocity": object_linear.tolist(),
            "object_angular_velocity": object_angular.tolist(),
            "controller_force": controller_force.tolist(),
            "wrist_controller_force_norm": float(np.linalg.norm(controller_force[:6])),
            "finger_controller_force_norm": float(np.linalg.norm(controller_force[6:])),
            "contacts": contacts,
        }

    def step(self, action_r, action_l=None):
        if not self._ready:
            raise RuntimeError("call reset/reset_state before step")
        action = np.asarray(action_r, np.float32)
        if action.ndim == 2:
            action = action[0]
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("Shadow action must be finite with shape (28,) or (1, 28)")
        current = self._read_raisim_hand_state()
        target = current + action * self._action_scale

        if self.root_guided:
            wrist_rot, object_rot, wrist_pos, object_pos, frame_world, _, _ = self._kinematics()
            center_world = wrist_pos + wrist_rot.apply(self._initial_hand_center_local)
            target_world = object_pos + object_rot.apply(self._goal_center)
            # The checkpoint was trained against the explicit x/y/z root-joint
            # coordinates. Preserve those coordinates across the backend bridge.
            position_error = target_world - center_world
            position_error[2] += self._gravity_compensation
            if self._lift:
                position_error = position_error + np.array(
                    [0.0, 0.0, self._lift_guidance_height * self._current_lift_ratio()],
                    np.float32,
                )
            target[:3] += self._root_guidance_gain * self._initial_root_rotation.inv().apply(
                position_error
            )
            desired = object_rot * self._initial_wrist_object_rotation
            # Match SHADOW_TEST exactly: Raisim forms target_wrist_in_wrist
            # as current_wrist^T * desired_wrist, converts that relative
            # rotation to Euler coordinates, and then adds the residual action.
            # Using reset_wrist^T * desired_wrist here produced an accumulating
            # absolute target and caused the floating wrist to keep rotating.
            target[3:6] = (
                wrist_rot.inv() * desired
            ).as_euler("XYZ") + action[3:6] * self._action_scale[3:6]
        else:
            gravity_world = np.array([0.0, 0.0, self._gravity_compensation], np.float32)
            if self._lift:
                gravity_world[2] += self._lift_guidance_height * self._current_lift_ratio()
            target[:3] += self._initial_root_rotation.inv().apply(gravity_world)

        if self.freeze_wrist_rotation:
            target[3:6] = 0.0

        # Match Raisim: scale the raw residual action, add the current pose,
        # then clip the resulting target to the URDF joint limits.
        target = np.clip(target, self._joint_low, self._joint_high)
        self._last_raisim_target = target.copy()
        control_target = self.backend.control_target.copy()
        control_target[6:] = self._raisim_target_to_superdex(target)
        self.backend.control_target = control_target
        self.backend.hand.reset_articulated_target_pose(control_target)
        substep_diagnostics = []
        for substep_index in range(self._physics_substeps):
            self.backend.scene.step(self._physics_dt)
            self._enforce_table_support()
            if self._record_substep_diagnostics:
                substep_diagnostics.append(
                    self._capture_substep_diagnostics(substep_index)
                )
        self._steps += 1
        self._update_observation()
        if self._record_substep_diagnostics:
            self._last_info["substeps"] = substep_diagnostics
        terminated = not np.isfinite(self._observation).all()
        truncated = self._steps >= self.max_steps
        reward = float(min(self._last_info["finger_groups"], 2) * 0.5)
        return self._observation.copy(), reward, terminated, truncated, self.last_info

    def observe(self, contain_non_aff=None, **unused):
        if contain_non_aff is not None:
            value = np.asarray(contain_non_aff, np.float32).reshape(-1)
            if value.size:
                enabled = bool(value[0] >= 0.5)
                if enabled != self._contain_non_aff:
                    self._contain_non_aff = enabled
                    if self._ready:
                        self._update_observation()
        return self._observation[None].copy(), np.zeros((1, 44), np.float32)

    def get_global_state(self):
        return self._global_state[None].copy()

    @property
    def last_info(self):
        return dict(self._last_info)

    def attach_debugger(self) -> bool:
        """Open SuperDex native debugger for the current physics scene."""
        physics.get_debug_server().set_coordinate_space(
            physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
        )
        return bool(physics.debugger.attach())

    @staticmethod
    def debugger_is_attached() -> bool:
        return bool(physics.debugger.is_attached())

    def close(self):
        backend = getattr(self, "backend", None)
        if backend is not None:
            backend.close()
            self.backend = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
