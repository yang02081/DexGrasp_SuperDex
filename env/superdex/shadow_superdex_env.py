"""Single-environment Shadow Hand control task using SuperDex physics."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
import trimesh

from .shadow_superdex_spec import (
    BASE_OBSERVATION_DIM, CONTACT_LINK_NAMES, POLICY_ACTION_DIM,
    SUPERDEX_DOF_DIM, SUPERDEX_FREE_ROOT_DOF, policy_to_superdex_action,
    validate_actor_layout,
)

DT = 0.0025
SUBSTEPS = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
SHADOW_URDF = REPOSITORY_ROOT / 'rsc/shadow/shadowhand_superdex.urdf'
SHADOW_RAISIM_GEOMETRY_URDF = (
    REPOSITORY_ROOT / 'rsc/shadow/shadowhand_raisim_geometry_superdex_valid_inertia.urdf'
)
YCB_ROOT = REPOSITORY_ROOT / 'rsc/ycb_crossDex'
DEFAULT_OBJECT_BOT = PACKAGE_ROOT / 'assets/14680ffbf.superdex_bot'
SHADOW_CONTACT_SEED = PACKAGE_ROOT / 'assets/shadow_contact_seed.npz'
DEFAULT_OBJECT_POSITION = np.array([0.35, 0.0, 0.35], np.float32)


def _read_pose(actor):
    out = np.zeros(actor.get_num_dofs(), np.float32)
    actor.get_articulated_pose(out)
    return out


def _read_velocity(actor):
    out = np.zeros(actor.get_num_dofs(), np.float32)
    actor.get_articulated_joint_velocities(out)
    return out


class ShadowSuperDexEnv:
    observation_shape = (BASE_OBSERVATION_DIM,)
    action_shape = (POLICY_ACTION_DIM,)

    def __init__(self, *, object_id='14680ffbf', object_root=None, seed=1, max_steps=450,
                 object_locked=True, gravity_z=-9.81, use_contact_seed=False, task='generic',
                 hand_contact_layer=None, object_contact_layer=None,
                 hand_urdf=None, ground_height=0.0, use_contact_layers=False,
                 tracking_stiffness=80.0, tracking_damping=10.0, tracking_saturation=20.0,
                 contact_friction=3.0, contact_normal_damping=20.60526466369629,
                 contact_friction_falloff=0.1, contact_penalty_coefficient=None,
                 contact_smoothing_half_distance=None, contact_max_alignment_normals=None):
        self.object_id = str(object_id)
        self.ycb_root = Path(object_root) if object_root is not None else YCB_ROOT
        default_object_urdf = self.ycb_root / self.object_id / f'{self.object_id}.urdf'
        superdex_object_urdf = self.ycb_root / self.object_id / f'{self.object_id}_superdex.urdf'
        self.object_urdf = (superdex_object_urdf if superdex_object_urdf.is_file()
                            else default_object_urdf)
        if not self.object_urdf.is_file():
            raise FileNotFoundError(self.object_urdf)
        self.rng = np.random.default_rng(seed)
        self.max_steps = int(max_steps)
        self.object_locked = bool(object_locked)
        self.gravity_z = float(gravity_z)
        if task not in ('generic', 'fixed_contact', 'lift'):
            raise ValueError('task must be generic, fixed_contact, or lift')
        self.task = task
        self.use_contact_seed = bool(use_contact_seed or task in ('fixed_contact', 'lift'))
        self.action_mask = np.array(([0.0] * 6 + [1.0] * 22) if task == 'fixed_contact' else [1.0] * 28, np.float32)
        self.contact_seed = None
        if self.use_contact_seed:
            if not SHADOW_CONTACT_SEED.is_file():
                raise FileNotFoundError(SHADOW_CONTACT_SEED)
            with np.load(SHADOW_CONTACT_SEED) as saved:
                self.contact_seed = {name: np.asarray(saved[name], np.float32).copy() for name in ('hand_pose', 'hand_target', 'object_pose')}
        self.closed = False
        physics.initialize(num_worker_threads=0)
        try:
            self.scene = physics.create_scene('DexGrasp Shadow SuperDex')
            ground_handle = self.scene.create_rigid_actor(
                name='ground',
                shape=physics.create_plane_shape(
                    normal=[0, 0, 1], distance=float(ground_height)
                ),
                is_static=True)
            self.ground = (ground_handle if hasattr(ground_handle, 'set_contact_layer')
                           else self.scene.get_actor(ground_handle))
            self.hand_urdf = Path(hand_urdf) if hand_urdf is not None else SHADOW_URDF
            self.context = robotics.create_context()
            self.hand_bot = robotics.create_bot(
                self.scene, robotics.load_bot_prefab_from_urdf_file(str(self.hand_urdf)), self.context)
            object_prefab = (
                robotics.load_bot_prefab_from_file(str(DEFAULT_OBJECT_BOT))
                if self.object_id == '14680ffbf' else
                robotics.load_bot_prefab_from_urdf_file(str(self.object_urdf))
            )
            self.object_bot = robotics.create_bot(self.scene, object_prefab, self.context)
            self.hand = self.hand_bot.get_articulated_actor()
            self.obj = self.object_bot.get_articulated_actor()
            # Raisim excludes the table collision bit from the hand mask,
            # while retaining object-table and hand-object collisions. Setting
            # layers on nested articulated actors crashes the current runtime;
            # actor-level filtering provides the same pair semantics safely.
            if use_contact_layers and self.ground is not None:
                self.scene.enable_actor_contact_symmetric(
                    self.hand.get_handle(), self.ground.get_handle(), False,
                    physics.IncludeNestedActors.YES,
                )
            # DexGrasp-Zero configures object/finger material friction to
            # 3.0 with an inelastic collision response. SuperDex defaults to
            # friction 0.5 and no normal damping, which caused grasp impacts to
            # launch the object. Apply the matched friction and a calibrated
            # low-restitution damping coefficient to both sides of each pair.
            for articulated in (self.hand, self.obj):
                for handle in articulated.get_nested_link_actors():
                    actor = self.scene.get_actor(handle)
                    if actor is None:
                        continue
                    try:
                        contact = actor.get_contact_params()
                    except Exception:
                        # Fixed/virtual links without collision geometry have
                        # no ContactParams component.
                        continue
                    contact.coulomb_friction_coefficient = float(contact_friction)
                    contact.friction_falloff_vel = float(contact_friction_falloff)
                    contact.normal_viscous_damping_coefficient = float(contact_normal_damping)
                    if contact_penalty_coefficient is not None:
                        contact.penalty_coefficient = float(contact_penalty_coefficient)
                    if contact_smoothing_half_distance is not None:
                        contact.penalty_smoothing_half_distance = float(contact_smoothing_half_distance)
                    if contact_max_alignment_normals is not None:
                        contact.max_alignment_normals = float(contact_max_alignment_normals)
                    actor.set_contact_params(contact)
            if self.ground is not None:
                contact = self.ground.get_contact_params()
                contact.coulomb_friction_coefficient = float(contact_friction)
                contact.friction_falloff_vel = float(contact_friction_falloff)
                contact.normal_viscous_damping_coefficient = float(contact_normal_damping)
                if contact_penalty_coefficient is not None:
                    contact.penalty_coefficient = float(contact_penalty_coefficient)
                if contact_smoothing_half_distance is not None:
                    contact.penalty_smoothing_half_distance = float(contact_smoothing_half_distance)
                if contact_max_alignment_normals is not None:
                    contact.max_alignment_normals = float(contact_max_alignment_normals)
                self.ground.set_contact_params(contact)
            if self.hand.get_num_dofs() != SUPERDEX_DOF_DIM or self.obj.get_num_dofs() != 7:
                raise RuntimeError('unexpected Shadow or object DoF count')
            hand_info = self.hand.get_articulated_shape_info()
            self.hand_indices = {name: i for i, name in enumerate(hand_info.link_names)}
            object_names = list(self.obj.get_articulated_shape_info().link_names)
            self.top_index = object_names.index('top')
            self.top_mesh = trimesh.load_mesh(self.ycb_root / self.object_id / 'top_watertight_tiny.obj', force='mesh')
            self.bottom_mesh = trimesh.load_mesh(self.ycb_root / self.object_id / 'bottom_watertight_tiny.obj', force='mesh')
            validate_actor_layout(list(hand_info.joint_names), list(hand_info.link_names))
            self.hand.add_boundary_condition_dofs_world_permanent(
                list(range(SUPERDEX_FREE_ROOT_DOF)), [0.0] * SUPERDEX_FREE_ROOT_DOF)
            # SuperDex controller gains are indexed by articulated link.
            # Shadow_Test uses D=0.1 for the three translational wrist joints
            # and D=0.2 for wrist rotations and finger joints; Kp=100 for all.
            num_links = len(hand_info.link_names)
            params = physics.PoseControllerParams(num_links)
            translation_joints = {"x_joint", "y_joint", "z_joint"}
            for index, (joint_name, dof) in enumerate(
                zip(hand_info.joint_names, hand_info.dof_info)
            ):
                if dof.get_size() <= 0 or joint_name == "world_joint":
                    continue
                params.joint_tracking[index] = physics.PoseTrackingParams(
                    stiffness=float(tracking_stiffness),
                    damping=(0.1 if joint_name in translation_joints
                             else float(tracking_damping)),
                    saturation=float(tracking_saturation),
                )
            self.hand.add_articulated_pose_controller(params)
            self.hand_links = self._query_by_name(self.hand, set(CONTACT_LINK_NAMES))
            self.object_links = self._query_by_name(self.obj, {'top', 'bottom'})
            if len(self.hand_links) != len(CONTACT_LINK_NAMES) or 'top' not in self.object_links:
                raise RuntimeError('required Shadow/object contact links are unavailable')
            self.query_handles = []
            if not self.hand.is_query_supported(physics.QueryType.ARTICULATED_CONTROLLER_FORCE):
                raise RuntimeError('Shadow hand does not support controller-force queries')
            self.query_handles.append(
                self.hand.register_query(physics.QueryType.ARTICULATED_CONTROLLER_FORCE)
            )
            for actor in [*self.hand_links.values(), *self.object_links.values()]:
                self.query_handles.append(actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE))
            self.initial_hand_pose = np.zeros(SUPERDEX_DOF_DIM, np.float32)
            self.initial_hand_pose[6:9] = np.array([0.25, 0.0, 0.35], np.float32)
            self.action_scale = np.array([0.003] * 6 + [0.02] * 22, np.float32)
            self.zero_hand_velocity = np.zeros(SUPERDEX_DOF_DIM, np.float32)
            self.zero_object_velocity = np.zeros(7, np.float32)
        except Exception:
            self.close()
            raise

    def _query_by_name(self, articulated, wanted):
        result = {}
        for handle in articulated.get_nested_link_actors():
            actor = self.scene.get_actor(handle)
            if actor is None:
                continue
            name = actor.get_name().rsplit('/', 1)[-1]
            if name in wanted and actor.is_query_supported(physics.QueryType.TOTAL_CONTACT_FORCE):
                result[name] = actor
        return result

    @staticmethod
    def _force(source, target):
        raw = source.get_contact_force_from_actor_world(target)
        return np.array([raw[0], raw[1], raw[2]], np.float32)

    def _contacts(self):
        top = np.zeros(16, np.float32)
        bottom = np.zeros(16, np.float32)
        top_actor = self.object_links['top']
        bottom_actor = self.object_links.get('bottom')
        for index, name in enumerate(CONTACT_LINK_NAMES):
            actor = self.hand_links[name]
            top[index] = np.linalg.norm(self._force(actor, top_actor))
            if bottom_actor is not None:
                bottom[index] = np.linalg.norm(self._force(actor, bottom_actor))
        return top, bottom

    def _base_observation(self, top_force, bottom_force):
        hand_q = _read_pose(self.hand)
        hand_v = _read_velocity(self.hand)
        obj_v = _read_velocity(self.obj)
        policy_q = hand_q[SUPERDEX_FREE_ROOT_DOF:]
        torque_proxy = self.control_target[SUPERDEX_FREE_ROOT_DOF:] - policy_q
        target_difference = DEFAULT_OBJECT_POSITION - policy_q[:3]
        orientation_difference = policy_q[3:6]
        top_contacts = (top_force > 0.1).astype(np.float32)
        bottom_contacts = (bottom_force > 0.1).astype(np.float32)
        impulse_scale = DT * SUBSTEPS
        obs = np.concatenate((
            target_difference, orientation_difference, policy_q[6:], torque_proxy,
            hand_v[6:9], hand_v[9:12], top_contacts, top_force * impulse_scale,
            obj_v[:3], obj_v[3:6], bottom_contacts, bottom_force * impulse_scale,
        )).astype(np.float32)
        if obs.shape != (BASE_OBSERVATION_DIM,) or not np.isfinite(obs).all():
            raise RuntimeError(f'invalid Shadow base observation {obs.shape}')
        return obs

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.scene.set_gravity([0.0, 0.0, 0.0 if self.object_locked else self.gravity_z])
        if self.contact_seed is not None:
            hand_pose = self.contact_seed['hand_pose'].copy()
            object_pose = self.contact_seed['object_pose'].copy()
        else:
            hand_pose = self.initial_hand_pose.copy()
            hand_pose[12:] += self.rng.uniform(-0.01, 0.01, 22).astype(np.float32)
            object_pose = np.zeros(7, np.float32)
            object_pose[:3] = DEFAULT_OBJECT_POSITION
        self.hand.set_articulated_pose_from_joints(hand_pose)
        self.hand.set_articulated_joint_velocities(self.zero_hand_velocity)
        self.obj.set_articulated_pose_from_joints(object_pose)
        self.obj.set_articulated_joint_velocities(self.zero_object_velocity)
        self.control_target = (self.contact_seed['hand_target'].copy()
                               if self.contact_seed is not None else hand_pose.copy())
        self.hand.reset_articulated_target_pose(self.control_target)
        self.fixed_object_pose = object_pose.copy()
        self.steps = 0
        self.hold_streak = 0
        self.initial_height = float(object_pose[2])
        for _ in range(25 if self.contact_seed is not None else 4):
            if self.object_locked:
                self.obj.set_articulated_pose_from_joints(self.fixed_object_pose)
                self.obj.set_articulated_joint_velocities(self.zero_object_velocity)
            self.scene.step(DT)
        top, bottom = self._contacts()
        return self._base_observation(top, bottom)

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (POLICY_ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError('action must be finite with shape (28,)')
        effective_action = np.clip(action, -1, 1) * self.action_mask
        policy_target = self.control_target[6:] + effective_action * self.action_scale
        self.control_target = policy_to_superdex_action(policy_target, self.control_target)
        self.hand.reset_articulated_target_pose(self.control_target)
        for _ in range(SUBSTEPS):
            if self.object_locked:
                self.obj.set_articulated_pose_from_joints(self.fixed_object_pose)
                self.obj.set_articulated_joint_velocities(self.zero_object_velocity)
            self.scene.step(DT)
        # The task contract requires one contact read per control step. Querying
        # every physics substep multiplies Shadow's 32 actor-pair queries by four.
        top, bottom = self._contacts()
        peak_force = float(np.sum(top + bottom))
        object_pose = _read_pose(self.obj)
        lift_height = float(object_pose[2] - self.initial_height)
        link_forces = top + bottom
        contact_count = int(np.count_nonzero(link_forces > 0.1))
        group_forces = np.array([link_forces[1:4].sum(), link_forces[4:7].sum(),
                                 link_forces[7:10].sum(), link_forces[10:13].sum(),
                                 link_forces[13:16].sum()], np.float32)
        finger_group_count = int(np.count_nonzero(group_forces > 0.1))
        total_force = float(group_forces.sum())
        valid_hold = finger_group_count >= 2 and 2.0 <= total_force <= 60.0
        self.hold_streak = self.hold_streak + 1 if valid_hold else 0
        ground = bool(object_pose[2] < 0.05)
        if self.task == 'fixed_contact':
            success = bool(self.hold_streak >= 20)
            reward = 0.25 * finger_group_count + (1.0 if finger_group_count >= 2 else 0.0)
            reward += 0.05 if valid_hold else 0.0
            reward -= 0.02 * max(total_force - 60.0, 0.0)
        else:
            success = bool(finger_group_count >= 2 and lift_height >= 0.04)
            reward = 0.1 * min(finger_group_count, 2) + 10.0 * max(lift_height, 0.0) + (10.0 if success else 0.0)
            if ground: reward -= 2.0
        self.steps += 1
        terminated = bool(success or (ground and not self.object_locked))
        truncated = bool(self.steps >= self.max_steps and not terminated)
        info = dict(contact_count=contact_count, finger_group_count=finger_group_count, group_forces=group_forces.copy(), hold_streak=self.hold_streak, stable_contact=self.hold_streak >= 20, top_forces=top.copy(), bottom_forces=bottom.copy(),
                    contact_force=float(np.sum(top + bottom)), peak_contact_force=peak_force,
                    lift_height=lift_height, is_success=success, ground=ground, step=self.steps)
        return self._base_observation(top, bottom), float(reward), terminated, truncated, info

    def close(self):
        if self.closed:
            return
        for handle in getattr(self, 'query_handles', []):
            del handle
        if getattr(self, 'object_bot', None) is not None:
            robotics.destroy_bot(self.scene, self.object_bot)
        if getattr(self, 'hand_bot', None) is not None:
            robotics.destroy_bot(self.scene, self.hand_bot)
        physics.shutdown()
        self.closed = True
