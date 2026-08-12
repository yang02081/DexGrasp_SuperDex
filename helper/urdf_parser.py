import math
from typing import Dict, List, Optional

import numpy as np
import torch
try:
    from .constants import (
    HandType,
    build_joint_semantic_map,
    build_node_to_joint_sequence,
    get_data_dict,
    FLEX,
    ABD,
    ROT,
    )
except:
    from helper.constants import (
    HandType,
    build_joint_semantic_map,
    build_node_to_joint_sequence,
    get_data_dict,
    FLEX,
    ABD,
    ROT,
    )
from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R
from collections import deque, defaultdict
import xml.etree.ElementTree as ET

class URDFJointPoseExtractor:
    def _patch_urdf(self, urdf_path):
        """
        Patch the URDF before parsing by adding missing `velocity` and `effort` attributes
        to `<limit>` tags. This avoids ParseError from strict `urdf_parser_py` validation.
        """
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            # Iterate over all <joint> tags
            for joint in root.findall('joint'):
                # Only process non-'fixed' joints, since fixed joints have no <limit>
                if joint.get('type') != 'fixed':
                    limit = joint.find('limit')
                    if limit is not None:
                        # Add a default velocity if it is missing
                        if 'velocity' not in limit.attrib:
                            limit.set('velocity', '0')
                        # Add a default effort if it is missing
                        if 'effort' not in limit.attrib:
                            limit.set('effort', '0')
            
            # Convert the patched XML tree back to a string
            return ET.tostring(root, encoding='unicode')
            
        except ET.ParseError as e:
            raise ValueError(f"URDF XML format error: {e}")
        except FileNotFoundError:
            raise

    def __init__(self, urdf_path, base_link="world"):
        """
        Initialize the URDF parser.
        :param urdf_path: URDF file path
        :param base_link: Base frame name
        """
        # 0. Patch the URDF before loading
        patched_urdf_string = self._patch_urdf(urdf_path)

        # 1. Load the URDF from the patched string
        self.robot = URDF.from_xml_string(patched_urdf_string)
        self.base_link = base_link
        
        # 2. Build joint/link mappings
        self.joint_map = {j.name: j for j in self.robot.joints}
        self.children_map = defaultdict(list)
        for joint in self.robot.joints:
            self.children_map[joint.parent].append(joint)
        
        # 3. Verify that base_link exists
        if self.base_link not in [link.name for link in self.robot.links]:
            # Try to find root nodes
            all_parents = set(j.parent for j in self.robot.joints)
            all_children = set(j.child for j in self.robot.joints)
            roots = all_parents - all_children
            print(f"Warning: Base link '{self.base_link}' is not in the URDF. Suggested root nodes: {roots}")

    def _origin_to_transform(self, origin):
        """Convert a URDF <origin> to a 4x4 homogeneous transform matrix."""
        if origin is None:
            return np.eye(4)
        
        xyz = np.array(origin.xyz) if origin.xyz is not None else np.zeros(3)
        rpy = np.array(origin.rpy) if origin.rpy is not None else np.zeros(3)
        
        T = np.eye(4)
        T[:3, 3] = xyz
        # In URDF/ROS, rpy(roll, pitch, yaw) usually corresponds to
        # R = Rz(yaw) * Ry(pitch) * Rx(roll).
        # scipy's from_euler('xyz', ...) (intrinsic) produces that form.
        rot = R.from_euler('xyz', rpy, degrees=False)
        T[:3, :3] = rot.as_matrix()
        
        return T

    def _joint_rotation_transform(self, joint, angle):
        """Compute the joint's 4x4 rotation transform from its type, axis, and angle."""
        if joint.type == 'fixed' or angle == 0:
            return np.eye(4)

        T_rot = np.eye(4)
        if joint.type in ['revolute', 'continuous']:
            axis = np.array(joint.axis, dtype=np.float64) if joint.axis is not None else np.array([1.0, 0.0, 0.0])
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0:
                axis = axis / axis_norm
            # Axis-angle rotation
            rot_vec = angle * axis
            rotation = R.from_rotvec(rot_vec)
            T_rot[:3, :3] = rotation.as_matrix()
        
        return T_rot

    def get_all_poses(self, joint_angles=None, *, joint_pose_frame: str = "child"):
        """
        Compute poses for all links and joints under the given joint angles.
        
        :param joint_pose_frame:
            - "origin": joint_poses returns the joint origin frame (without the joint's own rotation)
            - "child":  joint_poses returns the frame after applying the joint rotation
                        (equivalent to the child link pose at that joint)
        """
        if joint_angles is None:
            joint_angles = {}

        if joint_pose_frame not in {"origin", "child"}:
            raise ValueError(f"joint_pose_frame must be 'origin' or 'child', got: {joint_pose_frame}")

        link_poses = {self.base_link: np.eye(4)}
        joint_poses = {}
        
        queue = deque([self.base_link])
        visited_links = {self.base_link}

        while queue:
            parent_link_name = queue.popleft()
            
            for child_joint in self.children_map.get(parent_link_name, []):
                child_link_name = child_joint.child
                
                if child_link_name in visited_links:
                    continue
                
                # 1. Get the parent link pose
                T_base_to_parent = link_poses[parent_link_name]
                
                # 2. Compute the joint pose (joint frame = parent link frame + origin)
                T_parent_to_joint_origin = self._origin_to_transform(child_joint.origin)
                T_base_to_joint = T_base_to_parent @ T_parent_to_joint_origin

                # 3. Compute the child link / rotated joint pose (child link frame = joint frame + rotation)
                angle = joint_angles.get(child_joint.name, 0.0)
                T_joint_rotation = self._joint_rotation_transform(child_joint, angle)
                
                T_base_to_child = T_base_to_joint @ T_joint_rotation
                link_poses[child_link_name] = T_base_to_child

                # joint_poses semantics can be either origin (without self-rotation) or child (with self-rotation)
                joint_poses[child_joint.name] = T_base_to_joint if joint_pose_frame == "origin" else T_base_to_child
                
                # 4. Enqueue
                visited_links.add(child_link_name)
                queue.append(child_link_name)
                
        return link_poses, joint_poses

    def get_joint_pose_in_base(self, joint_name, joint_angles=None):
        """Get the pose of a single joint under the given joint angles."""
        _, joint_poses = self.get_all_poses(joint_angles, joint_pose_frame="child")
        if joint_name not in joint_poses:
            raise ValueError(f"Joint '{joint_name}' not found or not reachable from base_link '{self.base_link}'")
        T = joint_poses[joint_name]
        return T[:3, 3], T[:3, :3]

    def get_joint_axis_in_base(self, joint_name, joint_angles=None):
        """Return the joint axis direction in the base frame (unit vector) and a point on
        the axis (the joint origin position).

        Note: the URDF axis is defined in the joint frame (determined by origin).
        We use joint_pose_frame='origin' to compute that frame's orientation in base.
        """
        if joint_name not in self.joint_map:
            raise ValueError(f"Joint '{joint_name}' not found in URDF")

        joint = self.joint_map[joint_name]
        if joint.type not in {"revolute", "continuous"}:
            raise ValueError(f"Joint '{joint_name}' is type '{joint.type}', no rotation axis")

        _, joint_poses_origin = self.get_all_poses(joint_angles, joint_pose_frame="origin")
        if joint_name not in joint_poses_origin:
            raise ValueError(f"Joint '{joint_name}' not reachable from base_link '{self.base_link}'")

        T = joint_poses_origin[joint_name]
        R_base_joint = T[:3, :3]
        p_base = T[:3, 3]

        axis_local = np.array(joint.axis, dtype=np.float64) if joint.axis is not None else np.array([1.0, 0.0, 0.0])
        axis_norm = np.linalg.norm(axis_local)
        if axis_norm > 0:
            axis_local = axis_local / axis_norm
        axis_base = R_base_joint @ axis_local

        axis_base_norm = np.linalg.norm(axis_base)
        if axis_base_norm > 0:
            axis_base = axis_base / axis_base_norm

        return p_base, axis_base

    def get_all_joint_axes_in_base(self, joint_angles=None):
        """Return all revolute joint axes expressed in the base frame.

        Returns:
            axis_info: dict[joint_name] = {"position": (3,), "axis": (3,)}

        Semantics:
            - position: joint origin position in the base frame
            - axis: unit joint-axis direction in the base frame
        """
        if joint_angles is None:
            joint_angles = {}

        # Joint axes are defined in the joint origin frame, so the origin frame is sufficient.
        _, joint_poses_origin = self.get_all_poses(joint_angles, joint_pose_frame="origin")

        axis_info = {}
        for joint in self.robot.joints:
            if joint.type not in {"revolute", "continuous"}:
                continue
            if joint.name not in joint_poses_origin:
                continue

            T = joint_poses_origin[joint.name]
            R_base_joint = T[:3, :3]
            p_base = T[:3, 3]

            axis_local = np.array(joint.axis, dtype=np.float64) if joint.axis is not None else np.array([1.0, 0.0, 0.0])
            axis_norm = np.linalg.norm(axis_local)
            if axis_norm > 0:
                axis_local = axis_local / axis_norm

            axis_base = R_base_joint @ axis_local
            axis_base_norm = np.linalg.norm(axis_base)
            if axis_base_norm > 0:
                axis_base = axis_base / axis_base_norm

            axis_info[joint.name] = {"position": p_base, "axis": axis_base}

        return axis_info



class URDFGraphParser:
    """Builds URDF-derived node features for the graph state space.

    This parser implements the "specific graph" node_feature described in
    `version_ctrl.md`:

        - joint limits: lower/upper/velocity (semantic FLEX/ABD/ROT order) => 9
        - semantic axis direction vectors in wrist/base frame => 9
        - node vector (parent->child) using representative joints => 3
        - damping => 1

    Total: 22 dims per node. The root node is zero-padded.

    Notes
    -----
    - Semantic axis mapping and sign convention come from `build_joint_semantic_map()`.
    - Node->joints mapping comes from `build_node_to_joint_sequence()`.
    - Representative joints for node vectors come from `get_data_dict()[hand_type]["body_parts"]`.
    """

    LIMIT_RANGE = (-math.pi, math.pi)
    VELOCITY_RANGE = (0.0, 10.0)
    DAMPING_RANGE = (0.0, 10.0)

    def __init__(self, base_link: str = "world"):
        self.base_link = base_link
        self.features: Dict[int, Dict[str, torch.Tensor]] = {}

        self.joint_semantic_map = build_joint_semantic_map()
        self.node_to_joint_sequence = build_node_to_joint_sequence()

        # Pull representative joint lists + topology from the canonical data_dict.
        self._data_dict_all = get_data_dict(
            TRAIN_MANO=True,
            TRAIN_LEAP=True,
            TRAIN_ALLEGRO=True,
            TRAIN_SHADOW=True,
            TRAIN_SCHUNK=True,
            TRAIN_INSPIRE=True,
            TRAIN_ABILITY=True,
            TRAIN_BRAINCO=True,
            TRAIN_BARRETT=True,
        )

    @staticmethod
    def _minmax(value: float, vmin: float, vmax: float) -> float:
        if vmax <= vmin:
            return 0.0
        value = float(value)
        value = max(min(value, vmax), vmin)
        return (value - vmin) / (vmax - vmin)

    def _get_parent_map(self, hand_type: int) -> Dict[int, int]:
        d = self._data_dict_all.get(hand_type)
        if d is None:
            raise ValueError(f"hand_type={hand_type} not found in get_data_dict()")
        parent_of: Dict[int, int] = {}
        for parent, child in d["connection"]:
            parent_of[child] = parent
        return parent_of

    def _get_body_parts(self, hand_type: int) -> List[str]:
        d = self._data_dict_all.get(hand_type)
        if d is None:
            raise ValueError(f"hand_type={hand_type} not found in get_data_dict()")
        return d["body_parts"]

    def parse_urdf(self, urdf_path: str, hand_type: int):
        # Validate hand_type
        if hand_type not in self.node_to_joint_sequence:
            raise ValueError(f"hand type not supported (no node_to_joint_sequence): {hand_type}")
        if hand_type not in self.joint_semantic_map:
            raise ValueError(f"hand type not supported (no joint_semantic_map): {hand_type}")

        extractor = URDFJointPoseExtractor(urdf_path=urdf_path, base_link=self.base_link)

        # Static geometry: use zero angles.
        joint_angles: Dict[str, float] = {}

        # Joint origin poses for positions
        _, joint_poses_origin = extractor.get_all_poses(joint_angles, joint_pose_frame="origin")

        # Axes in base (from URDF <axis>, transformed by joint origin rotation)
        all_joint_axes = extractor.get_all_joint_axes_in_base(joint_angles)

        # Prepare node-vector normalization scale (hand span)
        body_parts = self._get_body_parts(hand_type)
        rep_positions = []
        for jn in body_parts:
            T = joint_poses_origin.get(jn)
            # print(f"jn:{jn}")
            rep_positions.append(T[:3, 3])
        if len(rep_positions) >= 2:
            rep_positions_np = np.stack(rep_positions, axis=0)
            span = float(np.linalg.norm(rep_positions_np.max(axis=0) - rep_positions_np.min(axis=0)))
        else:
            span = 0.0
        if span <= 1e-9:
            span = 1.0

        # Build per-node features
        node_joint_seq = self.node_to_joint_sequence[hand_type]
        joint_semantic = self.joint_semantic_map[hand_type]
        parent_of = self._get_parent_map(hand_type)

        num_nodes = len(node_joint_seq)
        # TODO: the 22-D feature size is hard-coded here; update carefully if you extend it.
        node_features = np.zeros((num_nodes, 22), dtype=np.float32)

        # Helper: fill limit/axis arrays in semantic order
        for node_id in range(num_nodes):
            if node_id == 0:
                continue  # root: all zeros

            limit_lower = np.zeros(3, dtype=np.float32)
            limit_upper = np.zeros(3, dtype=np.float32)
            velocity = np.zeros(3, dtype=np.float32)

            axis_sem = np.zeros((3, 3), dtype=np.float32)  # [FLEX, ABD, ROT] x xyz
            damping_val = 0.0

            joint_names = node_joint_seq[node_id]
            for jn in joint_names:
                sem = joint_semantic.get(jn)
                if sem is None: # maybe fixed joint
                    continue
                axis_type, sign = sem
                if axis_type not in (FLEX, ABD, ROT):
                    raise ValueError(f"Unexpected axis_type: {axis_type} for joint '{jn}'")

                joint_obj = extractor.joint_map.get(jn)
                if joint_obj is None:
                    raise ValueError(f"Joint '{jn}' not found in URDF")

                # Limits
                if getattr(joint_obj, "limit", None) is not None:
                    lo = joint_obj.limit.lower if joint_obj.limit.lower is not None else 0.0
                    up = joint_obj.limit.upper if joint_obj.limit.upper is not None else 0.0
                    vel = joint_obj.limit.velocity if joint_obj.limit.velocity is not None else 0.0
                else:
                    lo, up, vel = 0.0, 0.0, 0.0

                limit_lower[axis_type] = self._minmax(lo, *self.LIMIT_RANGE)
                limit_upper[axis_type] = self._minmax(up, *self.LIMIT_RANGE)
                velocity[axis_type] = self._minmax(vel, *self.VELOCITY_RANGE)

                # Axis direction (semantic positive direction)
                axis_info = all_joint_axes.get(jn)
                if axis_info is not None:
                    axis_vec = np.asarray(axis_info["axis"], dtype=np.float32) * float(sign)
                    axis_sem[axis_type, :] = axis_vec

                # Damping: pick max over joints on this node (stable, avoids order-dependence)
                dyn = getattr(joint_obj, "dynamics", None)
                if dyn is not None and getattr(dyn, "damping", None) is not None:
                    damping_val = max(damping_val, float(dyn.damping))

            damping_norm = self._minmax(damping_val, *self.DAMPING_RANGE)

            # Node vector (parent->child) using representative joints from body_parts
            parent_id = parent_of.get(node_id, 0)
            child_rep = body_parts[node_id] if node_id < len(body_parts) else None
            parent_rep = body_parts[parent_id] if parent_id < len(body_parts) else None
            if child_rep is not None and parent_rep is not None:
                Tc = joint_poses_origin.get(child_rep)
                Tp = joint_poses_origin.get(parent_rep)
                if Tc is not None and Tp is not None:
                    link_vec = (Tc[:3, 3] - Tp[:3, 3]) / span
                else:
                    link_vec = np.zeros(3, dtype=np.float32)
            else:
                link_vec = np.zeros(3, dtype=np.float32)

            # Pack: 9 + 9 + 3 + 1 = 22
            feat = np.concatenate(
                [
                    limit_lower, # 3
                    limit_upper, # 3
                    velocity, # 3
                    axis_sem.reshape(-1), # 9
                    link_vec.astype(np.float32), # 3
                    np.array([damping_norm], dtype=np.float32), # 1
                ],
                axis=0,
            )
            node_features[node_id, :] = feat

        feature = {
            "node_feature": torch.from_numpy(node_features),
        }
        self.features[hand_type] = feature

    def get_feature(self, hand_type: int) -> Dict[str, torch.Tensor]:
        if hand_type not in self.features:
            raise ValueError("hand type not found; call parse_urdf() first")
        return self.features[hand_type]
