from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os

import numpy as np
# Notes on historical fixes:
# - Allegro: adjust body_parts mapping for more accurate visualization.
# - Shadow: fix node semantics (14=MIDDLE->proximal, 15=DISTAL->middle) due to metacarpal.
# - Shadow: change body_parts "tips_back" to "tip" for visualization.
# - Schunk: fix node semantics (6=MIDDLE->proximal, 7=DISTAL->middle) due to metacarpal.
# - Allegro action primitive fix: joint_12.0 ROT->FLEX, joint_13.0 FLEX->ROT.

FLEX = 0   # Flexion/Extension (bending)
ABD  = 1   # Abduction/Adduction (spreading)
ROT  = 2   # Axial Rotation (twisting)

class HandType(Enum):
    """Hand type enum.

    Values are used as indices/identifiers in observation buffers.
    """
    ALLEGRO = 0
    MANO = 1
    SHADOW = 2
    LEAP = 3
    INSPIRE = 4
    ABILITY = 5
    SCHUNK = 6
    BRAINCO = 7
    BARRETT = 8


@dataclass(frozen=True)
class CrossHandSpec:
    """Single source of truth for cross_hand hands.
    - `urdf_relpath` is relative to the GraspXL root (the folder that contains `rsc/`).
      In this repo that root is usually passed around as `home_path`.
    - `hand_model_r` is the filename expected by the C++ env cfg (under the corresponding rsc subdir).
    """

    hand: str
    hand_type: int
    urdf_relpath: str
    hand_model_r: str


CROSS_HAND_SPECS: dict[str, CrossHandSpec] = {
    "allegro": CrossHandSpec(
        hand="allegro",
        hand_type=HandType.ALLEGRO.value,
        urdf_relpath="rsc/allegro/allegro_low_friction.urdf",
        hand_model_r="allegro_low_friction.urdf",
    ),
    "shadow": CrossHandSpec(
        hand="shadow",
        hand_type=HandType.SHADOW.value,
        urdf_relpath="rsc/shadow/shadowhand_large.urdf",
        hand_model_r="shadowhand_large.urdf",
    ),
    "leap": CrossHandSpec(
        hand="leap",
        hand_type=HandType.LEAP.value,
        urdf_relpath="rsc/leap_hand/leap_hand_right_envusing.urdf",
        hand_model_r="leap_hand_right_envusing.urdf",
    ),
    "inspire": CrossHandSpec(
        hand="inspire",
        hand_type=HandType.INSPIRE.value,
        urdf_relpath="rsc/inspire_hand/inspire_hand_right_envusing.urdf",
        hand_model_r="inspire_hand_right_envusing.urdf",
    ),
    "ability": CrossHandSpec(
        hand="ability",
        hand_type=HandType.ABILITY.value,
        urdf_relpath="rsc/ability_hand/ability_hand_right_envusing.urdf",
        hand_model_r="ability_hand_right_envusing.urdf",
    ),
    "schunk": CrossHandSpec(
        hand="schunk",
        hand_type=HandType.SCHUNK.value,
        urdf_relpath="rsc/schunk_hand/schunk_svh_hand_right_envusing.urdf",
        hand_model_r="schunk_svh_hand_right_envusing.urdf",
    ),
    "brainco": CrossHandSpec(
        hand="brainco",
        hand_type=HandType.BRAINCO.value,
        urdf_relpath="rsc/brainco/revo2_right_hand_envusing.urdf",
        hand_model_r="revo2_right_hand_envusing.urdf",
    ),
    "barrett": CrossHandSpec(
        hand="barrett",
        hand_type=HandType.BARRETT.value,
        urdf_relpath="rsc/barrett_hand/bhand_model_right_envusing.urdf",
        hand_model_r="bhand_model_right_envusing.urdf",
    ),
}


def list_cross_hand_names() -> list[str]:
    return sorted(CROSS_HAND_SPECS.keys())


def get_cross_hand_spec(hand: str) -> CrossHandSpec:
    if hand not in CROSS_HAND_SPECS:
        raise ValueError(f"Unknown hand: {hand}. Valid: {list_cross_hand_names()}")
    return CROSS_HAND_SPECS[hand]


def get_cross_hand_urdf_path(home_path: str, hand: str) -> str:
    """Return absolute URDF path based on the provided GraspXL root (`home_path`)."""
    spec = get_cross_hand_spec(hand)
    return os.path.join(home_path, spec.urdf_relpath)


# Unified registry: hand name -> base rotation matrix (3x3)
# Used by initial_pose_final.get_initial_pose_faive / get_initial_pose_faive_label
HAND_BASE_ROT_MAT: dict[str, np.ndarray] = {
    "faive": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32),
    "shadow": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
    "allegro": np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32),
    #"leap": np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
    "leap": np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
    "inspire": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float32),
    "ability": np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], dtype=np.float32),
    "schunk": np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], dtype=np.float32),
    "brainco": np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32),
    "barrett": np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32),
}


def get_hand_base_rot_mat(hand: str) -> np.ndarray:
    if hand not in HAND_BASE_ROT_MAT:
        raise ValueError(f"Unknown hand for base rot mat: {hand}")
    return HAND_BASE_ROT_MAT[hand]


def observe_kwargs_for_hand_type(hand_type: int) -> dict:
    return {
        "allegro": hand_type == HandType.ALLEGRO.value,
        "leap": hand_type == HandType.LEAP.value,
        "shadow": hand_type == HandType.SHADOW.value,
        "inspire": hand_type == HandType.INSPIRE.value,
        "ability": hand_type == HandType.ABILITY.value,
        "schunk": hand_type == HandType.SCHUNK.value,
        "brainco": hand_type == HandType.BRAINCO.value,
        "barrett": hand_type == HandType.BARRETT.value,
    }
    
class FingerIdType(Enum):
    """Finger id enum for each node (for one-hot encoding).

    0: Wrist
    1: Thumb
    2: Index
    3: Middle
    4: Ring
    5: Pinky
    """
    WRIST = 0
    THUMB = 1
    INDEX = 2
    MIDDLE = 3
    RING = 4
    PINKY = 5

class NodeTypeSemantic(Enum):
    """Node semantic type enum.

    0: Wrist
    1: Proximal phalanx
    2: Middle phalanx
    3: Distal phalanx
    4: Fingertip
    """
    WRIST = 0
    PROXIMAL = 1
    MIDDLE = 2
    DISTAL = 3
    FINGERTIP = 4
    #METACARPAL = 5




def get_finger_node_indices(data_dict_for_hand: dict):
    """Return node indices for each finger from a single-hand config of `get_data_dict()`.

    Args:
        data_dict_for_hand: a `data_dict[hand_type]` entry that must contain `finger_id_mapping`.

    Returns:
        dict: {
            "wrist": [..],
            "thumb": [..],
            "index": [..],
            "middle": [..],
            "ring": [..],
            "pinky": [..],
        }

    Notes:
        - Each element in `finger_id_mapping` is a `FingerIdType` (or an int convertible).
        - `keypoints_num` can differ across hand types; this function assumes no fixed length.
    """
    if "finger_id_mapping" not in data_dict_for_hand:
        raise KeyError("data_dict_for_hand must contain 'finger_id_mapping'")

    mapping = data_dict_for_hand["finger_id_mapping"]
    out = {
        "wrist": [],
        "thumb": [],
        "index": [],
        "middle": [],
        "ring": [],
        "pinky": [],
    }

    for idx, fid in enumerate(mapping):
        # `fid` may be a FingerIdType or already an int.
        if isinstance(fid, FingerIdType):
            fid_val = fid.value
        else:
            fid_val = int(fid)

        if fid_val == FingerIdType.WRIST.value:
            out["wrist"].append(idx)
        elif fid_val == FingerIdType.THUMB.value:
            out["thumb"].append(idx)
        elif fid_val == FingerIdType.INDEX.value:
            out["index"].append(idx)
        elif fid_val == FingerIdType.MIDDLE.value:
            out["middle"].append(idx)
        elif fid_val == FingerIdType.RING.value:
            out["ring"].append(idx)
        elif fid_val == FingerIdType.PINKY.value:
            out["pinky"].append(idx)
        else:
            raise ValueError(f"Unknown FingerIdType value: {fid_val}")

    return out


def get_data_dict(
    TRAIN_MANO: bool = False,
    TRAIN_LEAP: bool = False,
    TRAIN_ALLEGRO: bool = False,
    TRAIN_SHADOW: bool = False,
    TRAIN_SCHUNK: bool = False,
    TRAIN_INSPIRE: bool = False,
    TRAIN_ABILITY: bool = False,
    TRAIN_BRAINCO: bool = False,
    TRAIN_BARRETT: bool = False,
):
    data_dict = {}
    if TRAIN_MANO:
        data_dict[HandType.MANO.value] = {
            "obs_dim": 304,
            "act_dim": 51,
            "finger_num": 5,
            "keypoints_num": 21,
            "finger_weights_scale": 21.0,
            "fingertips_id": [4, 8, 12, 16, 20],
            "connection": [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
            ],
            "finger_id_mapping": [
                FingerIdType.WRIST,
                FingerIdType.THUMB, FingerIdType.THUMB, FingerIdType.THUMB, FingerIdType.THUMB,
                FingerIdType.INDEX, FingerIdType.INDEX, FingerIdType.INDEX, FingerIdType.INDEX,
                FingerIdType.MIDDLE, FingerIdType.MIDDLE, FingerIdType.MIDDLE, FingerIdType.MIDDLE,
                FingerIdType.RING, FingerIdType.RING, FingerIdType.RING, FingerIdType.RING,
                FingerIdType.PINKY, FingerIdType.PINKY, FingerIdType.PINKY, FingerIdType.PINKY,
            ],
            "node_semantic_mapping": [
                NodeTypeSemantic.WRIST,
                NodeTypeSemantic.PROXIMAL, NodeTypeSemantic.MIDDLE, NodeTypeSemantic.DISTAL, NodeTypeSemantic.FINGERTIP,
                NodeTypeSemantic.PROXIMAL, NodeTypeSemantic.MIDDLE, NodeTypeSemantic.DISTAL, NodeTypeSemantic.FINGERTIP,
                NodeTypeSemantic.PROXIMAL, NodeTypeSemantic.MIDDLE, NodeTypeSemantic.DISTAL, NodeTypeSemantic.FINGERTIP,
                NodeTypeSemantic.PROXIMAL, NodeTypeSemantic.MIDDLE, NodeTypeSemantic.DISTAL, NodeTypeSemantic.FINGERTIP,
                NodeTypeSemantic.PROXIMAL, NodeTypeSemantic.MIDDLE, NodeTypeSemantic.DISTAL, NodeTypeSemantic.FINGERTIP,
            ],
            "body_parts": [
                "right_wrist_0rz",
                "right_index1_x", "right_index2_x", "right_index3_x", "right_index_tip",
                "right_middle1_x", "right_middle2_x", "right_middle3_x", "right_middle_tip",
                "right_pinky1_x", "right_pinky2_x", "right_pinky3_x", "right_pinky_tip",
                "right_ring1_x", "right_ring2_x", "right_ring3_x", "right_ring_tip",
                "right_thumb1_x", "right_thumb2_x", "right_thumb3_x", "right_thumb_tip",
            ],
        }
    if TRAIN_LEAP:
        data_dict[HandType.LEAP.value] = {
            "obs_dim":210,
            "act_dim":22,
            "finger_num":4, 
            "keypoints_num":17,  # Number of keypoints used in graph modeling.
            "finger_weights_scale":17.0,
            "fingertips_id":[4,8,12,16],
            "connection":[
                (0,1), (1,2), (2,3), (3,4),    # index
                (0,5), (5,6), (6,7), (7,8),    # middle
                (0,9), (9,10), (10,11), (11,12), # ring
                (0,13), (13,14), (14,15), (15,16) # thumb
            ], 
            "finger_id_mapping": [
                FingerIdType.WRIST,  # 0
                FingerIdType.INDEX,  # 1
                FingerIdType.INDEX,  # 2
                FingerIdType.INDEX,  # 3
                FingerIdType.INDEX,  # 4 (fingertip)
                FingerIdType.MIDDLE, # 5
                FingerIdType.MIDDLE, # 6
                FingerIdType.MIDDLE, # 7
                FingerIdType.MIDDLE, # 8 (fingertip)
                FingerIdType.RING,   # 9
                FingerIdType.RING,   # 10
                FingerIdType.RING,   # 11
                FingerIdType.RING,   # 12 (fingertip)
                FingerIdType.THUMB,  # 13
                FingerIdType.THUMB,  # 14
                FingerIdType.THUMB,  # 15
                FingerIdType.THUMB,  # 16 (fingertip)
            ],
            "node_semantic_mapping": [
                NodeTypeSemantic.WRIST,     # 0
                NodeTypeSemantic.PROXIMAL,  # 1
                NodeTypeSemantic.MIDDLE,    # 2
                NodeTypeSemantic.DISTAL,    # 3
                NodeTypeSemantic.FINGERTIP, # 4
                NodeTypeSemantic.PROXIMAL,  # 5
                NodeTypeSemantic.MIDDLE,    # 6
                NodeTypeSemantic.DISTAL,    # 7
                NodeTypeSemantic.FINGERTIP, # 8
                NodeTypeSemantic.PROXIMAL,  # 9
                NodeTypeSemantic.MIDDLE,    # 10
                NodeTypeSemantic.DISTAL,    # 11
                NodeTypeSemantic.FINGERTIP, # 12
                NodeTypeSemantic.PROXIMAL,  # 13
                NodeTypeSemantic.MIDDLE,    # 14
                NodeTypeSemantic.DISTAL,    # 15
                NodeTypeSemantic.FINGERTIP, # 16
            ],
            "body_parts": ["z_rotation_joint",
                "1", "0", "3", "index_tip",  
                "5", "4",  "7","middle_tip",
                "9", "8", "11", "ring_tip",
                "12", "14","15", "thumb_tip"]
        }
    if TRAIN_ALLEGRO:
        data_dict[HandType.ALLEGRO.value] = {
            "obs_dim":210,
            "act_dim":22,
            "finger_num":4,
            "keypoints_num":17,  
            "finger_weights_scale":17.0,
            "fingertips_id":[4,8,12,16],
            "connection":[
                (0,1), (1,2), (2,3), (3,4),    # index
                (0,5), (5,6), (6,7), (7,8),    # middle
                (0,9), (9,10), (10,11), (11,12), # pinky
                (0,13), (13,14), (14,15), (15,16) # thumb
            ],
            "finger_id_mapping": [
                FingerIdType.WRIST,  # 0
                FingerIdType.INDEX,  # 1
                FingerIdType.INDEX,  # 2
                FingerIdType.INDEX,  # 3
                FingerIdType.INDEX,  # 4 (fingertip)
                FingerIdType.MIDDLE, # 5
                FingerIdType.MIDDLE, # 6
                FingerIdType.MIDDLE, # 7
                FingerIdType.MIDDLE, # 8 (fingertip)
                FingerIdType.PINKY,  # 9
                FingerIdType.PINKY,  # 10
                FingerIdType.PINKY,  # 11
                FingerIdType.PINKY,  # 12 (fingertip)
                FingerIdType.THUMB,  # 13
                FingerIdType.THUMB,  # 14
                FingerIdType.THUMB,  # 15
                FingerIdType.THUMB,  # 16 (fingertip)
            ],
            "node_semantic_mapping": [
                NodeTypeSemantic.WRIST,     # 0
                NodeTypeSemantic.PROXIMAL,  # 1
                NodeTypeSemantic.MIDDLE,    # 2
                NodeTypeSemantic.DISTAL,    # 3
                NodeTypeSemantic.FINGERTIP, # 4
                NodeTypeSemantic.PROXIMAL,  # 5
                NodeTypeSemantic.MIDDLE,    # 6
                NodeTypeSemantic.DISTAL,    # 7
                NodeTypeSemantic.FINGERTIP, # 8
                NodeTypeSemantic.PROXIMAL,  # 9
                NodeTypeSemantic.MIDDLE,    # 10
                NodeTypeSemantic.DISTAL,    # 11
                NodeTypeSemantic.FINGERTIP, # 12
                NodeTypeSemantic.PROXIMAL,  # 13
                NodeTypeSemantic.MIDDLE,    # 14
                NodeTypeSemantic.DISTAL,    # 15
                NodeTypeSemantic.FINGERTIP, # 16
            ],
            "body_parts": ["z_rotation_joint",
                                             "joint_0.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
                                             "joint_4.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
                                             "joint_8.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
                                             "joint_12.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"]

        }
    if TRAIN_SHADOW:
        data_dict[HandType.SHADOW.value]={
            # Shadow env now includes LFJ5 (+lfmetacarpal contact), so num_bodyparts=22:
            # obs_dim = obDim_r_(=12+2*gcDim+4*num_contacts=136) + 6*num_bodyparts(=132) = 268
            "obs_dim":268,
            "act_dim":28, 
            "finger_num":5,
            "keypoints_num":22,
            "finger_weights_scale":22.0,
            "fingertips_id":[4,8,12,17,21],
            "connection":[
                (0,1), (1,2), (2,3), (3,4),    # index
                (0,5), (5,6), (6,7), (7,8),    # middle
                (0,9), (9,10), (10,11), (11,12), # ring
                (0,13), (13,14), (14,15), (15,16),(16,17), # pinky
                (0,18),(18,19), (19,20), (20,21) # thumb
            ],
            "finger_id_mapping": [
                FingerIdType.WRIST,  # 0
                FingerIdType.INDEX,  # 1
                FingerIdType.INDEX,  # 2
                FingerIdType.INDEX,  # 3
                FingerIdType.INDEX,  # 4 (fingertip)
                FingerIdType.MIDDLE, # 5
                FingerIdType.MIDDLE, # 6
                FingerIdType.MIDDLE, # 7
                FingerIdType.MIDDLE, # 8 (fingertip)
                FingerIdType.RING,   # 9
                FingerIdType.RING,   # 10
                FingerIdType.RING,   # 11
                FingerIdType.RING,   # 12 (fingertip)
                FingerIdType.PINKY,  # 13
                FingerIdType.PINKY,  # 14
                FingerIdType.PINKY,  # 15
                FingerIdType.PINKY,  # 16
                FingerIdType.PINKY,  # 17 (fingertip)
                FingerIdType.THUMB,  # 18
                FingerIdType.THUMB,  # 19
                FingerIdType.THUMB,  # 20
                FingerIdType.THUMB,  # 21 (fingertip)
            ],
            "node_semantic_mapping": [
                NodeTypeSemantic.WRIST,     # 0
                NodeTypeSemantic.PROXIMAL,  # 1
                NodeTypeSemantic.MIDDLE,    # 2
                NodeTypeSemantic.DISTAL,    # 3
                NodeTypeSemantic.FINGERTIP, # 4
                NodeTypeSemantic.PROXIMAL,  # 5
                NodeTypeSemantic.MIDDLE,    # 6
                NodeTypeSemantic.DISTAL,    # 7
                NodeTypeSemantic.FINGERTIP, # 8
                NodeTypeSemantic.PROXIMAL,  # 9
                NodeTypeSemantic.MIDDLE,    # 10
                NodeTypeSemantic.DISTAL,    # 11
                NodeTypeSemantic.FINGERTIP, # 12
#                NodeTypeSemantic.METACARPAL, # 13
                NodeTypeSemantic.PROXIMAL, # 13
                NodeTypeSemantic.PROXIMAL,  # 14
                NodeTypeSemantic.MIDDLE,    # 15
                NodeTypeSemantic.DISTAL,    # 16
                NodeTypeSemantic.FINGERTIP, # 17
                NodeTypeSemantic.PROXIMAL,  # 18
                NodeTypeSemantic.MIDDLE,    # 19
                NodeTypeSemantic.DISTAL,    # 20
                NodeTypeSemantic.FINGERTIP, # 21
            ],
            "body_parts": ["z_rotation_joint", 
                "FFJ3", "FFJ2", "FFJ1","FFtip", 
                "MFJ3", "MFJ2", "MFJ1","MFtip", 
                "RFJ3", "RFJ2", "RFJ1","RFtip", 
                "LFJ5","LFJ3", "LFJ2", "LFJ1","LFtip", 
                "THJ4", "THJ2", "THJ1","thtip"  ]

               
        }
    if TRAIN_SCHUNK:
        data_dict[HandType.SCHUNK.value] = {
            "obs_dim":264,
            "act_dim":26,
            "finger_num":5,
            "keypoints_num":22,
            "finger_weights_scale":22.0,
            "fingertips_id":[4,9,13,17,21],    
            "connection":[
                (0,1), (1,2), (2,3), (3,4),    # thumb
                (0,5), 
                    (5,6), (6,7), (7,8),(8,9),    # ring
                    (5,10), (10,11), (11,12),(12,13), #pinky
                (0,14),  (14,15), (15,16),(16,17), # index
                (0,18),(18,19), (19,20), (20,21) # middle
            ],
            "finger_id_mapping": [
                FingerIdType.WRIST,  # 0
                FingerIdType.THUMB,  # 1
                FingerIdType.THUMB,  # 2
                FingerIdType.THUMB,  # 3
                FingerIdType.THUMB,  # 4 (fingertip)
                FingerIdType.RING,   # 5
                FingerIdType.RING,   # 6
                FingerIdType.RING,   # 7
                FingerIdType.RING,   # 8
                FingerIdType.RING,   # 9 (fingertip)
                FingerIdType.PINKY,  # 10
                FingerIdType.PINKY,  # 11
                FingerIdType.PINKY,  # 12
                FingerIdType.PINKY,  # 13 (fingertip)
                FingerIdType.INDEX,  # 14
                FingerIdType.INDEX,  # 15
                FingerIdType.INDEX,  # 16
                FingerIdType.INDEX,  # 17 (fingertip)
                FingerIdType.MIDDLE, # 18
                FingerIdType.MIDDLE, # 19
                FingerIdType.MIDDLE, # 20
                FingerIdType.MIDDLE, # 21 (fingertip)
            ],
            "node_semantic_mapping": [
                NodeTypeSemantic.WRIST,     # 0
                NodeTypeSemantic.PROXIMAL,  # 1
                NodeTypeSemantic.MIDDLE,    # 2
                NodeTypeSemantic.DISTAL,    # 3
                NodeTypeSemantic.FINGERTIP, # 4
                NodeTypeSemantic.PROXIMAL, # 5                
#                NodeTypeSemantic.METACARPAL, # 5
                NodeTypeSemantic.PROXIMAL,  # 6
                NodeTypeSemantic.MIDDLE,    # 7
                NodeTypeSemantic.DISTAL,    # 8
                NodeTypeSemantic.FINGERTIP, # 9
                NodeTypeSemantic.PROXIMAL,  # 10
                NodeTypeSemantic.MIDDLE,    # 11
                NodeTypeSemantic.DISTAL,    # 12
                NodeTypeSemantic.FINGERTIP, # 13
                NodeTypeSemantic.PROXIMAL,  # 14
                NodeTypeSemantic.MIDDLE,    # 15
                NodeTypeSemantic.DISTAL,    # 16
                NodeTypeSemantic.FINGERTIP, # 17
                NodeTypeSemantic.PROXIMAL,  # 18
                NodeTypeSemantic.MIDDLE,    # 19
                NodeTypeSemantic.DISTAL,    # 20
                NodeTypeSemantic.FINGERTIP, # 21
            ],
            "body_parts": ["z_rotation_joint", 
                            "right_hand_Thumb_Flexion", "right_hand_j3", "right_hand_j4", "thtip_joint",  
                            "right_hand_j5",         
                            "right_hand_Ring_Finger", "right_hand_j12", "right_hand_j16", "rftip_joint", 
                            "right_hand_Pinky", "right_hand_j13", "right_hand_j17", "lftip_joint",     
                            "right_hand_Index_Finger_Proximal","right_hand_Index_Finger_Distal", "right_hand_j14", "fftip_joint", 
                            "right_hand_Middle_Finger_Proximal","right_hand_Middle_Finger_Distal", "right_hand_j15", "mftip_joint",]
        }
    if TRAIN_INSPIRE:
        data_dict[HandType.INSPIRE.value] = {
            "obs_dim":198,
            "act_dim":18,
            "finger_num":5,
            "keypoints_num":17,
            "finger_weights_scale":17.0,
            "fingertips_id":[4,7,10,13,16],
            "connection":[
                (0,1), (1,2), (2,3), (3,4),    # thumb
                (0,5), (5,6), (6,7),    # index
                (0,8),(8,9), (9,10),  # middle
                (0,11), (11,12),(12,13), # ring
                (0,14), (14,15),(15,16), # pinky
            ],
            "finger_id_mapping": [
                FingerIdType.WRIST,  # 0
                FingerIdType.THUMB,  # 1
                FingerIdType.THUMB,  # 2
                FingerIdType.THUMB,  # 3
                FingerIdType.THUMB,  # 4 (fingertip)
                FingerIdType.INDEX,  # 5
                FingerIdType.INDEX,  # 6
                FingerIdType.INDEX,  # 7 (fingertip)
                FingerIdType.MIDDLE, # 8
                FingerIdType.MIDDLE, # 9
                FingerIdType.MIDDLE, # 10 (fingertip)
                FingerIdType.RING,   # 11
                FingerIdType.RING,   # 12
                FingerIdType.RING,   # 13 (fingertip)
                FingerIdType.PINKY,  # 14
                FingerIdType.PINKY,  # 15
                FingerIdType.PINKY,  # 16 (fingertip)
            ],
            "node_semantic_mapping": [
                NodeTypeSemantic.WRIST,     # 0
                NodeTypeSemantic.PROXIMAL,  # 1
                NodeTypeSemantic.MIDDLE,    # 2
                NodeTypeSemantic.DISTAL,    # 3
                NodeTypeSemantic.FINGERTIP, # 4
                NodeTypeSemantic.PROXIMAL,  # 5
                NodeTypeSemantic.MIDDLE,    # 6
                NodeTypeSemantic.FINGERTIP, # 7
                NodeTypeSemantic.PROXIMAL,  # 8
                NodeTypeSemantic.MIDDLE,    # 9
                NodeTypeSemantic.FINGERTIP, # 10
                NodeTypeSemantic.PROXIMAL,  # 11
                NodeTypeSemantic.MIDDLE,    # 12
                NodeTypeSemantic.FINGERTIP, # 13
                NodeTypeSemantic.PROXIMAL,  # 14
                NodeTypeSemantic.MIDDLE,    # 15
                NodeTypeSemantic.FINGERTIP, # 16
            ],
            "body_parts": ["z_rotation_joint",
                            "thumb_proximal_pitch_joint", "thumb_intermediate_joint", "thumb_distal_joint", "thumb_tip_joint",
                            "index_proximal_joint", "index_intermediate_joint", "index_tip_joint",
                            "middle_proximal_joint", "middle_intermediate_joint", "middle_tip_joint",
                            "ring_proximal_joint", "ring_intermediate_joint", "ring_tip_joint",
                            "pinky_proximal_joint", "pinky_intermediate_joint", "pinky_tip_joint"]
        }
    if TRAIN_ABILITY:
        data_dict[HandType.ABILITY.value] = {
            "obs_dim":184,
            "act_dim":16,
            "finger_num":5,
            "keypoints_num":16,
            "finger_weights_scale":16.0,
            "fingertips_id":[3,6,9,12,15],    
            "connection":[
                (0,1), (1,2), (2,3),    # thumb
                (0,4),(4,5), (5,6),     # index
                (0,7),(7,8), (8,9), # middle
                (0,10), (10,11), (11,12), #ring
                (0,13), (13,14), (14,15), #pinky
            ],
            "finger_id_mapping": [
                FingerIdType.WRIST,  # 0
                FingerIdType.THUMB,  # 1
                FingerIdType.THUMB,  # 2
                FingerIdType.THUMB,  # 3 (fingertip)
                FingerIdType.INDEX,  # 4
                FingerIdType.INDEX,  # 5
                FingerIdType.INDEX,  # 6 (fingertip)
                FingerIdType.MIDDLE, # 7
                FingerIdType.MIDDLE, # 8
                FingerIdType.MIDDLE, # 9 (fingertip)
                FingerIdType.RING,   # 10
                FingerIdType.RING,   # 11
                FingerIdType.RING,   # 12 (fingertip)
                FingerIdType.PINKY,  # 13
                FingerIdType.PINKY,  # 14
                FingerIdType.PINKY,  # 15 (fingertip)
            ],
            "node_semantic_mapping": [
                NodeTypeSemantic.WRIST,     # 0
                NodeTypeSemantic.PROXIMAL,  # 1
                NodeTypeSemantic.MIDDLE,    # 2
                NodeTypeSemantic.FINGERTIP, # 3
                NodeTypeSemantic.PROXIMAL,  # 4
                NodeTypeSemantic.MIDDLE,    # 5
                NodeTypeSemantic.FINGERTIP, # 6
                NodeTypeSemantic.PROXIMAL,  # 7
                NodeTypeSemantic.MIDDLE,    # 8
                NodeTypeSemantic.FINGERTIP, # 9
                NodeTypeSemantic.PROXIMAL,  # 10
                NodeTypeSemantic.MIDDLE,    # 11
                NodeTypeSemantic.FINGERTIP, # 12
                NodeTypeSemantic.PROXIMAL,  # 13
                NodeTypeSemantic.MIDDLE,    # 14
                NodeTypeSemantic.FINGERTIP, # 15
            ],
            "body_parts": ["z_rotation_joint",    
                            "thumb_q1", "thumb_q2", "thumb_tip_joint", 
                            "index_q1", "index_q2", "index_tip_joint",
                            "middle_q1", "middle_q2", "middle_tip_joint", 
                            "ring_q1", "ring_q2", "ring_tip_joint", 
                            "pinky_q1", "pinky_q2", "pinky_tip_joint"]
        }
    if TRAIN_BRAINCO:
        """obs_space
        BrainCo notes:
        DoF: 17 = 6 wrist + 6 actuated + 5 passive/coupled.
        Here, `torque_error` and `joint angle` include both actuated + passive joints.
        On real hardware, passive joint states may be unavailable, and the control interface
        may not expose passive control dimensions. Therefore, the output action may need to
        be truncated to remove the passive part.

        obs_space consists of:
            target_center_dif,           // 3
            wrist_ori_error,             // 3
            joint angle,                // 11
            torque_error,                // 17
            wrist_vel,                   // 3
            wrist_qvel,                  // 3
            contacts_r_af,               // 11 (excluding fingertips)
            impulses_r_af,               // 11
            obj_vel_in_wrist,            // 3
            obj_qvel_in_wrist,           // 3
            contacts_r_non_af,           // 11
            impulses_r_non_af;           // 11
            af_vec;  // 3*16=48
            non_aff_vec_obs; //3*16=48
        total=3+3+11+17+3+3+11+11+3+3+11+11+48+48=186
        """
        data_dict[HandType.BRAINCO.value] = {
            "obs_dim":186, 
            "act_dim":17,  # 6 wrist + 6 actuated + 5 passive/coupled
            "finger_num":5,
            "keypoints_num":16,
            "finger_weights_scale":17.0,
            "fingertips_id":[3,6,9,12,15],
            "connection":[
                (0,1), (1,2),(2,3),   # thumb
                (0,4), (4,5), (5,6),    # index
                (0,7),(7,8), (8,9),  # middle
                (0,10), (10,11),(11,12), # ring
                (0,13), (13,14),(14,15), # pinky
            ],
            "finger_id_mapping": [
                FingerIdType.WRIST,  # 0
                FingerIdType.THUMB,  # 1
                FingerIdType.THUMB,  # 2
                FingerIdType.THUMB,  # 3 (fingertip)
                FingerIdType.INDEX,  # 4
                FingerIdType.INDEX,  # 5
                FingerIdType.INDEX,  # 6 (fingertip)
                FingerIdType.MIDDLE, # 7
                FingerIdType.MIDDLE, # 8
                FingerIdType.MIDDLE, # 9(fingertip)
                FingerIdType.RING,   # 10
                FingerIdType.RING,   # 11
                FingerIdType.RING,   # 12 (fingertip)
                FingerIdType.PINKY,  # 13
                FingerIdType.PINKY,  # 14
                FingerIdType.PINKY,  # 15 (fingertip)
            ],
            "node_semantic_mapping": [
                NodeTypeSemantic.WRIST,     # 0
                NodeTypeSemantic.PROXIMAL,  # 1
                NodeTypeSemantic.MIDDLE,    # 2
                NodeTypeSemantic.FINGERTIP,    # 3
                NodeTypeSemantic.PROXIMAL,  # 4
                NodeTypeSemantic.MIDDLE,    # 5
                NodeTypeSemantic.FINGERTIP, # 6
                NodeTypeSemantic.PROXIMAL,  # 7
                NodeTypeSemantic.MIDDLE,    # 8
                NodeTypeSemantic.FINGERTIP, # 9
                NodeTypeSemantic.PROXIMAL,  # 10
                NodeTypeSemantic.MIDDLE,    # 11
                NodeTypeSemantic.FINGERTIP, # 12
                NodeTypeSemantic.PROXIMAL,  # 13
                NodeTypeSemantic.MIDDLE,    # 14
                NodeTypeSemantic.FINGERTIP, # 15
            ],
            "body_parts": ["z_rotation_joint",
                           "right_thumb_metacarpal_joint","right_thumb_distal_joint","right_thumb_tip_joint",
                           "right_index_proximal_joint","right_index_distal_joint","right_index_tip_joint",
                           "right_middle_proximal_joint","right_middle_distal_joint","right_middle_tip_joint",
                           "right_ring_proximal_joint","right_ring_distal_joint","right_ring_tip_joint",
                           "right_pinky_proximal_joint","right_pinky_distal_joint","right_pinky_tip_joint"
                           ],
            "contact_bodies_r_":[ 
                                "right_base_link", 
                                "right_thumb_proximal_link", "right_thumb_distal_link", 
                                "right_index_proximal_link", "right_index_distal_link", 
                                "right_middle_proximal_link", "right_middle_distal_link",
                                "right_ring_proximal_link", "right_ring_distal_link",
                                "right_pinky_proximal_link", "right_pinky_distal_link"],
                                
        }
    if TRAIN_BARRETT:
        """obs_space
        Barrett notes:
        DoF: 14 = 6 wrist + 8 actuated.
        Here, `torque_error` and `joint angle` include both actuated + passive joints.
        On real hardware, passive joint states may be unavailable, and the control interface
        may not expose passive control dimensions. Therefore, the output action may need to
        be truncated to remove the passive part.

        obs_space consists of:
            target_center_dif,           // 3
            wrist_ori_error,             // 3
            joint angle,                // 8
            torque_error,                // 14
            wrist_vel,                   // 3
            wrist_qvel,                  // 3
            contacts_r_af,               // 8 (excluding fingertips)
            impulses_r_af,               // 8
            obj_vel_in_wrist,            // 3
            obj_qvel_in_wrist,           // 3
            contacts_r_non_af,           // 8
            impulses_r_non_af;           // 8
            af_vec;  // 3*9=27
            non_aff_vec_obs; //3*9=27
        total=3+3+8+14+3+3+8+8+3+3+8+8+27+27=126
        """
        data_dict[HandType.BARRETT.value] = {
            "obs_dim":186, 
            "act_dim":14,  # 6 wrist + 8 actuated
            "finger_num":3,
            "keypoints_num":9,
            "finger_weights_scale":17.0,
            "fingertips_id":[],# no fingertips for Barrett hand
            "connection":[
                (0,1), (1,2),(2,3),   # thumb
                (0,4),(4,5), (5,6),  # ring
                (0,7), (7,8), # middle
            ],
            "finger_id_mapping": [
                FingerIdType.WRIST,  # 0
                FingerIdType.RING, # 1
                FingerIdType.RING, # 2
                FingerIdType.RING, # 3
                FingerIdType.THUMB,  # 4
                FingerIdType.THUMB,  # 5
                FingerIdType.THUMB,  # 6 
                FingerIdType.MIDDLE,   # 7
                FingerIdType.MIDDLE,   # 8
            ],
            "node_semantic_mapping": [
                NodeTypeSemantic.WRIST,     # 0
                NodeTypeSemantic.PROXIMAL,  # 1
                NodeTypeSemantic.MIDDLE,    # 2
                NodeTypeSemantic.DISTAL,    # 3
                NodeTypeSemantic.PROXIMAL,  # 4
                NodeTypeSemantic.MIDDLE,    # 5
                NodeTypeSemantic.DISTAL,    # 6
                NodeTypeSemantic.PROXIMAL,  # 7
                NodeTypeSemantic.DISTAL,    # 8
            ],
            # Only 8 DoF, 9 nodes.
            "body_parts": ["z_rotation_joint",
                           "finger_1_prox_joint","finger_1_med_joint","finger_1_dist_joint", # RING has no tip joint
                           "finger_2_prox_joint","finger_2_med_joint","finger_2_dist_joint", # thumb
                           "finger_3_med_joint","finger_3_dist_joint", # middle finger (no root rotation)
                            ],

            "contact_bodies_r_":[ 
                                "right_base_link", 
                                "finger_1_prox_link", "finger_1_med_liink","finger_1_dist_link", 
                                "finger_2_prox_link", "finger_2_med_link", "finger_2_dist_link",
                                "finger_3_med_link", "finger_3_dist_link", # middle finger (no root rotation)
                                ],
                                
        }
    
    return data_dict

def build_joint_semantic_map():
    """
    Returns:
        dict: {hand_type: {joint_name: semantic_axis}}
        ignore root axis
        FLEX = 0   # Flexion/Extension (bending)     +1 flex forward   -1 extend backward
        ABD  = 1   # Abduction/Adduction (spreading) +1 towards thumb  -1 away from thumb
        ROT  = 2   # Axial Rotation (twisting)       +1 follows right-hand rule (right hand)

        Rules:
            The thumb may require special-case definitions.
            Semantic axis  Name                  Anatomical meaning                                   Positive direction (+1)        Example robot task
            FLEX           Flexion               Thumb IP/MCP flexion (towards the palm)              Flex                             Hooking during grasp
            ABD            Abduction             Abduction                                             Lift/open                        Opposition prep / pinching
            ROT            Internal rotation     Rotate thumb around metacarpal, tip towards palm     Oppose (internal rotation)      Thumb-pinky pinch
    """
    return {
        HandType.ALLEGRO.value: {
            "x_joint":None ,
            "y_joint":None ,
            "z_joint":None ,
            "x_rotation_joint":None ,
            "y_rotation_joint":None ,
            "z_rotation_joint":None ,
            # index
            "joint_0.0":[ROT,1] ,
            "joint_1.0":[FLEX,1] ,
            "joint_2.0":[FLEX,1] ,
            "joint_3.0":[FLEX,1] ,
            # middle
            "joint_4.0":[ROT,1] ,
            "joint_5.0":[FLEX,1] ,
            "joint_6.0":[FLEX,1] ,
            "joint_7.0":[FLEX,1] ,
            # ring
            "joint_8.0":[ROT,1] ,
            "joint_9.0": [FLEX,1],
            "joint_10.0":[FLEX,1] ,
            "joint_11.0": [FLEX,1],
            # thumb
            "joint_12.0":[FLEX,1]  , 
            "joint_13.0":[ROT,1] , 
            "joint_14.0":[FLEX,1] ,
            "joint_15.0":[FLEX,1] ,
        },
        HandType.SHADOW.value: {
            "x_joint":None ,
            "y_joint":None ,
            "z_joint":None ,
            "x_rotation_joint":None ,
            "y_rotation_joint":None ,
            "z_rotation_joint":None ,
            # index
            "FFJ4":[ABD,-1] ,
            "FFJ3":[FLEX,1] ,
            "FFJ2": [FLEX,1] ,
            "FFJ1":[FLEX,1]  ,
            # middle
            "MFJ4":[ABD,-1] ,
            "MFJ3": [FLEX,1] ,
            "MFJ2": [FLEX,1] ,
            "MFJ1":[FLEX,1]  ,
            # ring
            "RFJ4": [ABD,1],
            "RFJ3": [FLEX,1] ,
            "RFJ2": [FLEX,1] ,
            "RFJ1": [FLEX,1] ,
            # little/pinky
            "LFJ5": [FLEX,1],
            "LFJ4": [ABD,1] ,
            "LFJ3": [FLEX,1] ,
            "LFJ2": [FLEX,1] ,
            "LFJ1": [FLEX,1] ,
            #thumb
            "THJ5":[ROT,-1] ,
            "THJ4": [FLEX,1],  # Default pose: extends roughly perpendicular to the palm.
            "THJ3": [FLEX,1], 
            "THJ2": [FLEX,1],
            "THJ1": [FLEX,1],
        },
        HandType.LEAP.value: {
            "x_joint":None ,
            "y_joint": None,
            "z_joint":None ,
            "x_rotation_joint":None ,
            "y_rotation_joint":None ,
            "z_rotation_joint": None,
            # index
            "1":[FLEX,1] ,
            "0":[ABD,-1],
            "2":[FLEX,1],
            "3":[FLEX,1] ,
            # middle
            "5":[FLEX,1] ,
            "4": [ABD,-1],
            "6":[FLEX,1] ,
            "7":[FLEX,1] ,
            # pinky
            "9":[FLEX,1] ,
            "8":[ABD,-1] ,
            "10":[FLEX,1] ,
            "11":[FLEX,1] ,
            # thumb
            "12":[FLEX,1] ,
            "13":[ROT,1] ,
            "14":[FLEX,1] ,
            "15":[FLEX,1] ,
        },
        HandType.INSPIRE.value:{
            "x_joint":None, 
            "y_joint":None, 
            "z_joint":None, 
            "x_rotation_joint":None, 
            "y_rotation_joint":None, 
            "z_rotation_joint":None, 
            # thumb
            "thumb_proximal_yaw_joint":[FLEX,1], 
            "thumb_proximal_pitch_joint":[ABD,-1], 
            "thumb_intermediate_joint":[FLEX,1], 
            "thumb_distal_joint":[FLEX,1], 
            # index
            "index_proximal_joint":[FLEX,1], 
            "index_intermediate_joint":[FLEX,1], 
            # middle
            "middle_proximal_joint":[FLEX,1], 
            "middle_intermediate_joint":[FLEX,1], 
            # ring
            "ring_proximal_joint":[FLEX,1], 
            "ring_intermediate_joint":[FLEX,1], 
            # pinky
            "pinky_proximal_joint":[FLEX,1], 
            "pinky_intermediate_joint":[FLEX,1], 
        },
        HandType.ABILITY.value:{
            "x_joint":None,
            "y_joint":None,
            "z_joint":None, 
            "x_rotation_joint":None, 
            "y_rotation_joint":None,
            "z_rotation_joint":None, 
            # thumb
            "thumb_q1":[FLEX,-1], 
            "thumb_q2":[ABD,-1], 
            # index
            "index_q1":[FLEX,1], 
            "index_q2":[FLEX,1], 
            # middle
            "middle_q1":[FLEX,1], 
            "middle_q2":[FLEX,1], 
            # ring
            "ring_q1":[FLEX,1], 
            "ring_q2":[FLEX,1], 
            # pinky
            "pinky_q1":[FLEX,1], 
            "pinky_q2":[FLEX,1], 
        },
        HandType.SCHUNK.value:{
            "x_joint":None,
            "y_joint":None,
            "z_joint":None, 
            "x_rotation_joint":None,
            "y_rotation_joint":None,
            "z_rotation_joint":None, 
            # thumb
            "right_hand_Thumb_Opposition":[ABD,1], 
            "right_hand_Thumb_Flexion":[FLEX,1], 
            "right_hand_j3":[FLEX,1], 
            "right_hand_j4":[FLEX,1], 
            # additional point
            "right_hand_j5":[ROT,1], 
            # ring
            "right_hand_ring_spread":[ABD,-1], 
            "right_hand_Ring_Finger":[FLEX,1], 
            "right_hand_j12":[FLEX,1],
            "right_hand_j16":[FLEX,1], 
            # pinky
            "right_hand_Finger_Spread":[ABD,-1], 
            "right_hand_Pinky":[FLEX,1], 
            "right_hand_j13":[FLEX,1], 
            "right_hand_j17":[FLEX,1], 
            # index
            "right_hand_index_spread":[ABD,1], 
            "right_hand_Index_Finger_Proximal":[FLEX,1], 
            "right_hand_Index_Finger_Distal":[FLEX,1], 
            "right_hand_j14":[FLEX,1], 
            # middle
            "right_hand_Middle_Finger_Proximal":[FLEX,1], 
            "right_hand_Middle_Finger_Distal":[FLEX,1], 
            "right_hand_j15":[FLEX,1]
        },
        HandType.BRAINCO.value:{
            "x_joint":None,
            "y_joint":None,
            "z_joint":None, 
            "x_rotation_joint":None, 
            "y_rotation_joint":None,
            "z_rotation_joint":None, 
            # thumb
            "right_thumb_metacarpal_joint":[ABD,1], #TODO:examine
            "right_thumb_proximal_joint":[FLEX,1],
            "right_thumb_distal_joint":[FLEX,1],#mimic
            # index
            "right_index_proximal_joint":[FLEX,1], 
            "right_index_distal_joint":[FLEX,1], #mimic
            # middle
            "right_middle_proximal_joint":[FLEX,1], 
            "right_middle_distal_joint":[FLEX,1], #mimic
            # ring
            "right_ring_proximal_joint":[FLEX,1], 
            "right_ring_distal_joint":[FLEX,1], #mimic
            # pinky
            "right_pinky_proximal_joint":[FLEX,1], 
            "right_pinky_distal_joint":[FLEX,1], #mimic
        }, 
        HandType.BARRETT.value:{
            "x_joint":None,
            "y_joint":None,
            "z_joint":None, 
            "x_rotation_joint":None, 
            "y_rotation_joint":None,
            "z_rotation_joint":None, 
            # thumb
            "finger_1_prox_joint":[ABD,-1], # TODO: direction needs verification
            "finger_1_med_joint":[FLEX,-1],
            "finger_1_dist_joint":[FLEX,-1],
            # ring
            "finger_2_prox_joint":[ABD,-1], # TODO: direction needs verification
            "finger_2_med_joint":[FLEX,-1], 
            "finger_2_dist_joint":[FLEX,-1], 
            # MIDDLE
            "finger_3_med_joint":[FLEX,-1], 
            "finger_3_dist_joint":[FLEX,-1], 
        }, 
    }




def build_node_to_joint_sequence():
    """
    Returns:
        dict: {hand_type: [ [joint_names_for_node_0], [joint_names_for_node_1], ... ]}
    """
    return {
        HandType.ALLEGRO.value: [
            ["x_joint" , "y_joint" , "z_joint" ,  "x_rotation_joint" ,  "y_rotation_joint" , "z_rotation_joint"] ,# 0: root
            # index
            ["joint_0.0" ,                "joint_1.0"], #1
            ["joint_2.0" ], #2
            ["joint_3.0" ], #3
            [], #4
            # middle
            ["joint_4.0" ,                "joint_5.0"], #5
            ["joint_6.0"] ,#6
            ["joint_7.0" ],#7
            [],#8
            # ring
            ["joint_8.0" ,                "joint_9.0" ],#9
            ["joint_10.0"],#10
            ["joint_11.0" ],#11
            [],#12
            # thumb
            ["joint_12.0"  ,                 "joint_13.0" ], #13
            ["joint_14.0"],#14
            ["joint_15.0"],#15
            [],#16
        ],
        HandType.SHADOW.value: [
            ["x_joint", "y_joint", "z_joint", "x_rotation_joint", "y_rotation_joint", "z_rotation_joint"],  # node 0
            # index
            ["FFJ4",  "FFJ3"], #1
            ["FFJ2"] , #2
            ["FFJ1"] , #3
            [], #4
            # middle
            ["MFJ4" , "MFJ3"] , #5
            ["MFJ2" ], #6
            ["MFJ1"] , #7
            [], #8
            # ring
            ["RFJ4", "RFJ3" ], #9
            ["RFJ2" ], #10
            ["RFJ1" ], #11
            [],#12
            # little/pinky
            ["LFJ5"], #13
            ["LFJ4" , "LFJ3" ], #14
            ["LFJ2" ], #15
            ["LFJ1" ], #16
            [], #17
            #thumb
            ["THJ5" ,"THJ4"],  #18
            ["THJ3", "THJ2"], #19
            ["THJ1"], #20
            [], #21
        ],
        HandType.LEAP.value: [
            ["x_joint", "y_joint", "z_joint", "x_rotation_joint", "y_rotation_joint", "z_rotation_joint"],  # node 0
            # index
            ["1" ,                "0"], #1
            ["2"], #2
            ["3" ], #3
            [], #4
            # middle
            ["5" ,                "4"] , #5
            ["6" ],#6
            ["7" ],#7
            [],#8
            # pinky
            ["9" ,                "8" ],#9
            ["10" ],#10
            ["11" ],#11
            [],#12
            # thumb
            ["12" ,                "13" ],#13
            ["14"] ,#14
            ["15"],#15
            [],#16
        ],
        HandType.INSPIRE.value:[
            ["x_joint", "y_joint","z_joint","x_rotation_joint","y_rotation_joint","z_rotation_joint"], 
            # thumb
            ["thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint"],
            ["thumb_intermediate_joint"], 
            ["thumb_distal_joint"], 
            [],
            # index
            ["index_proximal_joint"], 
            ["index_intermediate_joint"], 
            [],
            # middle
            ["middle_proximal_joint"], 
            ["middle_intermediate_joint"], 
            [],
            # ring
            ["ring_proximal_joint"], 
            ["ring_intermediate_joint"], 
            [],
            # pinky
            ["pinky_proximal_joint"], 
            ["pinky_intermediate_joint"], 
            [],
        ],
        HandType.ABILITY.value:[
            ["x_joint", "y_joint","z_joint","x_rotation_joint","y_rotation_joint","z_rotation_joint"], 
            # thumb
            ["thumb_q1"], 
            ["thumb_q2"], 
            [],
            # index
            ["index_q1"], 
            ["index_q2"], 
            [],
            # middle
            ["middle_q1"], 
            ["middle_q2"], 
            [],
            # ring
            ["ring_q1"], 
            ["ring_q2"], 
            [],
            # pinky
            ["pinky_q1"], 
            ["pinky_q2"], 
            [],
        ],
        HandType.SCHUNK.value:[
            ["x_joint", "y_joint","z_joint","x_rotation_joint","y_rotation_joint","z_rotation_joint"], 
            # thumb
            ["right_hand_Thumb_Opposition", "right_hand_Thumb_Flexion"],
            ["right_hand_j3"], 
            ["right_hand_j4"], 
            [],
            # additional point
            ["right_hand_j5"], 
            # ring
            ["right_hand_ring_spread","right_hand_Ring_Finger"], 
            ["right_hand_j12"],
            ["right_hand_j16"], 
            [],
            # pinky
            ["right_hand_Finger_Spread","right_hand_Pinky"], 
            ["right_hand_j13"], 
            ["right_hand_j17"], 
            [],
            # index
            ["right_hand_index_spread","right_hand_Index_Finger_Proximal"], 
            ["right_hand_Index_Finger_Distal"], 
            ["right_hand_j14"], 
            [],
            # middle
            ["right_hand_Middle_Finger_Proximal"], 
            ["right_hand_Middle_Finger_Distal"], 
            ["right_hand_j15"],
            [],
        ],
        HandType.BRAINCO.value:[
            ["x_joint", "y_joint","z_joint","x_rotation_joint","y_rotation_joint","z_rotation_joint"], 
            # thumb
            ["right_thumb_metacarpal_joint","right_thumb_proximal_joint"], 
            ["right_thumb_distal_joint"], 
            [],
            # index
            ["right_index_proximal_joint"], 
            ["right_index_distal_joint"], 
            [],
            # middle
            ["right_middle_proximal_joint"], 
            ["right_middle_distal_joint"], 
            [],
            # ring
            ["right_ring_proximal_joint"], 
            ["right_ring_distal_joint"], 
            [],
            # pinky
            ["right_pinky_proximal_joint"], 
            ["right_pinky_distal_joint"], 
            [],
        ],
        HandType.BARRETT.value:[
            ["x_joint", "y_joint","z_joint","x_rotation_joint","y_rotation_joint","z_rotation_joint"], 
            # thumb
            ["finger_1_prox_joint"], 
            ["finger_1_med_joint"], 
            ["finger_1_dist_joint"], 
            # ring
            ["finger_2_prox_joint"], 
            ["finger_2_med_joint"], 
            ["finger_2_dist_joint"],
            # middle
            ["finger_3_med_joint"], 
            ["finger_3_dist_joint"],
        ],
    }


def build_dof_to_finger_id_map(hand_type: int, *, exclude_wrist: bool = True):
    """Build a mapping from each DOF in `finger_joints` to `FingerIdType`.

    Design goals:
        - Fully replace any heuristic that infers finger ownership from joint-name strings.
        - When adding a new hand, only `finger_id_mapping` and `build_node_to_joint_sequence()`
          should need updates.

    Conventions:
        - `get_data_dict()[hand_type]["finger_id_mapping"]` provides a finger id per node.
        - `build_node_to_joint_sequence()[hand_type]` provides a joint-name list per node.
        - DOF order follows the flattened order of `build_node_to_joint_sequence()`.
        - If `exclude_wrist=True`, joints in node 0 (root 6-DOF) are skipped, matching the
          common env convention: `finger_joints = gc_r_.tail(act_dim-6)`.

    Args:
        hand_type: `HandType.*.value`
        exclude_wrist: Whether to exclude wrist/root DOFs (default: True).

    Returns:
        list[FingerIdType]: Length equals the DOF dimension of `finger_joints`.
    """
    data_dict = get_data_dict(
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
    if hand_type not in data_dict:
        raise KeyError(f"Unknown hand_type={hand_type} for get_data_dict()")

    data_dict_for_hand = data_dict[hand_type]
    if "finger_id_mapping" not in data_dict_for_hand:
        raise KeyError(f"data_dict[{hand_type}] must contain 'finger_id_mapping'")

    finger_id_mapping = data_dict_for_hand["finger_id_mapping"]
    node2seq = build_node_to_joint_sequence()[hand_type]

    if len(finger_id_mapping) != len(node2seq):
        raise ValueError(
            f"finger_id_mapping len ({len(finger_id_mapping)}) must match node2seq len ({len(node2seq)})"
        )

    out = []
    for node_id, joint_names in enumerate(node2seq):
        if exclude_wrist and node_id == 0:
            continue

        fid = finger_id_mapping[node_id]
        if not isinstance(fid, FingerIdType):
            fid = FingerIdType(int(fid))

        for jn in joint_names:
            if not isinstance(jn, str) or len(jn) == 0:
                raise ValueError(f"Invalid joint name in node2seq[{node_id}]: {jn!r}")
            out.append(fid)

    return out
