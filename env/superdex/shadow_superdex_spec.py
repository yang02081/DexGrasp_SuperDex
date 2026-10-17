"""Validated Shadow Hand dimensions and mappings for the SuperDex backend."""
from __future__ import annotations

import numpy as np

POLICY_ACTION_DIM = 28
SUPERDEX_DOF_DIM = 34
SUPERDEX_FREE_ROOT_DOF = 6
BASE_OBSERVATION_DIM = 132
GLOBAL_STATE_DIM = 136
GEOMETRY_FRAME_COUNT = 21
GEOMETRY_VECTOR_DIM = GEOMETRY_FRAME_COUNT * 3 * 2
POLICY_OBSERVATION_DIM = BASE_OBSERVATION_DIM + GEOMETRY_VECTOR_DIM

FRAME_NAMES = (
    'palm',
    'ffproximal', 'ffmiddle', 'ffdistal', 'fftip_back',
    'mfproximal', 'mfmiddle', 'mfdistal', 'mftip_back',
    'rfproximal', 'rfmiddle', 'rfdistal', 'rftip_back',
    'lfproximal', 'lfmiddle', 'lfdistal', 'lftip_back',
    'thproximal', 'thmiddle', 'thdistal', 'thtip_back',
)
CONTACT_LINK_NAMES = (
    'palm',
    'ffproximal', 'ffmiddle', 'ffdistal',
    'mfproximal', 'mfmiddle', 'mfdistal',
    'rfproximal', 'rfmiddle', 'rfdistal',
    'lfproximal', 'lfmiddle', 'lfdistal',
    'thproximal', 'thmiddle', 'thdistal',
)


def policy_to_superdex_action(action, current_dofs=None):
    """Insert a 28D policy target after SuperDex's automatically added free root."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (POLICY_ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError('Shadow policy action must be a finite (28,) array')
    if current_dofs is None:
        result = np.zeros(SUPERDEX_DOF_DIM, dtype=np.float32)
    else:
        result = np.asarray(current_dofs, dtype=np.float32).copy()
        if result.shape != (SUPERDEX_DOF_DIM,) or not np.isfinite(result).all():
            raise ValueError('Shadow SuperDex state must be a finite (34,) array')
    result[SUPERDEX_FREE_ROOT_DOF:] = action
    return result


def compose_policy_observation(base_observation, affordance_vectors, non_affordance_vectors):
    base = np.asarray(base_observation, dtype=np.float32)
    affordance = np.asarray(affordance_vectors, dtype=np.float32)
    non_affordance = np.asarray(non_affordance_vectors, dtype=np.float32)
    if base.shape != (BASE_OBSERVATION_DIM,):
        raise ValueError('Shadow base observation must have shape (132,)')
    expected = (GEOMETRY_FRAME_COUNT, 3)
    if affordance.shape != expected or non_affordance.shape != expected:
        raise ValueError('Shadow nearest-point vectors must each have shape (21, 3)')
    result = np.concatenate((base, affordance.reshape(-1), non_affordance.reshape(-1)))
    if result.shape != (POLICY_OBSERVATION_DIM,) or not np.isfinite(result).all():
        raise ValueError('Shadow policy observation must be finite with shape (258,)')
    return result


def validate_actor_layout(joint_names, link_names):
    joints = tuple(joint_names)
    links = set(link_names)
    expected_control_joints = (
        'x_joint', 'y_joint', 'z_joint',
        'x_rotation_joint', 'y_rotation_joint', 'z_rotation_joint',
    )
    if tuple(joints[2:8]) != expected_control_joints:
        raise RuntimeError('unexpected Shadow wrist joint order')
    missing = sorted(set(FRAME_NAMES + CONTACT_LINK_NAMES) - links)
    if missing:
        raise RuntimeError(f'Shadow actor is missing required links: {missing}')
    return dict(superdex_dofs=SUPERDEX_DOF_DIM, policy_dofs=POLICY_ACTION_DIM,
                free_root_dofs=SUPERDEX_FREE_ROOT_DOF)
