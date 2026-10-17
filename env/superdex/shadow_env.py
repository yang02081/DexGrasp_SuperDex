"""Public import for the SuperDex-backed Shadow cross-hand environment."""
from .shadow_crosshand_superdex_env import (
    ACTION_DIM,
    BASE_OBSERVATION_DIM,
    GLOBAL_STATE_DIM,
    HAND_TYPE,
    OBSERVATION_DIM,
    ShadowCrossHandSuperDexEnv,
)

__all__ = [
    "ACTION_DIM", "BASE_OBSERVATION_DIM", "GLOBAL_STATE_DIM", "HAND_TYPE",
    "OBSERVATION_DIM", "ShadowCrossHandSuperDexEnv",
]
