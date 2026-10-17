"""PPO package with inference-safe imports.

TensorBoard is a training-only dependency.  Keeping PPO lazy allows the MAGCN
model factory and checkpoints to be used in a minimal SuperDex runtime.
"""
from .module import Actor, Critic
from .storage import RolloutStorage

__all__ = ["Actor", "Critic", "RolloutStorage", "PPO"]


def __getattr__(name):
    if name == "PPO":
        from .ppo import PPO
        return PPO
    raise AttributeError(name)
