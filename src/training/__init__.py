"""Training modules for fine-tuning and GRPO."""


def train_finetune(*args, **kwargs):
    """Lazy wrapper — imports finetune only when called.

    The eager ``from .finetune import train`` pulled in transformers/sklearn/
    scipy at package import time.  Spawned worker processes re-import this
    package via spawn_main, causing OpenBLAS to try to create 64 threads and
    hit the OS RLIMIT_NPROC limit, crashing every worker before it could run.
    """
    from .finetune import train  # noqa: PLC0415
    return train(*args, **kwargs)

try:
    from .soundness_reward import SoundnessReward, RewardResult
    from .grpo import GRPOTrainer, GRPOConfig
except ImportError:
    SoundnessReward = None
    RewardResult = None
    GRPOTrainer = None
    GRPOConfig = None

__all__ = [
    "train_finetune",
    "SoundnessReward",
    "RewardResult",
    "GRPOTrainer",
    "GRPOConfig",
]


