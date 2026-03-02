"""Training modules for fine-tuning and GRPO."""

from .finetune import train as train_finetune

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


