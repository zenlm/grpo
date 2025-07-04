"""
GRPO: Guided Reinforcement Policy Optimization for LLM Fine-tuning
"""

__version__ = "0.1.0"

from .trainer import GRPOTrainer
from .dataset import GRPODataset
from .reward import RewardFunction

__all__ = ["GRPOTrainer", "GRPODataset", "RewardFunction"]