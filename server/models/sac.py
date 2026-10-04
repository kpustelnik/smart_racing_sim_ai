"""
SAC (Soft Actor-Critic) Trainer

An off-policy algorithm that maximizes both expected return and entropy.
Excellent sample efficiency and stable training for continuous control.

Hyperparameters:
- learning_rate: 0.0003
- batch_size: 512 (larger for off-policy replay buffer sampling)
- ent_coef: "auto" (automatic entropy tuning)
- buffer_size: 1_000_000 (replay buffer size)
"""

from typing import Any, Optional

from stable_baselines3 import SAC

from .base import ModelTrainer


class SACTrainer(ModelTrainer):
    """
    SAC Trainer - Soft Actor-Critic

    Best for:
    - Sample efficiency (off-policy)
    - Continuous action spaces
    - Automatic entropy tuning
    """

    NAME = "sac"
    ALGORITHM = SAC

    HYPERPARAMETERS = {
        "total_timesteps": 200_000,
        "learning_rate": 0.0003,
        "batch_size": 512,
        "ent_coef": "auto",
        "buffer_size": 1_000_000,
        "net_arch": [256, 256],
        "device": "cpu",
        "train_freq": 256,
        "gradient_steps": 256,
        "learning_starts": 5_000,
    }

    def create_model(self, env: Any, tensorboard_log: Optional[str] = None) -> SAC:
        """Create a new SAC model with configured hyperparameters."""
        hp = self.HYPERPARAMETERS

        return SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=hp["learning_rate"],
            batch_size=hp["batch_size"],
            ent_coef=hp["ent_coef"],
            buffer_size=hp["buffer_size"],
            train_freq=hp["train_freq"],
            gradient_steps=hp["gradient_steps"],
            learning_starts=hp["learning_starts"],
            policy_kwargs=dict(net_arch=hp["net_arch"]),
            device=hp["device"],
            tensorboard_log=tensorboard_log,
        )
