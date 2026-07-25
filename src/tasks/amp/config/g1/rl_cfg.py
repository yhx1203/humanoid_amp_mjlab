"""RL configuration for Unitree G1 omnidirectional AMP walking."""

from dataclasses import dataclass
from typing import Tuple

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

G1_DEFAULT_JOINT_POS: Tuple[float, ...] = (
  -0.1,
  0.0,
  0.0,
  0.3,
  -0.2,
  0.0,
  -0.1,
  0.0,
  0.0,
  0.3,
  -0.2,
  0.0,
  0.0,
  0.0,
  0.0,
  0.35,
  0.18,
  0.0,
  0.87,
  0.0,
  0.0,
  0.0,
  0.35,
  -0.18,
  0.0,
  0.87,
  0.0,
  0.0,
  0.0,
)

G1_SEED_MOTIONS: Tuple[str, ...] = (
  "src/assets/motions/g1_seed/*.csv",
)


@dataclass
class AmpRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config fields consumed by :class:`AmpOnPolicyRunner`."""

  amp_motion_files: Tuple[str, ...] = G1_SEED_MOTIONS
  amp_motion_input_fps: float = 30.0
  amp_default_joint_pos: Tuple[float, ...] = G1_DEFAULT_JOINT_POS
  amp_num_preload_transitions: int = 200000
  amp_replay_buffer_size: int = 100000
  amp_batch_size: int | None = None
  amp_reward_coef: float = 1.0
  amp_reward_scale_by_dt: bool = True
  amp_task_reward_lerp: float = 0.8
  amp_discr_hidden_dims: Tuple[int, ...] = (1024, 512)
  amp_discr_learning_rate: float = 1.0e-4
  amp_discr_weight_decay: float = 1.0e-4
  amp_discr_head_weight_decay: float = 1.0e-3
  amp_grad_penalty_coef: float = 10.0
  amp_max_grad_norm: float = 1.0
  amp_normalize: bool = True
  amp_normalizer_clip: float | None = 5.0


def unitree_g1_amp_ppo_runner_cfg() -> AmpRslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for omnidirectional AMP walking."""
  return AmpRslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_amp_walking",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
