"""Unitree G1 omnidirectional AMP walking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.amp.amp_env_cfg import add_amp_observation_group
from src.tasks.amp.config.g1.velocity_env_cfgs import unitree_g1_flat_env_cfg

AMP_COMMAND_RANGES = {
  "lin_vel_x": (-0.5, 1.2),
  "lin_vel_y": (-0.5, 0.5),
  "ang_vel_z": (-1.0, 1.0),
}
AMP_NUM_ENVS = 4096

AMP_DISABLED_REWARDS = (
  "foot_gait",
  "pose",
  "foot_clearance",
  "angular_momentum",
  "stand_still",
)


def unitree_g1_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a Unitree G1 flat omnidirectional walking config."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  cfg.scene.num_envs = 1 if play else AMP_NUM_ENVS

  # AMP supplies the gait prior, so remove explicit gait/posture shaping terms.
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms.pop("phase", None)
  for reward_name in AMP_DISABLED_REWARDS:
    cfg.rewards.pop(reward_name, None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.05
  # Sample yaw rate directly so vx, vy, and wz are independent commands.
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.ranges.lin_vel_x = AMP_COMMAND_RANGES["lin_vel_x"]
  twist_cmd.ranges.lin_vel_y = AMP_COMMAND_RANGES["lin_vel_y"]
  twist_cmd.ranges.ang_vel_z = AMP_COMMAND_RANGES["ang_vel_z"]

  cfg.curriculum.pop("command_vel", None)

  add_amp_observation_group(cfg)
  return cfg
