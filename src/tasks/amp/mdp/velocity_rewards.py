from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _log_metric(
  env: ManagerBasedRlEnv,
  name: str,
  value: torch.Tensor,
) -> None:
  env.extras.setdefault("log", {})[name] = value


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  return torch.exp(-(xy_error + 2 * z_error) / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-(z_error + 0.05 * xy_error) / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  _log_metric(
    env,
    "Metrics/angular_momentum_mean",
    torch.mean(torch.sqrt(angmom_magnitude_sq)),
  )
  return angmom_magnitude_sq


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  cost = torch.sum(torch.abs(foot_z - target_height) * torch.norm(foot_vel_xy, dim=-1), dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      cost *= (total_command > command_threshold).float()
  return cost


def feet_gait(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  is_contact = sensor.data.current_contact_time > 0
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
  is_stance = ((global_phase + offsets) % 1.0) < threshold
  reward = (is_stance == is_contact).float().mean(dim=1)
  command = env.command_manager.get_command(command_name)
  if command is not None:
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    reward *= (total_command > command_threshold).float()
  return reward


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
  cost = torch.sum(torch.square(vel_xy_norm) * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  _log_metric(
    env,
    "Metrics/slip_velocity_mean",
    torch.sum(vel_xy_norm * in_contact) / torch.clamp(num_in_contact, min=1),
  )
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  force_magnitude = torch.norm(sensor_data.force, dim=-1)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  cost = torch.sum(force_magnitude * first_contact.float(), dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      cost *= (total_command > command_threshold).float()
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(std_standing, device=env.device, dtype=torch.float32)
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()
    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.mean(torch.square(current_joint_pos - desired_joint_pos) / (std**2), dim=1))


def commanded_stillness(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  min_linear_speed: float = 0.12,
  min_linear_speed_fraction: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  command_linear_speed = torch.norm(command[:, :2], dim=1)
  active = (command_linear_speed > command_threshold).float()
  actual_linear_speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  required_linear_speed = torch.clamp(
    command_linear_speed * min_linear_speed_fraction,
    min=min_linear_speed,
  )
  missing_speed = torch.clamp(required_linear_speed - actual_linear_speed, min=0.0)
  penalty = torch.square(
    missing_speed / torch.clamp(required_linear_speed, min=1.0e-6)
  )
  _log_metric(
    env,
    "Metrics/anti_still_actual_lin_speed",
    torch.mean(actual_linear_speed),
  )
  _log_metric(
    env,
    "Metrics/anti_still_penalty",
    torch.mean(penalty * active),
  )
  return penalty * active


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  diff_angle = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  reward = torch.sum(torch.square(diff_angle), dim=1)
  command = env.command_manager.get_command(command_name)
  if command is not None:
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    reward *= (total_command <= command_threshold).float()
  return reward
