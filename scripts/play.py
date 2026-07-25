"""Script to play RL agent with RSL-RL."""

# ruff: noqa: E402

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

# Prefer this checkout over other editable projects that also expose a top-level
# package named ``src``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


class NativeVelocityKeyboardController:
  """Keyboard velocity override for native MuJoCo play mode."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    command_name: str = "twist",
    lin_vel_step: float = 0.05,
    yaw_vel_step: float = 0.05,
  ) -> None:
    self.command_name = command_name
    self.lin_vel_step = lin_vel_step
    self.yaw_vel_step = yaw_vel_step
    self.enabled = False
    self.command = torch.zeros(3, device=env.unwrapped.device)
    self.term = env.unwrapped.command_manager.get_term(command_name)
    self.ranges = getattr(self.term.cfg, "ranges", None)

    original_compute = self.term.compute

    def compute_with_keyboard(dt: float) -> None:
      original_compute(dt)
      self.apply()

    self.term.compute = compute_with_keyboard

  def handle_key(self, key: int) -> None:
    from mjlab.viewer.native.keys import (
      KEY_A,
      KEY_D,
      KEY_DOWN,
      KEY_KP_2,
      KEY_KP_4,
      KEY_KP_6,
      KEY_KP_8,
      KEY_LEFT,
      KEY_RIGHT,
      KEY_UP,
    )

    if key in (KEY_UP, KEY_KP_8):
      self.command[0] += self.lin_vel_step
    elif key in (KEY_DOWN, KEY_KP_2):
      self.command[0] -= self.lin_vel_step
    elif key == KEY_A:
      self.command[1] += self.lin_vel_step
    elif key == KEY_D:
      self.command[1] -= self.lin_vel_step
    elif key in (KEY_LEFT, KEY_KP_4):
      self.command[2] += self.yaw_vel_step
    elif key in (KEY_RIGHT, KEY_KP_6):
      self.command[2] -= self.yaw_vel_step
    else:
      return

    self.enabled = True
    self._clamp_command()
    self.apply()
    print(
      "[CMD] "
      f"lin_vel_x={self.command[0].item():+.2f} m/s, "
      f"lin_vel_y={self.command[1].item():+.2f} m/s, "
      f"ang_vel_z={self.command[2].item():+.2f} rad/s"
    )

  def apply(self) -> None:
    if not self.enabled:
      return
    self.term.vel_command_b[:, :] = self.command
    if hasattr(self.term, "is_heading_env"):
      self.term.is_heading_env[:] = False
    if hasattr(self.term, "is_standing_env"):
      self.term.is_standing_env[:] = False

  def _clamp_command(self) -> None:
    if self.ranges is None:
      return
    self.command[0] = torch.clamp(
      self.command[0],
      min=self.ranges.lin_vel_x[0],
      max=self.ranges.lin_vel_x[1],
    )
    self.command[1] = torch.clamp(
      self.command[1],
      min=self.ranges.lin_vel_y[0],
      max=self.ranges.lin_vel_y[1],
    )
    self.command[2] = torch.clamp(
      self.command[2],
      min=self.ranges.ang_vel_z[0],
      max=self.ranges.ang_vel_z[1],
    )


def _install_viser_velocity_gui(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
) -> None:
  """Create Viser controls only for velocity axes that are not locked."""
  term = env.command_manager.get_term(command_name)
  ranges = getattr(term.cfg, "ranges", None)
  if ranges is None:
    return

  axis_ranges = (
    (0, "lin_vel_x", ranges.lin_vel_x),
    (1, "lin_vel_y", ranges.lin_vel_y),
    (2, "ang_vel_z", ranges.ang_vel_z),
  )
  active_axes = [
    (index, label, max(abs(float(limits[0])), abs(float(limits[1]))))
    for index, label, limits in axis_ranges
    if max(abs(float(limits[0])), abs(float(limits[1]))) > 0.0
  ]

  original_compute = term.compute
  gui_state: dict[str, Any] = {
    "enabled": None,
    "sliders": [],
    "get_env_idx": None,
  }

  def compute_with_gui(dt: float) -> None:
    original_compute(dt)
    enabled = gui_state["enabled"]
    get_env_idx = gui_state["get_env_idx"]
    if enabled is None or not enabled.value or get_env_idx is None:
      return
    env_idx = get_env_idx()
    for axis_index, slider in gui_state["sliders"]:
      term.vel_command_b[env_idx, axis_index] = slider.value

  def create_gui(name: str, server: Any, get_env_idx: Any) -> None:
    from viser import Icon

    sliders: list[tuple[int, Any]] = []
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      for axis_index, label, max_val in active_axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max(max_val, 0.1),
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_event: Any, _slider=slider, _max_input=max_input) -> None:
          _slider.min = -_max_input.value
          _slider.max = _max_input.value

        sliders.append((axis_index, slider))

      zero_button = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_button.on_click
      def _(_event: Any) -> None:
        for _, slider in sliders:
          slider.value = 0.0

    gui_state["enabled"] = enabled
    gui_state["sliders"] = sliders
    gui_state["get_env_idx"] = get_env_idx

  term.compute = compute_with_gui
  term.create_gui = create_gui


class FootholdPlacementDebugAdapter:
  """Expose the foothold reward debug drawing to older MJLab viewers."""

  def __init__(self, env: ManagerBasedRlEnv, term_name: str) -> None:
    self.env = env
    self.term_name = term_name
    self._debug_vis_enabled = True

  def debug_vis(self, visualizer) -> None:
    if not self._debug_vis_enabled:
      return
    term = self._get_reward_func()
    if term is not None and hasattr(term, "debug_vis"):
      term.debug_vis(visualizer)

  def _get_reward_func(self) -> Any | None:
    reward_manager = self.env.reward_manager
    if not hasattr(reward_manager, "get_term_cfg"):
      return None
    try:
      return reward_manager.get_term_cfg(self.term_name).func
    except ValueError:
      return None


def _install_foothold_debug_visualizer(
  env: ManagerBasedRlEnv,
  term_name: str = "foothold_placement",
) -> None:
  """Register foothold reward visualization even on MJLab builds without reward GUI."""
  reward_manager = env.reward_manager
  if hasattr(reward_manager, "get_visualizable_terms"):
    try:
      if any(name == term_name for name, _ in reward_manager.get_visualizable_terms()):
        return
    except Exception:
      pass
  if not hasattr(reward_manager, "get_term_cfg"):
    return
  try:
    term = reward_manager.get_term_cfg(term_name).func
  except ValueError:
    return
  if not hasattr(term, "debug_vis"):
    return

  adapter = FootholdPlacementDebugAdapter(env, term_name)
  if hasattr(env, "manager_visualizers"):
    env.manager_visualizers[term_name] = adapter

  command_manager = env.command_manager
  if getattr(command_manager, "_foothold_debug_gui_installed", False):
    return
  original_create_debug_vis_gui = command_manager.create_debug_vis_gui

  def create_debug_vis_gui_with_foothold(server, *args, **kwargs):
    result = original_create_debug_vis_gui(server, *args, **kwargs)
    checkbox = server.gui.add_checkbox(
      "Foothold placement",
      initial_value=adapter._debug_vis_enabled,
    )

    @checkbox.on_update
    def _(_) -> None:
      adapter._debug_vis_enabled = checkbox.value
      on_change = kwargs.get("on_change")
      if on_change is not None:
        on_change()

    return result

  command_manager.create_debug_vis_gui = create_debug_vis_gui_with_foothold
  command_manager._foothold_debug_gui_installed = True
  print("[INFO]: Foothold placement debug visualization enabled for play.")


def _set_play_terrain_origin(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  terrain_level: int | None = None,
  terrain_type: int | None = None,
) -> None:
  """Select a fixed terrain row/type for play resets."""
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return

  device = terrain.terrain_levels.device
  if env_ids is None or isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=device, dtype=torch.long)

  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  num_envs = len(env_ids)

  if terrain_level is None:
    levels = torch.randint(0, num_rows, (num_envs,), device=device)
  else:
    if terrain_level < 0 or terrain_level >= num_rows:
      raise ValueError(
        f"terrain_level={terrain_level} is out of range [0, {num_rows - 1}]"
      )
    levels = torch.full((num_envs,), terrain_level, device=device, dtype=torch.long)

  if terrain_type is None:
    types = torch.randint(0, num_cols, (num_envs,), device=device)
  else:
    if terrain_type < 0 or terrain_type >= num_cols:
      raise ValueError(
        f"terrain_type={terrain_type} is out of range [0, {num_cols - 1}]"
      )
    types = torch.full((num_envs,), terrain_type, device=device, dtype=torch.long)

  terrain.terrain_levels[env_ids] = levels
  terrain.terrain_types[env_ids] = types
  terrain.env_origins[env_ids] = terrain.terrain_origins[levels, types]


def _configure_play_terrain_selection(env_cfg, cfg: "PlayConfig") -> None:
  if (
    cfg.terrain_level is None
    and cfg.terrain_type is None
    and cfg.terrain_name is None
  ):
    return

  if cfg.terrain_type is not None and cfg.terrain_name is not None:
    raise ValueError("Use either --terrain-type or --terrain-name, not both.")

  terrain_cfg = env_cfg.scene.terrain
  terrain_generator = None if terrain_cfg is None else terrain_cfg.terrain_generator
  if terrain_generator is None:
    raise ValueError("Terrain selection requires a generated terrain.")

  terrain_type = cfg.terrain_type
  terrain_names = tuple(terrain_generator.sub_terrains.keys())

  # Play config may use random terrain generation for visual variety. Fixed
  # terrain selection needs curriculum mode so rows represent difficulty again.
  terrain_generator.curriculum = True
  terrain_generator.num_rows = max(terrain_generator.num_rows, 10)

  if cfg.terrain_name is not None:
    if cfg.terrain_name not in terrain_names:
      options = ", ".join(terrain_names)
      raise ValueError(
        f"Unknown terrain_name={cfg.terrain_name!r}. Available: {options}"
      )
    terrain_generator.sub_terrains = {
      cfg.terrain_name: terrain_generator.sub_terrains[cfg.terrain_name]
    }
    terrain_generator.num_cols = 1
    terrain_names = (cfg.terrain_name,)
    terrain_type = 0
  else:
    # Older mjlab releases allocate curriculum columns from sub-terrain
    # proportions. Use equal proportions here so play exposes every terrain type
    # exactly once instead of dropping low-proportion types when num_cols is
    # small.
    terrain_generator.sub_terrains = {
      name: replace(sub_cfg, proportion=1.0)
      for name, sub_cfg in terrain_generator.sub_terrains.items()
    }
    terrain_generator.num_cols = len(terrain_names)

  if cfg.terrain_level is not None:
    num_rows = terrain_generator.num_rows
    if cfg.terrain_level < 0 or cfg.terrain_level >= num_rows:
      raise ValueError(
        f"terrain_level={cfg.terrain_level} is out of range [0, {num_rows - 1}]"
      )
  if terrain_type is not None:
    num_cols = terrain_generator.num_cols
    if terrain_type < 0 or terrain_type >= num_cols:
      raise ValueError(
        f"terrain_type={terrain_type} is out of range [0, {num_cols - 1}]"
      )

  env_cfg.events.pop("randomize_terrain", None)
  select_terrain_event = EventTermCfg(
    func=_set_play_terrain_origin,
    mode="reset",
    params={
      "terrain_level": cfg.terrain_level,
      "terrain_type": terrain_type,
    },
  )
  env_cfg.events = {
    "select_terrain": select_terrain_event,
    **env_cfg.events,
  }

  terrain_desc = (
    terrain_names[terrain_type] if terrain_type is not None else "random type"
  )
  level_desc = (
    str(cfg.terrain_level) if cfg.terrain_level is not None else "random level"
  )
  print(f"[INFO]: Play terrain selection: level={level_desc}, type={terrain_desc}")


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  terrain_level: int | None = None
  """Fixed generated-terrain difficulty row for play resets."""
  terrain_type: int | None = None
  """Fixed generated-terrain type column index for play resets."""
  terrain_name: str | None = None
  """Fixed generated-terrain name for play resets."""


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  _configure_play_terrain_selection(env_cfg, cfg)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  _install_foothold_debug_visualizer(env.unwrapped)

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    key_callback = None
    if "twist" in env.unwrapped.command_manager.active_terms:
      keyboard_controller = NativeVelocityKeyboardController(env)
      key_callback = keyboard_controller.handle_key
      print(
        "[INFO]: Native velocity keyboard enabled: "
        "Up/Down adjusts lin_vel_x, A/D adjusts lin_vel_y, "
        "Left/Right adjusts ang_vel_z."
      )
    NativeMujocoViewer(env, policy, key_callback=key_callback).run()
  elif resolved_viewer == "viser":
    if "twist" in env.unwrapped.command_manager.active_terms:
      _install_viser_velocity_gui(env.unwrapped)
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
