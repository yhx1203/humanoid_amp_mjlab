"""View a G1 motion CSV directly in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import re
import threading
import time
from pathlib import Path
from typing import Literal


KEY_SPACE = 32
KEY_R = 82
KEY_RIGHT = 262
KEY_LEFT = 263


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "motion_file",
    nargs="?",
    default="src/assets/motions/g1/walk1_subject1.csv",
    help="CSV motion file. Layout: root_pos(3), root_quat(4), joints(29).",
  )
  parser.add_argument(
    "--xml",
    default="src/assets/robots/unitree_g1/xmls/scene_g1.xml",
    help="MuJoCo scene XML path.",
  )
  parser.add_argument("--fps", type=float, default=30.0, help="CSV playback FPS.")
  parser.add_argument(
    "--speed", type=float, default=1.0, help="Playback speed scale."
  )
  parser.add_argument(
    "--quat-order",
    choices=("xyzw", "wxyz"),
    default="xyzw",
    help="Quaternion order in the CSV.",
  )
  parser.add_argument("--start", type=int, default=0, help="Start frame index.")
  parser.add_argument(
    "--end", type=int, default=None, help="End frame index, exclusive."
  )
  parser.add_argument(
    "--once",
    action="store_true",
    help="Play each selected motion once, then hold its last frame.",
  )
  parser.add_argument(
    "--single-file",
    action="store_true",
    help="Disable Left/Right switching between sibling CSV files.",
  )
  return parser.parse_args()


def natural_sort_key(path: Path) -> tuple[object, ...]:
  parts = re.split(r"(\d+)", path.name.casefold())
  return tuple(int(part) if part.isdigit() else part for part in parts)


def discover_motion_files(
  selected_path: Path, single_file: bool
) -> tuple[list[Path], int]:
  selected_path = selected_path.resolve()
  if selected_path.suffix.lower() != ".csv":
    raise ValueError(f"Motion file must have a .csv suffix: {selected_path}")
  if single_file:
    return [selected_path], 0

  motion_paths = sorted(selected_path.parent.glob("*.csv"), key=natural_sort_key)
  if not motion_paths:
    raise FileNotFoundError(f"No CSV motions found in: {selected_path.parent}")
  return motion_paths, motion_paths.index(selected_path)


def csv_quat_to_mujoco(
  quat,
  quat_order: Literal["xyzw", "wxyz"],
):
  import numpy as np

  if quat_order == "xyzw":
    quat = quat[[3, 0, 1, 2]]
  quat_norm = np.linalg.norm(quat)
  if quat_norm < 1.0e-8:
    raise ValueError("Encountered a near-zero root quaternion in the motion CSV.")
  return quat / quat_norm


def load_motion(path: Path, start: int, end: int | None, expected_cols: int):
  import numpy as np

  motion = np.loadtxt(path, delimiter=",", dtype=np.float64)
  if motion.ndim == 1:
    motion = motion[None, :]
  motion = motion[start:end]
  if motion.shape[0] == 0:
    raise ValueError(f"Motion has no frames in the requested range: {path}")
  if motion.shape[1] != expected_cols:
    raise ValueError(
      f"{path}: found {motion.shape[1]} columns, but model.nq is {expected_cols}. "
      "Expected root_pos(3), root_quat(4), then one value per joint qpos."
    )
  if not np.all(np.isfinite(motion)):
    raise ValueError(f"Motion contains NaN or infinity: {path}")
  return motion


class PlaybackControls:
  """Collect MuJoCo viewer key events for the playback loop."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._motion_delta = 0
    self._toggle_pause = False
    self._restart = False

  def handle_key(self, key: int) -> None:
    with self._lock:
      if key == KEY_LEFT:
        self._motion_delta -= 1
      elif key == KEY_RIGHT:
        self._motion_delta += 1
      elif key == KEY_SPACE:
        self._toggle_pause = True
      elif key == KEY_R:
        self._restart = True

  def consume(self) -> tuple[int, bool, bool]:
    with self._lock:
      result = (self._motion_delta, self._toggle_pause, self._restart)
      self._motion_delta = 0
      self._toggle_pause = False
      self._restart = False
    return result


def apply_frame(model, data, frame, quat_order: Literal["xyzw", "wxyz"]) -> None:
  import mujoco

  data.qpos[0:3] = frame[0:3]
  data.qpos[3:7] = csv_quat_to_mujoco(frame[3:7], quat_order)
  data.qpos[7:] = frame[7:]
  mujoco.mj_forward(model, data)


def describe_motion(index: int, paths: list[Path], motion, fps: float) -> None:
  print(
    f"[MOTION {index + 1}/{len(paths)}] {paths[index]} | "
    f"{motion.shape[0]} frames | {motion.shape[0] / fps:.2f}s"
  )


def main() -> None:
  args = parse_args()
  import mujoco
  import mujoco.viewer

  xml_path = Path(args.xml)
  motion_path = Path(args.motion_file)
  if not xml_path.exists():
    raise FileNotFoundError(f"XML file not found: {xml_path}")
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion CSV not found: {motion_path}")
  if args.fps <= 0.0:
    raise ValueError("--fps must be positive.")
  if args.speed <= 0.0:
    raise ValueError("--speed must be positive.")

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  motion_paths, motion_index = discover_motion_files(motion_path, args.single_file)
  motion = load_motion(
    motion_paths[motion_index],
    args.start,
    args.end,
    model.nq,
  )

  frame_dt = 1.0 / (args.fps * args.speed)
  controls = PlaybackControls()
  frame_index = 0
  paused = False
  completed = False

  apply_frame(model, data, motion[frame_index], args.quat_order)
  describe_motion(motion_index, motion_paths, motion, args.fps)
  print(f"[PLAYBACK] {args.fps:g} fps | speed {args.speed:g}x | scene {xml_path}")
  if len(motion_paths) > 1:
    print("[KEYS] Left/Right: previous/next | Space: pause/resume | R: restart")
  else:
    print("[KEYS] Space: pause/resume | R: restart")

  with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=controls.handle_key,
  ) as viewer:
    viewer.sync()
    next_frame_time = time.perf_counter() + frame_dt
    while viewer.is_running():
      motion_delta, toggle_pause, restart = controls.consume()
      if motion_delta and len(motion_paths) > 1:
        motion_index = (motion_index + motion_delta) % len(motion_paths)
        motion = load_motion(
          motion_paths[motion_index],
          args.start,
          args.end,
          model.nq,
        )
        frame_index = 0
        completed = False
        apply_frame(model, data, motion[frame_index], args.quat_order)
        describe_motion(motion_index, motion_paths, motion, args.fps)
        viewer.sync()
        next_frame_time = time.perf_counter() + frame_dt
      elif restart:
        frame_index = 0
        completed = False
        apply_frame(model, data, motion[frame_index], args.quat_order)
        print(f"[RESTART] {motion_paths[motion_index].name}")
        viewer.sync()
        next_frame_time = time.perf_counter() + frame_dt

      if toggle_pause:
        paused = not paused
        print("[PAUSED]" if paused else "[PLAYING]")
        next_frame_time = time.perf_counter() + frame_dt

      if paused or completed:
        time.sleep(0.01)
        continue

      now = time.perf_counter()
      if now < next_frame_time:
        time.sleep(min(next_frame_time - now, 0.01))
        continue

      if frame_index + 1 >= motion.shape[0]:
        if args.once:
          completed = True
          print(
            f"[DONE] {motion_paths[motion_index].name} | "
            "use Left/Right to select another motion or R to replay."
          )
          continue
        frame_index = 0
      else:
        frame_index += 1

      apply_frame(model, data, motion[frame_index], args.quat_order)
      viewer.sync()
      next_frame_time += frame_dt
      if next_frame_time < now - frame_dt:
        next_frame_time = now + frame_dt


if __name__ == "__main__":
  main()
