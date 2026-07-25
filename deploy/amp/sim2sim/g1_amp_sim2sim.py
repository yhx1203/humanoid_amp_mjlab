from __future__ import annotations

import argparse
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


NUM_JOINTS = 29
OBS_DIM = 96
LOWCMD_MOTOR_COUNT = 35
LOWCMD_TOPIC = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"
WIRELESS_TOPIC = "rt/wirelesscontroller"

# The order is both the MJLab natural joint order and the G1 29-DOF HG IDL
# motor order (indices 0..28).
JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Fallbacks are used only when no companion policy.onnx metadata file exists.
FALLBACK_DEFAULT_POS = np.array(
    [
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
    ],
    dtype=np.float32,
)
FALLBACK_ACTION_SCALE = np.array(
    [
        0.548,
        0.351,
        0.548,
        0.351,
        0.439,
        0.439,
        0.548,
        0.351,
        0.548,
        0.351,
        0.439,
        0.439,
        0.548,
        0.439,
        0.439,
        0.439,
        0.439,
        0.439,
        0.439,
        0.439,
        0.075,
        0.075,
        0.439,
        0.439,
        0.439,
        0.439,
        0.439,
        0.075,
        0.075,
    ],
    dtype=np.float32,
)
FALLBACK_KP = np.array(
    [
        40.179,
        99.098,
        40.179,
        99.098,
        28.501,
        28.501,
        40.179,
        99.098,
        40.179,
        99.098,
        28.501,
        28.501,
        40.179,
        28.501,
        28.501,
        14.251,
        14.251,
        14.251,
        14.251,
        14.251,
        16.778,
        16.778,
        14.251,
        14.251,
        14.251,
        14.251,
        14.251,
        16.778,
        16.778,
    ],
    dtype=np.float32,
)
FALLBACK_KD = np.array(
    [
        2.558,
        6.309,
        2.558,
        6.309,
        1.814,
        1.814,
        2.558,
        6.309,
        2.558,
        6.309,
        1.814,
        1.814,
        2.558,
        1.814,
        1.814,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
        1.068,
        1.068,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
        1.068,
        1.068,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class DeployParameters:
    default_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    metadata_path: Path | None


@dataclass(frozen=True)
class RobotState:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray
    mode_machine: int
    received_at: float


def _parse_metadata_array(
    metadata: dict[str, str], key: str, fallback: np.ndarray
) -> np.ndarray:
    value = metadata.get(key)
    if value is None:
        return fallback.copy()
    array = np.fromstring(value, sep=",", dtype=np.float32)
    if array.shape != (NUM_JOINTS,):
        raise ValueError(
            f"ONNX metadata '{key}' must contain {NUM_JOINTS} values, got {array.size}."
        )
    return array


def load_deploy_parameters(
    checkpoint_path: Path, metadata_path: Path | None
) -> DeployParameters:
    """Load controller parameters from the companion ONNX metadata when present."""
    resolved_metadata = metadata_path
    explicitly_requested = metadata_path is not None
    if resolved_metadata is None:
        candidate = checkpoint_path.parent / "policy.onnx"
        resolved_metadata = candidate if candidate.exists() else None

    if resolved_metadata is None:
        return DeployParameters(
            default_pos=FALLBACK_DEFAULT_POS.copy(),
            action_scale=FALLBACK_ACTION_SCALE.copy(),
            kp=FALLBACK_KP.copy(),
            kd=FALLBACK_KD.copy(),
            metadata_path=None,
        )
    if not resolved_metadata.exists():
        if explicitly_requested:
            raise FileNotFoundError(f"Metadata file not found: {resolved_metadata}")
        raise AssertionError("Implicit metadata path must exist.")

    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "A policy.onnx metadata file was found, but Python package 'onnx' is not "
            "installed. Install it or pass --ignore-metadata to use built-in values."
        ) from exc

    model = onnx.load(str(resolved_metadata), load_external_data=False)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata_joints = tuple(filter(None, metadata.get("joint_names", "").split(",")))
    if metadata_joints and metadata_joints != JOINT_NAMES:
        raise ValueError(
            "Policy joint order in ONNX metadata does not match the G1 29-DOF DDS "
            "order. Refusing to send commands with an unsafe joint mapping."
        )

    return DeployParameters(
        default_pos=_parse_metadata_array(
            metadata, "default_joint_pos", FALLBACK_DEFAULT_POS
        ),
        action_scale=_parse_metadata_array(
            metadata, "action_scale", FALLBACK_ACTION_SCALE
        ),
        kp=_parse_metadata_array(metadata, "joint_stiffness", FALLBACK_KP),
        kd=_parse_metadata_array(metadata, "joint_damping", FALLBACK_KD),
        metadata_path=resolved_metadata.resolve(),
    )


class CheckpointActor:
    """Minimal deterministic RSL-RL actor loader with no rsl_rl dependency."""

    def __init__(self, checkpoint_path: Path, device: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "actor_state_dict" not in checkpoint:
            raise KeyError(f"No actor_state_dict in checkpoint: {checkpoint_path}")
        state_dict = checkpoint["actor_state_dict"]

        # Current checkpoints use mlp.*, while older RSL-RL checkpoints may use
        # actor.*. Both contain numbered Linear layers.
        layer_pattern = re.compile(r"^(?:mlp|actor)\.(\d+)\.weight$")
        layer_entries: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for name, weight in state_dict.items():
            match = layer_pattern.match(name)
            if match is None:
                continue
            prefix = name.rsplit(".", 1)[0]
            bias_name = f"{prefix}.bias"
            if bias_name not in state_dict:
                raise KeyError(f"Missing actor layer bias '{bias_name}'.")
            layer_entries.append((int(match.group(1)), weight, state_dict[bias_name]))
        layer_entries.sort(key=lambda item: item[0])
        if not layer_entries:
            raise ValueError("Could not find actor MLP layers in checkpoint.")

        first_input_dim = int(layer_entries[0][1].shape[1])
        final_output_dim = int(layer_entries[-1][1].shape[0])
        if first_input_dim != OBS_DIM or final_output_dim != NUM_JOINTS:
            raise ValueError(
                "Checkpoint dimensions do not match this deployment: "
                f"expected {OBS_DIM}->{NUM_JOINTS}, got "
                f"{first_input_dim}->{final_output_dim}."
            )

        self.weights = [entry[1].to(device=device) for entry in layer_entries]
        self.biases = [entry[2].to(device=device) for entry in layer_entries]
        mean = state_dict.get("obs_normalizer._mean")
        std = state_dict.get("obs_normalizer._std")
        if mean is None:
            mean = state_dict.get("actor_obs_normalizer._mean")
        if std is None:
            std = state_dict.get("actor_obs_normalizer._std")
        if mean is None or std is None:
            self.mean = torch.zeros((1, OBS_DIM), device=device)
            self.std = torch.ones((1, OBS_DIM), device=device)
            self.uses_normalizer = False
        else:
            self.mean = mean.to(device=device)
            self.std = std.to(device=device)
            self.uses_normalizer = True
        self.device = device

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(observation).reshape(1, OBS_DIM).to(self.device)
        # EmpiricalNormalization in RSL-RL uses eps=1e-2.
        value = (obs - self.mean) / (self.std + 1.0e-2)
        with torch.inference_mode():
            for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                value = F.linear(value, weight, bias)
                if index + 1 < len(self.weights):
                    value = F.elu(value)
        action = value.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if action.shape != (NUM_JOINTS,) or not np.all(np.isfinite(action)):
            raise FloatingPointError("Policy produced an invalid action.")
        return action


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world gravity [0, 0, -1] into the pelvis frame."""
    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        raise FloatingPointError("Received a zero-norm IMU quaternion.")
    w, x, y, z = quat / norm
    # Third row of R(body->world), negated: R.T @ [0, 0, -1].
    return np.array(
        [
            -2.0 * (x * z - y * w),
            -2.0 * (y * z + x * w),
            -(1.0 - 2.0 * (x * x + y * y)),
        ],
        dtype=np.float32,
    )


def build_observation(
    state: RobotState,
    command: np.ndarray,
    last_action: np.ndarray,
    default_pos: np.ndarray,
) -> np.ndarray:
    observation = np.concatenate(
        (
            state.angular_velocity.astype(np.float32),
            projected_gravity(state.quaternion_wxyz),
            command.astype(np.float32),
            state.joint_pos.astype(np.float32) - default_pos,
            state.joint_vel.astype(np.float32),
            last_action.astype(np.float32),
        )
    )
    if observation.shape != (OBS_DIM,) or not np.all(np.isfinite(observation)):
        raise FloatingPointError("LowState produced an invalid policy observation.")
    return observation


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _apply_deadzone(value: float, deadzone: float) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    if abs(value) <= deadzone:
        return 0.0
    return float(np.sign(value) * (abs(value) - deadzone) / (1.0 - deadzone))


def _scale_unit_input(value: float, limits: np.ndarray) -> float:
    return value * float(limits[1] if value >= 0.0 else abs(limits[0]))


class VelocityCommand:
    def __init__(self, args: argparse.Namespace) -> None:
        self.fixed = np.array([args.cmd_x, args.cmd_y, args.cmd_yaw], dtype=np.float32)
        self.ranges = np.array(
            [args.cmd_x_range, args.cmd_y_range, args.cmd_yaw_range], dtype=np.float32
        )
        self.fixed = np.clip(self.fixed, self.ranges[:, 0], self.ranges[:, 1])
        self.use_wireless = args.wireless
        self.deadzone = args.wireless_deadzone
        self._wireless: np.ndarray | None = None
        self._lock = threading.Lock()

    def wireless_callback(self, message: WirelessController_) -> None:
        # unitree_mujoco already flips the physical gamepad Y axes. Mapping matches
        # the raw-MuJoCo runner: LY forward, -LX left/right, -RX yaw.
        unit_input = np.array([message.ly, -message.lx, -message.rx], dtype=np.float32)
        unit_input = np.array(
            [_apply_deadzone(float(v), self.deadzone) for v in unit_input],
            dtype=np.float32,
        )
        command = np.array(
            [_scale_unit_input(float(unit_input[i]), self.ranges[i]) for i in range(3)],
            dtype=np.float32,
        )
        with self._lock:
            self._wireless = command

    def get(self) -> np.ndarray:
        if not self.use_wireless:
            return self.fixed.copy()
        with self._lock:
            if self._wireless is None:
                return np.zeros(3, dtype=np.float32)
            return self._wireless.copy()


class G1AmpSim2Sim:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        checkpoint_path = Path(args.checkpoint_file).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        metadata_path = (
            None
            if args.ignore_metadata or args.metadata_file is None
            else Path(args.metadata_file).expanduser().resolve()
        )
        if args.ignore_metadata:
            # A non-existent sentinel prevents implicit companion-file discovery.
            self.params = DeployParameters(
                default_pos=FALLBACK_DEFAULT_POS.copy(),
                action_scale=FALLBACK_ACTION_SCALE.copy(),
                kp=FALLBACK_KP.copy(),
                kd=FALLBACK_KD.copy(),
                metadata_path=None,
            )
        else:
            self.params = load_deploy_parameters(checkpoint_path, metadata_path)

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.actor = CheckpointActor(checkpoint_path, args.device)
        self.checkpoint_path = checkpoint_path
        self.command = VelocityCommand(args)
        self._state: RobotState | None = None
        self._state_lock = threading.Lock()
        self._publisher: ChannelPublisher | None = None
        self._lowstate_subscriber: ChannelSubscriber | None = None
        self._wireless_subscriber: ChannelSubscriber | None = None
        self._crc = CRC()
        self._lowcmd = unitree_hg_msg_dds__LowCmd_()
        self._initialize_lowcmd()

    def _initialize_lowcmd(self) -> None:
        self._lowcmd.mode_pr = 0  # PR: pitch/roll joint coordinates.
        for index in range(LOWCMD_MOTOR_COUNT):
            motor = self._lowcmd.motor_cmd[index]
            motor.mode = 1 if index < NUM_JOINTS else 0
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def lowstate_callback(self, message: LowState_) -> None:
        state = RobotState(
            joint_pos=np.fromiter(
                (message.motor_state[i].q for i in range(NUM_JOINTS)),
                dtype=np.float32,
                count=NUM_JOINTS,
            ),
            joint_vel=np.fromiter(
                (message.motor_state[i].dq for i in range(NUM_JOINTS)),
                dtype=np.float32,
                count=NUM_JOINTS,
            ),
            quaternion_wxyz=np.asarray(
                message.imu_state.quaternion, dtype=np.float32
            ).copy(),
            angular_velocity=np.asarray(
                message.imu_state.gyroscope, dtype=np.float32
            ).copy(),
            mode_machine=int(message.mode_machine),
            received_at=time.perf_counter(),
        )
        with self._state_lock:
            self._state = state

    def get_state(self) -> RobotState | None:
        with self._state_lock:
            return self._state

    def connect(self) -> None:
        ChannelFactoryInitialize(self.args.domain_id, self.args.interface)
        self._publisher = ChannelPublisher(LOWCMD_TOPIC, LowCmd_)
        self._publisher.Init()
        self._lowstate_subscriber = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        self._lowstate_subscriber.Init(self.lowstate_callback, 10)
        if self.args.wireless:
            self._wireless_subscriber = ChannelSubscriber(
                self.args.wireless_topic, WirelessController_
            )
            self._wireless_subscriber.Init(self.command.wireless_callback, 10)

    def wait_for_state(self) -> RobotState:
        print(f"[INFO] Waiting for G1 LowState on {LOWSTATE_TOPIC} ...")
        deadline = time.perf_counter() + self.args.connect_timeout
        while time.perf_counter() < deadline:
            state = self.get_state()
            if state is not None:
                if not np.all(np.isfinite(state.joint_pos)):
                    raise FloatingPointError(
                        "First LowState contains invalid joint positions."
                    )
                print("[INFO] Connected to G1 29-DOF LowState.")
                return state
            time.sleep(0.01)
        raise TimeoutError(
            "No G1 LowState received. Check that unitree_mujoco is running G1 "
            "scene_29dof.xml with the same DDS domain/interface."
        )

    def publish_target(self, target_pos: np.ndarray, mode_machine: int) -> None:
        if self._publisher is None:
            raise RuntimeError("DDS publisher is not initialized.")
        if target_pos.shape != (NUM_JOINTS,) or not np.all(np.isfinite(target_pos)):
            raise FloatingPointError("Refusing to publish an invalid joint target.")
        self._lowcmd.mode_pr = 0
        self._lowcmd.mode_machine = mode_machine
        for index in range(NUM_JOINTS):
            motor = self._lowcmd.motor_cmd[index]
            motor.mode = 1
            motor.q = float(target_pos[index])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(self.params.kp[index])
            motor.kd = float(self.params.kd[index])
        self._lowcmd.crc = self._crc.Crc(self._lowcmd)
        self._publisher.Write(self._lowcmd)

    def publish_damping(self, mode_machine: int) -> None:
        if self._publisher is None:
            return
        self._lowcmd.mode_pr = 0
        self._lowcmd.mode_machine = mode_machine
        for index in range(NUM_JOINTS):
            motor = self._lowcmd.motor_cmd[index]
            motor.mode = 1
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = float(self.params.kd[index])
        self._lowcmd.crc = self._crc.Crc(self._lowcmd)
        self._publisher.Write(self._lowcmd)

    def print_summary(self) -> None:
        print(f"[INFO] Checkpoint: {self.checkpoint_path}")
        if self.params.metadata_path is None:
            print(
                "[WARN] Using built-in deployment parameters (no ONNX metadata loaded)."
            )
        else:
            print(f"[INFO] Metadata:   {self.params.metadata_path}")
        print(
            f"[INFO] DDS domain={self.args.domain_id}, interface={self.args.interface}, "
            f"low-level={self.args.low_level_hz:.1f} Hz, policy={self.args.policy_hz:.1f} Hz"
        )
        source = (
            f"wireless ({self.args.wireless_topic})"
            if self.args.wireless
            else f"fixed {tuple(float(v) for v in self.command.fixed)}"
        )
        print(f"[INFO] Velocity command source: {source}")

    def run(self) -> None:
        self.print_summary()
        self.connect()
        initial_state = self.wait_for_state()
        initial_pos = initial_state.joint_pos.copy()
        last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        target_pos = self.params.default_pos.copy()
        low_level_period = 1.0 / self.args.low_level_hz
        policy_period = 1.0 / self.args.policy_hz
        start_time = time.perf_counter()
        next_tick = start_time
        next_policy_time = (
            start_time + self.args.stand_up_duration + self.args.stand_hold_duration
        )
        next_status_time = start_time
        stage = "stand-up"
        last_mode_machine = initial_state.mode_machine

        try:
            while True:
                now = time.perf_counter()
                if (
                    self.args.duration is not None
                    and now - start_time >= self.args.duration
                ):
                    break
                state = self.get_state()
                if state is None:
                    raise RuntimeError("LowState disappeared after connection.")
                last_mode_machine = state.mode_machine
                state_age = now - state.received_at
                if state_age > self.args.state_timeout:
                    raise TimeoutError(
                        f"LowState is stale by {state_age:.3f}s; stopping command output."
                    )

                elapsed = now - start_time
                stand_up_end = self.args.stand_up_duration
                policy_start = stand_up_end + self.args.stand_hold_duration
                if elapsed < stand_up_end:
                    stage = "stand-up"
                    ratio = elapsed / max(self.args.stand_up_duration, 1.0e-6)
                    blend = _smoothstep(ratio)
                    target_pos = (
                        1.0 - blend
                    ) * initial_pos + blend * self.params.default_pos
                elif elapsed < policy_start:
                    stage = "hold"
                    target_pos = self.params.default_pos
                else:
                    stage = "policy"
                    if now >= next_policy_time:
                        command = self.command.get()
                        policy_elapsed = elapsed - policy_start
                        if self.args.command_warmup > 0.0:
                            command = command * min(
                                1.0, policy_elapsed / self.args.command_warmup
                            )
                        observation = build_observation(
                            state, command, last_action, self.params.default_pos
                        )
                        last_action = self.actor(observation)
                        if self.args.action_clip is not None:
                            last_action = np.clip(
                                last_action,
                                -self.args.action_clip,
                                self.args.action_clip,
                            )
                        target_pos = (
                            self.params.default_pos
                            + last_action * self.params.action_scale
                        )
                        next_policy_time += policy_period
                        if next_policy_time < now - policy_period:
                            next_policy_time = now + policy_period

                self.publish_target(target_pos.astype(np.float32), state.mode_machine)

                if now >= next_status_time:
                    command = self.command.get() if stage == "policy" else np.zeros(3)
                    print(
                        f"[STATE] stage={stage:8s} cmd=({command[0]:+.2f}, "
                        f"{command[1]:+.2f}, {command[2]:+.2f}) "
                        f"gravity_z={projected_gravity(state.quaternion_wxyz)[2]:+.3f} "
                        f"state_age={state_age * 1000.0:.1f}ms"
                    )
                    next_status_time = now + self.args.status_interval

                next_tick += low_level_period
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                elif sleep_time < -5.0 * low_level_period:
                    next_tick = time.perf_counter()
        finally:
            # Remove position stiffness so unitree_mujoco does not keep executing the
            # last policy target after Ctrl+C or an exception.
            for _ in range(10):
                self.publish_damping(last_mode_machine)
                time.sleep(low_level_period)
            print("[INFO] Controller stopped; position gains released.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an MJLab G1 AMP checkpoint through Unitree SDK2 DDS."
    )
    parser.add_argument(
        "--checkpoint-file",
        required=True,
        help="RSL-RL model_*.pt checkpoint produced by this repository.",
    )
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Optional policy.onnx containing deployment metadata (default: beside checkpoint).",
    )
    parser.add_argument(
        "--ignore-metadata",
        action="store_true",
        help="Use built-in G1 parameters even if a companion policy.onnx exists.",
    )
    parser.add_argument("--device", default="cpu", help="Torch inference device.")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--low-level-hz", type=float, default=200.0)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--state-timeout", type=float, default=0.2)
    parser.add_argument("--stand-up-duration", type=float, default=0.0)
    parser.add_argument("--stand-hold-duration", type=float, default=0.0)
    parser.add_argument("--command-warmup", type=float, default=1.0)
    parser.add_argument("--cmd-x", type=float, default=0.5)
    parser.add_argument("--cmd-y", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument("--cmd-x-range", type=float, nargs=2, default=(-0.5, 1.0))
    parser.add_argument("--cmd-y-range", type=float, nargs=2, default=(-0.5, 0.5))
    parser.add_argument("--cmd-yaw-range", type=float, nargs=2, default=(-1.0, 1.0))
    parser.add_argument(
        "--wireless",
        action="store_true",
        help="Use unitree_mujoco's wireless-controller topic instead of fixed commands.",
    )
    parser.add_argument("--wireless-topic", default=WIRELESS_TOPIC)
    parser.add_argument("--wireless-deadzone", type=float, default=0.08)
    parser.add_argument(
        "--action-clip",
        type=float,
        default=None,
        help="Optional symmetric action clip; omitted to match training.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional total runtime in seconds.",
    )
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate checkpoint/metadata without opening DDS channels.",
    )
    args = parser.parse_args()
    if args.low_level_hz <= 0.0 or args.policy_hz <= 0.0:
        parser.error("--low-level-hz and --policy-hz must be positive.")
    if args.policy_hz > args.low_level_hz:
        parser.error("--policy-hz cannot exceed --low-level-hz.")
    if args.state_timeout <= 0.0 or args.connect_timeout <= 0.0:
        parser.error("DDS timeouts must be positive.")
    if args.stand_up_duration < 0.0 or args.stand_hold_duration < 0.0:
        parser.error("Stand-up durations cannot be negative.")
    if not 0.0 <= args.wireless_deadzone < 1.0:
        parser.error("--wireless-deadzone must be in [0, 1).")
    if args.status_interval <= 0.0:
        parser.error("--status-interval must be positive.")
    if args.action_clip is not None and args.action_clip <= 0.0:
        parser.error("--action-clip must be positive.")
    return args


def main() -> None:
    args = parse_args()
    controller = G1AmpSim2Sim(args)
    if args.validate_only:
        controller.print_summary()
        zero_obs = np.zeros(OBS_DIM, dtype=np.float32)
        action = controller.actor(zero_obs)
        print(
            f"[OK] checkpoint validated: obs={OBS_DIM}, actions={action.size}, "
            f"zero_obs_action_norm={np.linalg.norm(action):.4f}"
        )
        return
    try:
        controller.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
