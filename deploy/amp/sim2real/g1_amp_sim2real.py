from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.amp.sim2sim.g1_amp_sim2sim import (  # noqa: E402
    CheckpointActor,
    DeployParameters,
    FALLBACK_ACTION_SCALE,
    FALLBACK_DEFAULT_POS,
    FALLBACK_KD,
    FALLBACK_KP,
    LOWCMD_MOTOR_COUNT,
    LOWCMD_TOPIC,
    LOWSTATE_TOPIC,
    NUM_JOINTS,
    OBS_DIM,
    build_observation,
    load_deploy_parameters,
    projected_gravity,
)


REAL_DDS_DOMAIN_ID = 0
MODE_PR = 0

# These are the G1 joint limits in the same 29-DOF HG motor order used by the
# training model, policy metadata, LowState, and LowCmd.
JOINT_LIMIT_LOWER = np.array(
    [
        -2.5307,
        -0.5236,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.5307,
        -2.9671,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.618,
        -0.52,
        -0.52,
        -3.0892,
        -1.5882,
        -2.618,
        -1.0472,
        -1.97222,
        -1.61443,
        -1.61443,
        -3.0892,
        -2.2515,
        -2.618,
        -1.0472,
        -1.97222,
        -1.61443,
        -1.61443,
    ],
    dtype=np.float32,
)
JOINT_LIMIT_UPPER = np.array(
    [
        2.8798,
        2.9671,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.8798,
        0.5236,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.618,
        0.52,
        0.52,
        2.6704,
        2.2515,
        2.618,
        2.0944,
        1.97222,
        1.61443,
        1.61443,
        2.6704,
        1.5882,
        2.618,
        2.0944,
        1.97222,
        1.61443,
        1.61443,
    ],
    dtype=np.float32,
)

BUTTON_START = 2
BUTTON_A = 8
BUTTON_B = 9


class SafetyViolation(RuntimeError):
    """A robot-state or controller-output safety invariant was violated."""


class EmergencyStop(RuntimeError):
    """The operator requested software emergency damping."""


@dataclass(frozen=True)
class RemoteState:
    lx: float
    ly: float
    rx: float
    ry: float
    keys: int

    def pressed(self, bit: int) -> bool:
        return bool(self.keys & (1 << bit))


@dataclass(frozen=True)
class RealRobotState:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray
    motor_errors: np.ndarray
    remote: RemoteState
    mode_machine: int
    tick: int
    received_at: float


def parse_wireless_remote(data: Sequence[int]) -> RemoteState:
    """Decode the 40-byte Unitree remote payload embedded in HG LowState."""

    raw = bytes(data)
    if len(raw) != 40:
        raise ValueError(f"Expected 40 wireless-remote bytes, got {len(raw)}.")
    keys = int(struct.unpack_from("<H", raw, 2)[0])
    lx = float(struct.unpack_from("<f", raw, 4)[0])
    rx = float(struct.unpack_from("<f", raw, 8)[0])
    ry = float(struct.unpack_from("<f", raw, 12)[0])
    ly = float(struct.unpack_from("<f", raw, 20)[0])
    axes = np.array((lx, ly, rx, ry), dtype=np.float32)
    if not np.all(np.isfinite(axes)):
        raise FloatingPointError("Wireless remote contains NaN or infinity.")
    if np.any(np.abs(axes) > 1.25):
        raise ValueError(f"Wireless remote axes are outside [-1.25, 1.25]: {axes}.")
    return RemoteState(lx=lx, ly=ly, rx=rx, ry=ry, keys=keys)


def _apply_deadzone(value: float, deadzone: float) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    if abs(value) <= deadzone:
        return 0.0
    return float(np.sign(value) * (abs(value) - deadzone) / (1.0 - deadzone))


def _scale_unit_input(value: float, limits: np.ndarray) -> float:
    return value * float(limits[1] if value >= 0.0 else abs(limits[0]))


def command_from_remote(
    remote: RemoteState,
    command_ranges: np.ndarray,
    deadzone: float,
) -> np.ndarray:
    """Map Unitree sticks to [forward, left, yaw] training commands."""

    unit_input = np.array(
        (remote.ly, -remote.lx, -remote.rx),
        dtype=np.float32,
    )
    unit_input = np.array(
        [_apply_deadzone(float(value), deadzone) for value in unit_input],
        dtype=np.float32,
    )
    return np.array(
        [
            _scale_unit_input(float(unit_input[index]), command_ranges[index])
            for index in range(3)
        ],
        dtype=np.float32,
    )


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _load_parameters(
    args: argparse.Namespace, checkpoint_path: Path
) -> DeployParameters:
    if args.ignore_metadata:
        return DeployParameters(
            default_pos=FALLBACK_DEFAULT_POS.copy(),
            action_scale=FALLBACK_ACTION_SCALE.copy(),
            kp=FALLBACK_KP.copy(),
            kd=FALLBACK_KD.copy(),
            metadata_path=None,
        )
    metadata_path = (
        None
        if args.metadata_file is None
        else Path(args.metadata_file).expanduser().resolve()
    )
    return load_deploy_parameters(checkpoint_path, metadata_path)


class G1AmpSim2Real:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.checkpoint_path = Path(args.checkpoint_file).expanduser().resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.params = _load_parameters(args, self.checkpoint_path)
        self._validate_deploy_parameters()
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.actor = CheckpointActor(self.checkpoint_path, args.device)

        self.command_ranges = np.asarray(
            (args.cmd_x_range, args.cmd_y_range, args.cmd_yaw_range),
            dtype=np.float32,
        )
        self.safe_joint_lower = JOINT_LIMIT_LOWER + args.joint_limit_margin
        self.safe_joint_upper = JOINT_LIMIT_UPPER - args.joint_limit_margin
        if np.any(self.safe_joint_lower >= self.safe_joint_upper):
            raise ValueError("--joint-limit-margin leaves an empty joint range.")

        self._state: RealRobotState | None = None
        self._state_lock = threading.Lock()
        self._callback_error: str | None = None
        self._invalid_crc_count = 0
        self._publisher: ChannelPublisher | None = None
        self._subscriber: ChannelSubscriber | None = None
        self._crc = CRC()
        self._lowcmd = unitree_hg_msg_dds__LowCmd_()
        self._owns_low_level = False
        self._last_mode_machine = 0
        self._previous_high_level_mode: str | None = None
        self._initialize_lowcmd()

    def _validate_deploy_parameters(self) -> None:
        arrays = {
            "default_pos": self.params.default_pos,
            "action_scale": self.params.action_scale,
            "kp": self.params.kp,
            "kd": self.params.kd,
        }
        for name, array in arrays.items():
            if array.shape != (NUM_JOINTS,) or not np.all(np.isfinite(array)):
                raise ValueError(f"Deployment parameter '{name}' is invalid.")
        if np.any(self.params.action_scale <= 0.0):
            raise ValueError("Deployment action scales must be positive.")
        if np.any(self.params.kp < 0.0) or np.any(self.params.kd < 0.0):
            raise ValueError("Deployment gains cannot be negative.")
        if np.any(self.params.default_pos <= JOINT_LIMIT_LOWER) or np.any(
            self.params.default_pos >= JOINT_LIMIT_UPPER
        ):
            raise ValueError("Default pose is outside the G1 joint limits.")

    def _initialize_lowcmd(self) -> None:
        self._lowcmd.mode_pr = MODE_PR
        for index in range(LOWCMD_MOTOR_COUNT):
            motor = self._lowcmd.motor_cmd[index]
            motor.mode = 1 if index < NUM_JOINTS else 0
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def lowstate_callback(self, message: LowState_) -> None:
        try:
            expected_crc = int(self._crc.Crc(message))
            if int(message.crc) != expected_crc:
                with self._state_lock:
                    self._invalid_crc_count += 1
                    self._callback_error = (
                        f"LowState CRC mismatch: received={int(message.crc)}, "
                        f"expected={expected_crc}"
                    )
                return

            state = RealRobotState(
                joint_pos=np.fromiter(
                    (message.motor_state[index].q for index in range(NUM_JOINTS)),
                    dtype=np.float32,
                    count=NUM_JOINTS,
                ),
                joint_vel=np.fromiter(
                    (message.motor_state[index].dq for index in range(NUM_JOINTS)),
                    dtype=np.float32,
                    count=NUM_JOINTS,
                ),
                quaternion_wxyz=np.asarray(
                    message.imu_state.quaternion,
                    dtype=np.float32,
                ).copy(),
                angular_velocity=np.asarray(
                    message.imu_state.gyroscope,
                    dtype=np.float32,
                ).copy(),
                motor_errors=np.fromiter(
                    (
                        int(message.motor_state[index].motorstate)
                        for index in range(NUM_JOINTS)
                    ),
                    dtype=np.uint32,
                    count=NUM_JOINTS,
                ),
                remote=parse_wireless_remote(message.wireless_remote),
                mode_machine=int(message.mode_machine),
                tick=int(message.tick),
                received_at=time.perf_counter(),
            )
        except (FloatingPointError, TypeError, ValueError) as exc:
            with self._state_lock:
                self._callback_error = f"Invalid LowState: {exc}"
            return

        with self._state_lock:
            self._state = state
            self._callback_error = None

    def get_state(self) -> RealRobotState | None:
        with self._state_lock:
            return self._state

    def _get_callback_diagnostics(self) -> tuple[str | None, int]:
        with self._state_lock:
            return self._callback_error, self._invalid_crc_count

    def connect(self, state_only: bool = False) -> None:
        ChannelFactoryInitialize(REAL_DDS_DOMAIN_ID, self.args.network_interface)
        if not state_only:
            self._publisher = ChannelPublisher(LOWCMD_TOPIC, LowCmd_)
            self._publisher.Init()
        self._subscriber = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        self._subscriber.Init(self.lowstate_callback, 10)

    def wait_for_state(self) -> RealRobotState:
        print(f"[INFO] Waiting for CRC-valid G1 LowState on {LOWSTATE_TOPIC} ...")
        deadline = time.perf_counter() + self.args.connect_timeout
        while time.perf_counter() < deadline:
            state = self.get_state()
            if state is not None:
                self._check_state(state, require_upright=False)
                self._last_mode_machine = state.mode_machine
                print(
                    "[INFO] Connected to G1 HG LowState: "
                    f"tick={state.tick}, mode_machine={state.mode_machine}."
                )
                return state
            time.sleep(0.01)
        callback_error, invalid_crc_count = self._get_callback_diagnostics()
        detail = (
            f" Last callback error: {callback_error} "
            f"(CRC failures={invalid_crc_count})."
            if callback_error is not None
            else ""
        )
        raise TimeoutError(
            "No valid G1 LowState received. Check the robot network interface, "
            f"power, SDK version, and DDS domain.{detail}"
        )

    def _fresh_state(self, require_upright: bool = True) -> RealRobotState:
        state = self.get_state()
        if state is None:
            raise SafetyViolation("LowState is unavailable.")
        age = time.perf_counter() - state.received_at
        if age > self.args.state_timeout:
            callback_error, invalid_crc_count = self._get_callback_diagnostics()
            detail = (
                f"; last callback error: {callback_error}" if callback_error else ""
            )
            raise SafetyViolation(
                f"LowState is stale by {age:.3f}s "
                f"(CRC failures={invalid_crc_count}){detail}."
            )
        self._check_state(state, require_upright=require_upright)
        self._last_mode_machine = state.mode_machine
        return state

    def _check_state(
        self,
        state: RealRobotState,
        *,
        require_upright: bool,
    ) -> None:
        arrays = (
            state.joint_pos,
            state.joint_vel,
            state.quaternion_wxyz,
            state.angular_velocity,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise SafetyViolation("LowState contains NaN or infinity.")
        if state.joint_pos.shape != (NUM_JOINTS,) or state.joint_vel.shape != (
            NUM_JOINTS,
        ):
            raise SafetyViolation("LowState does not contain 29 ordered joints.")

        quat_norm = float(np.linalg.norm(state.quaternion_wxyz))
        if not 0.8 <= quat_norm <= 1.2:
            raise SafetyViolation(f"IMU quaternion norm is invalid: {quat_norm:.3f}.")
        if np.any(np.abs(state.joint_vel) > self.args.max_joint_speed):
            joint = int(np.argmax(np.abs(state.joint_vel)))
            raise SafetyViolation(
                f"Joint {joint} speed is {state.joint_vel[joint]:.2f} rad/s, "
                f"over the {self.args.max_joint_speed:.2f} rad/s limit."
            )

        tolerance = self.args.state_joint_limit_tolerance
        below = state.joint_pos < JOINT_LIMIT_LOWER - tolerance
        above = state.joint_pos > JOINT_LIMIT_UPPER + tolerance
        if np.any(below | above):
            joint = int(np.flatnonzero(below | above)[0])
            raise SafetyViolation(
                f"Joint {joint} position {state.joint_pos[joint]:.3f} rad is "
                "outside the configured G1 hard limit."
            )
        if np.any(state.motor_errors):
            joint = int(np.flatnonzero(state.motor_errors)[0])
            raise SafetyViolation(
                f"Motor {joint} reports error code {int(state.motor_errors[joint])}."
            )

        gravity_z = float(projected_gravity(state.quaternion_wxyz)[2])
        if require_upright and gravity_z > self.args.upright_gravity_z_threshold:
            raise SafetyViolation(
                f"Robot tilt is unsafe: projected gravity z={gravity_z:.3f}, "
                f"required <= {self.args.upright_gravity_z_threshold:.3f}."
            )

    def _clip_target(self, target_pos: np.ndarray) -> tuple[np.ndarray, int]:
        target = np.asarray(target_pos, dtype=np.float32)
        if target.shape != (NUM_JOINTS,) or not np.all(np.isfinite(target)):
            raise SafetyViolation("Controller produced an invalid joint target.")
        clipped = np.clip(target, self.safe_joint_lower, self.safe_joint_upper)
        clip_count = int(np.count_nonzero(np.abs(clipped - target) > 1.0e-6))
        return clipped.astype(np.float32), clip_count

    def publish_target(self, target_pos: np.ndarray, mode_machine: int) -> None:
        if self._publisher is None:
            raise RuntimeError("LowCmd publisher is not initialized.")
        if target_pos.shape != (NUM_JOINTS,) or not np.all(np.isfinite(target_pos)):
            raise SafetyViolation("Refusing to publish an invalid joint target.")
        self._lowcmd.mode_pr = MODE_PR
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
        self._lowcmd.mode_pr = MODE_PR
        self._lowcmd.mode_machine = mode_machine
        for index in range(NUM_JOINTS):
            motor = self._lowcmd.motor_cmd[index]
            motor.mode = 1
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = self.args.damping_kd
        self._lowcmd.crc = self._crc.Crc(self._lowcmd)
        self._publisher.Write(self._lowcmd)

    def _wait_for_button_edge(
        self,
        button: int,
        label: str,
        *,
        publish_target: np.ndarray | None,
        allow_emergency_stop: bool,
    ) -> RealRobotState:
        print(f"[WAIT] Release, then press {label}.")
        saw_released = False
        period = 1.0 / self.args.low_level_hz
        next_tick = time.perf_counter()
        while True:
            state = self._fresh_state(require_upright=True)
            if allow_emergency_stop and state.remote.pressed(BUTTON_B):
                raise EmergencyStop("B pressed by operator")
            is_pressed = state.remote.pressed(button)
            if not is_pressed:
                saw_released = True
            elif saw_released:
                return state

            if publish_target is not None:
                self.publish_target(publish_target, state.mode_machine)
            next_tick = self._sleep_until_next_tick(next_tick, period)

    def _release_high_level_mode(self, hold_target: np.ndarray) -> None:
        switcher = MotionSwitcherClient()
        switcher.SetTimeout(self.args.motion_switch_timeout)
        switcher.Init()

        status, result = switcher.CheckMode()
        if status != 0 or result is None:
            raise RuntimeError(f"MotionSwitcher CheckMode failed with code {status}.")
        active_name = str(result.get("name", ""))
        self._previous_high_level_mode = active_name or None
        if active_name:
            print(f"[INFO] Releasing active high-level mode: {active_name!r}.")
        else:
            print("[INFO] No active high-level mode reported.")

        for attempt in range(1, self.args.motion_switch_retries + 1):
            if active_name:
                code, _ = switcher.ReleaseMode()
                if code != 0:
                    raise RuntimeError(
                        f"MotionSwitcher ReleaseMode failed with code {code}."
                    )
            self._owns_low_level = True

            # Start holding the measured pose immediately after release so there
            # is no deliberate zero-torque gap while the service state settles.
            settle_deadline = time.perf_counter() + self.args.motion_switch_retry_delay
            while time.perf_counter() < settle_deadline:
                state = self._fresh_state(require_upright=True)
                if state.remote.pressed(BUTTON_B):
                    raise EmergencyStop("B pressed during low-level takeover")
                self.publish_target(hold_target, state.mode_machine)
                time.sleep(1.0 / self.args.low_level_hz)

            status, result = switcher.CheckMode()
            if status != 0 or result is None:
                raise RuntimeError(
                    f"MotionSwitcher CheckMode failed after release with code {status}."
                )
            active_name = str(result.get("name", ""))
            if not active_name:
                print(
                    "[INFO] High-level mode released; low-level control owns the robot."
                )
                return
            print(
                f"[WARN] High-level mode still active after attempt {attempt}: "
                f"{active_name!r}."
            )

        raise RuntimeError(
            "High-level motion mode could not be released after "
            f"{self.args.motion_switch_retries} attempts."
        )

    def _stand_up(self, initial_pos: np.ndarray) -> np.ndarray:
        print(
            "[STAGE] stand-up: interpolating measured pose to policy default over "
            f"{self.args.stand_up_duration:.1f}s."
        )
        period = 1.0 / self.args.low_level_hz
        start = time.perf_counter()
        next_tick = start
        target = initial_pos.copy()
        while True:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= self.args.stand_up_duration:
                break
            state = self._fresh_state(require_upright=True)
            if state.remote.pressed(BUTTON_B):
                raise EmergencyStop("B pressed during stand-up")
            blend = _smoothstep(elapsed / self.args.stand_up_duration)
            desired = (1.0 - blend) * initial_pos + blend * self.params.default_pos
            target, _ = self._clip_target(desired)
            self.publish_target(target, state.mode_machine)
            next_tick = self._sleep_until_next_tick(next_tick, period)

        target, _ = self._clip_target(self.params.default_pos)
        self.publish_target(target, self._fresh_state().mode_machine)
        return target

    def _hold_default(self, target: np.ndarray) -> None:
        print(
            f"[STAGE] hold: stabilizing default pose for {self.args.stand_hold_duration:.1f}s."
        )
        period = 1.0 / self.args.low_level_hz
        deadline = time.perf_counter() + self.args.stand_hold_duration
        next_tick = time.perf_counter()
        while time.perf_counter() < deadline:
            state = self._fresh_state(require_upright=True)
            if state.remote.pressed(BUTTON_B):
                raise EmergencyStop("B pressed during stand hold")
            self.publish_target(target, state.mode_machine)
            next_tick = self._sleep_until_next_tick(next_tick, period)

    def _run_policy(self, initial_target: np.ndarray) -> None:
        print("[STAGE] policy: B = emergency damping and exit.")
        period = 1.0 / self.args.low_level_hz
        policy_period = 1.0 / self.args.policy_hz
        start = time.perf_counter()
        next_tick = start
        next_policy = start
        next_status = start
        last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        target = initial_target.copy()
        command = np.zeros(3, dtype=np.float32)
        clipped_joints = 0

        while True:
            now = time.perf_counter()
            if self.args.duration is not None and now - start >= self.args.duration:
                print("[INFO] Requested policy duration completed.")
                return
            state = self._fresh_state(require_upright=True)
            if state.remote.pressed(BUTTON_B):
                raise EmergencyStop("B pressed by operator")

            if now >= next_policy:
                command = command_from_remote(
                    state.remote,
                    self.command_ranges,
                    self.args.wireless_deadzone,
                )
                if self.args.command_warmup > 0.0:
                    command *= min(1.0, (now - start) / self.args.command_warmup)
                observation = build_observation(
                    state,
                    command,
                    last_action,
                    self.params.default_pos,
                )
                action = self.actor(observation)
                last_action = np.clip(
                    action,
                    -self.args.action_clip,
                    self.args.action_clip,
                ).astype(np.float32)
                desired = (
                    self.params.default_pos + last_action * self.params.action_scale
                )
                desired, clipped_joints = self._clip_target(desired)
                max_delta = self.args.max_target_speed / self.args.policy_hz
                target += np.clip(desired - target, -max_delta, max_delta)
                target, additional_clips = self._clip_target(target)
                clipped_joints = max(clipped_joints, additional_clips)

                next_policy += policy_period
                if next_policy < now - policy_period:
                    next_policy = now + policy_period

            self.publish_target(target, state.mode_machine)
            if now >= next_status:
                gravity_z = float(projected_gravity(state.quaternion_wxyz)[2])
                state_age_ms = (now - state.received_at) * 1000.0
                print(
                    f"[STATE] cmd=({command[0]:+.2f}, {command[1]:+.2f}, "
                    f"{command[2]:+.2f}) gravity_z={gravity_z:+.3f} "
                    f"state_age={state_age_ms:.1f}ms "
                    f"target_clips={clipped_joints}"
                )
                next_status = now + self.args.status_interval
            next_tick = self._sleep_until_next_tick(next_tick, period)

    @staticmethod
    def _sleep_until_next_tick(next_tick: float, period: float) -> float:
        next_tick += period
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        elif sleep_time < -5.0 * period:
            next_tick = time.perf_counter()
        return next_tick

    def _damping_shutdown(self) -> None:
        if not self._owns_low_level or self._publisher is None:
            return
        print(
            "[SAFE] Sending damping-only commands for "
            f"{self.args.damping_duration:.1f}s."
        )
        period = 1.0 / self.args.low_level_hz
        deadline = time.perf_counter() + self.args.damping_duration
        while time.perf_counter() < deadline:
            try:
                state = self.get_state()
                mode_machine = (
                    state.mode_machine if state is not None else self._last_mode_machine
                )
                self.publish_damping(mode_machine)
            except Exception as exc:  # Keep trying for the bounded shutdown window.
                print(f"[WARN] Failed to publish one damping command: {exc}")
            time.sleep(period)
        print(
            "[SAFE] Low-level output stopped in damping mode. "
            "High-level mode was not restored automatically."
        )

    def print_summary(self) -> None:
        print(f"[INFO] Checkpoint: {self.checkpoint_path}")
        if self.params.metadata_path is None:
            print(
                "[WARN] Using built-in deployment parameters; no ONNX metadata loaded."
            )
        else:
            print(f"[INFO] Metadata:   {self.params.metadata_path}")
        if self.args.network_interface:
            print(
                f"[INFO] Real DDS domain={REAL_DDS_DOMAIN_ID}, "
                f"interface={self.args.network_interface}, "
                f"low-level={self.args.low_level_hz:.1f}Hz, "
                f"policy={self.args.policy_hz:.1f}Hz"
            )

    def validate_only(self) -> None:
        self.print_summary()
        zero_obs = np.zeros(OBS_DIM, dtype=np.float32)
        action = self.actor(zero_obs)
        target, clip_count = self._clip_target(
            self.params.default_pos
            + np.clip(action, -self.args.action_clip, self.args.action_clip)
            * self.params.action_scale
        )
        if not np.all(np.isfinite(target)):
            raise AssertionError("Offline target validation failed.")
        print(
            f"[OK] sim2real offline validation: obs={OBS_DIM}, actions={action.size}, "
            f"zero_obs_action_norm={np.linalg.norm(action):.4f}, "
            f"target_clips={clip_count}"
        )

    def check_state_only(self) -> None:
        self.print_summary()
        self.connect(state_only=True)
        self.wait_for_state()
        print(
            "[INFO] Read-only state check: no LowCmd publisher was created and no "
            "motion mode will be released."
        )
        deadline = time.perf_counter() + self.args.state_check_duration
        next_status = time.perf_counter()
        while time.perf_counter() < deadline:
            state = self._fresh_state(require_upright=False)
            now = time.perf_counter()
            if now >= next_status:
                gravity_z = float(projected_gravity(state.quaternion_wxyz)[2])
                max_speed = float(np.max(np.abs(state.joint_vel)))
                print(
                    f"[STATE] tick={state.tick} mode_machine={state.mode_machine} "
                    f"gravity_z={gravity_z:+.3f} max_joint_speed={max_speed:.2f}rad/s "
                    f"keys=0x{state.remote.keys:04x}"
                )
                next_status = now + self.args.status_interval
            time.sleep(0.01)
        print("[OK] Read-only real-robot state check passed.")

    def run(self) -> None:
        self.print_summary()
        self.connect(state_only=False)
        initial_state = self.wait_for_state()
        self._check_state(initial_state, require_upright=True)
        print(
            "[SAFE] No LowCmd has been sent. Ensure the robot is supported, the area "
            "is clear, and an operator can use the physical emergency stop."
        )

        try:
            takeover_state = self._wait_for_button_edge(
                BUTTON_START,
                "START to release high-level mode and begin stand-up",
                publish_target=None,
                allow_emergency_stop=False,
            )
            initial_target, _ = self._clip_target(takeover_state.joint_pos)
            self._release_high_level_mode(initial_target)
            current_state = self._fresh_state(require_upright=True)
            target = self._stand_up(current_state.joint_pos.copy())
            self._hold_default(target)
            self._wait_for_button_edge(
                BUTTON_A,
                "A to enable the policy",
                publish_target=target,
                allow_emergency_stop=True,
            )
            self._run_policy(target)
        finally:
            self._damping_shutdown()


def _validate_range(
    parser: argparse.ArgumentParser,
    name: str,
    values: Sequence[float],
) -> None:
    if len(values) != 2 or not np.all(np.isfinite(values)) or values[0] >= values[1]:
        parser.error(
            f"{name} must contain finite LOWER UPPER values with LOWER < UPPER."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy a 29-DOF G1 AMP checkpoint to real hardware with explicit "
            "takeover and software safety checks."
        )
    )
    parser.add_argument(
        "--checkpoint-file",
        required=True,
        help="RSL-RL model_*.pt checkpoint produced by this repository.",
    )
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Optional policy.onnx containing deployment metadata.",
    )
    parser.add_argument(
        "--ignore-metadata",
        action="store_true",
        help="Use built-in parameters even if policy.onnx exists beside the checkpoint.",
    )
    parser.add_argument("--device", default="cpu", help="Torch inference device.")
    parser.add_argument(
        "--network-interface",
        default=None,
        help="Physical network interface connected to the G1, for example enp3s0.",
    )
    parser.add_argument(
        "--robot-model",
        choices=("g1-29dof",),
        default="g1-29dof",
        help="Connected hardware variant (default: g1-29dof).",
    )
    parser.add_argument(
        "--waist-mode",
        choices=("unlocked",),
        default="unlocked",
        help=(
            "Waist configuration; roll and pitch must be mechanically available "
            "(default: unlocked)."
        ),
    )
    parser.add_argument(
        "--acknowledge-risk",
        action="store_true",
        help="Acknowledge that this command will take low-level control of real hardware.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate checkpoint and targets without opening DDS.",
    )
    mode.add_argument(
        "--state-check-only",
        action="store_true",
        help="Read and validate real LowState without publishing or releasing control.",
    )
    parser.add_argument("--state-check-duration", type=float, default=5.0)
    parser.add_argument("--low-level-hz", type=float, default=500.0)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--state-timeout", type=float, default=0.1)
    parser.add_argument("--stand-up-duration", type=float, default=5.0)
    parser.add_argument("--stand-hold-duration", type=float, default=2.0)
    parser.add_argument("--command-warmup", type=float, default=2.0)
    parser.add_argument("--cmd-x-range", type=float, nargs=2, default=(-0.5, 1.0))
    parser.add_argument("--cmd-y-range", type=float, nargs=2, default=(-0.5, 0.5))
    parser.add_argument("--cmd-yaw-range", type=float, nargs=2, default=(-1.0, 1.0))
    parser.add_argument("--wireless-deadzone", type=float, default=0.08)
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument(
        "--max-target-speed",
        type=float,
        default=4.0,
        help="Maximum per-joint policy target slew rate in rad/s.",
    )
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--state-joint-limit-tolerance", type=float, default=0.1)
    parser.add_argument("--max-joint-speed", type=float, default=35.0)
    parser.add_argument(
        "--upright-gravity-z-threshold",
        type=float,
        default=-0.5,
        help="Stop when projected gravity z rises above this value (upright is -1).",
    )
    parser.add_argument("--damping-kd", type=float, default=8.0)
    parser.add_argument("--damping-duration", type=float, default=1.0)
    parser.add_argument("--motion-switch-timeout", type=float, default=5.0)
    parser.add_argument("--motion-switch-retries", type=int, default=3)
    parser.add_argument("--motion-switch-retry-delay", type=float, default=0.5)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional policy runtime in seconds before damping shutdown.",
    )
    parser.add_argument("--status-interval", type=float, default=1.0)
    args = parser.parse_args()

    if not args.validate_only:
        if not args.network_interface:
            parser.error("--network-interface is required for real DDS access.")
        if args.network_interface == "lo":
            parser.error(
                "The sim2real entry point refuses loopback interface 'lo'; "
                "use the physical interface connected to G1."
            )
        interface_path = Path("/sys/class/net") / args.network_interface
        if not interface_path.exists():
            parser.error(f"Network interface does not exist: {args.network_interface}")
        if args.robot_model != "g1-29dof":
            parser.error("--robot-model g1-29dof is required.")
        if args.waist_mode != "unlocked":
            parser.error(
                "--waist-mode unlocked is required because the policy commands "
                "waist roll and pitch."
            )
    if (
        not args.validate_only
        and not args.state_check_only
        and not args.acknowledge_risk
    ):
        parser.error("--acknowledge-risk is required before real low-level control.")

    if args.low_level_hz <= 0.0 or args.policy_hz <= 0.0:
        parser.error("Control frequencies must be positive.")
    if args.policy_hz > args.low_level_hz:
        parser.error("--policy-hz cannot exceed --low-level-hz.")
    if args.connect_timeout <= 0.0 or args.state_timeout <= 0.0:
        parser.error("DDS timeouts must be positive.")
    if args.stand_up_duration <= 0.0 or args.stand_hold_duration < 0.0:
        parser.error(
            "Stand-up duration must be positive and hold duration non-negative."
        )
    if args.command_warmup < 0.0:
        parser.error("--command-warmup cannot be negative.")
    if not 0.0 <= args.wireless_deadzone < 1.0:
        parser.error("--wireless-deadzone must be in [0, 1).")
    if args.action_clip <= 0.0 or args.max_target_speed <= 0.0:
        parser.error("Action clip and target speed must be positive.")
    if args.joint_limit_margin < 0.0:
        parser.error("--joint-limit-margin cannot be negative.")
    if args.state_joint_limit_tolerance < 0.0:
        parser.error("--state-joint-limit-tolerance cannot be negative.")
    if args.max_joint_speed <= 0.0:
        parser.error("--max-joint-speed must be positive.")
    if not -1.0 < args.upright_gravity_z_threshold < 0.0:
        parser.error("--upright-gravity-z-threshold must be between -1 and 0.")
    if args.damping_kd <= 0.0 or args.damping_duration <= 0.0:
        parser.error("Damping gain and duration must be positive.")
    if args.motion_switch_timeout <= 0.0:
        parser.error("--motion-switch-timeout must be positive.")
    if args.motion_switch_retries <= 0 or args.motion_switch_retry_delay <= 0.0:
        parser.error("Motion-switch retries and retry delay must be positive.")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive.")
    if args.state_check_duration <= 0.0 or args.status_interval <= 0.0:
        parser.error("State-check duration and status interval must be positive.")
    _validate_range(parser, "--cmd-x-range", args.cmd_x_range)
    _validate_range(parser, "--cmd-y-range", args.cmd_y_range)
    _validate_range(parser, "--cmd-yaw-range", args.cmd_yaw_range)
    return args


def main() -> None:
    args = parse_args()
    controller = G1AmpSim2Real(args)
    if args.validate_only:
        controller.validate_only()
        return
    if args.state_check_only:
        controller.check_state_only()
        return
    try:
        controller.run()
    except EmergencyStop as exc:
        print(f"[SAFE] Operator emergency stop: {exc}.")
    except SafetyViolation as exc:
        print(f"[SAFE] Safety violation: {exc}.")
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        print("[SAFE] Ctrl+C received.")


if __name__ == "__main__":
    main()
