from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mjlab.entity import Entity  # noqa: E402
from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (  # noqa: E402
    unitree_go_msg_dds__WirelessController_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402

from deploy.amp.sim2sim.g1_amp_sim2sim import (  # noqa: E402
    FALLBACK_DEFAULT_POS,
    JOINT_NAMES,
    LOWCMD_TOPIC,
    LOWSTATE_TOPIC,
    NUM_JOINTS,
    WIRELESS_TOPIC,
)
from src.assets.robots.unitree_g1.g1_constants import (  # noqa: E402
    get_g1_robot_cfg,
)


SIM_TIMESTEP = 0.005
SOLVER_ITERATIONS = 10
SOLVER_LS_ITERATIONS = 20
CCD_ITERATIONS = 50
SCENE_G1_XML = (
    REPO_ROOT
    / "src"
    / "assets"
    / "robots"
    / "unitree_g1"
    / "xmls"
    / "scene_g1_flat.xml"
)


@dataclass(frozen=True)
class ModelIndex:
    """Natural-joint-order addresses into a compiled MuJoCo model."""

    joint_qpos: np.ndarray
    joint_dof: np.ndarray
    actuator: np.ndarray
    root_qpos: int
    imu_gyro_sensor: int
    imu_acc_sensor: int


@dataclass(frozen=True)
class LowLevelCommand:
    """Thread-safe copy of the latest Unitree LowCmd."""

    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    mode_machine: int


GAMEPAD_AXIS = {
    "xbox": {"LX": 0, "LY": 1, "RX": 3, "RY": 4, "LT": 2, "RT": 5},
    "switch": {"LX": 0, "LY": 1, "RX": 2, "RY": 3, "LT": 5, "RT": 4},
}

GAMEPAD_BUTTON = {
    "xbox": {
        "A": 0,
        "B": 1,
        "X": 2,
        "Y": 3,
        "LB": 4,
        "RB": 5,
        "SELECT": 6,
        "START": 7,
    },
    "switch": {
        "A": 0,
        "B": 1,
        "X": 3,
        "Y": 4,
        "LB": 6,
        "RB": 7,
        "SELECT": 10,
        "START": 11,
    },
}

UNITREE_KEY_BIT = {
    "R1": 0,
    "L1": 1,
    "start": 2,
    "select": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
    "up": 12,
    "right": 13,
    "down": 14,
    "left": 15,
}


@dataclass(frozen=True)
class WirelessState:
    """Unitree wireless-controller values sampled from a local gamepad."""

    lx: float
    ly: float
    rx: float
    ry: float
    keys: int


class Gamepad:
    """Pygame gamepad reader using unitree_mujoco's axis convention."""

    def __init__(self, device_id: int, gamepad_type: str) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "Gamepad control requires pygame. Install it with 'pip install pygame'."
            ) from exc

        self._pygame = pygame
        self._axis = GAMEPAD_AXIS[gamepad_type]
        self._button = GAMEPAD_BUTTON[gamepad_type]
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("No gamepad detected by pygame.")
        if device_id >= count:
            raise ValueError(
                f"--joystick-device {device_id} is unavailable; detected {count} gamepad(s)."
            )

        self._joystick = pygame.joystick.Joystick(device_id)
        self._joystick.init()
        print(
            f"[INFO] Gamepad: id={device_id}, type={gamepad_type}, "
            f"name={self._joystick.get_name()!r}, "
            f"axes={self._joystick.get_numaxes()}, "
            f"buttons={self._joystick.get_numbuttons()}"
        )

    def _get_axis(self, name: str) -> float:
        axis_id = self._axis[name]
        if axis_id >= self._joystick.get_numaxes():
            return 0.0
        return float(np.clip(self._joystick.get_axis(axis_id), -1.0, 1.0))

    def _get_button(self, name: str) -> bool:
        button_id = self._button[name]
        if button_id >= self._joystick.get_numbuttons():
            return False
        return bool(self._joystick.get_button(button_id))

    def sample(self) -> WirelessState:
        self._pygame.event.pump()
        hat = self._joystick.get_hat(0) if self._joystick.get_numhats() > 0 else (0, 0)
        pressed = {
            "R1": self._get_button("RB"),
            "L1": self._get_button("LB"),
            "start": self._get_button("START"),
            "select": self._get_button("SELECT"),
            "R2": self._get_axis("RT") > 0.0,
            "L2": self._get_axis("LT") > 0.0,
            "F1": False,
            "F2": False,
            "A": self._get_button("A"),
            "B": self._get_button("B"),
            "X": self._get_button("X"),
            "Y": self._get_button("Y"),
            "up": hat[1] > 0,
            "right": hat[0] > 0,
            "down": hat[1] < 0,
            "left": hat[0] < 0,
        }
        keys = sum(
            int(is_pressed) << UNITREE_KEY_BIT[name]
            for name, is_pressed in pressed.items()
        )
        return WirelessState(
            lx=self._get_axis("LX"),
            ly=-self._get_axis("LY"),
            rx=self._get_axis("RX"),
            ry=-self._get_axis("RY"),
            keys=keys,
        )


def _joint_index(model: mujoco.MjModel) -> ModelIndex:
    joint_qpos: list[int] = []
    joint_dof: list[int] = []
    actuator: list[int] = []

    actuator_by_joint: dict[str, int] = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is not None:
            actuator_by_joint[joint_name] = actuator_id

    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Training model is missing joint '{joint_name}'.")
        if joint_name not in actuator_by_joint:
            raise ValueError(f"Training model is missing actuator for '{joint_name}'.")
        joint_qpos.append(int(model.jnt_qposadr[joint_id]))
        joint_dof.append(int(model.jnt_dofadr[joint_id]))
        actuator.append(actuator_by_joint[joint_name])

    root_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    if root_joint_id < 0:
        raise ValueError("Training model is missing floating_base_joint.")

    gyro_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    acc_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_lin_acc")
    if gyro_sensor_id < 0 or acc_sensor_id < 0:
        raise ValueError("Training model is missing pelvis IMU sensors.")

    return ModelIndex(
        joint_qpos=np.asarray(joint_qpos, dtype=np.int32),
        joint_dof=np.asarray(joint_dof, dtype=np.int32),
        actuator=np.asarray(actuator, dtype=np.int32),
        root_qpos=int(model.jnt_qposadr[root_joint_id]),
        imu_gyro_sensor=int(model.sensor_adr[gyro_sensor_id]),
        imu_acc_sensor=int(model.sensor_adr[acc_sensor_id]),
    )


def _convert_position_actuators_to_torque(model: mujoco.MjModel) -> None:
    """Keep training limits/dynamics while accepting Unitree-style torque commands."""

    if model.nu != NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} actuators, got {model.nu}.")
    if not np.all(model.actuator_forcelimited):
        raise ValueError("All training actuators must have effort limits.")

    force_range = model.actuator_forcerange.copy()
    model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
    model.actuator_gainprm[:] = 0.0
    model.actuator_gainprm[:, 0] = 1.0
    model.actuator_biasprm[:] = 0.0
    model.actuator_ctrllimited[:] = 1
    model.actuator_ctrlrange[:] = force_range


def build_training_model(
    foot_friction: float | None = None,
) -> tuple[mujoco.MjModel, ModelIndex]:
    """Build the training entity from the include-only flat scene."""

    cfg = get_g1_robot_cfg()
    cfg.spec_fn = lambda: mujoco.MjSpec.from_file(str(SCENE_G1_XML))
    robot = Entity(cfg)
    spec = robot.spec
    floor = spec.geom("floor")
    if floor is None:
        raise ValueError("scene_g1_flat.xml is missing the floor geom.")
    # FULL_COLLISION intentionally disables every geom outside *_collision.
    # Re-enable the scene floor after that editor has configured the robot.
    floor.contype = 1
    floor.conaffinity = 1
    floor.condim = 3
    spec.option.timestep = SIM_TIMESTEP
    spec.option.iterations = SOLVER_ITERATIONS
    spec.option.ls_iterations = SOLVER_LS_ITERATIONS
    spec.option.ccd_iterations = CCD_ITERATIONS
    model = spec.compile()

    if foot_friction is not None:
        for side in ("left", "right"):
            for number in range(1, 8):
                geom_name = f"{side}_foot{number}_collision"
                geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if geom_id < 0:
                    raise ValueError(f"Training model is missing geom '{geom_name}'.")
                model.geom_friction[geom_id, 0] = foot_friction

    index = _joint_index(model)
    _convert_position_actuators_to_torque(model)
    return model, index


def validate_model(model: mujoco.MjModel, index: ModelIndex) -> None:
    """Fail early if the simulator no longer matches the deployment contract."""

    if (model.nq, model.nv, model.nu) != (36, 35, NUM_JOINTS):
        raise ValueError(
            f"Unexpected G1 dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}."
        )
    if len(set(index.actuator.tolist())) != NUM_JOINTS:
        raise ValueError("Natural joint order does not map one-to-one to actuators.")
    if (model.ngeom, model.nsensor, model.nsensordata) != (69, 4, 12):
        raise ValueError(
            "Unexpected flat scene structure: "
            f"ngeom={model.ngeom}, nsensor={model.nsensor}, "
            f"nsensordata={model.nsensordata}."
        )
    expected_options = (
        SIM_TIMESTEP,
        SOLVER_ITERATIONS,
        SOLVER_LS_ITERATIONS,
        CCD_ITERATIONS,
    )
    actual_options = (
        model.opt.timestep,
        model.opt.iterations,
        model.opt.ls_iterations,
        model.opt.ccd_iterations,
    )
    if actual_options != expected_options:
        raise ValueError(f"Unexpected scene solver options: got {actual_options}.")
    if np.any(model.dof_damping) or np.any(model.dof_frictionloss):
        raise ValueError("Training model must not add passive damping/frictionloss.")

    for actuator_id in index.actuator:
        if model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED:
            raise ValueError("Actuator torque conversion failed.")
        if model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_NONE:
            raise ValueError("Actuator still contains built-in position feedback.")

    foot_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{side}_foot{number}_collision",
        )
        for side in ("left", "right")
        for number in range(1, 8)
    ]
    if any(geom_id < 0 for geom_id in foot_ids):
        raise ValueError("Expected all fourteen training foot collision geoms.")
    if not all(
        model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
        for geom_id in foot_ids
    ):
        raise ValueError("Training foot collision geoms must be capsules.")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0 or model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("scene_g1_flat.xml checker floor is missing.")
    if model.geom_contype[floor_id] == 0 or model.geom_conaffinity[floor_id] == 0:
        raise ValueError("scene_g1_flat.xml floor collision is disabled.")


class DdsBridge:
    """Minimal G1 LowCmd/LowState bridge."""

    def __init__(
        self,
        domain_id: int,
        interface: str,
        publish_wireless: bool,
        wireless_topic: str,
    ) -> None:
        self.domain_id = domain_id
        self.interface = interface
        self.publish_wireless = publish_wireless
        self.wireless_topic = wireless_topic
        self._lock = threading.Lock()
        self._command: LowLevelCommand | None = None
        self._publisher: ChannelPublisher | None = None
        self._subscriber: ChannelSubscriber | None = None
        self._wireless_publisher: ChannelPublisher | None = None
        self._state = unitree_hg_msg_dds__LowState_()
        self._wireless_state = unitree_go_msg_dds__WirelessController_()

    def connect(self) -> None:
        ChannelFactoryInitialize(self.domain_id, self.interface)
        self._publisher = ChannelPublisher(LOWSTATE_TOPIC, LowState_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber(LOWCMD_TOPIC, LowCmd_)
        self._subscriber.Init(self._lowcmd_callback, 10)
        if self.publish_wireless:
            self._wireless_publisher = ChannelPublisher(
                self.wireless_topic, WirelessController_
            )
            self._wireless_publisher.Init()

    def _lowcmd_callback(self, message: LowCmd_) -> None:
        command = LowLevelCommand(
            q=np.fromiter(
                (message.motor_cmd[i].q for i in range(NUM_JOINTS)),
                dtype=np.float64,
                count=NUM_JOINTS,
            ),
            dq=np.fromiter(
                (message.motor_cmd[i].dq for i in range(NUM_JOINTS)),
                dtype=np.float64,
                count=NUM_JOINTS,
            ),
            tau=np.fromiter(
                (message.motor_cmd[i].tau for i in range(NUM_JOINTS)),
                dtype=np.float64,
                count=NUM_JOINTS,
            ),
            kp=np.fromiter(
                (message.motor_cmd[i].kp for i in range(NUM_JOINTS)),
                dtype=np.float64,
                count=NUM_JOINTS,
            ),
            kd=np.fromiter(
                (message.motor_cmd[i].kd for i in range(NUM_JOINTS)),
                dtype=np.float64,
                count=NUM_JOINTS,
            ),
            mode_machine=int(message.mode_machine),
        )
        arrays = (command.q, command.dq, command.tau, command.kp, command.kd)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            return
        with self._lock:
            self._command = command

    def command(self) -> LowLevelCommand | None:
        with self._lock:
            return self._command

    def publish_state(
        self,
        data: mujoco.MjData,
        index: ModelIndex,
        tick: int,
        mode_machine: int,
    ) -> None:
        if self._publisher is None:
            raise RuntimeError("DDS bridge is not connected.")

        q = data.qpos[index.joint_qpos]
        dq = data.qvel[index.joint_dof]
        tau = data.actuator_force[index.actuator]
        for motor_id in range(NUM_JOINTS):
            motor = self._state.motor_state[motor_id]
            motor.q = float(q[motor_id])
            motor.dq = float(dq[motor_id])
            motor.tau_est = float(tau[motor_id])

        quat = data.qpos[index.root_qpos + 3 : index.root_qpos + 7]
        gyro = data.sensordata[index.imu_gyro_sensor : index.imu_gyro_sensor + 3]
        acc = data.sensordata[index.imu_acc_sensor : index.imu_acc_sensor + 3]
        self._state.imu_state.quaternion[:] = [float(value) for value in quat]
        self._state.imu_state.gyroscope[:] = [float(value) for value in gyro]
        self._state.imu_state.accelerometer[:] = [float(value) for value in acc]
        self._state.mode_pr = 0
        self._state.mode_machine = mode_machine
        self._state.tick = tick & 0xFFFFFFFF
        self._publisher.Write(self._state)

    def publish_gamepad(self, state: WirelessState) -> None:
        if self._wireless_publisher is None:
            raise RuntimeError("Wireless DDS publisher is not initialized.")
        self._wireless_state.lx = state.lx
        self._wireless_state.ly = state.ly
        self._wireless_state.rx = state.rx
        self._wireless_state.ry = state.ry
        self._wireless_state.keys = state.keys
        self._wireless_publisher.Write(self._wireless_state)


def _apply_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: ModelIndex,
    command: LowLevelCommand,
) -> None:
    q = data.qpos[index.joint_qpos]
    dq = data.qvel[index.joint_dof]
    torque = command.tau + command.kp * (command.q - q) + command.kd * (command.dq - dq)
    control_range = model.actuator_ctrlrange[index.actuator]
    torque = np.clip(torque, control_range[:, 0], control_range[:, 1])
    data.ctrl[index.actuator] = torque


def _reset_to_training_state(
    model: mujoco.MjModel, data: mujoco.MjData, index: ModelIndex
) -> None:
    """Reset to the same HOME pose used by the training entity."""

    mujoco.mj_resetData(model, data)
    data.qpos[index.root_qpos : index.root_qpos + 7] = (
        0.0,
        0.0,
        0.8,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    data.qpos[index.joint_qpos] = FALLBACK_DEFAULT_POS
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def run_simulator(args: argparse.Namespace) -> None:
    model, index = build_training_model(args.foot_friction)
    validate_model(model, index)
    data = mujoco.MjData(model)
    _reset_to_training_state(model, data, index)

    gamepad = (
        Gamepad(args.joystick_device, args.joystick_type) if args.joystick else None
    )
    bridge = DdsBridge(
        args.domain_id,
        args.interface,
        publish_wireless=gamepad is not None,
        wireless_topic=args.wireless_topic,
    )
    bridge.connect()

    mass = float(np.sum(model.body_mass))
    friction_text = (
        "training default (0.6)"
        if args.foot_friction is None
        else f"{args.foot_friction:.3f}"
    )
    print(
        f"[INFO] MJLab scene_g1_flat model: mass={mass:.3f} kg, "
        f"dt={model.opt.timestep:.4f}s, foot_friction={friction_text}"
    )
    print(
        f"[INFO] DDS domain={args.domain_id}, interface={args.interface}, "
        f"publish={LOWSTATE_TOPIC}, subscribe={LOWCMD_TOPIC}"
    )
    if gamepad is not None:
        print(
            f"[INFO] Gamepad publish={args.wireless_topic}, "
            f"rate={args.wireless_hz:.1f} Hz"
        )
    if not args.start_immediately:
        print("[INFO] Physics is paused until the first LowCmd arrives.")

    viewer_context = (
        contextlib.nullcontext(None) if args.headless else _viewer_context(model, data)
    )
    with viewer_context as viewer:
        start_wall = time.perf_counter()
        next_tick = start_wall
        next_status = start_wall
        tick = 0
        started = args.start_immediately
        announced_start = started
        mode_machine = 0
        viewer_sync_steps = max(1, round(0.02 / model.opt.timestep))
        wireless_steps = max(1, round(1.0 / (args.wireless_hz * model.opt.timestep)))

        while viewer is None or viewer.is_running():
            now = time.perf_counter()
            if args.duration is not None and now - start_wall >= args.duration:
                break

            command = bridge.command()
            if command is not None:
                started = True
                mode_machine = command.mode_machine
                _apply_command(model, data, index, command)
            elif started:
                data.ctrl[:] = 0.0

            if started:
                if not announced_start:
                    print("[INFO] First LowCmd received; physics started.")
                    announced_start = True
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            bridge.publish_state(data, index, tick, mode_machine)
            if gamepad is not None and tick % wireless_steps == 0:
                bridge.publish_gamepad(gamepad.sample())
            tick += 1

            if viewer is not None and tick % viewer_sync_steps == 0:
                viewer.sync()

            if now >= next_status:
                stage = "running" if started else "waiting-lowcmd"
                root_z = float(data.qpos[index.root_qpos + 2])
                speed = float(np.linalg.norm(data.qvel[index.joint_dof]))
                print(
                    f"[STATE] stage={stage:14s} root_z={root_z:.3f} "
                    f"joint_speed_norm={speed:.3f}"
                )
                next_status = now + args.status_interval

            next_tick += model.opt.timestep / args.realtime_rate
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            elif sleep_time < -5.0 * model.opt.timestep:
                next_tick = time.perf_counter()


def _viewer_context(
    model: mujoco.MjModel, data: mujoco.MjData
) -> contextlib.AbstractContextManager:
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.lookat[:] = (0.0, 0.0, 0.8)
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -15.0
    return viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Unitree LowCmd/LowState simulator using the MJLab G1 "
            "training model inside a lightweight flat scene."
        )
    )
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument(
        "--foot-friction",
        type=float,
        default=None,
        help="Override the training foot friction (default: keep 0.6).",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without a MuJoCo viewer."
    )
    parser.add_argument(
        "--joystick",
        action="store_true",
        help="Read a local gamepad and publish Unitree WirelessController.",
    )
    parser.add_argument(
        "--joystick-type",
        choices=tuple(GAMEPAD_AXIS),
        default="xbox",
        help="Gamepad axis/button layout.",
    )
    parser.add_argument("--joystick-device", type=int, default=0)
    parser.add_argument("--wireless-topic", default=WIRELESS_TOPIC)
    parser.add_argument("--wireless-hz", type=float, default=100.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional wall-clock runtime in seconds.",
    )
    parser.add_argument(
        "--start-immediately",
        action="store_true",
        help="Advance physics before receiving the first LowCmd.",
    )
    parser.add_argument("--realtime-rate", type=float, default=1.0)
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Compile and validate the training model without opening DDS.",
    )
    args = parser.parse_args()
    if args.domain_id < 0:
        parser.error("--domain-id must be non-negative.")
    if args.joystick_device < 0:
        parser.error("--joystick-device must be non-negative.")
    if args.wireless_hz <= 0.0 or args.wireless_hz > 1.0 / SIM_TIMESTEP:
        parser.error(f"--wireless-hz must be in (0, {1.0 / SIM_TIMESTEP:.0f}].")
    if args.foot_friction is not None and args.foot_friction <= 0.0:
        parser.error("--foot-friction must be positive.")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive.")
    if args.realtime_rate <= 0.0:
        parser.error("--realtime-rate must be positive.")
    if args.status_interval <= 0.0:
        parser.error("--status-interval must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if args.validate_only:
        model, index = build_training_model(args.foot_friction)
        validate_model(model, index)
        foot_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot1_collision"
        )
        print(
            "[OK] scene_g1_flat model validated: "
            f"nq={model.nq}, nv={model.nv}, nu={model.nu}, "
            f"mass={np.sum(model.body_mass):.3f}kg, "
            f"foot_friction={model.geom_friction[foot_id, 0]:.3f}"
        )
        return
    try:
        run_simulator(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
