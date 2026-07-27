from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort


NUM_JOINTS = 29
OBS_DIM = 96

# The ONNX metadata, policy output, HG LowState, and HG LowCmd must all use this
# exact G1 29-DOF motor order.
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
OBSERVATION_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
)


class PolicyState(Protocol):
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray


@dataclass(frozen=True)
class DeployParameters:
    default_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


def _parse_required_metadata_array(
    metadata: dict[str, str],
    key: str,
) -> np.ndarray:
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"ONNX metadata is missing required key '{key}'.")
    parts = value.split(",")
    if len(parts) != NUM_JOINTS:
        raise ValueError(
            f"ONNX metadata '{key}' must contain {NUM_JOINTS} values, "
            f"got {len(parts)}."
        )
    try:
        array = np.asarray([float(part) for part in parts], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"ONNX metadata '{key}' is not numeric.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"ONNX metadata '{key}' contains NaN or infinity.")
    return array


class OnnxPolicy:
    """Validated CPU-only ONNX actor and its required deployment metadata."""

    provider = "CPUExecutionProvider"

    def __init__(self, policy_path: Path) -> None:
        self.policy_path = policy_path.expanduser().resolve()
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {self.policy_path}")
        if self.policy_path.suffix.lower() != ".onnx":
            raise ValueError(f"Policy must be an .onnx file: {self.policy_path}")
        if self.provider not in ort.get_available_providers():
            raise RuntimeError(
                f"ONNX Runtime provider {self.provider!r} is unavailable; "
                f"available providers: {ort.get_available_providers()}."
            )

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                str(self.policy_path),
                sess_options=options,
                providers=[self.provider],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ONNX policy: {self.policy_path}"
            ) from exc

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "ONNX policy must have exactly one input and one output; "
                f"got {len(inputs)} inputs and {len(outputs)} outputs."
            )
        policy_input = inputs[0]
        policy_output = outputs[0]
        if policy_input.type != "tensor(float)" or tuple(policy_input.shape) != (
            1,
            OBS_DIM,
        ):
            raise ValueError(
                "ONNX input must be float32 with shape "
                f"[1, {OBS_DIM}], got type={policy_input.type!r}, "
                f"shape={policy_input.shape!r}."
            )
        if policy_output.type != "tensor(float)" or tuple(policy_output.shape) != (
            1,
            NUM_JOINTS,
        ):
            raise ValueError(
                "ONNX output must be float32 with shape "
                f"[1, {NUM_JOINTS}], got type={policy_output.type!r}, "
                f"shape={policy_output.shape!r}."
            )
        self.input_name = policy_input.name
        self.output_name = policy_output.name

        metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        metadata_joints = tuple(
            filter(None, metadata.get("joint_names", "").split(","))
        )
        if metadata_joints != JOINT_NAMES:
            raise ValueError(
                "ONNX joint_names metadata does not match the G1 29-DOF DDS "
                "motor order."
            )
        observation_names = tuple(
            filter(None, metadata.get("observation_names", "").split(","))
        )
        if observation_names != OBSERVATION_NAMES:
            raise ValueError(
                "ONNX observation_names metadata does not match the deployed "
                f"observation order: expected {OBSERVATION_NAMES}, "
                f"got {observation_names}."
            )
        command_names = tuple(
            filter(None, metadata.get("command_names", "").split(","))
        )
        if command_names != ("twist",):
            raise ValueError(
                "ONNX command_names metadata must be exactly 'twist', "
                f"got {command_names}."
            )

        self.params = DeployParameters(
            default_pos=_parse_required_metadata_array(
                metadata,
                "default_joint_pos",
            ),
            action_scale=_parse_required_metadata_array(metadata, "action_scale"),
            kp=_parse_required_metadata_array(metadata, "joint_stiffness"),
            kd=_parse_required_metadata_array(metadata, "joint_damping"),
        )
        if np.any(self.params.action_scale <= 0.0):
            raise ValueError("ONNX action_scale values must be positive.")
        if np.any(self.params.kp < 0.0) or np.any(self.params.kd < 0.0):
            raise ValueError("ONNX joint stiffness and damping cannot be negative.")
        self.run_path = metadata.get("run_path", "")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (OBS_DIM,) or not np.all(np.isfinite(observation)):
            raise FloatingPointError("Policy received an invalid observation.")
        inputs = np.ascontiguousarray(observation.reshape(1, OBS_DIM))
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: inputs},
            )
        except Exception as exc:
            raise RuntimeError("ONNX policy inference failed.") from exc
        action = np.asarray(outputs[0], dtype=np.float32)
        if action.shape != (1, NUM_JOINTS) or not np.all(np.isfinite(action)):
            raise FloatingPointError("ONNX policy produced an invalid action.")
        return action[0].copy()


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world gravity [0, 0, -1] into the pelvis frame."""

    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        raise FloatingPointError("Received a zero-norm IMU quaternion.")
    w, x, y, z = quat / norm
    return np.array(
        [
            -2.0 * (x * z - y * w),
            -2.0 * (y * z + x * w),
            -(1.0 - 2.0 * (x * x + y * y)),
        ],
        dtype=np.float32,
    )


def build_observation(
    state: PolicyState,
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
