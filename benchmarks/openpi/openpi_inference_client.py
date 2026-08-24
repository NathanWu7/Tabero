# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import hashlib
import json
import os
import re
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
import tyro
from isaaclab.app import AppLauncher
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaacsim import SimulationApp
from openpi_client import websocket_client_policy as _websocket_client_policy

# Utilize the common utility functions from gr00t for OpenPI inference
from benchmarks.common.closedloop_policy_inference import (
    ClosedLoopArguments,
    ClosedLoopPolicyInference,
)
from benchmarks.common.episode_metrics import (
    TrajectoryForceTracker,
    aggregate_success_force_metrics,
    extract_damage_threshold_snapshot,
    extract_friction_snapshot,
    extract_object_damage_details,
    resolve_episode_termination,
    summarize_episode_force_metrics,
    summarize_force_tracking_metrics,
    summarize_trajectory_force_metrics,
)
from benchmarks.common.episode_lift import EpisodeLiftTracker
from benchmarks.common.episode_video import EpisodeVideoWriter
from benchmarks.common.metrics import (
    compute_contact_force_series_from_lr_forces,
    compute_topk_mean,
)
from benchmarks.openpi.openpi_payload import infer_openpi_step

TARGET_IMAGE_HW = (224, 224)

_SAFE_DIR_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_dirname(name: str) -> str:
    """Sanitize a string to be safe as a single path component."""
    s = (name or "").strip()
    if not s:
        return "none"
    s = s.replace(" ", "_")
    s = _SAFE_DIR_RE.sub("_", s)
    s = s.strip("._-")
    return s or "none"


def _to_uint8_rgb(img) -> np.ndarray:
    """Convert an image tensor/ndarray to uint8 RGB numpy array."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _vector3_or_none(value) -> list[float] | None:
    """Return one JSON-safe 3D vector, or None when unavailable."""
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        vector = np.asarray(value, dtype=np.float32).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(vector)):
        return None
    return [float(component) for component in vector]


def _finite_float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _bool_or_none(value) -> bool | None:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        scalar = np.asarray(value).reshape(()).item()
    except (TypeError, ValueError):
        return None
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    return None


def _integer_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        scalar = np.asarray(value).reshape(()).item()
    except (TypeError, ValueError):
        return None
    if isinstance(scalar, (int, np.integer)) and not isinstance(
        scalar, (bool, np.bool_)
    ):
        return int(scalar)
    return None


def _contact_or_none(left: list[float] | None, right: list[float] | None) -> bool | None:
    if left is None or right is None:
        return None
    return bool(np.linalg.norm(left) + np.linalg.norm(right) > 1e-6)


def _pad_history_front(items: list[np.ndarray], target_len: int) -> list[np.ndarray]:
    """Pad a history list by repeating the earliest item (front-padding)."""
    if target_len <= 0:
        return []
    if len(items) == 0:
        raise ValueError('Cannot pad empty history.')
    if len(items) >= target_len:
        return items[-target_len:]
    pad_n = target_len - len(items)
    return [items[0]] * pad_n + items


def _build_tactile_mosaic(
    left_hist: list[np.ndarray],
    right_hist: list[np.ndarray],
    *,
    out_hw: tuple[int, int] = TARGET_IMAGE_HW,
) -> np.ndarray:
    """Build the 4x4 tactile mosaic (left 4x2 + right 4x2), matching convert_all_libero_to_tabero.py."""
    H_out, W_out = out_hw
    cell_h, cell_w = H_out // 4, W_out // 4
    canvas = np.zeros((H_out, W_out, 3), dtype=np.uint8)

    # Layout:
    # - Left finger: 4x2 grid in columns 0..1
    # - Right finger: 4x2 grid in columns 2..3
    for k in range(8):
        r = k // 2  # 0..3
        c = k % 2  # 0..1
        y0, y1 = r * cell_h, (r + 1) * cell_h

        # left
        x0, x1 = c * cell_w, (c + 1) * cell_w
        canvas[y0:y1, x0:x1] = cv2.resize(left_hist[k], (cell_w, cell_h))

        # right
        x0, x1 = (c + 2) * cell_w, (c + 3) * cell_w
        canvas[y0:y1, x0:x1] = cv2.resize(right_hist[k], (cell_w, cell_h))

    return canvas


class _OnlineTactileBuffer:
    """Maintain online tactile/force/marker histories to match Tabero dataset fields."""

    def __init__(
        self,
        *,
        tactile_sensors: tuple[str, str],
        tactile_output_type: str,
        tactile_history_len: int = 8,
        force_history_len: int = 8,
        marker_history_len: int = 8,
    ) -> None:
        if tactile_history_len != 8:
            raise ValueError('tactile_history_len must be 8 to match the 4x4 mosaic layout.')
        self.tactile_sensors = tactile_sensors
        self.tactile_output_type = tactile_output_type
        self.force_history_len = force_history_len
        self.marker_history_len = marker_history_len
        self.reset()

    def reset(self) -> None:
        self._left_frames: deque[np.ndarray] = deque(maxlen=8)
        self._right_frames: deque[np.ndarray] = deque(maxlen=8)
        self._force_hist: deque[np.ndarray] = deque(maxlen=self.force_history_len)
        self._marker_hist: deque[np.ndarray] = deque(maxlen=self.marker_history_len)
        self._marker_init: np.ndarray | None = None

    def update_tactile_frames(self, env, env_id: int = 0) -> None:
        left_name, right_name = self.tactile_sensors
        left_img = env.unwrapped.scene.sensors[left_name].data.output[self.tactile_output_type][env_id]
        right_img = env.unwrapped.scene.sensors[right_name].data.output[self.tactile_output_type][env_id]
        self._left_frames.append(_to_uint8_rgb(left_img))
        self._right_frames.append(_to_uint8_rgb(right_img))

    def update_force(self, obs: dict) -> None:
        policy_obs = obs.get('policy', {}) if isinstance(obs, dict) else {}
        if not isinstance(policy_obs, dict) or 'gripper_net_force' not in policy_obs:
            return
        gnf = policy_obs['gripper_net_force']
        if isinstance(gnf, torch.Tensor):
            gnf = gnf.detach().cpu().numpy()
        gnf = np.asarray(gnf)
        gnf0 = np.squeeze(gnf, axis=0)
        if gnf0.ndim == 2:
            # (2,3)
            inst = gnf0.reshape(6).astype(np.float32)
        else:
            # (H,2,3): take current step at index 0
            inst = gnf0[0].reshape(6).astype(np.float32)
        self._force_hist.append(inst)

    def update_marker_motion(self, obs: dict) -> None:
        policy_obs = obs.get('policy', {}) if isinstance(obs, dict) else {}
        if not isinstance(policy_obs, dict) or 'gripper_marker_motion' not in policy_obs:
            return
        gmm = policy_obs['gripper_marker_motion']
        if isinstance(gmm, torch.Tensor):
            gmm = gmm.detach().cpu().numpy()
        gmm = np.asarray(gmm)
        gmm0 = np.squeeze(gmm, axis=0)
        if gmm0.ndim != 4:
            return
        # (2,2,M,2): sensor, (init/current), marker, xy
        init_pos = gmm0[:, 0, :, :].reshape(-1, 2).astype(np.float32)  # (2*M,2)
        curr_pos = gmm0[:, 1, :, :].reshape(-1, 2).astype(np.float32)  # (2*M,2)
        if self._marker_init is None:
            self._marker_init = init_pos
        self._marker_hist.append(curr_pos)

    def append_control_sample(
        self,
        obs: dict,
        *,
        env=None,
        include_tactile: bool = False,
        env_id: int = 0,
    ) -> None:
        """Append one synchronized sample from an observation-producing control step."""
        self.update_force(obs)
        if not include_tactile:
            return
        if env is None:
            raise ValueError("env is required when include_tactile=True")
        try:
            self.update_tactile_frames(env, env_id=env_id)
        except Exception:
            # Preserve the existing behavior: unavailable tactile RGB must not stop evaluation.
            pass
        self.update_marker_motion(obs)

    def get_tactile_image(self) -> np.ndarray | None:
        if len(self._left_frames) == 0 or len(self._right_frames) == 0:
            return None
        left_hist = _pad_history_front(list(self._left_frames), 8)
        right_hist = _pad_history_front(list(self._right_frames), 8)
        return _build_tactile_mosaic(left_hist, right_hist, out_hw=TARGET_IMAGE_HW)

    def get_force_history(self) -> np.ndarray | None:
        if len(self._force_hist) == 0:
            return None
        hist = _pad_history_front([x.astype(np.float32) for x in self._force_hist], self.force_history_len)
        return np.stack(hist, axis=0).astype(np.float32)  # (H,6)

    def get_marker_motion(self) -> np.ndarray | None:
        if self._marker_init is None or len(self._marker_hist) == 0:
            return None
        hist = _pad_history_front([x.astype(np.float32) for x in self._marker_hist], self.marker_history_len)
        out = np.zeros((1 + self.marker_history_len, self._marker_init.shape[0], 2), dtype=np.float32)
        out[0] = self._marker_init
        out[1:] = np.stack(hist, axis=0)
        return out


@dataclass
class OpenpiClientArguments(ClosedLoopArguments):

    record_images: bool = False
    record_videos: bool = False
    num_envs: int = 1
    background_env_usd_path: str | None = None
    record_camera_output_path: str | None = None

    # Server connection parameters
    server_host: str = "127.0.1.1"
    server_port: int = 8000
    target_image_size: tuple[int, int, int] = (224, 224, 3)
    # Send both 256x256 DSRL raw views (agentview and eye-in-hand).
    send_dsrl_raw_image: bool = False

    # Simulator specific parameters
    # Default to headless to avoid X11/GLX BadMatch crashes on servers or misconfigured displays.
    # If you want a GUI window, pass: --no-headless
    headless: bool = True
    # IsaacLab/Kit device. Unlike CUDA_VISIBLE_DEVICES, AppLauncher's active_gpu/physics_gpu
    # can be interpreted as a physical GPU id by the renderer, so keep this explicit.
    sim_device: str = os.environ.get("ISAACLAB_DEVICE", "cuda:0")
    # Keep single-GPU app launch by default. Multi-GPU mode can probe or initialize extra GPUs.
    sim_multi_gpu: bool = False
    # Extra Kit args, for example: --/renderer/activeGpu=8
    sim_kit_args: str = os.environ.get("ISAACLAB_KIT_ARGS", "")
    seed: int = 11
    randomize_light: bool = False
    # debug_mode:
    #   0: 关闭所有额外调试，仅打印基础统计信息
    #   1: 在 0 的基础上额外保存动作 (action_XXXX.npy)
    #   2: 在 1 的基础上额外保存相机帧到 debug_path
    #   3: 在 2 的基础上额外 dump 关节状态 / 图像序列
    #   4: 在 0 的基础上开启 Hybrid 力–位混合可视化（不依赖 1-3 的其它 dump）
    #   5: 不实时画图；逐帧记录挤压力（预测/实测）到 benchmarks/tabero/gripper_force/<task_id>/
    #   6: 逐帧保存：
    #        - 双相机 RGB（第三人称 agentview + 腕部 eye_in_hand）
    #        - 左右触觉 markers_rgb（gsmini_left/right）
    #        - 夹持/外力的预测量与实测量（含 3D 向量与 squeeze/ap 派生指标）
    #      输出目录：<debug_path>/capture_mode6/<suite>/task_<id>/<adverb_tag>/<timestamp>/exp_XXX/...
    debug_mode: int = 0
    # Default to a repo-local folder for full debug records (images + tactile + forces).
    # You can override via CLI: --debug_path /abs/path/to/dir
    debug_path: str = str(project_root / "full_records")
    # Optional force-only JSONL. This is independent from debug_mode=6 image capture.
    step_trace_path: Optional[Path] = None

    camera_names: tuple[str, ...] = ("agentview_cam", "eye_in_hand_cam")
    tactile_sensor_names: tuple[str, str] = ("gsmini_left", "gsmini_right")
    tactile_output_type: str = "tactile_rgb"  # or "markers_rgb"
    tactile_history_len: int = 8
    force_history_len: int = 8
    marker_history_len: int = 8
    num_steps_wait: int = 5  # Number of steps to wait for objects to stabilize i n sim
    replan_steps: int = 10  # For each action, will execute replan_steps times
    max_inference_steps: int = 30  # max number of inference steps to run
    num_success_steps: int = 8  # continuous success steps to consider the policy as successful
    num_total_experiments: int = 50  # total number of experiments to do policy evaluation
    # A lift is a post-contact height increase held for consecutive env steps.
    lift_height_threshold_m: float = 0.03
    lift_hold_steps: int = 5
    # Match RLinf success.force_bonus trajectory-force eligibility.
    reward_force_contact_epsilon_n: float = 1.0
    reward_force_min_valid_samples: int = 4

    # Control mode parameters
    # Supported modes:
    #   - "diffik": Task-space control via Differential IK
    #   - "osc":    Task-space control via OSC
    #   - "hybrid":  Hybrid force–position control (ContactForce)
    #   - "tactile": Hybrid force–position + tactile observations (GelSight)
    #   - "binary": IK + tactile observations (GelSight), but execute 7D actions with **binary gripper**
    #
    # OpenPI server always returns a 32D action vector (padded), but:
    #   - For "diffik": we use the first 7D
    #       (x, y, z, rx, ry, rz, gripper) - axis-angle + gripper
    #       and convert it to 8D quaternion before sending to the env:
    #       (x, y, z, qw, qx, qy, qz, gripper)
    #   - For "osc": we use the first 7D directly:
    #       (x, y, z, rx, ry, rz, gripper)
    #   - For "hybrid"/"tactile": we use the first 13D **directly** as the Hybrid action:
    #       (x, y, z, rx, ry, rz, gripper, fL(3), fR(3))  -- no zero padding on the client side
    control_mode: str = "diffik"
    task: str = ""  # Will be auto-set based on control_mode if not provided

    # Ablation (short flag): tactile obs/model branch, but execute absolute 7D task-space actions.
    # - Env still expects 13D in tactile mode: pad force dims with zeros
    # - Disable pos_kp/squeeze_kp corrections at runtime
    abs7d: bool = False

    # Task setup parameters
    task_suite: str = "libero_goal"
    task_id: int = 1
    task_config_path: Path = Path(__file__).parent.parent.resolve() / "datasets" / "libero" / "config"
    language_instruction: str = ""

    # Optional: prompt adverb augmentation (Tabero-style).
    # Keep CLI flags compatible with scripts/tools/run_task_evaluations.py:
    #   --prompt-adverb, --prompt-adverbs, --prompt-seed
    prompt_adverb: str = ""
    prompt_adverbs: tuple[str, ...] = ()
    prompt_seed: int = 0

    # HDF5 dataset parameters for initial state loading（目录内需含 {task_suite}_task{id}_*_demo.hdf5）
    # 未指定时唯一来源：环境变量 HDF5_TRAJ_SOURCE_DIR（与 set_replay_env.sh / task_configs 一致）
    hdf5_folder: Optional[Path] = None


# Parse arguments first to get task_suite and task_id
args = tyro.cli(OpenpiClientArguments)
if not np.isfinite(args.lift_height_threshold_m) or args.lift_height_threshold_m <= 0.0:
    raise ValueError("lift_height_threshold_m must be finite and positive")
if isinstance(args.lift_hold_steps, bool) or args.lift_hold_steps <= 0:
    raise ValueError("lift_hold_steps must be a positive integer")
os.environ["LIBERO_RANDOMIZE_LIGHT"] = "1" if args.randomize_light else "0"


def _choose_adverb(seed: int, key: str, adverbs: tuple[str, ...]) -> str:
    """Deterministically choose one adverb from a list (match convert_all_libero_to_tabero.py)."""
    if not adverbs:
        return ""
    digest = hashlib.blake2b(f"{int(seed)}:{key}".encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest, "big") % len(adverbs)
    return (adverbs[idx] or "").strip()


def _rewrite_instruction(instruction: str, adverb: str, seed: int, key: str) -> str:
    """Rewrite instruction with an adverb in a more natural English style (deterministic).

    Strategy (deterministic per key):
    - randomly choose between:
      * prefix:  "{adverb} {instruction}"   (e.g. "gently open the drawer")
      * suffix:  "{instruction} {adverb}"   (e.g. "open the drawer gently")
    """
    instruction = (instruction or "").strip()
    adverb = (adverb or "").strip()
    if not adverb:
        return instruction
    if not instruction:
        return adverb

    lower = instruction.lower()
    if lower.startswith(f"{adverb} "):
        return instruction
    if lower.endswith(f" {adverb}"):
        return instruction

    style = _choose_adverb(int(seed), f"{key}:style", ("prefix", "suffix"))
    if style == "suffix":
        return f"{instruction} {adverb}"
    return f"{adverb} {instruction}"


# Set USE_RELATIVE_MODE environment variable for DiffIK controller
# For OpenPI inference with absolute pose control, we always use absolute mode (False)
if "USE_RELATIVE_MODE" not in os.environ:
    os.environ["USE_RELATIVE_MODE"] = "False"
    print("Set USE_RELATIVE_MODE=False for absolute pose control (OpenPI default)")

# Map control mode to corresponding environment if task not explicitly set
if not args.task:
    control_mode_to_env = {
        "diffik": "Isaac-Libero-Franka-IK-v0",  # Differential IK control
        "osc": "Isaac-Libero-Franka-OscPose-v0",  # OSC control
        # 兼容模式：
        # - hybrid  -> 纯 Hybrid-ContactForce 环境（无 GelSight），保持与旧版一致
        # - tactile -> Hybrid-Tactile 环境（推荐，用于触觉+力评估）
        "hybrid": "Isaac-Libero-Franka-Hybrid-ContactForce-v0",
        "tactile": "Isaac-Libero-Franka-Hybrid-Tactile-v0",
        # binary -> IK + tactile env (non-hybrid). Action execution: 8D pose + **binary** gripper.
        "binary": "Isaac-Libero-Franka-IK-Camera-Tactile-v0",
    }
    if args.control_mode not in control_mode_to_env:
        raise ValueError(f"Invalid control mode: {args.control_mode}. Supported modes: {list(control_mode_to_env.keys())}")
    args.task = control_mode_to_env[args.control_mode]
    print(f"Using task environment: {args.task} for control mode: {args.control_mode}")
else:
    print(f"Using explicitly specified task environment: {args.task}")

# HDF5 目录：仅认 HDF5_TRAJ_SOURCE_DIR；可选 CLI --hdf5_folder 覆盖并写回该环境变量。
if args.hdf5_folder is None:
    traj = (os.environ.get("HDF5_TRAJ_SOURCE_DIR") or "").strip()
    if not traj:
        raise ValueError(
            "Missing HDF5 folder for OpenPI inference.\n"
            "  export HDF5_TRAJ_SOURCE_DIR=/path/to/assembled_hdf5\n"
            "  # 或: source scripts/tools/set_replay_env.sh inference\n"
            "Or pass: --hdf5-folder /path/to/assembled_hdf5"
        )
    args.hdf5_folder = Path(traj)
    print(f"Using HDF5 folder from HDF5_TRAJ_SOURCE_DIR: {args.hdf5_folder}")
else:
    os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(args.hdf5_folder)
    print(f"Using HDF5 folder from command line (--hdf5.folder): {args.hdf5_folder}")

# Launch the simulator FIRST before importing tac_manip modules
print(f"Using IsaacLab simulation device: {args.sim_device}")
app_launcher = AppLauncher(
    headless=args.headless,
    enable_cameras=True,
    num_envs=1,
    device=args.sim_device,
    multi_gpu=args.sim_multi_gpu,
    kit_args=args.sim_kit_args,
)
simulation_app = app_launcher.app

# add configs for dataset generation for various task_suite and task_id,
# supported task_suites: [xhumanoid, libero, etc.]
# NOTE: Import tac_manip modules AFTER AppLauncher is initialized
if args.task_suite is not None:
    from tac_manip.utils.task_configs import setup_task_objects

    setup_task_objects(args.task_suite, args.task_id)

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import import_packages

from benchmarks.openpi.env import (
    axisangle2quat,
    quat2axisangle,
)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp", "pick_place"]
# Import all configs in this package
import_packages("isaaclab_tasks", _BLACKLIST_PKGS)


def get_episode_map(names):
    """Get a mapping of episode indices to their names.

    Args:
        names: List or dict of episode names

    Returns:
        dict: Mapping of episode indices to their names (e.g., {0: 'episode_0', 2: 'episode_2', 5: 'episode_5'})
    """
    import re

    def extract_episode_index(name):
        """Extract the episode index from the name."""
        match = re.search(r"(\d+)", name)
        if match:
            return int(match.group(1))
        return 0

    # Create a mapping of episode index to episode name
    episode_map = {}
    for name in names:
        idx = extract_episode_index(name)
        episode_map[idx] = name

    return episode_map


def find_hdf5_file(hdf5_folder: Path, task_suite: str, task_id: int) -> Path | None:
    """Find the HDF5 file for the given task_suite and task_id.

    Args:
        hdf5_folder: Path to the folder containing HDF5 files
        task_suite: Task suite name (e.g., "libero_10", "xhumanoid")
        task_id: Task ID number

    Returns:
        Path to the HDF5 file if found, None otherwise
    """
    if not hdf5_folder.exists():
        print(f"HDF5 folder does not exist: {hdf5_folder}")
        return None

    # Create pattern to match the HDF5 file
    pattern = f"{task_suite}_task{task_id}_*_demo.hdf5"

    # Find matching files
    matching_files = list(hdf5_folder.glob(pattern))

    if matching_files:
        hdf5_file = matching_files[0]
        print(f"Found HDF5 file: {hdf5_file}")
        return hdf5_file
    else:
        print(f"No HDF5 file found matching pattern: {pattern}")
        print(f"Searched in: {hdf5_folder}")
        # List available files for debugging
        available_files = list(hdf5_folder.glob("*.hdf5"))
        if available_files:
            print("Available HDF5 files:")
            for file in available_files:
                print(f"  - {file.name}")
        return None


def run_closed_loop_policy(  # noqa: C901
    args: OpenpiClientArguments,
    simulation_app: SimulationApp,
    env: gym.Env,
    env_cfg: ManagerBasedRLEnvCfg,
    success_term: Callable[[gym.Env], bool] | None,
):
    """Run the closed loop policy evaluation."""
    if (
        not np.isfinite(args.reward_force_contact_epsilon_n)
        or args.reward_force_contact_epsilon_n < 0.0
    ):
        raise ValueError(
            "--reward-force-contact-epsilon-n must be finite and non-negative"
        )
    if (
        isinstance(args.reward_force_min_valid_samples, bool)
        or args.reward_force_min_valid_samples < 1
    ):
        raise ValueError(
            "--reward-force-min-valid-samples must be a positive integer"
        )
    tactile_buf = _OnlineTactileBuffer(
        tactile_sensors=args.tactile_sensor_names,
        tactile_output_type=args.tactile_output_type,
        tactile_history_len=args.tactile_history_len,
        force_history_len=args.force_history_len,
        marker_history_len=args.marker_history_len,
    )

    # debug_mode=1/2/3 才使用 debug_path 做本地 dump
    if args.debug_mode in (1, 2, 3):
        os.makedirs(args.debug_path, exist_ok=True)

    # debug_mode=5: 逐帧挤压力记录目录
    force_dump_dir: Path | None = None
    if args.debug_mode == 5:
        # 统一副词：推荐用 --prompt-adverb firmly/gently；若使用 --prompt-adverbs，则标记为 mixed
        adverb_tag = "mixed" if args.prompt_adverbs else _sanitize_dirname(args.prompt_adverb)
        force_dump_dir = (
            project_root
            / "benchmarks"
            / "tabero"
            / "gripper_force"
            / _sanitize_dirname(str(args.task_suite))
            / f"task_{int(args.task_id)}"
            / adverb_tag
        )
        force_dump_dir.mkdir(parents=True, exist_ok=True)

    # debug_mode=6: 保存相机+触觉 markers_rgb + 预测/实测夹持力（逐帧）
    capture_mode6_root: Path | None = None
    if args.debug_mode == 6:
        # 统一副词标签，便于不同 prompt 版本的对照
        adverb_tag = "mixed" if args.prompt_adverbs else _sanitize_dirname(args.prompt_adverb)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_mode6_root = (
            Path(args.debug_path)
            / "capture_mode6"
            / _sanitize_dirname(str(args.task_suite))
            / f"task_{int(args.task_id)}"
            / adverb_tag
            / ts
        )
        capture_mode6_root.mkdir(parents=True, exist_ok=True)
        # 写一份 run 级别 meta，方便回溯配置
        try:
            meta = {
                "task_suite": args.task_suite,
                "task_id": int(args.task_id),
                "task": args.task,
                "control_mode": args.control_mode,
                "camera_names": list(args.camera_names),
                "tactile_sensor_names": list(args.tactile_sensor_names),
                "tactile_output_type": args.tactile_output_type,
                "debug_mode": int(args.debug_mode),
                "debug_path": str(args.debug_path),
                "prompt_adverb": (args.prompt_adverb or "").strip(),
                "prompt_adverbs": list(args.prompt_adverbs) if args.prompt_adverbs else [],
                "prompt_seed": int(args.prompt_seed),
                "mode6_scalar_schema_version": 3,
                "force_target_semantics": {
                    "predicted": "raw_model_squeeze_target",
                    "effective": "controller_target_after_feed_forward_or_override",
                    "measured_ema": "controller_filtered_measured_squeeze",
                    "measured_raw": "unfiltered_gripper_local_squeeze",
                },
                "gripper_command_semantics": "controller_d_cmd",
                "gripper_measured_semantics": "mean_raw_finger_joint_position",
                "timestamp": ts,
            }
            with open(capture_mode6_root / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DebugMode 6] Failed to write run_meta.json: {e}")

    video_output_dir: Path | None = None
    video_fps = 20.0
    if args.record_videos:
        video_output_dir = Path(
            args.record_camera_output_path
            or (project_root / "evaluation_results" / "episode_videos")
        )
        video_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            step_dt = float(env.unwrapped.step_dt)
            if np.isfinite(step_dt) and step_dt > 0.0:
                video_fps = 1.0 / step_dt
        except Exception:
            pass
        print(
            f"[Video] Recording every env step to {video_output_dir} "
            f"at {video_fps:.3f} FPS."
        )

    # Hybrid 力–位混合在线可视化（仅在 debug_mode == 4 时启用）
    force_viz = None
    if args.debug_mode == 4 and "Isaac-Libero-Franka-Hybrid-" in args.task and args.num_envs == 1:
        try:
            # 复用 scripts/tools 中的调试可视化工具
            # 注意：此处从工程根目录下的 scripts.tools.common 导入，而不是相对路径的 common。
            from scripts.tools.common.force_position_debug_viz import (
                ForcePositionDebugVisualizer,
            )

            force_viz = ForcePositionDebugVisualizer()
            print("[DebugMode 4] Enabled Hybrid force-position debug visualizer.")
        except Exception as e:
            print(f"[DebugMode 4] Failed to initialize ForcePositionDebugVisualizer: {e}")
            force_viz = None
    elif args.debug_mode == 4:
        print(
            "[DebugMode 4] Force-position visualization is only available for Hybrid environments "
            "with num_envs == 1. Skipping visualizer initialization."
        )

    step_trace_fh = None
    step_trace_open_error: str | None = None
    if args.step_trace_path is not None:
        try:
            trace_path = Path(args.step_trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            step_trace_fh = open(trace_path, "x", encoding="utf-8", buffering=1)
        except Exception as exc:
            step_trace_open_error = str(exc)
            print(f"[Step-Trace] Failed to create {args.step_trace_path}: {exc}")

    successful_experiments = 0
    episode_metrics_records: list[dict] = []

    # Find HDF5 file based on task_suite and task_id
    hdf5_file = find_hdf5_file(args.hdf5_folder, args.task_suite, args.task_id)

    # Load dataset and episode information if HDF5 file is found
    episode_indices_to_use = []
    episode_map = {}
    dataset_file_handler = None

    if hdf5_file and hdf5_file.exists():
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(str(hdf5_file))
        episode_count = dataset_file_handler.get_num_episodes()
        episode_map = get_episode_map(dataset_file_handler.get_episode_names())
        # Use actual episode indices from episode_map instead of assuming they're consecutive
        episode_indices_to_use = sorted(episode_map.keys())
        print(f"Loaded {episode_count} initial_states of episodes from dataset: {hdf5_file}")
        print(f"Available episode indices: {episode_indices_to_use}")
    else:
        print(
            f"No valid HDF5 file found for {args.task_suite}_task{args.task_id}, will use default reset for all"
            " experiments"
        )

    # Read language instruction from task_suite_config as a fallback.
    # If the user provided --language-instruction, do NOT override it.
    task_config_path = args.task_config_path / f"{args.task_suite}.json"
    if not task_config_path.exists():
        raise FileNotFoundError(f"Task config file not found: {task_config_path}")
    with open(task_config_path) as f:
        task_suite_config = json.load(f)

    selected_task_config: dict | None = None
    cli_instruction = (args.language_instruction or "").strip()
    if cli_instruction:
        print(f"\nUsing language instruction (from CLI): {cli_instruction}")
        args.language_instruction = cli_instruction
    else:
        for task in task_suite_config["tasks"]:
            task_id = task["task_id"]
            if task_id == args.task_id:
                selected_task_config = task
                args.language_instruction = task["language_instruction"]
                print(f"\nUsing language instruction (from task config): {args.language_instruction}")
                break

    if selected_task_config is None:
        selected_task_config = next(
            (
                task
                for task in task_suite_config["tasks"]
                if task["task_id"] == args.task_id
            ),
            None,
        )
    lift_target_object: str | None = None
    if selected_task_config is not None:
        target_override = (
            selected_task_config.get("control_overrides", {})
            .get("target_contact_squeeze", {})
        )
        if target_override.get("enabled") is True:
            target_object = target_override.get("target_object")
            if not isinstance(target_object, str) or not target_object:
                raise ValueError(
                    "enabled target_contact_squeeze requires a non-empty "
                    "target_object for lift tracking"
                )
            lift_target_object = target_object

    client = _websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        for exp_idx in range(args.num_total_experiments):
            print(f"\n[{exp_idx + 1}/{args.num_total_experiments}] Starting experiment...", end=" ", flush=True)
            success_step_count = 0
            experiment_success = False
            total_steps_taken = 0
            inference_chunks_taken = 0
            end_reason = "max_inference_steps"
            terminal_term: str | None = None
            damage_details: dict | None = None
            damage_threshold_snapshot: dict | None = None
            friction_snapshot: dict | None = None
            dataset_episode_index: int | None = None

            # 当前 experiment 的逐帧挤压力 / 加持力记录（用于 Top5% 统计）
            exp_fsq_pred_values: list[float] = []
            exp_fsq_meas_values: list[float] = []
            exp_ap_pred_values: list[float] = []
            exp_ap_meas_values: list[float] = []
            # 缓存逐步左右指实测 3D 力，用于接触步数与覆盖率统计。
            exp_fL_meas_values: list[np.ndarray] = []
            exp_fR_meas_values: list[np.ndarray] = []
            exp_tracking_reward_valid: list[bool] = []
            exp_tracking_squeeze_pred: list[float | None] = []
            exp_tracking_squeeze_target_eff: list[float | None] = []
            exp_tracking_squeeze_meas_raw: list[float | None] = []
            exp_tracking_gripper_pred: list[float | None] = []
            exp_tracking_gripper_cmd: list[float | None] = []

            # RLinf success.force_bonus-equivalent trajectory force tracking.
            trajectory_force_tracker: TrajectoryForceTracker | None = None
            trajectory_force_error: str | None = None

            # JSON-configured target-contact squeeze override audit.
            exp_override_configured = False
            exp_override_activated = False
            exp_override_activation_step: int | None = None
            exp_override_active_steps = 0
            exp_override_contact_steps = 0
            exp_override_single_finger_target_n: float | None = None
            exp_override_squeeze_target_n: float | None = None
            exp_override_left_normal_abs: list[float] = []
            exp_override_right_normal_abs: list[float] = []
            lift_tracker: EpisodeLiftTracker | None = None
            lift_object = None

            # 当前 experiment 的 Hybrid 13D 动作缓存。
            exp_actions_13d: list[torch.Tensor] = []
            exp_step_trace_rows: list[dict] = []

            episode_video = (
                EpisodeVideoWriter(video_output_dir, exp_idx, video_fps)
                if video_output_dir is not None
                else None
            )

            # debug_mode=6: per-experiment capture directories + force log (JSONL)
            mode6_exp_dir: Path | None = None
            mode6_cam_dir: Path | None = None
            mode6_tac_dir: Path | None = None
            mode6_force_fh = None
            if capture_mode6_root is not None:
                try:
                    mode6_exp_dir = capture_mode6_root / f"exp_{exp_idx:03d}"
                    mode6_cam_dir = mode6_exp_dir / "camera_rgb"
                    mode6_tac_dir = mode6_exp_dir / "tactile_markers_rgb"
                    mode6_cam_dir.mkdir(parents=True, exist_ok=True)
                    mode6_tac_dir.mkdir(parents=True, exist_ok=True)
                    mode6_force_fh = open(mode6_exp_dir / "forces.jsonl", "w", encoding="utf-8")
                except Exception as e:
                    print(f"[DebugMode 6] Failed to init exp dir/log for exp_{exp_idx:03d}: {e}")
                    mode6_exp_dir = mode6_cam_dir = mode6_tac_dir = None
                    mode6_force_fh = None

            # 每个 experiment 开始时重置力–位可视化
            if force_viz is not None:
                try:
                    force_viz.reset()
                except Exception:
                    pass

            # reset environment with initial state from HDF5 if available
            if episode_indices_to_use:
                # Use episode index from the list (cycling through all episodes)
                episode_index = episode_indices_to_use[exp_idx % len(episode_indices_to_use)]
                dataset_episode_index = int(episode_index)
                episode_data = dataset_file_handler.load_episode(episode_map[episode_index], env.unwrapped.device)

                if "initial_state" in episode_data.data:
                    # reset environment
                    obs, info = env.reset()
                    # Set initial state for the environment
                    initial_state = episode_data.get_initial_state()
                    # print("---- initial_state: ", initial_state)
                    obs, info = env.reset_to(
                        initial_state, torch.arange(args.num_envs, device=env.unwrapped.device), is_relative=True
                    )

                else:
                    # Fallback to default reset if no initial state available
                    obs, info = env.reset()
            else:
                # Fallback to default reset if no dataset file specified or doesn't exist
                obs, info = env.reset()

            if lift_target_object is not None:
                if lift_target_object not in env.unwrapped.scene.keys():
                    raise RuntimeError(
                        f"Lift target {lift_target_object!r} is missing from the scene"
                    )
                lift_object = env.unwrapped.scene[lift_target_object]
                initial_height_m = float(
                    lift_object.data.root_pos_w[0, 2].detach().cpu().item()
                )
                lift_tracker = EpisodeLiftTracker(
                    target_object=lift_target_object,
                    initial_height_m=initial_height_m,
                    height_threshold_m=float(args.lift_height_threshold_m),
                    hold_steps_required=int(args.lift_hold_steps),
                )

            friction_snapshot = extract_friction_snapshot(info=info, env_index=0)
            damage_threshold_snapshot = extract_damage_threshold_snapshot(
                info=info, env_index=0
            )

            # Reset online histories per experiment to match dataset windowing.
            tactile_buf.reset()
            tactile_buf.append_control_sample(
                obs,
                env=env,
                include_tactile=args.control_mode in ("tactile", "binary"),
                env_id=0,
            )

            frame_count = 0
            terminated = torch.tensor([False])  # Initialize to handle case where inner loop doesn't execute
            truncated = torch.tensor([False])

            # Build prompt once per experiment (Tabero-style adverb augmentation).
            base_instruction = (args.language_instruction or "").strip()
            exp_adv = ""
            if args.prompt_adverbs:
                # Deterministic per experiment; include task identifiers for stability.
                key = f"{args.task_suite}:{args.task_id}:{exp_idx}"
                exp_adv = _choose_adverb(int(args.prompt_seed), key, tuple(args.prompt_adverbs))
            else:
                exp_adv = (args.prompt_adverb or "").strip()
            exp_prompt = _rewrite_instruction(
                base_instruction, exp_adv, seed=int(args.prompt_seed), key=f"{args.task_suite}:{args.task_id}:{exp_idx}"
            )

            # debug_mode=6: write per-experiment meta once prompt is decided
            if mode6_exp_dir is not None:
                try:
                    meta = {
                        "task_suite": args.task_suite,
                        "task_id": int(args.task_id),
                        "exp_idx": int(exp_idx),
                        "prompt_adverb": (args.prompt_adverb or "").strip(),
                        "prompt_adverbs": list(args.prompt_adverbs) if args.prompt_adverbs else [],
                        "adverb_used": exp_adv,
                        "prompt": exp_prompt,
                        "camera_names": list(args.camera_names),
                        "tactile_sensor_names": list(args.tactile_sensor_names),
                        "tactile_output_type_saved": "markers_rgb",
                        "mode6_scalar_schema_version": 3,
                        "force_target_semantics": {
                            "predicted": "raw_model_squeeze_target",
                            "effective": "controller_target_after_feed_forward_or_override",
                            "measured_ema": "controller_filtered_measured_squeeze",
                            "measured_raw": "unfiltered_gripper_local_squeeze",
                        },
                        "gripper_command_semantics": "controller_d_cmd",
                        "gripper_measured_semantics": "mean_raw_finger_joint_position",
                    }
                    with open(mode6_exp_dir / "exp_meta.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            for action_idx in range(args.max_inference_steps):
                # Get camera images from live cameras
                camera_frames = []
                for cam_name in args.camera_names:
                    cam = env.unwrapped.scene[cam_name]
                    rgb = cam.data.output["rgb"]
                    camera_frames.append((cam_name, rgb))

                # Run model inference to get predicted actions (for comparison or execution)
                inference_actions = None

                # pi0-style **task-space** observation for OpenPI:
                #   - base_state: [x, y, z, ax, ay, az, gripper_abs] -> 7D
                #   - hybrid   : base_state plus separate H×6 finger forces (sent via 'observation/gripper_force')
                #
                # Get current EEF pose from policy observations: (x, y, z, qw, qx, qy, qz)
                eef_pose = obs["policy"]["eef_pose"].cpu().numpy()
                eef_pose = np.squeeze(eef_pose, axis=0)  # (7,)
                pos = eef_pose[:3]                       # (3,)
                quat = eef_pose[3:7]                     # (4,) (w,x,y,z)

                # Convert quaternion to axis-angle (ax, ay, az)
                axis_angle = quat2axisangle(quat.copy())  # (3,)

                # Gripper scalar: use first component of gripper_pos observation (abs position)
                gripper_pos = obs["policy"]["gripper_pos"].cpu().numpy()
                gripper_pos = np.squeeze(gripper_pos, axis=0)
                if gripper_pos.ndim == 1:
                    gripper_scalar = np.array([gripper_pos[0]], dtype=np.float32)
                else:
                    gripper_scalar = np.array([gripper_pos[0]], dtype=np.float32)

                # Base 7D state: [x, y, z, ax, ay, az, gripper_abs]
                task_state_7 = np.concatenate((pos, axis_angle, gripper_scalar), axis=0).astype(np.float32)

                # All modes: state is pure task-space 7D; forces are sent separately for hybrid
                eef_pose_states = task_state_7

                # Print modified instruction once so you can verify what is sent to the server.
                if action_idx == 0 and (exp_idx == 0 or args.debug_mode > 0):
                    if exp_adv:
                        print(f"[Prompt] {exp_prompt}   (adverb='{exp_adv}')")
                    else:
                        print(f"[Prompt] {exp_prompt}")

                element = {
                    # Image keys are added by the CPU-testable production payload builder.
                    "state": eef_pose_states,
                    "observation/state": eef_pose_states,
                    "prompt": exp_prompt,
                }
                if args.control_mode == "hybrid":
                    gf = tactile_buf.get_force_history()
                    if gf is not None:
                        # Duplicate both top-level and nested keys
                        element["gripper_force"] = gf
                        element["observation/gripper_force"] = gf
                elif args.control_mode in ("tactile", "binary"):
                    tac_img = tactile_buf.get_tactile_image()
                    tac_force = tactile_buf.get_force_history()
                    tac_mm = tactile_buf.get_marker_motion()
                    if tac_img is not None:
                        # OpenPI's Libero tactile policy expects `tactile_image` at top-level
                        element["tactile_image"] = tac_img
                        element["observation/tactile_image"] = tac_img
                    if tac_force is not None:
                        element["tactile_gripper_force"] = tac_force
                        element["observation/tactile_gripper_force"] = tac_force
                    if tac_mm is not None:
                        element["tactile_marker_motion"] = tac_mm
                        element["observation/tactile_marker_motion"] = tac_mm

                before_infer = None
                if args.debug_mode in (2, 3):
                    def _save_debug_frames(resized_frames):
                        for (cam_name, _), rgb in zip(camera_frames, resized_frames):
                            cam_id = cam_name.split("_")[0]
                            rgb_np = (rgb * 255).astype(np.uint8) if rgb.dtype == np.float32 else rgb.copy()
                            cv2.imwrite(
                                str(f"{args.debug_path}/frame_{frame_count:04d}_{cam_id}.png"),
                                cv2.cvtColor(rgb_np[0], cv2.COLOR_RGB2BGR),
                            )

                    before_infer = _save_debug_frames

                inference_response, _ = infer_openpi_step(
                    client,
                    camera_frames=camera_frames,
                    target_image_size=args.target_image_size,
                    base_payload=element,
                    send_dsrl_raw_image=args.send_dsrl_raw_image,
                    before_infer=before_infer,
                )

                # Get action predictions from OpenPI
                # OpenPI outputs 32D (padded). We slice out the **effective** dims:
                #   - diffik/osc: first 7D   (x, y, z, rx, ry, rz, gripper)
                #   - hybrid/tactile: first 13D (x, y, z, rx, ry, rz, gripper, fL(3), fR(3))
                action_chunk = inference_response["actions"]
                assert len(action_chunk) >= args.replan_steps, (
                    f"We want to replan every {args.replan_steps} steps, but policy only predicts"
                    f" {len(action_chunk)} steps."
                )

                if args.control_mode in ("hybrid", "tactile"):
                    # Hybrid force–position + binary gripper control:
                    #   [x, y, z, rx, ry, rz, gripper, fL(3), fR(3)]  -> 13D
                    n = action_chunk.shape[0]
                    d = action_chunk.shape[1]
                    if args.control_mode == "tactile" and args.abs7d:
                        if d < 7:
                            raise ValueError(
                                f"abs7d expects at least 7D actions from OpenPI, "
                                f"but got shape {action_chunk.shape}."
                            )
                        # Force ablation: ignore force outputs (even if present) and pad zeros to 13D.
                        zeros6 = np.zeros((n, 6), dtype=np.float32)
                        hybrid_actions = np.concatenate([action_chunk[:, :7].astype(np.float32), zeros6], axis=1)
                    else:
                        if d < 13:
                            raise ValueError(
                                f"Hybrid control_mode expects at least 13D actions from OpenPI, "
                                f"but got shape {action_chunk.shape}."
                            )
                        hybrid_actions = action_chunk[:, :13].astype(np.float32)  # (N, 13)
                    inference_actions = torch.from_numpy(hybrid_actions).float()
                    inference_actions = inference_actions[: args.replan_steps, :]
                elif args.control_mode == "osc":
                    # OSC env action shape is 7D:
                    #   Input from OpenPI: (x, y, z, rx, ry, rz, gripper)
                    #   Output to env:     (x, y, z, rx, ry, rz, gripper)
                    if action_chunk.shape[1] < 7:
                        raise ValueError(
                            f"osc control_mode expects at least 7D actions from OpenPI, "
                            f"but got shape {action_chunk.shape}."
                        )
                    inference_actions = torch.from_numpy(action_chunk[:, :7].astype(np.float32)).float()
                    inference_actions = inference_actions[: args.replan_steps, :]
                elif args.control_mode == "binary":
                    # IK + tactile (non-hybrid) with **binary** gripper:
                    #   Input from OpenPI: (x, y, z, rx, ry, rz, gripper) - 7D axis-angle
                    #   Output to env:     (x, y, z, qw, qx, qy, qz, gripper_binary) - 8D quaternion
                    if action_chunk.shape[1] < 7:
                        raise ValueError(
                            f"binary control_mode expects at least 7D actions from OpenPI, "
                            f"but got shape {action_chunk.shape}."
                        )
                    action_chunk_7d = action_chunk[:, :7].astype(np.float32)

                    # Binarize gripper:
                    # - If model outputs in [-1, 1], sign() works.
                    # - If model outputs in [0, 1], threshold at 0.5 (open=-1, close=+1).
                    g = action_chunk_7d[:, 6]
                    if np.all(g >= 0.0) and np.all(g <= 1.0):
                        g_bin = np.where(g >= 0.5, 1.0, -1.0).astype(np.float32)
                    else:
                        g_bin = np.where(g >= 0.0, 1.0, -1.0).astype(np.float32)

                    eef_pose_quat = np.array([axisangle2quat(act[3:6]) for act in action_chunk_7d], dtype=np.float32)
                    eef_pose_with_gripper = np.concatenate(
                        (action_chunk_7d[:, :3], eef_pose_quat, g_bin.reshape(-1, 1)), axis=1
                    )  # (N, 8)
                    inference_actions = torch.from_numpy(eef_pose_with_gripper).float()
                    inference_actions = inference_actions[: args.replan_steps, :]
                else:
                    # DiffIK task-space control:
                    #   Input from OpenPI: (x, y, z, rx, ry, rz, gripper) - 7D axis-angle
                    #   Output to env:     (x, y, z, qw, qx, qy, qz, gripper) - 8D quaternion
                    if action_chunk.shape[1] < 7:
                        raise ValueError(
                            f"diffik control_mode expects at least 7D actions from OpenPI, "
                            f"but got shape {action_chunk.shape}."
                        )
                    action_chunk_7d = action_chunk[:, :7]
                    eef_pose_quat = np.array([axisangle2quat(act[3:6]) for act in action_chunk_7d])
                    eef_pose_with_gripper = np.concatenate(
                        (action_chunk_7d[:, :3], eef_pose_quat, action_chunk_7d[:, 6:7]), axis=1
                    )  # (N, 8)
                    inference_actions = torch.from_numpy(eef_pose_with_gripper).float()
                    inference_actions = inference_actions[: args.replan_steps, :]

                # Execute inference actions
                action = inference_actions
                inference_chunks_taken += 1

                # 仅在 debug_mode 1/2/3 时保存动作
                if args.debug_mode in (1, 2, 3):
                    np.save(str(f"{args.debug_path}/action_{frame_count:04d}.npy"), action.cpu().numpy())

                # Execute actions step by step
                # NOTE: We limit to the actual number of actions we have (might be less than replan_steps)
                num_actions_to_execute = min(action.shape[0], args.replan_steps)
                for i in range(num_actions_to_execute):
                    obs, reward, terminated, truncated, info = env.step(action[i].reshape([1, -1]))
                    tactile_buf.append_control_sample(
                        obs,
                        env=env,
                        include_tactile=args.control_mode in ("tactile", "binary"),
                        env_id=0,
                    )

                    step_fL_pred = None
                    step_fR_pred = None
                    step_fL_meas = None
                    step_fR_meas = None
                    step_fL_meas_raw = None
                    step_fR_meas_raw = None
                    step_squeeze_pred = None
                    step_squeeze_meas = None
                    step_ap_pred = None
                    step_ap_meas = None
                    step_override_enabled = False
                    step_target_contact_detected = False
                    step_override_latched = False
                    step_override_activation_step = None
                    step_effective_single_finger_target = None
                    step_effective_squeeze_target = None
                    step_target_contact_force_norm = None
                    step_gripper_pred = None
                    step_gripper_cmd = None
                    step_gripper_meas = None
                    step_gripper_closed_limit = 0.0
                    step_gripper_open_limit = _finite_float_or_none(
                        getattr(env.unwrapped.cfg, "gripper_open_val", 0.04)
                    )
                    if step_gripper_open_limit is None:
                        step_gripper_open_limit = 0.04
                    step_gripper_lower_saturated = None
                    step_gripper_upper_saturated = None
                    step_reward_force = {
                        "reward_grasp_observed": None,
                        "reward_grasp_terms_active": [],
                        "reward_grasp_started": False,
                        "reward_force_valid_step": False,
                        "reward_squeeze_meas_raw": None,
                    }
                    step_lift_state = {
                        "target_object_height_m": None,
                        "target_object_lift_delta_m": None,
                        "target_object_above_lift_threshold": False,
                        "target_object_lift_hold_count": 0,
                        "target_object_lifted": False,
                    }

                    # 若为 Hybrid 控制模式，则缓存 13D 动作以便后续计算 metrics
                    if args.control_mode in ("hybrid", "tactile"):
                        try:
                            if action[i].shape[-1] == 13:
                                exp_actions_13d.append(action[i].detach().cpu())
                                action_np = action[i].detach().cpu().numpy().astype(np.float32)
                                step_fL_pred = _vector3_or_none(action_np[7:10])
                                step_fR_pred = _vector3_or_none(action_np[10:13])
                                pred_series = compute_contact_force_series_from_lr_forces(
                                    fL=np.asarray([step_fL_pred], dtype=np.float32),
                                    fR=np.asarray([step_fR_pred], dtype=np.float32),
                                )
                                step_squeeze_pred = _finite_float_or_none(pred_series.squeeze[0])
                                step_ap_pred = _finite_float_or_none(pred_series.external_norm[0])
                        except Exception:
                            pass

                    # 从 ForcePositionAction.debug_info 统计当前 step 的挤压力（若可用）
                    try:
                        term = env.action_manager.get_term("arm_action")
                        debug = getattr(term, "debug_info", None)
                    except Exception:
                        debug = None
                    if debug:
                        try:
                            if debug.get("f_sq_pred") is not None:
                                step_squeeze_pred = _finite_float_or_none(debug["f_sq_pred"])
                                if step_squeeze_pred is not None:
                                    exp_fsq_pred_values.append(step_squeeze_pred)
                            if debug.get("f_sq_meas") is not None:
                                step_squeeze_meas = _finite_float_or_none(debug["f_sq_meas"])
                                if step_squeeze_meas is not None:
                                    exp_fsq_meas_values.append(step_squeeze_meas)

                            # 加持力模长（在 base frame 下），用于 Ap 相关统计
                            ap_pred = debug.get("F_app_norm_pred", None)
                            ap_meas = debug.get("F_app_norm_meas", None)
                            try:
                                if ap_pred is not None:
                                    step_ap_pred = _finite_float_or_none(ap_pred)
                                    if step_ap_pred is not None:
                                        exp_ap_pred_values.append(step_ap_pred)
                                if ap_meas is not None:
                                    step_ap_meas = _finite_float_or_none(ap_meas)
                                    if step_ap_meas is not None:
                                        exp_ap_meas_values.append(step_ap_meas)
                            except Exception:
                                pass

                            step_fL_pred = _vector3_or_none(debug.get("fL_pred_local")) or step_fL_pred
                            step_fR_pred = _vector3_or_none(debug.get("fR_pred_local")) or step_fR_pred
                            step_fL_meas = _vector3_or_none(debug.get("fL_meas_local"))
                            step_fR_meas = _vector3_or_none(debug.get("fR_meas_local"))
                            step_fL_meas_raw = _vector3_or_none(
                                debug.get("fL_meas_local_raw")
                            ) or step_fL_meas
                            step_fR_meas_raw = _vector3_or_none(
                                debug.get("fR_meas_local_raw")
                            ) or step_fR_meas
                            if step_fL_meas is not None and step_fR_meas is not None:
                                exp_fL_meas_values.append(np.asarray(step_fL_meas, dtype=np.float32))
                                exp_fR_meas_values.append(np.asarray(step_fR_meas, dtype=np.float32))

                            step_override_enabled = bool(
                                _bool_or_none(
                                    debug.get("target_contact_override_enabled")
                                )
                            )
                            step_target_contact_detected = bool(
                                _bool_or_none(debug.get("target_contact_detected"))
                            )
                            step_override_latched = bool(
                                _bool_or_none(
                                    debug.get(
                                        "target_contact_override_latched"
                                    )
                                )
                            )
                            step_override_activation_step = _integer_or_none(
                                debug.get("target_contact_activation_step")
                            )
                            step_effective_single_finger_target = (
                                _finite_float_or_none(
                                    debug.get(
                                        "target_contact_single_finger_target"
                                    )
                                )
                            )
                            step_effective_squeeze_target = _finite_float_or_none(
                                debug.get("target_contact_squeeze_target")
                            )
                            step_gripper_pred = _finite_float_or_none(
                                debug.get("d_pred")
                            )
                            step_gripper_cmd = _finite_float_or_none(
                                debug.get("d_cmd")
                            )
                            step_gripper_meas = _finite_float_or_none(
                                debug.get("d_actual")
                            )
                            if step_gripper_cmd is not None:
                                step_gripper_lower_saturated = bool(
                                    step_gripper_cmd
                                    <= step_gripper_closed_limit + 1.0e-8
                                )
                                step_gripper_upper_saturated = bool(
                                    step_gripper_cmd
                                    >= step_gripper_open_limit - 1.0e-8
                                )
                            step_target_contact_force_norm = _finite_float_or_none(
                                debug.get("target_contact_force_norm")
                            )

                            if step_override_enabled:
                                exp_override_configured = True
                                exp_override_single_finger_target_n = (
                                    _finite_float_or_none(
                                        debug.get(
                                            "target_contact_configured_single_finger_force"
                                        )
                                    )
                                )
                                exp_override_squeeze_target_n = (
                                    _finite_float_or_none(
                                        debug.get(
                                            "target_contact_configured_squeeze_force"
                                        )
                                    )
                                )
                                exp_override_contact_steps += int(
                                    step_target_contact_detected
                                )
                                if step_override_latched:
                                    exp_override_activated = True
                                    exp_override_active_steps += 1
                                    if (
                                        step_override_activation_step is not None
                                        and step_override_activation_step >= 0
                                    ):
                                        exp_override_activation_step = (
                                            step_override_activation_step
                                        )
                                    if step_fL_meas is not None:
                                        exp_override_left_normal_abs.append(
                                            abs(float(step_fL_meas[2]))
                                        )
                                    if step_fR_meas is not None:
                                        exp_override_right_normal_abs.append(
                                            abs(float(step_fR_meas[2]))
                                        )
                        except Exception:
                            pass
                    elif args.control_mode == "binary":
                        # Binary (IK+tactile) env doesn't use ForcePositionAction, so `debug_info` may be absent.
                        # For compatibility with existing evaluation parsers:
                        # - We still track squeeze_pred/squeeze_meas, but set pred to 0.0 (sentinel; should be ignored).
                        # - We additionally track applied force magnitude (ap_meas) from `gripper_net_force`,
                        #   using the EXACT same definition as ForcePositionAction / benchmarks.common.metrics.
                        try:
                            gnf = obs["policy"]["gripper_net_force"]  # (N, H=1, 2, 3) typically
                            # pick env0, current frame 0: (2,3)
                            f_lr = gnf[0, 0].detach().cpu().numpy().astype(np.float32)
                            f_left = f_lr[0]
                            f_right = f_lr[1]
                            # Reuse the canonical hybrid metric definition:
                            # - squeeze: 2*min(|fL_z|,|fR_z|)
                            # - applied force vector: Fx=fLx+fRx, Fy=fLy+fRy,
                            #   Fz=a+b-common*(sign(a)+sign(b)), then ap = ||F_app||_2
                            series = compute_contact_force_series_from_lr_forces(
                                fL=np.asarray([f_left], dtype=np.float32),
                                fR=np.asarray([f_right], dtype=np.float32),
                            )
                            squeeze_meas = _finite_float_or_none(series.squeeze[0])
                            ap_meas = _finite_float_or_none(series.external_norm[0])

                            # pred placeholders (0.0) for log/regex compatibility
                            squeeze_pred = 0.0
                            ap_pred = 0.0

                            exp_fsq_pred_values.append(squeeze_pred)
                            if squeeze_meas is not None:
                                exp_fsq_meas_values.append(squeeze_meas)

                            exp_ap_pred_values.append(ap_pred)
                            if ap_meas is not None:
                                exp_ap_meas_values.append(ap_meas)

                            step_squeeze_pred = squeeze_pred
                            step_squeeze_meas = squeeze_meas
                            step_ap_pred = ap_pred
                            step_ap_meas = ap_meas
                            step_fL_meas = _vector3_or_none(f_left)
                            step_fR_meas = _vector3_or_none(f_right)
                            step_fL_meas_raw = step_fL_meas
                            step_fR_meas_raw = step_fR_meas

                            # Keep raw 3D forces for strict per-episode metrics aggregation.
                            if step_fL_meas is not None and step_fR_meas is not None:
                                exp_fL_meas_values.append(np.asarray(step_fL_meas, dtype=np.float32))
                                exp_fR_meas_values.append(np.asarray(step_fR_meas, dtype=np.float32))
                        except Exception:
                            # If force obs is missing, skip silently (do not break main loop).
                            pass

                    # debug_mode=4: 在线更新 Hybrid 力–位混合可视化
                    if force_viz is not None and args.debug_mode == 4:
                        try:
                            force_viz.update(debug)
                        except Exception:
                            # 可视化失败不应中断主流程
                            pass

                    if trajectory_force_error is None:
                        try:
                            if (
                                step_fL_meas_raw is None
                                or step_fR_meas_raw is None
                            ):
                                policy_obs = obs.get("policy", {})
                                gnf = policy_obs.get("gripper_net_force")
                                if gnf is None:
                                    raise RuntimeError(
                                        "current raw gripper force is unavailable"
                                    )
                                if isinstance(gnf, torch.Tensor):
                                    gnf = gnf.detach().cpu().numpy()
                                force_history = np.asarray(gnf, dtype=np.float32)
                                if (
                                    force_history.ndim != 4
                                    or force_history.shape[0] != 1
                                    or force_history.shape[2:] != (2, 3)
                                    or force_history.shape[1] < 1
                                ):
                                    raise RuntimeError(
                                        "gripper_net_force must have shape "
                                        f"(1, H>=1, 2, 3), got {force_history.shape}"
                                    )
                                force_lr_raw = force_history[0, -1]
                                step_fL_meas_raw = _vector3_or_none(
                                    force_lr_raw[0]
                                )
                                step_fR_meas_raw = _vector3_or_none(
                                    force_lr_raw[1]
                                )
                            if (
                                step_fL_meas_raw is None
                                or step_fR_meas_raw is None
                            ):
                                raise RuntimeError(
                                    "raw gripper force contains invalid values"
                                )

                            grasp_observations = (
                                env.unwrapped.observation_manager.compute_group(
                                    "subtask_terms", update_history=False
                                )
                            )
                            if not isinstance(grasp_observations, dict):
                                raise RuntimeError(
                                    "subtask_terms observations are not a dictionary"
                                )
                            if trajectory_force_tracker is None:
                                grasp_term_names = sorted(
                                    name
                                    for name in grasp_observations
                                    if name.startswith("grasp_")
                                )
                                trajectory_force_tracker = TrajectoryForceTracker(
                                    grasp_term_names,
                                    contact_epsilon_n=(
                                        args.reward_force_contact_epsilon_n
                                    ),
                                    min_valid_samples=(
                                        args.reward_force_min_valid_samples
                                    ),
                                )
                            step_reward_force = trajectory_force_tracker.update(
                                step_index=total_steps_taken,
                                grasp_observations=grasp_observations,
                                force_lr_raw=np.asarray(
                                    [step_fL_meas_raw, step_fR_meas_raw],
                                    dtype=np.float32,
                                ),
                            )
                        except Exception as exc:
                            trajectory_force_error = str(exc)
                            print(
                                "[Trajectory-Force] Failed to update reward-aligned "
                                f"metric for experiment {exp_idx}: {exc}"
                            )

                    exp_tracking_reward_valid.append(
                        bool(step_reward_force["reward_force_valid_step"])
                    )
                    exp_tracking_squeeze_pred.append(step_squeeze_pred)
                    exp_tracking_squeeze_target_eff.append(
                        step_effective_squeeze_target
                    )
                    exp_tracking_squeeze_meas_raw.append(
                        step_reward_force["reward_squeeze_meas_raw"]
                    )
                    exp_tracking_gripper_pred.append(step_gripper_pred)
                    exp_tracking_gripper_cmd.append(step_gripper_cmd)

                    if lift_tracker is not None and lift_object is not None:
                        object_height_m = float(
                            lift_object.data.root_pos_w[0, 2]
                            .detach()
                            .cpu()
                            .item()
                        )
                        step_lift_state = lift_tracker.update(
                            step_index=total_steps_taken,
                            object_height_m=object_height_m,
                            contact_latched=step_override_latched,
                        )

                    total_steps_taken += 1

                    if args.step_trace_path is not None:
                        exp_step_trace_rows.append(
                            {
                                "schema_version": 3,
                                "task_suite": args.task_suite,
                                "task_id": int(args.task_id),
                                "experiment_index": int(exp_idx),
                                "hdf5_episode_index": dataset_episode_index,
                                "env_step_index": int(total_steps_taken - 1),
                                "inference_chunk_index": int(action_idx),
                                "action_in_chunk_index": int(i),
                                "fL_pred_local": step_fL_pred,
                                "fR_pred_local": step_fR_pred,
                                "fL_meas_local": step_fL_meas,
                                "fR_meas_local": step_fR_meas,
                                "squeeze_pred": step_squeeze_pred,
                                "squeeze_meas": step_squeeze_meas,
                                "ap_pred": step_ap_pred,
                                "ap_meas": step_ap_meas,
                                "predicted_contact": _contact_or_none(step_fL_pred, step_fR_pred),
                                "measured_contact": _contact_or_none(step_fL_meas, step_fR_meas),
                                "target_contact_override_enabled": step_override_enabled,
                                "target_contact_detected": step_target_contact_detected,
                                "target_contact_override_latched": step_override_latched,
                                "target_contact_activation_step": step_override_activation_step,
                                "target_contact_force_norm": step_target_contact_force_norm,
                                "effective_single_finger_target_n": step_effective_single_finger_target,
                                "effective_squeeze_target_n": step_effective_squeeze_target,
                                "gripper_pred_m": step_gripper_pred,
                                "gripper_cmd_m": step_gripper_cmd,
                                "gripper_meas_m": step_gripper_meas,
                                "gripper_closed_limit_m": step_gripper_closed_limit,
                                "gripper_open_limit_m": step_gripper_open_limit,
                                "gripper_lower_saturated": step_gripper_lower_saturated,
                                "gripper_upper_saturated": step_gripper_upper_saturated,
                                **step_reward_force,
                                **step_lift_state,
                            }
                        )

                    if episode_video is not None:
                        try:
                            video_frames: list[tuple[str, np.ndarray]] = []
                            for camera_name in args.camera_names:
                                camera = env.unwrapped.scene[camera_name]
                                rgb = camera.data.output["rgb"][0]
                                video_frames.append(
                                    (camera_name, _to_uint8_rgb(rgb))
                                )
                            left_normal = (
                                abs(float(step_fL_meas[2]))
                                if step_fL_meas is not None
                                else float("nan")
                            )
                            right_normal = (
                                abs(float(step_fR_meas[2]))
                                if step_fR_meas is not None
                                else float("nan")
                            )
                            lift_delta = step_lift_state[
                                "target_object_lift_delta_m"
                            ]
                            lift_delta_text = (
                                f"{float(lift_delta):.3f}"
                                if lift_delta is not None
                                else "N/A"
                            )
                            episode_video.write(
                                video_frames,
                                [
                                    f"episode={exp_idx:03d} step={total_steps_taken - 1}",
                                    (
                                        "target_contact="
                                        f"{int(step_target_contact_detected)} "
                                        f"latched={int(step_override_latched)}"
                                    ),
                                    (
                                        "target_single="
                                        f"{step_effective_single_finger_target!s} N "
                                        "meas_abs_z="
                                        f"({left_normal:.2f},{right_normal:.2f}) N"
                                    ),
                                    (
                                        f"lift_delta={lift_delta_text} m "
                                        "lifted="
                                        f"{int(step_lift_state['target_object_lifted'])}"
                                    ),
                                ],
                            )
                        except Exception as exc:
                            episode_video.error = str(exc)

                    if terminated[0] or truncated[0]:
                        experiment_success = False
                        end_reason, terminal_term = resolve_episode_termination(
                            info=info,
                            terminated=bool(terminated[0]),
                            truncated=bool(truncated[0]),
                        )
                        damage_details = extract_object_damage_details(
                            info=info,
                            terminal_term=terminal_term,
                        )
                        break

                    if success_term is not None:
                        if bool(success_term.func(env, **success_term.params)[0]):
                            success_step_count += 1
                            if success_step_count >= args.num_success_steps:
                                experiment_success = True
                                end_reason = "success"
                                break
                        else:
                            success_step_count = 0

                    # debug_mode=6: dump camera RGB + tactile markers_rgb + (pred/meas) gripper position + (pred/meas) squeeze
                    if mode6_force_fh is not None and mode6_cam_dir is not None and mode6_tac_dir is not None:
                        try:
                            # --- Images (post-step) ---
                            for cam_name in list(args.camera_names):
                                cam_id = cam_name.split("_")[0]
                                cam = env.unwrapped.scene[cam_name]
                                rgb = cam.data.output["rgb"][0]
                                rgb_u8 = _to_uint8_rgb(rgb)
                                cv2.imwrite(
                                    str(mode6_cam_dir / f"frame_{frame_count:04d}_{cam_id}.png"),
                                    cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR),
                                )

                            # tactile markers_rgb (left/right)
                            for tac_name in list(args.tactile_sensor_names):
                                try:
                                    sensor = env.unwrapped.scene.sensors[tac_name]
                                    outputs = sensor.data.output
                                    if "markers_rgb" not in outputs:
                                        continue
                                    tac_img = outputs["markers_rgb"][0]
                                    tac_u8 = _to_uint8_rgb(tac_img)
                                    cv2.imwrite(
                                        str(mode6_tac_dir / f"frame_{frame_count:04d}_{tac_name}_markers_rgb.png"),
                                        cv2.cvtColor(tac_u8, cv2.COLOR_RGB2BGR),
                                    )
                                except Exception:
                                    # tactile sensor may not exist in non-tactile envs
                                    continue

                            # --- Scalars (pred/meas) ---
                            # Gripper position semantics for force-position control:
                            # - pred: raw policy d_pred before force correction/clamping
                            # - cmd: controller-corrected and clamped d_cmd actually sent to both joints
                            # - meas: mean of the two raw finger joint positions after env.step()
                            # Do not average obs["policy"]["gripper_pos"] here: that observation
                            # contains [q_left, -q_right], whose mean is a signed asymmetry rather
                            # than a gripper opening position.
                            gripper_pred = None
                            gripper_cmd = None
                            gripper_meas = None
                            gripper_joint_left = None
                            gripper_joint_right = None

                            # squeeze_pred: prefer ForcePositionAction.debug_info if available, else derive from 13D action forces
                            # squeeze_meas: prefer ForcePositionAction.debug_info if available, else derive from obs["policy"]["gripper_net_force"]
                            squeeze_pred = None
                            squeeze_meas = None

                            # Raw model prediction from the executed action. This is retained for
                            # audit, but the video command curve uses the final controller d_cmd.
                            try:
                                a_np = action[i].detach().cpu().numpy().astype(np.float32)
                                # - hybrid/tactile: 13D => gripper at index 6
                                # - diffik/osc/binary: 8D => gripper at index 7
                                if a_np.shape[-1] >= 13:
                                    gripper_pred = float(a_np[6])
                                elif a_np.shape[-1] >= 8:
                                    gripper_pred = float(a_np[7])
                                elif a_np.shape[-1] >= 7:
                                    gripper_pred = float(a_np[6])
                            except Exception:
                                pass

                            # Controller command and raw joint measurements. ForcePositionAction
                            # exposes these values after applying the current env step.
                            try:
                                if debug:
                                    debug_gripper_pred = _finite_float_or_none(
                                        debug.get("d_pred")
                                    )
                                    if debug_gripper_pred is not None:
                                        gripper_pred = debug_gripper_pred
                                    gripper_cmd = _finite_float_or_none(
                                        debug.get("d_cmd")
                                    )
                                    gripper_meas = _finite_float_or_none(
                                        debug.get("d_actual")
                                    )
                                    gripper_joint_left = _finite_float_or_none(
                                        debug.get("d_actual_left")
                                    )
                                    gripper_joint_right = _finite_float_or_none(
                                        debug.get("d_actual_right")
                                    )
                            except Exception:
                                pass

                            # Measured squeeze from policy obs; gripper position intentionally
                            # does not use the signed gripper_pos observation.
                            try:
                                policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
                                gnf = policy_obs.get("gripper_net_force", None)
                                if gnf is not None:
                                    f_lr = gnf[0, 0].detach().cpu().numpy().astype(np.float32)  # (2,3)
                                    meas_series = compute_contact_force_series_from_lr_forces(
                                        fL=np.asarray([f_lr[0].copy()], dtype=np.float32),
                                        fR=np.asarray([f_lr[1].copy()], dtype=np.float32),
                                    )
                                    squeeze_meas = float(meas_series.squeeze[0])
                            except Exception:
                                pass

                            # Prefer debug_info squeeze values when available (matches existing reporting semantics)
                            try:
                                if debug:
                                    # These are scalars per-step
                                    squeeze_pred = float(debug.get("f_sq_pred", squeeze_pred or 0.0))
                                    squeeze_meas = float(debug.get("f_sq_meas", squeeze_meas or 0.0))
                            except Exception:
                                pass

                            # If no debug squeeze_pred but action is 13D, derive squeeze_pred from predicted forces
                            if squeeze_pred is None:
                                try:
                                    a_np = action[i].detach().cpu().numpy().astype(np.float32)
                                    if a_np.shape[-1] >= 13:
                                        fL_pred = a_np[7:10].copy()
                                        fR_pred = a_np[10:13].copy()
                                        pred_series = compute_contact_force_series_from_lr_forces(
                                            fL=np.asarray([fL_pred], dtype=np.float32),
                                            fR=np.asarray([fR_pred], dtype=np.float32),
                                        )
                                        squeeze_pred = float(pred_series.squeeze[0])
                                except Exception:
                                    pass

                            payload = {
                                "task_suite": args.task_suite,
                                "task_id": int(args.task_id),
                                "exp_idx": int(exp_idx),
                                "action_idx": int(action_idx),
                                "replan_i": int(i),
                                "frame": int(frame_count),
                                "mode6_scalar_schema_version": 3,
                                "gripper_pred": gripper_pred,
                                "gripper_cmd": gripper_cmd,
                                "gripper_meas": gripper_meas,
                                "gripper_joint_left": gripper_joint_left,
                                "gripper_joint_right": gripper_joint_right,
                                "squeeze_pred": squeeze_pred,
                                "squeeze_target_eff": step_effective_squeeze_target,
                                "squeeze_meas": squeeze_meas,
                                "squeeze_meas_raw": step_reward_force[
                                    "reward_squeeze_meas_raw"
                                ],
                                "gripper_lower_saturated": step_gripper_lower_saturated,
                                "gripper_upper_saturated": step_gripper_upper_saturated,
                            }
                            mode6_force_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        except Exception:
                            # Never break evaluation due to debug capture
                            pass

                    # 仅在 debug_mode=3 时额外 dump 关节状态 / 图像序列
                    if args.debug_mode == 3:
                        # get joint states
                        cam = env.unwrapped.scene["agentview_cam"]
                        rgb = cam.data.output["rgb"][0]
                        # get joint states
                        robot = env.unwrapped.scene["robot"]
                        states = robot.data.joint_pos
                        states = states.cpu().numpy()

                        np.save(str(f"{args.debug_path}/state_{frame_count:04d}_{i:02d}.npy"), states)
                        # Convert to numpy if it's a tensor
                        if isinstance(rgb, torch.Tensor):
                            rgb = rgb.cpu().numpy()
                        # Ensure correct format for saving
                        if rgb.dtype == np.float32:
                            rgb = (rgb * 255).astype(np.uint8)
                        # Save RGB image
                        cv2.imwrite(
                            str(f"{args.debug_path}/frame_{frame_count:04d}_{i:02d}.png"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        )
                    frame_count += 1

                if experiment_success:
                    successful_experiments += 1
                    current_sr = (successful_experiments / (exp_idx + 1)) * 100
                    if exp_fsq_pred_values and exp_fsq_meas_values:
                        avg_pred = float(np.mean(exp_fsq_pred_values))
                        avg_meas = float(np.mean(exp_fsq_meas_values))
                        print(
                            f"✓ Success | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%) "
                            f"| squeeze_pred={avg_pred:.4f}, squeeze_meas={avg_meas:.4f}"
                        )
                    else:
                        print(f"✓ Success | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%)")
                    break

                # Check if we broke out of inner loop due to unexpected termination
                if terminated[0] or truncated[0]:
                    current_sr = (successful_experiments / (exp_idx + 1)) * 100
                    print(
                        f"✗ Failed ({end_reason}) | Current SR: "
                        f"{successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%)"
                    )
                    break

                if action_idx >= args.max_inference_steps - 1:
                    current_sr = (successful_experiments / (exp_idx + 1)) * 100
                    print(f"✗ Failed (max steps) | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%)")

            if args.control_mode in ("hybrid", "tactile"):
                force_mode = "full"
            elif args.control_mode == "binary":
                force_mode = "measured_only"
            else:
                force_mode = "not_applicable"

            force_summary = summarize_episode_force_metrics(
                force_mode=force_mode,
                env_steps=total_steps_taken,
                actions_13d=exp_actions_13d,
                squeeze_pred_values=exp_fsq_pred_values,
                squeeze_meas_values=exp_fsq_meas_values,
                ap_pred_values=exp_ap_pred_values,
                ap_meas_values=exp_ap_meas_values,
                fL_meas_values=exp_fL_meas_values,
                fR_meas_values=exp_fR_meas_values,
            )
            if trajectory_force_tracker is not None:
                trajectory_force_summary = trajectory_force_tracker.summary()
            else:
                trajectory_force_summary = {
                    "trajectory_mean_measured_squeeze": None,
                    "trajectory_force_valid_samples": 0,
                    "trajectory_force_status": "unavailable",
                    "trajectory_force_error": None,
                    "grasp_started": False,
                    "grasp_start_step": None,
                }
            if trajectory_force_error is not None:
                trajectory_force_summary["trajectory_force_status"] = "error"
                trajectory_force_summary["trajectory_force_error"] = (
                    trajectory_force_error
                )

            try:
                force_tracking_summary = summarize_force_tracking_metrics(
                    expected_steps=total_steps_taken,
                    reward_valid_steps=exp_tracking_reward_valid,
                    predicted_squeeze_values=exp_tracking_squeeze_pred,
                    effective_squeeze_target_values=(
                        exp_tracking_squeeze_target_eff
                    ),
                    measured_squeeze_raw_values=(
                        exp_tracking_squeeze_meas_raw
                    ),
                    gripper_pred_values=exp_tracking_gripper_pred,
                    gripper_cmd_values=exp_tracking_gripper_cmd,
                    gripper_closed_limit_m=0.0,
                    gripper_open_limit_m=float(
                        getattr(env.unwrapped.cfg, "gripper_open_val", 0.04)
                    ),
                )
            except Exception as exc:
                force_tracking_summary = {
                    "status": "error",
                    "error": str(exc),
                }

            def _mean_or_none(values: list[float]) -> float | None:
                return float(np.mean(values)) if values else None

            force_override_summary = {
                "configured": bool(exp_override_configured),
                "activated": bool(exp_override_activated),
                "activation_step": exp_override_activation_step,
                "active_steps": int(exp_override_active_steps),
                "target_contact_steps": int(exp_override_contact_steps),
                "single_finger_target_n": exp_override_single_finger_target_n,
                "squeeze_target_n": exp_override_squeeze_target_n,
                "left_normal_abs_avg_n": _mean_or_none(
                    exp_override_left_normal_abs
                ),
                "right_normal_abs_avg_n": _mean_or_none(
                    exp_override_right_normal_abs
                ),
                "left_normal_abs_top5_n": (
                    compute_topk_mean(exp_override_left_normal_abs, frac=0.05)
                    if exp_override_left_normal_abs
                    else None
                ),
                "right_normal_abs_top5_n": (
                    compute_topk_mean(exp_override_right_normal_abs, frac=0.05)
                    if exp_override_right_normal_abs
                    else None
                ),
            }
            lift_summary = (
                lift_tracker.summary()
                if lift_tracker is not None
                else {
                    "enabled": False,
                    "target_object": None,
                    "initial_height_m": None,
                    "max_height_m": None,
                    "max_height_delta_m": None,
                    "height_threshold_m": float(args.lift_height_threshold_m),
                    "hold_steps_required": int(args.lift_hold_steps),
                    "above_threshold_steps": 0,
                    "max_consecutive_above_threshold_steps": 0,
                    "first_above_threshold_step": None,
                    "lift_confirmed_step": None,
                    "lifted": False,
                }
            )

            if episode_video is not None:
                video_summary = episode_video.finalize(
                    success=experiment_success,
                    end_reason=end_reason,
                    expected_frames=total_steps_taken,
                )
            else:
                video_summary = {
                    "enabled": False,
                    "status": "disabled",
                    "path": None,
                    "frames": 0,
                    "fps": None,
                    "error": None,
                }

            trace_status = "disabled"
            trace_rows_written = 0
            trace_error = step_trace_open_error
            if args.step_trace_path is not None:
                trace_status = "partial"
                if step_trace_fh is not None:
                    try:
                        for trace_row in exp_step_trace_rows:
                            step_trace_fh.write(json.dumps(trace_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                            trace_rows_written += 1
                        step_trace_fh.flush()
                        if trace_rows_written == total_steps_taken:
                            trace_status = "complete"
                        else:
                            trace_error = (
                                f"trace rows {trace_rows_written} do not match env_steps {total_steps_taken}"
                            )
                    except Exception as exc:
                        trace_error = str(exc)
                        print(f"[Step-Trace] Failed while writing experiment {exp_idx}: {exc}")

            episode_record = {
                "experiment_index": int(exp_idx),
                "hdf5_episode_index": dataset_episode_index,
                "success": bool(experiment_success),
                "end_reason": end_reason,
                "terminal_term": terminal_term,
                "object_damage": damage_details,
                "damage_threshold": damage_threshold_snapshot,
                "friction": friction_snapshot,
                "env_steps": int(total_steps_taken),
                "inference_chunks": int(inference_chunks_taken),
                **force_summary,
                **trajectory_force_summary,
                "force_tracking": force_tracking_summary,
                "force_override": force_override_summary,
                "lift": lift_summary,
                "video": video_summary,
                "trace_status": trace_status,
                "trace_rows": int(trace_rows_written),
                "trace_error": trace_error,
            }
            episode_metrics_records.append(episode_record)
            print("[Episode-Metrics] " + json.dumps(episode_record, ensure_ascii=False, separators=(",", ":")))

            # debug_mode=5: 每个 experiment 结束后落盘一份逐帧挤压力序列
            if force_dump_dir is not None:
                try:
                    payload = {
                        "task_suite": args.task_suite,
                        "task_id": int(args.task_id),
                        "exp_idx": int(exp_idx),
                        "prompt_adverb": (args.prompt_adverb or "").strip(),
                        "prompt_adverbs": list(args.prompt_adverbs) if args.prompt_adverbs else [],
                        "adverb_used": exp_adv,
                        "prompt": exp_prompt,
                        "success": bool(experiment_success),
                        "terminated": bool(terminated[0]) if hasattr(terminated, "__len__") else bool(terminated),
                        "truncated": bool(truncated[0]) if hasattr(truncated, "__len__") else bool(truncated),
                        "num_frames": int(len(exp_fsq_pred_values)),
                        # 逐帧挤压力：与 env.step() 次数一一对应
                        "squeeze_pred": [float(x) for x in exp_fsq_pred_values],
                        "squeeze_meas": [float(x) for x in exp_fsq_meas_values],
                        # 逐帧加持力模长（ap_pred/ap_meas，与 ForcePositionAction / metrics.py 一致）：
                        # - hybrid/tactile: from ForcePositionAction.debug_info when available
                        # - binary: pred is always 0.0 (sentinel), meas derived from gripper_net_force
                        "ap_pred": [float(x) for x in exp_ap_pred_values],
                        "ap_meas": [float(x) for x in exp_ap_meas_values],
                    }
                    out_path = force_dump_dir / f"exp_{exp_idx:03d}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[DebugMode 5] Failed to dump gripper force for exp_{exp_idx:03d}: {e}")

            # debug_mode=6: close per-experiment file handle
            try:
                if mode6_force_fh is not None:
                    mode6_force_fh.close()
            except Exception:
                pass

    if step_trace_fh is not None:
        try:
            step_trace_fh.close()
        except Exception:
            pass

    success_rate = (successful_experiments / args.num_total_experiments) * 100
    print("\nEvaluation Results:")
    print(f"Total experiments: {args.num_total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Success rate: {success_rate:.2f}%")

    force_aggregates, force_metric_counts = aggregate_success_force_metrics(episode_metrics_records)
    trajectory_force_aggregates = summarize_trajectory_force_metrics(
        episode_metrics_records
    )
    success_trajectory_force = trajectory_force_aggregates[
        "success_trajectory_mean_measured_squeeze"
    ]
    success_trajectory_force_count = trajectory_force_aggregates[
        "success_trajectory_force_episode_count"
    ]
    if success_trajectory_force is not None:
        print(
            "[Trajectory-Force] Task reward_aligned "
            f"success_trajectory_mean_measured_squeeze={success_trajectory_force:.4f} "
            f"over {success_trajectory_force_count} eligible successes"
        )
    task_avg_pred = force_aggregates["squeeze_avg_pred"]
    task_avg_meas = force_aggregates["squeeze_avg_meas"]
    if task_avg_pred is not None and task_avg_meas is not None:
        print(
            f"[Hybrid] Task avg squeeze_pred={task_avg_pred:.4f}, squeeze_meas={task_avg_meas:.4f} "
            f"over {force_metric_counts['squeeze_avg_pred']} successes"
        )

    contact_metric_count = max(
        force_metric_counts["squeeze_max_pred"],
        force_metric_counts["squeeze_max_meas"],
        force_metric_counts["ap_avg_pred"],
        force_metric_counts["ap_avg_meas"],
        force_metric_counts["ap_max_pred"],
        force_metric_counts["ap_max_meas"],
    )
    if contact_metric_count > 0:
        fragments: list[str] = []
        if task_avg_pred is not None:
            fragments.append(f"squeeze_avg_pred={task_avg_pred:.4f}")
        if task_avg_meas is not None:
            fragments.append(f"squeeze_avg_meas={task_avg_meas:.4f}")

        output_key_map = {
            "squeeze_max_pred": "squeeze_max_mean",
            "ap_max_pred": "app_max_mean",
            "ap_avg_pred": "app_mean_mean",
            "squeeze_max_meas": "squeeze_max_meas_mean",
            "ap_max_meas": "ap_max_meas_mean",
            "ap_avg_meas": "ap_mean_meas_mean",
        }
        for metric_key, output_key in output_key_map.items():
            value = force_aggregates[metric_key]
            if value is not None:
                fragments.append(f"{output_key}={value:.4f}")

        print(
            "[Hybrid-Metrics] Task contact_metrics "
            + ", ".join(fragments)
            + f" over {contact_metric_count} successes"
        )
    # 关闭 Hybrid 力–位可视化窗口
    if force_viz is not None:
        try:
            force_viz.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("args", args)

    # Initialize the closed loop policy inference
    # Only support task space / hybrid control (diffik, osc, hybrid, tactile, binary)
    if args.control_mode in ["diffik", "osc", "hybrid", "tactile", "binary"]:
        inferencer = ClosedLoopPolicyInference(args)
    else:
        raise ValueError(
            f"Invalid control mode: {args.control_mode}. "
            f"Supported modes: ['diffik', 'osc', 'hybrid', 'tactile', 'binary']"
        )

    # Initialize client policy inference
    env, env_cfg, success_term = inferencer.create_sim_environment()

    # Ablation: tactile obs/model, but pure position actions (no force) and no corrections.
    if args.control_mode == "tactile" and args.abs7d:
        try:
            term = env.action_manager.get_term("arm_action")
            term.cfg.pos_kp = (0.0, 0.0, 0.0)
            term.cfg.squeeze_kp = 0.0
            print("[Ablation] abs7d enabled: pos_kp=(0,0,0), squeeze_kp=0, force dims zeroed.")
        except Exception as e:
            print(f"[Ablation] Failed to disable pos_kp/squeeze_kp on arm_action: {e}")

    # Run the closed loop policy
    run_closed_loop_policy(
        args=args, simulation_app=simulation_app, env=env, env_cfg=env_cfg, success_term=success_term
    )

    # Close environment and simulation app after replay is complete
    env.close()
    simulation_app.close()
