# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import hashlib
import json
import os
import sys
import re
from datetime import datetime
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
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
from benchmarks.common.metrics import (
    compute_contact_force_metrics_from_13d,
    compute_contact_force_metrics_from_lr_forces,
    compute_contact_force_series_from_lr_forces,
    compute_topk_mean,
)


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

    # Simulator specific parameters
    # Default to headless to avoid X11/GLX BadMatch crashes on servers or misconfigured displays.
    # If you want a GUI window, pass: --no-headless
    headless: bool = True
    seed: int = 11
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

    camera_names: tuple[str] = ("agentview_cam", "eye_in_hand_cam")
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

    # Control mode parameters
    # Supported modes:
    #   - "diffik": Task-space control via Differential IK
    #   - "osc":    Task-space control via OSC
    #   - "hybrid":  Hybrid force–position control (ContactForce)
    #   - "tactile": Hybrid force–position + tactile observations (GelSight)
    #   - "binary": IK + tactile observations (GelSight), but execute 7D actions with **binary gripper**
    #
    # OpenPI server always returns a 32D action vector (padded), but:
    #   - For "diffik"/"osc": we use the first 7D
    #       (x, y, z, rx, ry, rz, gripper) - axis-angle + gripper
    #       and convert it to 8D quaternion before sending to the env:
    #       (x, y, z, qw, qx, qy, qz, gripper)
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


def _add_bytes_key_aliases(d: dict, keys: tuple[str, ...]) -> None:
    """Add bytes-key aliases for servers that decode msgpack map keys as bytes."""
    for k in keys:
        if k in d and isinstance(k, str):
            d[k.encode("utf-8")] = d[k]


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
app_launcher = AppLauncher(headless=args.headless, enable_cameras=True, num_envs=1)
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
    resize_frames_with_padding,
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
                "timestamp": ts,
            }
            with open(capture_mode6_root / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DebugMode 6] Failed to write run_meta.json: {e}")

    # Hybrid 力–位混合在线可视化（仅在 debug_mode == 4 时启用）
    force_viz = None
    if args.debug_mode == 4 and "Isaac-Libero-Franka-Hybrid-" in args.task and args.num_envs == 1:
        try:
            # 复用 scripts/tools 中的调试可视化工具
            # 注意：此处从工程根目录下的 scripts.tools.common 导入，而不是相对路径的 common。
            from scripts.tools.common.force_position_debug_viz import ForcePositionDebugVisualizer

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

    successful_experiments = 0

    # 统计：跨所有成功 experiment 的夹爪挤压力均值（仅在 Hybrid 环境中有效）
    succ_squeeze_pred_sum = 0.0
    succ_squeeze_meas_sum = 0.0
    succ_squeeze_count = 0

    # 统计：跨所有成功 experiment 的挤压力 / 加持力 metrics（仅在 Hybrid + 13D action 时有效）
    succ_metrics_count = 0
    succ_squeeze_max_sum = 0.0
    succ_app_max_sum = 0.0
    succ_app_mean_sum = 0.0

    # 统计：跨所有成功 experiment 的「实测」Top5% 最大挤压力 / 加持力及平均加持力
    succ_squeeze_max_meas_sum = 0.0
    succ_squeeze_max_meas_count = 0
    succ_ap_mean_meas_sum = 0.0
    succ_ap_mean_meas_count = 0
    succ_ap_max_meas_sum = 0.0
    succ_ap_max_meas_count = 0

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

    cli_instruction = (args.language_instruction or "").strip()
    if cli_instruction:
        print(f"\nUsing language instruction (from CLI): {cli_instruction}")
        args.language_instruction = cli_instruction
    else:
        for task in task_suite_config["tasks"]:
            task_id = task["task_id"]
            if task_id == args.task_id:
                args.language_instruction = task["language_instruction"]
                print(f"\nUsing language instruction (from task config): {args.language_instruction}")
                break

    client = _websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        for exp_idx in range(args.num_total_experiments):
            print(f"\n[{exp_idx + 1}/{args.num_total_experiments}] Starting experiment...", end=" ", flush=True)
            success_step_count = 0
            experiment_success = False
            total_steps_taken = 0

            # 当前 experiment 的挤压力统计（均值）
            exp_fsq_pred_sum = 0.0
            exp_fsq_meas_sum = 0.0
            exp_fsq_count = 0

            # 当前 experiment 的逐帧挤压力 / 加持力记录（用于 Top5% 统计）
            exp_fsq_pred_values: list[float] = []
            exp_fsq_meas_values: list[float] = []
            exp_ap_pred_values: list[float] = []
            exp_ap_meas_values: list[float] = []
            # binary 模式：额外缓存逐步左右指 3D 力序列（用于严格复用 metrics.py 的统计定义）
            exp_fL_meas_values: list[np.ndarray] = []
            exp_fR_meas_values: list[np.ndarray] = []

            # 当前 experiment 的 Hybrid 13D 动作缓存（仅在 control_mode == "hybrid" 时使用）
            exp_actions_13d: list[torch.Tensor] = []

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

            # Reset online histories per experiment to match dataset windowing.
            tactile_buf.reset()

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
                    }
                    with open(mode6_exp_dir / "exp_meta.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            for action_idx in range(args.max_inference_steps):
                # Get camera images from live cameras
                rgbs = []
                for cam_name in list(args.camera_names):
                    cam_id = cam_name.split("_")[0]
                    cam = env.unwrapped.scene[cam_name]
                    rgb = cam.data.output["rgb"]
                    rgb = resize_frames_with_padding(rgb, args.target_image_size, bgr_conversion=False, pad_img=True)
                    rgbs.append(rgb)

                    # 仅在 debug_mode=2/3 时保存相机帧到本地
                    if args.debug_mode in (2, 3):
                        rgb_np = (rgb * 255).astype(np.uint8) if rgb.dtype == np.float32 else rgb.copy()
                        cv2.imwrite(
                            str(f"{args.debug_path}/frame_{frame_count:04d}_{cam_id}.png"),
                            cv2.cvtColor(rgb_np[0], cv2.COLOR_RGB2BGR),
                        )

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

                # For Hybrid force–position control, compute finger force history (left/right, 3D each)
                tactile_buf.update_force(obs)
                if args.control_mode in ("tactile", "binary"):
                    # Tactile modalities (Tabero-style): tactile_image + tactile_gripper_force + tactile_marker_motion
                    try:
                        tactile_buf.update_tactile_frames(env, env_id=0)
                    except Exception:
                        pass
                    tactile_buf.update_marker_motion(obs)

                # All modes: state is pure task-space 7D; forces are sent separately for hybrid
                eef_pose_states = task_state_7

                image = _to_uint8_rgb(np.squeeze(rgbs[0], axis=0))
                wrist_image = _to_uint8_rgb(np.squeeze(rgbs[1], axis=0))

                # Print modified instruction once so you can verify what is sent to the server.
                if action_idx == 0 and (exp_idx == 0 or args.debug_mode > 0):
                    if exp_adv:
                        print(f"[Prompt] {exp_prompt}   (adverb='{exp_adv}')")
                    else:
                        print(f"[Prompt] {exp_prompt}")

                element = {
                    # Top-level keys (image / state) for OpenPI transforms
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": eef_pose_states,
                    # Nested "observation/*" keys to keep Tabero-style compatibility
                    "observation/image": image,
                    "observation/wrist_image": wrist_image,
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

                # Add bytes-key aliases for server-side msgpack decoders that return bytes keys.
                _add_bytes_key_aliases(
                    element,
                    (
                        "image",
                        "wrist_image",
                        "state",
                        "prompt",
                        "gripper_force",
                        "tactile_image",
                        "tactile_gripper_force",
                        "tactile_marker_motion",
                        "observation/image",
                        "observation/wrist_image",
                        "observation/state",
                        "observation/gripper_force",
                        "observation/tactile_image",
                        "observation/tactile_gripper_force",
                        "observation/tactile_marker_motion",
                    ),
                )

                # Get action predictions from OpenPI
                # OpenPI outputs 32D (padded). We slice out the **effective** dims:
                #   - diffik/osc: first 7D   (x, y, z, rx, ry, rz, gripper)
                #   - hybrid/tactile: first 13D (x, y, z, rx, ry, rz, gripper, fL(3), fR(3))
                action_chunk = client.infer(element)["actions"]
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
                    # DiffIK / OSC task-space control:
                    #   Input from OpenPI: (x, y, z, rx, ry, rz, gripper) - 7D axis-angle
                    #   Output to env:     (x, y, z, qw, qx, qy, qz, gripper) - 8D quaternion
                    if action_chunk.shape[1] < 7:
                        raise ValueError(
                            f"diffik/osc control_mode expects at least 7D actions from OpenPI, "
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

                # 仅在 debug_mode 1/2/3 时保存动作
                if args.debug_mode in (1, 2, 3):
                    np.save(str(f"{args.debug_path}/action_{frame_count:04d}.npy"), action.cpu().numpy())

                # Execute actions step by step
                # NOTE: We limit to the actual number of actions we have (might be less than replan_steps)
                num_actions_to_execute = min(action.shape[0], args.replan_steps)
                for i in range(num_actions_to_execute):
                    obs, reward, terminated, truncated, info = env.step(action[i].reshape([1, -1]))

                    # 若为 Hybrid 控制模式，则缓存 13D 动作以便后续计算 metrics
                    if args.control_mode in ("hybrid", "tactile"):
                        try:
                            if action[i].shape[-1] == 13:
                                exp_actions_13d.append(action[i].detach().cpu())
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
                            f_sq_pred = float(debug.get("f_sq_pred", 0.0))
                            f_sq_meas = float(debug.get("f_sq_meas", 0.0))
                            exp_fsq_pred_sum += f_sq_pred
                            exp_fsq_meas_sum += f_sq_meas
                            exp_fsq_count += 1
                            exp_fsq_pred_values.append(f_sq_pred)
                            exp_fsq_meas_values.append(f_sq_meas)

                            # 加持力模长（在 base frame 下），用于 Ap 相关统计
                            ap_pred = debug.get("F_app_norm_pred", None)
                            ap_meas = debug.get("F_app_norm_meas", None)
                            try:
                                if ap_pred is not None:
                                    exp_ap_pred_values.append(float(ap_pred))
                                if ap_meas is not None:
                                    exp_ap_meas_values.append(float(ap_meas))
                            except Exception:
                                pass
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
                            squeeze_meas = float(series.squeeze[0])
                            ap_meas = float(series.external_norm[0])

                            # pred placeholders (0.0) for log/regex compatibility
                            squeeze_pred = 0.0
                            ap_pred = 0.0

                            exp_fsq_pred_sum += squeeze_pred
                            exp_fsq_meas_sum += squeeze_meas
                            exp_fsq_count += 1
                            exp_fsq_pred_values.append(squeeze_pred)
                            exp_fsq_meas_values.append(squeeze_meas)

                            exp_ap_pred_values.append(ap_pred)
                            exp_ap_meas_values.append(ap_meas)

                            # Keep raw 3D forces for strict per-episode metrics aggregation.
                            exp_fL_meas_values.append(np.asarray(f_left, dtype=np.float32))
                            exp_fR_meas_values.append(np.asarray(f_right, dtype=np.float32))
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

                    total_steps_taken += 1

                    if terminated[0] or truncated[0]:
                        experiment_success = False
                        break

                    if success_term is not None:
                        if bool(success_term.func(env, **success_term.params)[0]):
                            success_step_count += 1
                            if success_step_count >= args.num_success_steps:
                                experiment_success = True
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
                            # gripper_cmd: from executed action (1D)
                            # gripper_meas: from obs["policy"]["gripper_pos"] (2,) -> mean (1D)
                            gripper_cmd = None
                            gripper_meas = None

                            # squeeze_pred: prefer ForcePositionAction.debug_info if available, else derive from 13D action forces
                            # squeeze_meas: prefer ForcePositionAction.debug_info if available, else derive from obs["policy"]["gripper_net_force"]
                            squeeze_pred = None
                            squeeze_meas = None

                            # commanded gripper position from executed action (shape depends on control_mode)
                            try:
                                a_np = action[i].detach().cpu().numpy().astype(np.float32)
                                # - hybrid/tactile: 13D => gripper at index 6
                                # - diffik/osc/binary: 8D => gripper at index 7
                                if a_np.shape[-1] >= 13:
                                    gripper_cmd = float(a_np[6])
                                elif a_np.shape[-1] >= 8:
                                    gripper_cmd = float(a_np[7])
                                elif a_np.shape[-1] >= 7:
                                    gripper_cmd = float(a_np[6])
                            except Exception:
                                pass

                            # measured gripper position and measured squeeze from policy obs
                            try:
                                policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
                                gp = policy_obs.get("gripper_pos", None)
                                if gp is not None:
                                    gp0 = gp[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
                                    if gp0.size > 0:
                                        gripper_meas = float(np.mean(gp0))
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
                                "gripper_cmd": gripper_cmd,
                                "gripper_meas": gripper_meas,
                                "squeeze_pred": squeeze_pred,
                                "squeeze_meas": squeeze_meas,
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

                    # 累加当前成功 experiment 的平均挤压力
                    if exp_fsq_count > 0:
                        avg_pred = exp_fsq_pred_sum / exp_fsq_count
                        avg_meas = exp_fsq_meas_sum / exp_fsq_count
                        succ_squeeze_pred_sum += avg_pred
                        succ_squeeze_meas_sum += avg_meas
                        succ_squeeze_count += 1
                        print(
                            f"✓ Success | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%) "
                            f"| squeeze_pred={avg_pred:.4f}, squeeze_meas={avg_meas:.4f}"
                        )
                    else:
                        print(f"✓ Success | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%)")

                    # Binary mode: record measured squeeze/apply metrics (no predicted forces).
                    # We compute Top5% stats from per-step measured sequences to mirror hybrid reporting.
                    if args.control_mode == "binary":
                        # Strict: reuse metrics.py aggregation on the per-step LR force series.
                        try:
                            if exp_fL_meas_values and exp_fR_meas_values:
                                fL = np.stack(exp_fL_meas_values, axis=0)  # (T,3)
                                fR = np.stack(exp_fR_meas_values, axis=0)  # (T,3)
                                meas_metrics = compute_contact_force_metrics_from_lr_forces(fL, fR)

                                # Fill the "pred-style" metrics slots with measured metrics in binary mode
                                # (there is no force prediction in this control mode).
                                succ_metrics_count += 1
                                succ_squeeze_max_sum += float(meas_metrics.squeeze_max)
                                succ_app_max_sum += float(meas_metrics.external_norm_max)
                                succ_app_mean_sum += float(meas_metrics.external_norm_mean)

                                # Also populate measured aggregates (same definitions).
                                succ_squeeze_max_meas_sum += float(meas_metrics.squeeze_max)
                                succ_squeeze_max_meas_count += 1
                                succ_ap_mean_meas_sum += float(meas_metrics.external_norm_mean)
                                succ_ap_mean_meas_count += 1
                                succ_ap_max_meas_sum += float(meas_metrics.external_norm_max)
                                succ_ap_max_meas_count += 1
                        except Exception:
                            pass

                    # 若为 Hybrid 控制模式且缓存到了 13D 动作，则为该成功 experiment 计算一次力学 metrics
                    if args.control_mode in ("hybrid", "tactile") and exp_actions_13d:
                        try:
                            actions_13d = torch.stack(exp_actions_13d, dim=0).numpy()  # (T, 13)
                            metrics = compute_contact_force_metrics_from_13d(actions_13d)
                            succ_metrics_count += 1
                            # squeeze_max / external_norm_max 已在 metrics.py 中按 Top5% 帧均值定义
                            succ_squeeze_max_sum += metrics.squeeze_max
                            succ_app_max_sum += metrics.external_norm_max
                            succ_app_mean_sum += metrics.external_norm_mean

                            # 统计当前成功 experiment 的「实测」挤压力 / 加持力指标
                            # 1) 实测挤压力 Top5% 最大值（均值）
                            sq_max_meas_top5 = compute_topk_mean(exp_fsq_meas_values, frac=0.05)
                            if sq_max_meas_top5 is not None:
                                succ_squeeze_max_meas_sum += sq_max_meas_top5
                                succ_squeeze_max_meas_count += 1

                            # 2) 实测加持力平均值（直接在该 demo 内求均值）
                            if exp_ap_meas_values:
                                ap_mean_meas = float(np.mean(exp_ap_meas_values))
                                succ_ap_mean_meas_sum += ap_mean_meas
                                succ_ap_mean_meas_count += 1

                            # 3) 实测加持力 Top5% 最大值（均值）
                            ap_max_meas_top5 = compute_topk_mean(exp_ap_meas_values, frac=0.05)
                            if ap_max_meas_top5 is not None:
                                succ_ap_max_meas_sum += ap_max_meas_top5
                                succ_ap_max_meas_count += 1

                            print(
                                "    [Hybrid-Metrics] "
                                f"squeeze_max={metrics.squeeze_max:.4f}, "
                                f"squeeze_mean={metrics.squeeze_mean:.4f}, "
                                f"app_max={metrics.external_norm_max:.4f}, "
                                f"app_mean={metrics.external_norm_mean:.4f}"
                            )
                        except Exception:
                            # metrics 计算失败不影响主流程
                            pass

                    break

                # Check if we broke out of inner loop due to unexpected termination
                if i < args.replan_steps - 1 and (terminated[0] or truncated[0]):
                    current_sr = (successful_experiments / (exp_idx + 1)) * 100
                    print(f"✗ Failed (terminated) | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%)")
                    break

                if action_idx >= args.max_inference_steps - 1:
                    current_sr = (successful_experiments / (exp_idx + 1)) * 100
                    print(f"✗ Failed (max steps) | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%)")

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

    success_rate = (successful_experiments / args.num_total_experiments) * 100
    print("\nEvaluation Results:")
    print(f"Total experiments: {args.num_total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Success rate: {success_rate:.2f}%")

    # 1) 挤压力平均值（预测 / 实测）——仅用于后续 metrics 行中输出
    task_avg_pred = None
    task_avg_meas = None
    if succ_squeeze_count > 0:
        task_avg_pred = succ_squeeze_pred_sum / succ_squeeze_count
        task_avg_meas = succ_squeeze_meas_sum / succ_squeeze_count

        # Keep backward-compatible line for scripts/tools/run_task_evaluations.py parser.
        if args.control_mode in ("hybrid", "tactile", "binary"):
            print(
                f"[Hybrid] Task avg squeeze_pred={task_avg_pred:.4f}, squeeze_meas={task_avg_meas:.4f} "
                f"over {succ_squeeze_count} successes"
            )

    # 2) Top5% 最大挤压力 / 最大加持力 + 平均加持力（预测 / 实测）——统一在 Hybrid-Metrics 一行输出
    if succ_metrics_count > 0:
        task_squeeze_max_mean = succ_squeeze_max_sum / succ_metrics_count
        task_app_max_mean = succ_app_max_sum / succ_metrics_count
        task_app_mean_mean = succ_app_mean_sum / succ_metrics_count

        # 实测挤压力 / 加持力（若有）
        task_squeeze_max_meas_mean = (
            succ_squeeze_max_meas_sum / succ_squeeze_max_meas_count
            if succ_squeeze_max_meas_count > 0
            else None
        )
        task_ap_mean_meas_mean = (
            succ_ap_mean_meas_sum / succ_ap_mean_meas_count if succ_ap_mean_meas_count > 0 else None
        )
        task_ap_max_meas_mean = (
            succ_ap_max_meas_sum / succ_ap_max_meas_count if succ_ap_max_meas_count > 0 else None
        )

        fragments: list[str] = []
        if task_avg_pred is not None:
            fragments.append(f"squeeze_avg_pred={task_avg_pred:.4f}")
        if task_avg_meas is not None:
            fragments.append(f"squeeze_avg_meas={task_avg_meas:.4f}")

        fragments.extend(
            [
                f"squeeze_max_mean={task_squeeze_max_mean:.4f}",
                f"app_max_mean={task_app_max_mean:.4f}",
                f"app_mean_mean={task_app_mean_mean:.4f}",
            ]
        )
        if task_squeeze_max_meas_mean is not None:
            fragments.append(f"squeeze_max_meas_mean={task_squeeze_max_meas_mean:.4f}")
        if task_ap_max_meas_mean is not None:
            fragments.append(f"ap_max_meas_mean={task_ap_max_meas_mean:.4f}")
        if task_ap_mean_meas_mean is not None:
            fragments.append(f"ap_mean_meas_mean={task_ap_mean_meas_mean:.4f}")

        print(
            "[Hybrid-Metrics] Task contact_metrics "
            + ", ".join(fragments)
            + f" over {succ_metrics_count} successes"
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
