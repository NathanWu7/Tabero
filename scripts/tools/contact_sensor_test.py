#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Contact sensor test script.

Goals:
- Replay a dataset demo like replay_demos_with_camera.py
- Perform detailed checks for contact sensor binding, USD prims, and force data flow
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# Ensure project root on sys.path for top-level imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


parser = argparse.ArgumentParser(
    description="Replay demos and run detailed contact sensor checks."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task env name.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="datasets/dataset.hdf5",
    help="HDF5 dataset path (or rely on task_suite/task_id + env vars).",
)
parser.add_argument(
    "--task_suite",
    type=str,
    nargs="+",
    default=None,
    help="Task suite(s), e.g., libero_goal libero_10 libero_spatial libero_object.",
)
parser.add_argument("--task_id", type=int, nargs="+", default=0, help="Task ID.")
parser.add_argument("--demo_id", type=int, default=None, help="Replay a single demo id.")
parser.add_argument(
    "--contact_sensor_name",
    type=str,
    default="contact_gripper",
    help="Contact sensor name in env.scene.",
)
parser.add_argument(
    "--prim_regex_hint",
    type=str,
    default="gelpad",
    help="Substring/regex hint to locate USD prims (e.g., gelpad).",
)
parser.add_argument(
    "--check_only",
    action="store_true",
    default=False,
    help="Only run checks without replaying actions.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Max steps to replay per episode (0 = full length).",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=50,
    help="Print force stats every N steps during replay.",
)
parser.add_argument(
    "--strict",
    action="store_true",
    default=False,
    help="Exit if contact sensor is missing or has no bound bodies.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# Append Isaac Sim launcher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Normalize task_id list
if isinstance(args_cli.task_id, list) and len(args_cli.task_id) == 1:
    args_cli.task_id = int(args_cli.task_id[0])

# Normalize to a single suite for single-run mode.
if args_cli.task_suite is not None and isinstance(args_cli.task_suite, list):
    args_cli.task_suite = args_cli.task_suite[0]

# Always enable cameras for Camera envs.
args_cli.enable_cameras = True
if args_cli.task is not None and "Camera" in args_cli.task:
    args_cli.enable_cameras = True

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.task_suite is not None:
    from tac_manip.utils.task_configs import setup_task_objects
    from common.replay_utils import resolve_input_hdf5

    setup_task_objects(args_cli.task_suite, args_cli.task_id)
    if args_cli.dataset_file == "datasets/dataset.hdf5":
        prefer = "auto" if args_cli.task_suite.startswith("libero") else "replayed"
        dataset_file, desc = resolve_input_hdf5(
            task=args_cli.task,
            task_suite=args_cli.task_suite,
            task_id=args_cli.task_id,
            prefer=prefer,
        )
        args_cli.dataset_file = dataset_file
        print(f"📊 {desc}")

if args_cli.dataset_file == "datasets/dataset.hdf5":
    raise ValueError(
        "Input HDF5 is not set.\n"
        "Set --dataset_file, or export proper env vars (REPLAYED_DEMOS_DIR / HDF5_TRAJ_SOURCE_DIR).\n"
    )


import gymnasium as gym
import torch
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from common.replay_utils import (
    convert_action_axisangle_to_quat,
    get_episode_map,
)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _tensor_stats(name: str, t: Any) -> str:
    if not isinstance(t, torch.Tensor):
        return f"{name}: None"
    if t.numel() == 0:
        return f"{name}: shape={tuple(t.shape)}, empty"
    t_abs = torch.abs(t)
    max_abs = float(t_abs.max().item())
    mean_abs = float(t_abs.mean().item())
    return f"{name}: shape={tuple(t.shape)}, max|x|={max_abs:.6f}, mean|x|={mean_abs:.6f}"


def _maybe_list_body_names(sensor: Any) -> list[str]:
    names: list[str] = []
    try:
        view = _safe_getattr(sensor, "body_physx_view", None)
        if view is not None and hasattr(view, "get_body_names"):
            names = list(view.get_body_names())
            return names
    except Exception:
        pass

    for key in ("body_names", "_body_names", "_body_paths", "body_paths"):
        val = _safe_getattr(sensor, key, None)
        if isinstance(val, (list, tuple)) and val:
            names = [str(x) for x in val]
            return names
    return names


def _check_stage_prims(prim_regex_hint: str, sensor_prim_expr: str | None) -> dict[str, list[str]]:
    results = {
        "hint_matches": [],
        "sensor_expr_matches": [],
        "rigid_matches": [],
        "collision_matches": [],
    }
    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[Stage] No USD stage available.")
            return results

        hint_re = re.compile(prim_regex_hint)
        expr_re = None
        if sensor_prim_expr:
            expr_pat = sensor_prim_expr.replace("{ENV_REGEX_NS}", ".+")
            try:
                expr_re = re.compile(expr_pat)
            except re.error:
                expr_re = None

        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if hint_re.search(path):
                results["hint_matches"].append(path)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    results["rigid_matches"].append(path)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    results["collision_matches"].append(path)
            if expr_re is not None and expr_re.match(path):
                results["sensor_expr_matches"].append(path)
    except Exception as exc:
        print(f"[Stage] Failed to traverse USD stage: {exc}")
    return results


def _print_contact_sensor_checks(env, env_cfg, contact_sensor_name: str, prim_regex_hint: str) -> bool:
    print("\n=== Contact Sensor Checks ===")
    print(f"- contact_sensor_name: {contact_sensor_name}")

    scene_keys = []
    try:
        scene_keys = list(env.scene.keys())
    except Exception:
        pass
    if scene_keys:
        print(f"- env.scene keys: {scene_keys}")

    sensor = None
    try:
        if contact_sensor_name in env.scene.keys():
            sensor = env.scene[contact_sensor_name]
    except Exception:
        sensor = None

    if sensor is None:
        print(f"❌ Sensor '{contact_sensor_name}' not found in env.scene.")
        return False

    print(f"✅ Sensor '{contact_sensor_name}' found: {type(sensor)}")

    cfg = _safe_getattr(sensor, "cfg", None)
    if cfg is not None:
        print(f"- cfg.prim_path: {_safe_getattr(cfg, 'prim_path', None)}")
        print(f"- cfg.update_period: {_safe_getattr(cfg, 'update_period', None)}")
        print(f"- cfg.history_length: {_safe_getattr(cfg, 'history_length', None)}")
        print(f"- cfg.debug_vis: {_safe_getattr(cfg, 'debug_vis', None)}")
        print(f"- cfg.visualize_triaxial_forces: {_safe_getattr(cfg, 'visualize_triaxial_forces', None)}")
        print(f"- cfg.track_pose: {_safe_getattr(cfg, 'track_pose', None)}")

    sensor_data = _safe_getattr(sensor, "data", None)
    if sensor_data is not None:
        print(_tensor_stats("data.net_forces_w", _safe_getattr(sensor_data, "net_forces_w", None)))
        print(_tensor_stats("data.net_forces_w_history", _safe_getattr(sensor_data, "net_forces_w_history", None)))
        print(_tensor_stats("data.pos_w", _safe_getattr(sensor_data, "pos_w", None)))
        print(_tensor_stats("data.quat_w", _safe_getattr(sensor_data, "quat_w", None)))

    body_names = _maybe_list_body_names(sensor)
    if body_names:
        print(f"- bound bodies ({len(body_names)}): {body_names}")
    else:
        print("⚠️  No bound body names found (sensor may be unbound or introspection failed).")

    # Try to infer USD prims for gelpads/fingers.
    prim_expr = _safe_getattr(cfg, "prim_path", None) if cfg is not None else None
    stage_hits = _check_stage_prims(prim_regex_hint, prim_expr)
    if stage_hits["hint_matches"]:
        print(f"- USD prims matching hint '{prim_regex_hint}': {stage_hits['hint_matches']}")
        if stage_hits["rigid_matches"]:
            print(f"- RigidBody prims: {stage_hits['rigid_matches']}")
        if stage_hits["collision_matches"]:
            print(f"- Collision prims: {stage_hits['collision_matches']}")
    else:
        print(f"⚠️  No USD prims matched hint '{prim_regex_hint}'.")

    if stage_hits["sensor_expr_matches"]:
        print(f"- USD prims matching sensor expr: {stage_hits['sensor_expr_matches']}")
    elif prim_expr:
        print("⚠️  No USD prims matched sensor prim_path expression.")

    # Print sim physx config hints
    physx = _safe_getattr(_safe_getattr(env_cfg, "sim", None), "physx", None)
    if physx is not None:
        print(f"- sim.physx.enable_ccd: {_safe_getattr(physx, 'enable_ccd', None)}")
        print(f"- sim.physx.solver_position_iteration_count: {_safe_getattr(physx, 'solver_position_iteration_count', None)}")
        print(f"- sim.physx.solver_velocity_iteration_count: {_safe_getattr(physx, 'solver_velocity_iteration_count', None)}")

    # Print robot USD path if available
    robot_cfg = _safe_getattr(_safe_getattr(env_cfg, "scene", None), "robot", None)
    if robot_cfg is not None:
        usd_path = _safe_getattr(_safe_getattr(robot_cfg, "spawn", None), "usd_path", None)
        if usd_path:
            print(f"- robot USD: {usd_path}")

    return bool(body_names)


def _update_force_stats(stats: dict, sensor, obs_policy: dict | None):
    stats["steps"] += 1

    net_w = None
    if sensor is not None:
        data = _safe_getattr(sensor, "data", None)
        net_w = _safe_getattr(data, "net_forces_w", None) if data is not None else None
    obs_force = obs_policy.get("gripper_net_force", None) if obs_policy else None

    if isinstance(net_w, torch.Tensor) and net_w.numel() > 0:
        max_abs = float(torch.abs(net_w).max().item())
        stats["sensor_max_abs"] = max(stats["sensor_max_abs"], max_abs)
        if max_abs > stats["eps"]:
            stats["sensor_nonzero_steps"] += 1

    if isinstance(obs_force, torch.Tensor) and obs_force.numel() > 0:
        max_abs = float(torch.abs(obs_force).max().item())
        stats["obs_max_abs"] = max(stats["obs_max_abs"], max_abs)
        if max_abs > stats["eps"]:
            stats["obs_nonzero_steps"] += 1


def main():  # noqa: C901
    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"Dataset file does not exist: {args_cli.dataset_file}")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()
    if episode_count == 0:
        raise ValueError("No episodes found in the dataset.")

    # Determine replay episode list
    episode_map = get_episode_map(dataset_file_handler.get_episode_names())
    episode_indices_to_replay = [args_cli.demo_id] if args_cli.demo_id is not None else []
    if not episode_indices_to_replay:
        episode_indices_to_replay = sorted(episode_map.keys())
        print(f"Found {len(episode_indices_to_replay)} episodes: {episode_indices_to_replay}")
    else:
        print(f"Selected demo_id: {episode_indices_to_replay[0]}")

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name is missing.")

    # Parse env cfg
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.sim.physx.enable_ccd = True
    env_cfg.observations.policy.concatenate_terms = False

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Run checks before replay
    sensor_bound = _print_contact_sensor_checks(
        env, env_cfg, args_cli.contact_sensor_name, args_cli.prim_regex_hint
    )
    if args_cli.strict and not sensor_bound:
        raise SystemExit("Contact sensor missing or no bound bodies; exiting due to --strict.")

    if args_cli.check_only:
        print("\n[CheckOnly] Done.")
        return

    # Prepare idle action
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(args_cli.num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)
        action_dim = env.action_space.shape[-1]
        is_eef_space = args_cli.task is not None and ("IK" in args_cli.task or "RmpFlow" in args_cli.task)
        if is_eef_space and action_dim >= 8:
            idle_action[..., 3] = 1.0
            idle_action[..., 4:7] = 0.0

    # Reset
    env.reset()

    # Replay (single env recommended)
    stats = {
        "steps": 0,
        "sensor_nonzero_steps": 0,
        "obs_nonzero_steps": 0,
        "sensor_max_abs": 0.0,
        "obs_max_abs": 0.0,
        "eps": 1e-8,
    }

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        for episode_id in episode_indices_to_replay:
            print(f"\n=== Replaying Episode {episode_id} ===")
            episode_data: EpisodeData = dataset_file_handler.load_episode(
                episode_map[episode_id], env.device
            )

            actions_tensor = episode_data.data.get("actions", None)
            if actions_tensor is None or not isinstance(actions_tensor, torch.Tensor):
                raise ValueError("Episode actions missing or invalid.")

            # IK env expects 8D (pos+quat+gripper); dataset may be 7D (pos+axisangle+gripper).
            if (
                args_cli.task
                and "IK" in args_cli.task
                and actions_tensor.ndim == 2
                and actions_tensor.shape[1] == 7
                and env.action_space.shape[-1] == 8
            ):
                converted = []
                for a in actions_tensor:
                    converted.append(convert_action_axisangle_to_quat(a))
                actions_tensor = torch.stack(converted, dim=0)

            # Hybrid env may expect 13D; if only 7D, try to augment from obs/gripper_net_force.
            if (
                args_cli.task
                and "Isaac-Libero-Franka-Hybrid-" in args_cli.task
                and actions_tensor.ndim == 2
                and actions_tensor.shape[1] == 7
                and env.action_space.shape[-1] == 13
            ):
                inst_force = None
                obs_group = episode_data.data.get("obs", None)
                if isinstance(obs_group, dict):
                    gnf = obs_group.get("gripper_net_force", None)
                    if isinstance(gnf, torch.Tensor) and gnf.ndim == 4:
                        T, _, _, _ = gnf.shape
                        inst_force = gnf[:, 0, :, :].reshape(T, 6).to(
                            device=actions_tensor.device, dtype=actions_tensor.dtype
                        )
                if inst_force is None:
                    inst_force = torch.zeros(
                        (actions_tensor.shape[0], 6),
                        device=actions_tensor.device,
                        dtype=actions_tensor.dtype,
                    )
                    print("⚠️  Hybrid actions are 7D; using zero forces to expand to 13D.")
                actions_tensor = torch.cat([actions_tensor, inst_force], dim=1)

            # Apply initial state if available
            if "initial_state" in episode_data.data:
                initial_state = episode_data.get_initial_state()
                env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)

            # Replay
            steps = actions_tensor.shape[0]
            if args_cli.max_steps > 0:
                steps = min(steps, args_cli.max_steps)

            for i in range(steps):
                action = actions_tensor[i]
                actions = idle_action.clone()
                actions[0] = action
                env.step(actions)

                obs_policy = None
                try:
                    obs_policy = env.obs_buf.get("policy", {})
                except Exception:
                    obs_policy = None

                sensor = None
                try:
                    sensor = env.scene[args_cli.contact_sensor_name]
                except Exception:
                    sensor = None

                _update_force_stats(stats, sensor, obs_policy)

                if args_cli.print_every > 0 and (i + 1) % args_cli.print_every == 0:
                    print(
                        f"[Step {i+1:04d}] "
                        f"sensor_max_abs={stats['sensor_max_abs']:.6f}, "
                        f"obs_max_abs={stats['obs_max_abs']:.6f}, "
                        f"sensor_nonzero_steps={stats['sensor_nonzero_steps']}, "
                        f"obs_nonzero_steps={stats['obs_nonzero_steps']}"
                    )

            print(
                f"[Episode {episode_id}] steps={stats['steps']}, "
                f"sensor_nonzero_steps={stats['sensor_nonzero_steps']}, "
                f"obs_nonzero_steps={stats['obs_nonzero_steps']}, "
                f"sensor_max_abs={stats['sensor_max_abs']:.6f}, "
                f"obs_max_abs={stats['obs_max_abs']:.6f}"
            )

    print("\n=== Summary ===")
    print(
        f"Total steps={stats['steps']}, "
        f"sensor_nonzero_steps={stats['sensor_nonzero_steps']}, "
        f"obs_nonzero_steps={stats['obs_nonzero_steps']}, "
        f"sensor_max_abs={stats['sensor_max_abs']:.6f}, "
        f"obs_max_abs={stats['obs_max_abs']:.6f}"
    )


if __name__ == "__main__":
    main()
