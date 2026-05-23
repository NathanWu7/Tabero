# Tools Guide

English | [中文](TOOLS.zh-CN.md)

This document is the central reference for scripts under `scripts/tools/`. The root [`README.md`](../README.md) only lists the main workflow entry points; this page focuses on what each tool does and when to use it.

## Script Groups

- **Data collection**: replay open-source trajectories in Isaac, recollect camera/tactile data, or record manual teleoperation demos.
- **Data evaluation**: replay HDF5 demos, compute success rates, and run policy evaluation in batch.
- **Visualization and utilities**: inspect LeRobot datasets, debug force signals, and upload converted datasets.

## Environment Variables

Keep input and output directories separate:

- `HDF5_TRAJ_SOURCE_DIR`: source trajectories, usually LIBERO `assembled_hdf5`.
- `OUTPUT_REPLAYED_DEMOS_DIR`: HDF5 output written by replay recollection.
- `OUTPUT_REPLAYED_VIDEOS_DIR`: video output written by replay recollection.
- `REPLAYED_DEMOS_DIR`: replayed demos used by evaluation and debug tools.
- `RECORDED_DEMOS_PATH` / `RECORDED_DEMOS_DIR`: manual teleoperation data inputs.

## Main Tools

### `set_replay_env.sh`

Sets profile-based environment variables in the current shell:

```bash
source scripts/tools/set_replay_env.sh inference
```

Use manual exports when recollecting into a custom output directory:

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets
export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"
```

### `replay_demos_with_camera.py`

Replays source trajectories in Isaac and writes new successful demos, camera videos, and optional tactile videos:

```bash
python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
  --task_suite libero_goal \
  --task_id 2 \
  --num_envs 1 \
  --video \
  --recorder_type 7dpf \
  --dump_data \
  --headless
```

### `record_demos.py`

Records manual demos with keyboard or SpaceMouse:

```bash
python scripts/tools/record_demos.py \
  --task Isaac-Libero-Franka-IK-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --teleop_device spacemouse \
  --num_demos 5 \
  --dataset_file ./output/manual_demo.hdf5
```

### `replay_demos.py`

Lightweight replay and validation for existing HDF5 demos:

```bash
python scripts/tools/replay_demos.py \
  --task Isaac-Libero-Franka-Replay-Camera-v0 \
  --dataset_file /path/to/demo.hdf5 \
  --demo_id 0
```

### `run_data_evaluations.py`

Evaluates replayed data quality from `REPLAYED_DEMOS_DIR`:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/run_data_evaluations.py --control_mode tactile --headless
```

### `run_task_evaluations.py`

Runs policy inference evaluation, typically through the OpenPI client:

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --task_ids 1 \
  --num_total_experiments 5 \
  --headless
```

### Visualization and Utilities

Use these scripts for dataset inspection, force debugging, and publishing:

- `visualize_lerobot_dataset.py`
- `lerobot_viewer_ui.py`
- `force_debug_playground.py`
- `raw_data_retention_analysis.py`
- `upload_lerobot_to_hf.py`

For the full Chinese command reference, see [TOOLS.zh-CN.md](TOOLS.zh-CN.md).
