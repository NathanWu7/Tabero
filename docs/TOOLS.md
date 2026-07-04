# Tools (Collection / Evaluation / Visualization)

English | [中文](TOOLS.zh-CN.md)

This page is the central reference for scripts under `scripts/tools/`. The root [`README.md`](../README.md) is only a high-level overview; this document explains what each tool does and how to run it.

## Common Conventions

### 1. Path And Environment Variables

- **Input: source trajectories**
  - `HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5`
- **Output: replay recollection**
  - `OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos`
  - `OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets`
- **Input: replayed demos for replay/evaluation**
  - `REPLAYED_DEMOS_DIR=/path/to/replayed_demos`
  - Usually: `export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"`
- **Input: manually recorded demos**
  - Single file: `RECORDED_DEMOS_PATH=/path/to/single_recorded_demo.hdf5`
  - Directory mode: `RECORDED_DEMOS_DIR=/path/to/recorded_demos`

Keep input directories and output directories separate.

### 2. Batch Execution

- `replay_demos_with_camera.py` / `replay_demos.py`: if you pass multiple `--task_suite` values, or pass `--task_suite` without `--task_id`, the script launches child processes for each `(suite, task_id)` pair. This avoids repeatedly rebuilding Isaac/Kit environments in one process.
- `run_data_evaluations.py`: launches `replay_demos.py` for each task, parses stdout, and aggregates success/metric results until `max_episodes` is reached.
- `run_task_evaluations.py`: launches the OpenPI or other policy inference client for each task and parses stdout for success rate and force metrics.

### 3. Optional Libero Light Randomization

Libero DomeLight randomization is disabled by default. Add `--randomize_light` to replay or evaluation commands when you want reset-time randomization of intensity, color, and HDR sky texture.

```bash
python scripts/tools/replay_demos.py \
  --task Isaac-Libero-Franka-Replay-Camera-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --randomize_light \
  --headless
```

The CLI flag sets `LIBERO_RANDOMIZE_LIGHT=1` before the environment cfg is parsed. `EventCfgFrankaPanda` then registers the `randomize_light` reset event; tactile and contact-force Libero environments inherit the same event wiring from the base Franka Libero cfg.

Batch wrappers also accept the same flag and pass it to child processes:

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode tactile \
  --task_suites libero_goal \
  --task_ids 1 \
  --randomize_light \
  --headless
```

### 4. Short Commands Versus Full Commands

- **Short commands** use minimal arguments and rely on the current shell environment/defaults.
- **Full commands** spell out important paths and common options, which is better for reproducibility and team use.

## 1. Data Collection

### `set_replay_env.sh`

Sets profile-based environment variables in the current shell.

Short command:

```bash
source scripts/tools/set_replay_env.sh tabero_force_gentle
```

Full/manual exports:

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets
export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"
export USE_TABERO_TASKS=0
```

### `replay_demos_with_camera.py`

Reads source trajectories, replays them in Isaac, and writes:

- `replayed_demos/*.hdf5` for successful episodes
- optional camera videos under `video_datasets/.../videos/*.mp4`
- optional tactile videos under `video_datasets/.../tactile_outputs/*.mp4`

`--recorder_type 7dpf` writes Force(6) into actions, producing 13D actions.

Short command:

```bash
python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
  --task_suite libero_goal \
  --task_id 2 \
  --dump_data \
  --recorder_type 7dpf \
  --video \
  --headless
```

Full command:

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets

python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
  --task_suite libero_goal \
  --task_id 2 \
  --num_envs 1 \
  --video \
  --camera_view_list agentview eye_in_hand \
  --tactile_sensor_list gsmini_left gsmini_right \
  --tactile_output_type tactile_rgb \
  --recorder_type 7dpf \
  --dump_data \
  --headless
```

### `record_demos.py`

Records manual demos with keyboard or SpaceMouse and writes one HDF5 file.

For LIBERO tasks with `--task_suite` / `--task_id`, set `HDF5_TRAJ_SOURCE_DIR` first. SpaceMouse also requires the device to be connected; use `--teleop_device keyboard` if no SpaceMouse is available.

Short command:

```bash
source scripts/tools/set_replay_env.sh inference
python scripts/tools/record_demos.py \
  --task Isaac-Libero-Franka-IK-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --teleop_device keyboard \
  --num_demos 1 \
  --dataset_file ./output/manual_demo.hdf5
```

Full command:

```bash
source scripts/tools/set_replay_env.sh inference

python scripts/tools/record_demos.py \
  --task Isaac-Libero-Franka-IK-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --teleop_device spacemouse \
  --step_hz 30 \
  --num_demos 5 \
  --num_success_steps 10 \
  --recorder_type 8d2 \
  --dataset_file ./output/manual_demo.hdf5
```

## 2. Data Evaluation

### `replay_demos.py`

Lightweight replay/validation for an HDF5 demo. Use it to:

- verify that a demo can replay correctly
- debug headless/camera/force observations
- optionally run `--validate_states` with `--num_envs 1`

Short command:

```bash
python scripts/tools/replay_demos.py --task Isaac-Libero-Franka-Replay-Camera-v0 --dataset_file /path/to/demo.hdf5 --demo_id 0
```

Full command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos

python scripts/tools/replay_demos.py \
  --task Isaac-Libero-Franka-Replay-Camera-ContactForce-v0 \
  --task_suite libero_goal \
  --task_id 5 \
  --num_envs 1 \
  --validate_states \
  --dump_data \
  --headless
```

### `run_data_evaluations.py`

Batch-evaluates replayed data quality from `REPLAYED_DEMOS_DIR`.

It resolves each task HDF5 from `REPLAYED_DEMOS_DIR`, launches `replay_demos.py` as a child process, parses stdout for success/Hybrid metrics, and writes JSON/TXT summaries.

Short command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/run_data_evaluations.py --control_mode tactile --headless
```

Full command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos

python scripts/tools/run_data_evaluations.py \
  --task_suites libero_10 libero_spatial libero_object libero_goal \
  --task_ids 0 1 2 \
  --control_mode tactile \
  --max_episodes 50 \
  --num_envs 1 \
  --output_dir ./evaluation_results \
  --output_format both \
  --replay_script scripts/tools/replay_demos.py \
  --headless
```

### `run_task_evaluations.py`

Runs policy inference evaluation for each task, usually through the OpenPI client, then parses success rate and Hybrid force metrics from stdout.

Short command:

```bash
python scripts/tools/run_task_evaluations.py --policy_model openpi --control_mode diffik --task_suites libero_goal --task_ids 1 --num_total_experiments 5 --headless
```

Full command:

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode tactile \
  --server_host 127.0.1.1 \
  --server_port 8000 \
  --task_suites libero_goal libero_10 \
  --task_ids 0 1 2 \
  --num_total_experiments 50 \
  --num_success_steps 8 \
  --max_inference_steps 80 \
  --replan_steps 10 \
  --hdf5_folder /path/to/libero/assembled_hdf5 \
  --debug_mode 0 \
  --output_dir ./evaluation_results \
  --output_format both \
  --headless
```

### `raw_data_retention_analysis.py`

Counts successful demos under each HDF5 in `REPLAYED_DEMOS_DIR` and compares the count with the expected number of episodes.

Short command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/raw_data_retention_analysis.py
```

Full command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos

python scripts/tools/raw_data_retention_analysis.py \
  --task_suites libero_10 libero_spatial libero_goal libero_object \
  --expected_episodes_per_file 50 \
  --output_dir ./evaluation_results
```

## 3. Visualization And Utilities

### `force_debug_playground.py`

Replays one demo and extracts local left/right finger forces from `obs["policy"]["gripper_net_force"]` as `(2, 3)` values for printing, plotting, or saving.

Use it with ContactForce / Tactile environments via `--env_variant`.

Short command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/force_debug_playground.py --env_variant contactforce --task_suite libero_goal --task_id 5 --demo_id 0 --save_plot ./force.png --headless
```

Full command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos

python scripts/tools/force_debug_playground.py \
  --env_variant tactile \
  --task_suite libero_goal \
  --task_id 5 \
  --demo_id 0 \
  --num_steps 300 \
  --print_every 5 \
  --save_npz ./force_series.npz \
  --save_plot ./force.png \
  --headless
```

### `upload_lerobot_to_hf.py`

Uploads a local LeRobot dataset to Hugging Face Hub. By default, it only uploads `data/` and `meta/`.

Short command:

```bash
python scripts/tools/upload_lerobot_to_hf.py --local-path /path/to/lerobot_dataset_root --repo-id your_username/your_dataset
```

Full command:

```bash
python scripts/tools/upload_lerobot_to_hf.py \
  --local-path /path/to/lerobot_dataset_root \
  --repo-id your_username/your_dataset \
  --repo-type dataset \
  --private
```

Omit `--private` for a public dataset. If the Hugging Face repository already exists and you do not want the script to create it, add `--no-create-repo`.
