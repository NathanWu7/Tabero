# Isaac-Libero Workflow Guide

English | [中文](LIBERO_WORKFLOW.md)

This guide covers the standard Isaac-Libero workflow in this repository: standard Franka environments, standard LIBERO data, 7D task-space states/actions, and OpenPI inference with `diffik` as the recommended default.

Before using this guide, finish the root [README](../README.md) `Quick Setup`: install the TacManip extension, install the OpenPI client, link `NathanWu7/Isaaclab_Libero`, and link tactile calibration assets if needed.

## 1. Understand the Main Environments

The Isaac-Libero workflow mainly uses three environments:

- `Isaac-Libero-Franka-Replay-Camera-v0`
  - Replays existing LIBERO trajectories in Isaac and exports new HDF5 files with camera observations.
- `Isaac-Libero-Franka-IK-v0`
  - Standard task-space control environment. This is the recommended default for OpenPI inference.
- `Isaac-Libero-Franka-OscPose-v0`
  - Optional task-space control environment. Use it only when you explicitly want to test OSC control.

The recommended order is to use the downloaded data directly for training, inference, and evaluation first. Recollect replay data only when you need your own replay outputs.

## 2. Set Isaac-Libero Environment Variables

This section assumes all data downloads and symlinks have already been configured through the root README.

**Use downloaded data for training, inference, and evaluation.** If you use the downloaded `assembled_hdf5/`, `replayed_demos/`, and `video_datasets/`, replay recollection is usually unnecessary. Run:

```bash
source scripts/tools/set_replay_env.sh inference
```

This profile points `HDF5_TRAJ_SOURCE_DIR` to the default `benchmarks/datasets/libero/assembled_hdf5` path and clears replay output variables. Use it for:

- Converting or training with existing Isaac-Libero data
- OpenPI inference
- Batch evaluation

Verify the path:

```bash
echo "$HDF5_TRAJ_SOURCE_DIR"
```

You should see a path similar to:

```text
.../benchmarks/datasets/libero/assembled_hdf5
```

The directory should contain files similar to:

```text
libero_goal_task1_..._demo.hdf5
libero_10_task0_..._demo.hdf5
```

**Recollect your own data.** If you want to replay and recollect your own Isaac-Libero data, run:

```bash
source scripts/tools/set_replay_env.sh libero
```

This profile points the following variables to the default `benchmarks/datasets/libero` layout:

- `HDF5_TRAJ_SOURCE_DIR`
- `OUTPUT_REPLAYED_DEMOS_DIR`
- `OUTPUT_REPLAYED_VIDEOS_DIR`
- `REPLAYED_DEMOS_DIR`

If you do not want to overwrite the downloaded `replayed_demos/` and `video_datasets/`, manually set separate output directories before recollection. The recollection workflow is intentionally placed later in this guide.

### Optional: Reset-Time Light Randomization

By default, Libero environments keep deterministic lighting for reproducibility. To evaluate or recollect under varied lighting, add `--randomize_light` to replay, OpenPI inference, or batch evaluation commands:

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --randomize_light \
  --headless
```

Runtime flow: the script sets `LIBERO_RANDOMIZE_LIGHT=1`, `setup_task_objects()` sets `TASK_SUITE` and `TASK_ID`, then `parse_env_cfg()` instantiates the Libero cfg. `EventCfgFrankaPanda` registers `randomize_light` only when the flag is enabled, and the reset event randomizes `/World/light` DomeLight intensity, color, and HDR texture on every `env.reset()`. Contact-force and tactile Libero environments inherit the same event configuration from the base Franka Libero cfg.

## 3. Convert Directly to LeRobot / OpenPI Training Format

Run the LeRobot/OpenPI conversion in the isolated `tabero_lerobot` environment, not in the Isaac runtime environment. `lerobot` pulls dependencies that can conflict with Isaac Sim / Isaac Lab pins, so do not install it back into the Isaac runtime environment.

### 3.1 Configure the `tabero_lerobot` Environment

If `tabero_lerobot` does not exist on the machine yet, create it from the exported environment file from the repository root:

```bash
conda env create -f envs/environment-tabero-lerobot.yml
conda activate tabero_lerobot
```

If the environment already exists but is missing conversion dependencies such as `lerobot` or `tyro`, repair it with the frozen requirements file:

```bash
conda activate tabero_lerobot
python -m pip install -r envs/requirements-tabero-lerobot.txt
```

After creating or repairing the environment, run the minimal checks:

```bash
python -c "import lerobot, tyro; print('lerobot/tyro ok')"
python -m pip check
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py --help
```

This environment is only for LeRobot conversion and related upload/checking tools. It does not need Isaac Sim / Isaac Lab and should not be used to start simulation.

### 3.2 Run the Conversion

Run the command from the `Tabero` repository root:

```bash
conda activate tabero_lerobot
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --data_root benchmarks/datasets/libero \
  --output_dir /tmp/tabero_lerobot_openpi
```

If you prefer `conda run`, use:

```bash
conda run --no-capture-output -n tabero_lerobot python \
  benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --data_root benchmarks/datasets/libero \
  --output_dir /tmp/tabero_lerobot_openpi
```

The converter reads `replayed_demos/` and `video_datasets/` under `--data_root`. It does not read `HDF5_TRAJ_SOURCE_DIR` directly, and `assembled_hdf5/` is used for replay input or inference initial states, not as the direct LeRobot conversion input.

The minimal usable input is:

- `benchmarks/datasets/libero/replayed_demos`
- `benchmarks/datasets/libero/video_datasets`

If those directories are symlinks, use `find -L` when checking file counts:

```bash
find -L benchmarks/datasets/libero/replayed_demos -maxdepth 1 -name '*.hdf5' | wc -l
find -L benchmarks/datasets/libero/video_datasets -maxdepth 2 -name '*.mp4' | wc -l
```

The conversion script produces:

- 7D `state`
- 7D `action`
- Camera image sequences

For a quick validation pass, convert only a small suite or a few tasks first.

Common environment errors:

- `ModuleNotFoundError: No module named 'lerobot'`: the command is running in the wrong environment; switch to `tabero_lerobot`.
- `ModuleNotFoundError: No module named 'tyro'`: the conversion environment is incomplete; verify it with `python -c "import lerobot, tyro"`.

## 4. Run OpenPI Inference Directly

`openpi_inference_client.py` is the repository-side client. The actual model inference service must be started from the OpenPI service repository.

For Tabero, the modified OpenPI service is maintained in [`NathanWu7/T2-VLA`](https://github.com/NathanWu7/T2-VLA). That repository provides the model training/inference service side; this repository provides the Isaac Lab environments and the Isaac-side client at `benchmarks/openpi/openpi_inference_client.py`. The corresponding weights are available at [`NathanWu7/pi0_lora_tacfield_tabero`](https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero).

Start the service from the T2-VLA repository with this template:

```bash
cd /path/to/T2-VLA
uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config <config_name> \
  --policy.dir /path/to/checkpoint
```

Make sure the server-side `--policy.config` / checkpoint matches the TacManip client `--control_mode`. For example, tacfield / tactile models usually require the client to use `--control_mode tactile`; the `diffik` example below is the standard Isaac-Libero visual-policy smoke-test path.

Default server settings are:

```text
server_host = 127.0.1.1
server_port = 8000
```

If your server uses another address, pass it explicitly in the inference command.

### 4.1 Recommended First Run: `diffik`

`diffik` is a visual-only / 7D-action smoke test. It does not send tactile fields to the OpenPI server, so do not use tactile checkpoints such as `pi0_lora_tacfield_tabero` for this path. Use the no-tactile checkpoint [`NathanWu7/pi0_lora_notac_tabero`](https://huggingface.co/NathanWu7/pi0_lora_notac_tabero).

Download only the checkpoint files needed for serving:

```bash
hf download NathanWu7/pi0_lora_notac_tabero \
  --local-dir /path/to/models/pi0_lora_notac_tabero \
  --include 'checkpoints/pi0_lora_notac_tabero/pi0_lora_notac_tabero/49999/params/**' \
  --include 'checkpoints/pi0_lora_notac_tabero/pi0_lora_notac_tabero/49999/assets/**' \
  --include 'norm_stats/**'
```

Start the no-tactile OpenPI service from the T2-VLA repository:

```bash
cd /path/to/T2-VLA

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi0_lora_notac_tabero \
  --policy.dir=/path/to/models/pi0_lora_notac_tabero/checkpoints/pi0_lora_notac_tabero/pi0_lora_notac_tabero/49999
```

If port `8000` is already used by another OpenPI service, start this server on `8001` and pass `--server_port 8001` to the client command.

Then run the `diffik` experiment from the Tabero repository:

```bash
source scripts/tools/set_replay_env.sh inference

python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0 \
  --server_host 127.0.1.1 \
  --server_port 8000
```

This uses:

- `Isaac-Libero-Franka-IK-v0`
- the `pi0_lora_notac_tabero` server-side model
- initial states from `libero_goal_task1_put_the_bowl_on_the_stove_demo.hdf5`

With `debug_mode=0`, the command does not write debug images or action dumps by default; inspect stdout. A completed run prints:

- `Found HDF5 file: ...`
- `[Prompt] put the bowl on the stove`
- the per-experiment result, such as `✓ Success` or `✗ Failed (max steps)`
- `Evaluation Results`, including `Total experiments`, `Successful experiments`, and `Success rate`

### 4.2 Optional: Use `osc`

`osc` is also a visual-only / 7D-action path and uses the same `pi0_lora_notac_tabero` OpenPI service as `diffik`; no extra model is needed. The Isaac-side environment is different: `osc` uses `Isaac-Libero-Franka-OscPose-v0`, and actions are sent directly to the OSC environment as 7D `(x, y, z, rx, ry, rz, gripper)` commands.

If the no-tactile OpenPI service from 4.1 is still running, reuse it. Otherwise, start `pi0_lora_notac_tabero` with the server command from 4.1.

Then run the OSC experiment:

```bash
source scripts/tools/set_replay_env.sh inference

python benchmarks/openpi/openpi_inference_client.py \
  --control_mode osc \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0 \
  --server_host 127.0.1.1 \
  --server_port 8000
```

This uses:

- `Isaac-Libero-Franka-OscPose-v0`
- the `pi0_lora_notac_tabero` server-side model
- initial states from `libero_goal_task1_put_the_bowl_on_the_stove_demo.hdf5`

A completed run prints the per-experiment result and `Evaluation Results` to stdout. For a one-run smoke test, `✗ Failed (max steps)` means the workflow completed but that particular episode did not succeed; it does not indicate a client/server connection failure.

### 4.3 Core Inputs Sent by the Client

For Isaac-Libero inference, the client sends these core fields to OpenPI:

- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `prompt`

Here:

- `observation/state` is a 7D task-space state.
- `prompt` comes from the language instruction in the task config.

## 5. Run Batch Testing / Evaluation Directly

After single-task inference works, run batch evaluation.

### 5.1 Evaluate One Task

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --task_ids 1 \
  --num_total_experiments 5 \
  --headless
```

### 5.2 Evaluate One Suite

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --num_total_experiments 5 \
  --headless
```

### 5.3 Evaluate Multiple Suites

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal libero_10 libero_spatial libero_object \
  --num_total_experiments 5 \
  --headless
```

The evaluation script starts `benchmarks/openpi/openpi_inference_client.py` for each task, computes success rates, and writes results to `evaluation_results/`.

## 6. Isaac-Libero Data Format

For the Isaac-Libero 7D workflow, the important fields are:

- `data/demo_<k>/actions`
- `data/demo_<k>/obs/eef_pose`
- `data/demo_<k>/obs/gripper_pos`

Typical meanings:

- `eef_pose` is usually shaped `(T, 7)` and stores `pos(3) + quat(4)`.
- Conversion scripts normalize orientation into axis-angle.
- Final states and actions are organized as 7D vectors: `[x, y, z, ax, ay, az, gripper]`.

## 7. Recollect Isaac-Libero Data Only When Needed

If you use the downloaded data directly for training, inference, and evaluation, skip this section. Recollection is useful when:

- You want to regenerate videos.
- You want to create your own `replayed_demos`.
- You want to validate the replay pipeline.

### 7.1 Recommended Method: Replay Existing LIBERO Trajectories

For standard Isaac-Libero 7D collection, use:

- Environment: `Isaac-Libero-Franka-Replay-Camera-v0`
- `recorder_type`: `7dp`

`7dp` means `position(3) + axis-angle(3) + abs gripper(1)`, together forming a 7D action.

### 7.2 Set a Separate Output Directory First

If you do not want to overwrite downloaded data, manually set output directories that are separate from the default symlinked dataset directory:

```bash
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/libero_replay/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/libero_replay/video_datasets
export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"
```

A recommended layout is:

```text
/path/to/libero_replay/
  replayed_demos/
  video_datasets/
```

### 7.3 Collect One Task

```bash
python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-v0 \
  --task_suite libero_10 \
  --task_id 0 \
  --num_envs 1 \
  --video \
  --recorder_type 7dp \
  --dump_data
```

This command:

- Finds the HDF5 demo for `libero_10` task `0` from `HDF5_TRAJ_SOURCE_DIR`.
- Replays the trajectory in Isaac.
- Writes new HDF5 output to `OUTPUT_REPLAYED_DEMOS_DIR`.
- Writes camera videos to `OUTPUT_REPLAYED_VIDEOS_DIR`.

### 7.4 Collect an Entire Suite

```bash
python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-v0 \
  --task_suite libero_goal \
  --num_envs 1 \
  --video \
  --recorder_type 7dp \
  --dump_data
```

The script will iterate over all tasks in the suite.

### 7.5 Manual Teleoperation Recording

If you want to record your own data instead of replaying open-source trajectories, use:

```bash
python scripts/tools/record_demos.py \
  --task Isaac-Libero-Franka-IK-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --teleop_device spacemouse \
  --num_demos 5 \
  --dataset_file ./output/manual_demo.hdf5
```

This path is useful for recording a few manual demonstrations, validating the environment/controller, or adding small amounts of extra data.

## 8. Recommended End-to-End Flow

If you use the downloaded Isaac-Libero data, the recommended order is:

1. Finish data download and symlink setup in the root README.
2. Run `source scripts/tools/set_replay_env.sh inference`.
3. Convert directly to LeRobot/OpenPI training format.
4. Start the OpenPI server.
5. Run single-task `diffik` inference.
6. Run batch evaluation.

If you need to recollect data, use `source scripts/tools/set_replay_env.sh libero` in step 2 and follow Section 7 for separate output directories and replay commands.

## 9. FAQ

### 9.1 Why Did Inference Not Reset from the Dataset Initial State?

Usually `HDF5_TRAJ_SOURCE_DIR` is not set correctly, or the HDF5 file for the task does not exist.

Check:

```bash
echo "$HDF5_TRAJ_SOURCE_DIR"
```

Then check whether the directory contains files like:

```text
libero_goal_task1_..._demo.hdf5
```

### 9.2 Why Did the Conversion Script Skip Some Trajectories?

Check whether `actions` are standard 7D or compatible 8D actions. This Isaac-Libero conversion script targets standard 7D/8D task-space actions.

The most stable collection setup is:

- `Isaac-Libero-Franka-Replay-Camera-v0`
- `--recorder_type 7dp`

### 9.3 Why Did Batch Evaluation Run Many Tasks?

`run_task_evaluations.py` evaluates all available tasks in the selected suites by default. To narrow the scope, pass:

- `--task_suites`
- `--task_ids`
