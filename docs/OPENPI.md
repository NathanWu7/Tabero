# OpenPI Inference Guide

English | [中文](OPENPI.zh-CN.md)

This guide explains how TacManip connects Isaac Lab environments to an external OpenPI model server for closed-loop inference evaluation.

OpenPI inference has two parts:

- **OpenPI server**: the model inference service, started outside this repository and exposed as `host:port`.
- **TacManip client**: `benchmarks/openpi/openpi_inference_client.py`, which starts Isaac Sim / Isaac Lab, collects observations, calls the server, and executes returned actions.

For Tabero, the recommended OpenPI-side service is the modified repository [`NathanWu7/T2-VLA`](https://github.com/NathanWu7/T2-VLA). [`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi) is the upstream reference, not the recommended service to run directly for Tabero.

## Quickstart

### 1. Prepare the TacManip-side environment

Run from the TacManip repository root with the Isaac Lab Python environment active:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -m pip install -e benchmarks/openpi/openpi-client
```

### 2. Prepare data

Set the LIBERO source trajectory directory if you want reproducible resets from dataset initial states:

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
# or:
source scripts/tools/set_replay_env.sh inference
```

The directory should contain task HDF5 files named like `{task_suite}_task{task_id}_*_demo.hdf5`. You can also pass `--hdf5-folder /path/to/...`, which updates `HDF5_TRAJ_SOURCE_DIR` for the client run.

### 3. Start the T2-VLA OpenPI service

Download the no-tactile model used by the `diffik` / `osc` smoke tests:

```bash
hf download NathanWu7/pi0_lora_notac_tabero \
  --local-dir /path/to/models/pi0_lora_notac_tabero \
  --include 'checkpoints/pi0_lora_notac_tabero/pi0_lora_notac_tabero/49999/params/**' \
  --include 'checkpoints/pi0_lora_notac_tabero/pi0_lora_notac_tabero/49999/assets/**' \
  --include 'norm_stats/**'
```

Start the service from the T2-VLA repository:

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

T2-VLA's `serve_policy.py` listens on `0.0.0.0`. The TacManip client defaults to:

```text
server_host = 127.0.1.1
server_port = 8000
```

If the server uses a different port, pass the same value with `--server_port`. If the server is on another machine, pass that machine's IP with `--server_host`.

### 4. Run one `diffik` inference experiment

```bash
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

Expected terminal output includes the prompt, one experiment result, and a summary with `Success rate`. A failed single smoke-test episode does not necessarily mean the client/server link is broken; first confirm that inference completes end to end.

## OpenPI Service And Model Selection

### Generic service command

From the T2-VLA repository, start any checkpoint with this template:

```bash
cd /path/to/T2-VLA

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=<config_name> \
  --policy.dir=/path/to/checkpoint_step
```

`--policy.config` and `--policy.dir` must refer to the same model. `--policy.dir` must point to a concrete checkpoint step directory containing `params/` and `assets/`.

### Match server model to client `control_mode`

| TacManip client `control_mode` | Recommended server model | Notes |
| --- | --- | --- |
| `diffik` | [`NathanWu7/pi0_lora_notac_tabero`](https://huggingface.co/NathanWu7/pi0_lora_notac_tabero) | Visual-only / 7D action path; no tactile fields are sent |
| `osc` | [`NathanWu7/pi0_lora_notac_tabero`](https://huggingface.co/NathanWu7/pi0_lora_notac_tabero) | Visual-only / 7D action path; reuses the same server as `diffik` |
| `tactile` | [`NathanWu7/pi0_lora_tacfield_tabero`](https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero) | Uses `tactile_marker_motion`, tactile image, and force history |
| `hybrid` | force-compatible checkpoint | Requires a model that reads `gripper_force`; do not use tacfield/no-tactile checkpoints by accident |

If a `diffik` / `osc` client connects to `pi0_lora_tacfield_tabero`, the server will fail because `tactile_marker_motion` is missing. If a `tactile` client connects to the no-tactile model, the tactile inputs are ignored by the model.

### `tactile` server example

Download the tacfield weights:

```bash
hf download NathanWu7/pi0_lora_tacfield_tabero \
  --local-dir /path/to/models/pi0_lora_tacfield_tabero \
  --include 'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params/**' \
  --include 'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/assets/**' \
  --include 'norm_stats/**'
```

Start the tacfield service:

```bash
cd /path/to/T2-VLA

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi0_lora_tacfield_tabero \
  --policy.dir=/path/to/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999
```

## Inference Loop

The client:

1. Starts Isaac Sim / Isaac Lab and creates the selected environment.
2. Optionally loads the task HDF5 from `HDF5_TRAJ_SOURCE_DIR` or `--hdf5-folder` for reset initial states.
3. Reads camera, state, force, tactile, and marker-motion observations from the environment.
4. Sends an OpenPI input dictionary to the server.
5. Receives a padded action chunk and executes the relevant 7D or 13D slice.
6. Counts an experiment as successful after `num_success_steps` consecutive success steps.

## Observation Fields

The TacManip client sends both top-level keys and `observation/...` compatibility keys. T2-VLA currently reads the top-level keys.

All modes send:

- `image` / `observation/image`: main RGB camera, `uint8`, `(224, 224, 3)`.
- `wrist_image` / `observation/wrist_image`: wrist RGB camera, `uint8`, `(224, 224, 3)`.
- `state` / `observation/state`: 7D task-space state `[x, y, z, ax, ay, az, gripper_abs]`, `float32`.
- `prompt`: language instruction from the task config.

Additional fields:

- `control_mode=hybrid`: `gripper_force` / `observation/gripper_force`, force history `(H, 6)` as `[fL(3), fR(3)]`.
- `control_mode=tactile`: `tactile_image` / `observation/tactile_image`, `tactile_gripper_force` / `observation/tactile_gripper_force`, and `tactile_marker_motion` / `observation/tactile_marker_motion`.

## Common Modes

### `diffik`

- **Server model**: `pi0_lora_notac_tabero`
- **Use case**: visual-only 7D task-space control

```bash
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

### `osc`

- **Server model**: `pi0_lora_notac_tabero`
- **Use case**: visual-only 7D OSC task-space control

```bash
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

### `hybrid`

- **Server model**: force-compatible checkpoint
- **Use case**: Tabero-force / ContactForce models with force history

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode hybrid \
  --task_suite libero_10 \
  --task_id 0 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0 \
  --server_host 127.0.1.1 \
  --server_port 8000
```

### `tactile`

- **Server model**: `pi0_lora_tacfield_tabero`
- **Use case**: tactile image, marker motion, and force history
- **Dependency**: tactile sensors must exist in the environment, usually `gsmini_left` and `gsmini_right`

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode tactile \
  --task_suite libero_10 \
  --task_id 0 \
  --tactile_output_type tactile_rgb \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0 \
  --server_host 127.0.1.1 \
  --server_port 8000
```

Useful tactile options:

- `--tactile_sensor_names gsmini_left gsmini_right`
- `--force_history_len 8`
- `--marker_history_len 8`

## Common Arguments

- `--server_host / --server_port`: OpenPI server address, default `127.0.1.1:8000`.
- `--task_suite / --task_id`: task selection, used for prompt and HDF5 initial state lookup.
- `--num_total_experiments`: number of independent attempts.
- `--max_inference_steps`: maximum number of action chunks per attempt.
- `--replan_steps`: number of steps executed from each chunk, default 10.
- `--debug_mode`: `0` is the clean smoke-test mode; higher values save more debug files.
- `--replay_mode`: comparison mode that executes GT actions while still running inference.

## FAQ

- **HDF5 reset states are not loaded**
  - Check `HDF5_TRAJ_SOURCE_DIR`, or pass `--hdf5-folder`.
  - Confirm the directory contains `{task_suite}_task{task_id}_*_demo.hdf5`.

- **`KeyError: "TaberoTacFieldInputs expects 'tactile_marker_motion' in data."`**
  - Cause: the server is using `pi0_lora_tacfield_tabero`, but the client is not running with `--control_mode tactile`.
  - Fix: use `pi0_lora_notac_tabero` for `diffik` / `osc`, or switch the client to `--control_mode tactile` when using the tacfield model.

- **Tactile mode cannot find sensors or output keys**
  - Confirm the client is running with `--control_mode tactile`.
  - Confirm the sensor names are `gsmini_left` / `gsmini_right`, or pass the correct names with `--tactile_sensor_names`.
  - Confirm `--tactile_output_type` matches an available sensor output, usually `tactile_rgb` or `markers_rgb`.

- **Checkpoint path is wrong**
  - `--policy.dir` must point to a concrete step directory, such as `.../49999`.
  - The directory should contain `params/` and `assets/`.

- **OpenPI server is unreachable**
  - Check that the T2-VLA service is still running.
  - Check that server `--port` and client `--server_port` match.
  - If running across machines, pass the server machine's IP with `--server_host` instead of relying on `127.0.1.1`.

## Training And Fine-Tuning

For Tabero model training, fine-tuning, and OpenPI service-side code, see:

- [`NathanWu7/T2-VLA`](https://github.com/NathanWu7/T2-VLA)
- Upstream reference: [`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi)
