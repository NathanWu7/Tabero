# OpenPI Inference Guide

English | [中文](OPENPI.zh-CN.md)

This guide explains how TacManip connects Isaac Lab environments to an external OpenPI model server for closed-loop inference evaluation.

## Server and Client

OpenPI inference has two parts:

- **OpenPI server**: the model inference service, started outside this repository by following [`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi).
- **TacManip client**: `benchmarks/openpi/openpi_inference_client.py`, which starts Isaac Lab environments, packages observations, calls the OpenPI server, and executes returned actions.

## Quickstart

Run from the TacManip repository root with the Isaac Lab Python environment active:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -m pip install -e benchmarks/openpi/openpi-client
```

Set the LIBERO source trajectory directory if you want reproducible resets from dataset initial states:

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
# or:
source scripts/tools/set_replay_env.sh inference
```

Start the OpenPI server separately, then run a diffik inference experiment:

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0
```

Default server settings are:

```text
server_host = 127.0.1.1
server_port = 8000
```

Override them with `--server_host` and `--server_port` when needed.

## Inference Loop

The client:

1. Starts Isaac Sim / Isaac Lab and creates the selected environment.
2. Optionally loads the task HDF5 from `HDF5_TRAJ_SOURCE_DIR` or `--hdf5-folder` for reset initial states.
3. Reads camera, state, force, tactile, and marker-motion observations from the environment.
4. Sends an OpenPI input dictionary to the server.
5. Receives a padded action chunk and executes the relevant 7D or 13D slice.
6. Counts an experiment as successful after `num_success_steps` consecutive success steps.

## Observation Fields

All modes send:

- `observation/image`: main RGB camera, `uint8`, `(224, 224, 3)`.
- `observation/wrist_image`: wrist RGB camera, `uint8`, `(224, 224, 3)`.
- `observation/state`: 7D task-space state `[x, y, z, ax, ay, az, gripper_abs]`, `float32`.
- `prompt`: language instruction from the task config.

Additional fields:

- `control_mode=hybrid`: `observation/gripper_force`, force history `(H, 6)` as `[fL(3), fR(3)]`.
- `control_mode=tactile`: `observation/tactile_image`, `observation/tactile_gripper_force`, and `observation/tactile_marker_motion`.

These fields match the Tabero conversion scripts.

## Common Modes

### `diffik`

Use this first for vision-only 7D task-space control:

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 5
```

### `hybrid`

Use this for Tabero-force / ContactForce models with force history:

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode hybrid \
  --task_suite libero_10 \
  --task_id 0 \
  --num_total_experiments 5
```

### `tactile`

Use this for Tabero tactile models with tactile images, marker motion, and force history:

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode tactile \
  --task_suite libero_10 \
  --task_id 0 \
  --tactile_output_type tactile_rgb \
  --num_total_experiments 5
```

Useful tactile options:

- `--tactile_sensor_names gsmini_left gsmini_right`
- `--force_history_len 8`
- `--marker_history_len 8`

## FAQ

- If HDF5 reset states are not loaded, check `HDF5_TRAJ_SOURCE_DIR` or pass `--hdf5-folder`.
- If tactile mode cannot find sensors or output keys, confirm the environment is tactile and the output type matches available sensor outputs.
- If the OpenPI server is unreachable, verify host, port, firewall, and container port mappings.

Training and fine-tuning belong to the upstream OpenPI repository. See [`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi).
