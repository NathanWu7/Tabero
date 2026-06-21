# TacManip

English | [中文](docs/README.zh-CN.md)

TacManip is a manipulation data collection, replay, conversion, and inference-evaluation project built on NVIDIA **Isaac Sim + Isaac Lab**. It focuses on replaying open-source or teleoperated trajectories in Isaac, exporting consistent HDF5/video data, converting the results to LeRobot/OpenPI formats, and evaluating OpenPI policies with visual, force, and tactile observations.

## Documentation

Detailed workflows are kept under `docs/`. The root README is intentionally short and indexes the English documentation first:

- [Isaac-Libero workflow](docs/LIBERO_WORKFLOW.en.md)
- [Tools guide](docs/TOOLS.md)
- [Benchmarks and data conversion](docs/BENCHMARKS.md)
- [OpenPI inference guide](docs/OPENPI.md)

Chinese versions are linked from each English document.

## Repository Map

- `source/tac_manip/`: TacManip Isaac Lab extension, tasks, assets, and environment registration.
- `scripts/tools/`: collection, replay, evaluation, visualization, and upload scripts.
- `benchmarks/common/`: converters from Isaac-side HDF5/video outputs to LeRobot/OpenPI datasets.
- `benchmarks/openpi/`: TacManip OpenPI inference client and debug utilities.
- `benchmarks/datasets/`: expected local data layout for LIBERO, Tabero, Tabero-force, and converted datasets.
- `docs/`: all user-facing documentation.

## Quick Setup

Install the TacManip extension:

```bash
python -m pip install -e source/tac_manip
```

Install the repository-side OpenPI inference client:

```bash
python -m pip install -e benchmarks/openpi/openpi-client
```

Download the LIBERO data from [`NathanWu7/Isaaclab_Libero`](https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero). The repository expects at least `assembled_hdf5/` and `USD/`; preprocessed `replayed_demos/` and `video_datasets/` can also be used directly for training or inference.

```bash
huggingface-cli download NathanWu7/Isaaclab_Libero \
  --repo-type dataset \
  --local-dir /path/to/Isaaclab_Libero
```

Prepare the default LIBERO data symlink:

```bash
ln -sfn /path/to/Isaaclab_Libero benchmarks/datasets/libero
```

Download the tactile calibration assets from [`china-sae-robotics/Tactile_Manipulation_Dataset`](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset) when using tactile environments:

```bash
huggingface-cli download china-sae-robotics/Tactile_Manipulation_Dataset \
  --repo-type dataset \
  --local-dir /path/to/Tactile_manipulation_dataset
```

Prepare the tactile calibration asset symlink:

```bash
ln -sfn /path/to/Tactile_manipulation_dataset source/tac_manip/tac_manip/assets/data
```

For replay recollection, use separate output directories instead of writing back to the default symlinked dataset path. See the [Isaac-Libero workflow](docs/LIBERO_WORKFLOW.en.md) and [Tools guide](docs/TOOLS.md).

## Model Code and Weights

The OpenPI-side model code for Tabero is maintained in [`NathanWu7/T2-VLA`](https://github.com/NathanWu7/T2-VLA). This repository provides the model-serving/training side; TacManip provides the Isaac Lab environments, data conversion tools, and inference client.

The corresponding model weights are available at [`NathanWu7/pi0_lora_tacfield_tabero`](https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero):

```bash
hf download NathanWu7/pi0_lora_tacfield_tabero \
  --local-dir /path/to/pi0_lora_tacfield_tabero
```

During closed-loop evaluation, start the model service from the model-code repository, then run TacManip's `benchmarks/openpi/openpi_inference_client.py` or `scripts/tools/run_task_evaluations.py` as the Isaac-side client.

## Main Workflows

You can choose either the **Isaac-Libero** path or the **Tabero** path:

- **Isaac-Libero**: use standard LIBERO data and standard Franka environments. If you need this path, follow the dedicated [Isaac-Libero workflow](docs/LIBERO_WORKFLOW.en.md).
- **Tabero**: use the force or tactile data path, including ContactForce and GelSight-based environments, 13D `7dpf` actions, Tabero conversion scripts, and OpenPI inference with force/tactile observations. See [Tools guide](docs/TOOLS.md), [Benchmarks and data conversion](docs/BENCHMARKS.md), and [OpenPI inference guide](docs/OPENPI.md).

## Common Environment IDs

- `Isaac-Libero-Franka-Replay-Camera-v0`: standard Franka replay with cameras.
- `Isaac-Libero-Franka-IK-v0`: standard task-space DiffIK environment.
- `Isaac-Libero-Franka-OscPose-v0`: OSC pose-control environment.
- `Isaac-Libero-Franka-Replay-Camera-ContactForce-v0`: replay with contact-force observations.
- `Isaac-Libero-Franka-Hybrid-ContactForce-v0`: hybrid force-position control with contact force.
- `Isaac-Libero-Franka-Replay-Camera-Tactile-v0`: replay with GelSight tactile sensors.
- `Isaac-Libero-Franka-Hybrid-Tactile-v0`: hybrid tactile environment.

## Data and Model Notes

- Standard LIBERO data usually uses 7D/8D task-space actions.
- ContactForce and tactile Tabero data use 13D `7dpf` actions when force is included.
- The OpenPI client sends RGB images, wrist images, task-space state, language prompt, and optional force/tactile fields to the model server.

For exact command templates and troubleshooting, use the dedicated documentation under `docs/`.
