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
hf download NathanWu7/Isaaclab_Libero \
  --repo-type dataset \
  --local-dir /path/to/Isaaclab_Libero
```

Prepare the default LIBERO data symlinks. Keep `benchmarks/datasets/libero/config` and `benchmarks/datasets/libero/utils` from this repository; only link the downloaded data subdirectories.

```bash
LIBERO_DATA=/path/to/Isaaclab_Libero

ln -sfn "$LIBERO_DATA/assembled_hdf5" benchmarks/datasets/libero/assembled_hdf5
ln -sfn "$LIBERO_DATA/USD" benchmarks/datasets/libero/USD

# Optional, useful if you use preprocessed replay/video data directly.
ln -sfn "$LIBERO_DATA/replayed_demos" benchmarks/datasets/libero/replayed_demos
ln -sfn "$LIBERO_DATA/video_datasets" benchmarks/datasets/libero/video_datasets
```

Verify the LIBERO links:

```bash
ls -l benchmarks/datasets/libero
test -d benchmarks/datasets/libero/assembled_hdf5
test -d benchmarks/datasets/libero/USD
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

## Experiment Results

For full reproduction commands, see the [reproduction guide](docs/REPRODUCTION.md).

### Paper Table 3

F/G refer to firm/gentle language prompts. SR is success rate, and AG is the average grip-force metric. `None` means no tactile input, `Img` means tactile image input, `Field` means force-field input, `Force E` means force input through an MLP encoder, `Force D` means force input through a decoder, and `FS` means force-supervision loss is enabled.

The paper numbers use the paper-style runtime setting: Isaac Lab 2.2 with Isaac Sim 5.0, all `contact_gripper` sensors bound to `panda_.*finger`, and `squeeze_ff_k_load_z = 0.6`.

| Model | F SR | G SR | F AG | G AG |
| --- | ---: | ---: | ---: | ---: |
| None | 0.00 | 0.00 | 0.0 | 0.0 |
| Img | 0.37 | 0.01 | 3.0 | 1.1 |
| Field | 0.40 | 0.01 | 2.9 | 2.0 |
| Force E | 0.40 | 0.01 | 2.5 | 1.8 |
| FS | 0.82 | 0.45 | 30.4 | 3.1 |
| Force D+FS | 0.82 | 0.31 | 28.5 | 3.3 |
| Force E+FS | 0.84 | 0.49 | 30.3 | 3.4 |
| Img+FS | 0.87 | 0.48 | 30.6 | 3.6 |
| Field+FS | 0.86 | 0.52 | 32.4 | 3.7 |

To reproduce the paper-style force-sensor setting in this environment, set `squeeze_ff_k_load_z = 0.6` in [force_position_action.py](source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py), and set every `contact_gripper.prim_path` in [franka_tactile_libero_env_cfg.py](source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_tactile_libero_env_cfg.py) to `"{ENV_REGEX_NS}/Robot/panda_.*finger"`.

### Local Minicase Rerun

The following local rerun uses Isaac Lab 2.3 with Isaac Sim 5.1, `gelsight_mini_case_.*` contact binding, `squeeze_ff_k_load_z = 0.9`, and `squeeze_ff_contact_threshold = 1.0`. Each firm or gentle value is aggregated over the Tabero LIBERO object subset, with 9 tasks and 450 total trials.

The exported package snapshot for the local `tabero` conda environment is kept as a reference at [environment-tabero-isaaclab23-isaacsim51.yml](envs/environment-tabero-isaaclab23-isaacsim51.yml). It corresponds to the Isaac Lab 2.3 / Isaac Sim 5.1 reproduction stack and is not intended to replace the normal Isaac Lab / Isaac Sim installation steps.

`AG pred` is the model-side predicted grip-force metric from the evaluation summary. `AG meas` is the measured contact-force metric reported by the environment.

| Variant | Model | F SR | G SR | F AG pred | G AG pred | F AG meas | G AG meas |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| minicase_k09 | Force E+FS enc10 | 0.789 | 0.316 | 29.06 | 3.73 | 20.19 | 1.87 |
| minicase_k09 | Img+FS | 0.860 | 0.331 | 31.91 | 3.97 | 20.57 | 2.45 |
| minicase_k09 | Field+FS | 0.911 | 0.358 | 33.77 | 6.58 | 20.76 | 4.49 |

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
- Libero light randomization is off by default for reproducibility. Add `--randomize_light` to replay or evaluation commands to randomize DomeLight intensity, color, and HDR texture on each environment reset.

For exact command templates and troubleshooting, use the dedicated documentation under `docs/`.
