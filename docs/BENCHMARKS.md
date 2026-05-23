# Benchmarks and Data Conversion

English | [中文](BENCHMARKS.zh-CN.md)

This document describes the `benchmarks/` area: dataset layout, HDF5/video to LeRobot conversion, and the OpenPI client entry point. The root [`README.md`](../README.md) keeps only a high-level index.

## What Lives Under `benchmarks/`

- `benchmarks/datasets/`: LIBERO assets, Tabero/Tabero-force replay outputs, and converted LeRobot/OpenPI datasets.
- `benchmarks/common/`: conversion scripts from Isaac-side HDF5 + videos to LeRobot format.
- `benchmarks/openpi/`: OpenPI inference client and debug helpers.

For collection, replay, and evaluation scripts, see [Tools Guide](TOOLS.md).

## Dataset Layout

Typical LIBERO input layout:

- `benchmarks/datasets/libero/config/`: task configs.
- `benchmarks/datasets/libero/assembled_hdf5/`: source trajectories used as replay input.
- `benchmarks/datasets/libero/USD/`: scene and object assets.

Typical replay outputs:

- `replayed_demos/`: HDF5 files containing successful demos.
- `video_datasets/`: RGB, wrist, or tactile videos.

Typical LeRobot output:

- `data/`: parquet chunks.
- `images/`: image storage.
- `meta/`: episodes, tasks, and dataset metadata.

## Conversion Scripts

All conversion scripts live under `benchmarks/common/` and expose tyro dataclass CLIs. Use `--help` to inspect exact options:

```bash
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py --help
```

### Standard LIBERO to LeRobot

Use `convert_all_libero_to_lerobot_openpi.py` for standard Franka data with 7D/8D actions and no force action:

```bash
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --task_suites libero_goal libero_10 \
  --data_root benchmarks/datasets/libero \
  --output_dir benchmarks/datasets/tabero_pi0 \
  --repo_name lerobot_all_libero_suites
```

This script does not support 13D force actions.

### ContactForce / Tabero-force to LeRobot

Use `convert_all_libero_to_tabero_force.py` for ContactForce data collected with 13D `7dpf` actions:

```bash
python benchmarks/common/convert_all_libero_to_tabero_force.py \
  --data_root benchmarks/datasets/tabero_force \
  --output_dir benchmarks/datasets/tabero_pi0 \
  --repo_name tabero_force_all_libero_suites \
  --force_history_len 8
```

### Tactile / Tabero to LeRobot

Use `convert_all_libero_to_tabero.py` for tactile data collected with 13D `7dpf` actions:

```bash
python benchmarks/common/convert_all_libero_to_tabero.py \
  --data_root benchmarks/datasets/tabero \
  --output_dir benchmarks/datasets/tabero_pi0 \
  --repo_name tabero_all_libero_suites \
  --tactile_output_type tactile_rgb \
  --force_history_len 8 \
  --marker_history_len 8
```

## OpenPI Client Entry Point

Install the repository-side OpenPI client from the repository root:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -m pip install -e benchmarks/openpi/openpi-client
```

Run one diffik inference experiment:

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1
```

For detailed model-server and observation-format guidance, see [OpenPI Inference Guide](OPENPI.md).

## Common Issues

- If conversion skips many trajectories, check that `video_datasets/` exists and aligns with action lengths.
- If action dimensions do not match, use standard conversion for 7D/8D actions and Tabero converters for 13D `7dpf` actions.
- If OpenPI inference does not load reset states, check `HDF5_TRAJ_SOURCE_DIR` or `--hdf5-folder`.
