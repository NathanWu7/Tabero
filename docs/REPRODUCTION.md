# Reproducing Table 3: Tactile Modality Ablation

This document records the Table 3 tactile-modality ablation results and gives
reproduction commands for the last three rows:

- `Force E+FS`
- `Img+FS`
- `Field+FS`

The OpenPI-side model service should be started from the modified
[`NathanWu7/T2-VLA`](https://github.com/NathanWu7/T2-VLA) repository. The
Isaac-side evaluation client runs from this Tabero repository.

## Table 3 Results

F/G refer to firm/gentle language prompts. SR is success rate. AG is the average
grip-force metric reported in the paper. `None` means no tactile input, `Img`
means tactile image input, `Field` means force-field input, `Force E` means force
input through an MLP encoder, `Force D` means force input through a decoder, and
`FS` means force-supervision loss is enabled.

The paper numbers use the paper-style runtime setting: Isaac Lab 2.2 with Isaac
Sim 5.0, all `contact_gripper` sensors bound to `panda_.*finger`, and
`squeeze_ff_k_load_z = 0.6`.

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

## Local Reproduction Results

The table below records the local `minicase_k09` rerun under Isaac Lab 2.3 with
Isaac Sim 5.1. In this setting, all `contact_gripper` sensors are bound to
`gelsight_mini_case_.*`, `squeeze_ff_k_load_z = 0.9`, and
`squeeze_ff_contact_threshold = 1.0`. Each firm or gentle value is aggregated
over the Tabero LIBERO object subset, with 9 tasks and 450 total trials.

`AG pred` is the model-side predicted grip-force metric from the evaluation
summary. `AG meas` is the measured contact-force metric reported by the
environment.

| Variant | Model | F SR | G SR | F AG pred | G AG pred | F AG meas | G AG meas |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| minicase_k09 | Force E+FS enc10 | 0.789 | 0.316 | 29.06 | 3.73 | 20.19 | 1.87 |
| minicase_k09 | Img+FS | 0.860 | 0.331 | 31.91 | 3.97 | 20.57 | 2.45 |
| minicase_k09 | Field+FS | 0.911 | 0.358 | 33.77 | 6.58 | 20.76 | 4.49 |

## Common Setup

Use three local roots:

```bash
TABERO_ROOT=/path/to/Tabero
T2_VLA_ROOT=/path/to/T2-VLA
MODEL_ROOT=/path/to/models
```

The Tabero client expects LIBERO initial-state HDF5 files under
`benchmarks/datasets/libero/assembled_hdf5`. From the Tabero repository root:

```bash
cd "$TABERO_ROOT"
source scripts/tools/set_replay_env.sh inference
```

The OpenPI service and the Tabero client must use the same host and port. The
commands below use:

```text
server_host = 127.0.1.1
```

For each model, run the server command from `T2_VLA_ROOT`, wait until the server
prints `server listening on 0.0.0.0:<PORT>`, then run the firm and gentle
evaluation commands from `TABERO_ROOT`.

The evaluation commands use the Tabero task subset and the downloaded LIBERO
initial states:

```bash
--task-suites libero_object
--use-tabero-tasks
--hdf5-folder benchmarks/datasets/libero/assembled_hdf5
--require-hdf5
```

## Force E+FS

### Model

| Item | Value |
| --- | --- |
| HF repo | [`NathanWu7/pi0_lora_tacforce_tabero_enc_10`](https://huggingface.co/NathanWu7/pi0_lora_tacforce_tabero_enc_10) |
| T2-VLA config | `pi0_lora_tacforce_tabero_enc` |
| Checkpoint step | `49999` |
| Checkpoint dir | `$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10/checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999` |

Download the weights:

```bash
hf download NathanWu7/pi0_lora_tacforce_tabero_enc_10 \
  --local-dir "$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10" \
  --include 'checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999/params/**' \
  --include 'checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999/assets/**' \
  --include 'norm_stats/**'
```

This T2-VLA config expects `assets/NathanWu7/tabero` under the checkpoint step.
If the downloaded checkpoint does not contain that assets directory, link the
downloaded norm stats into the checkpoint assets directory:

```bash
CHECKPOINT_DIR="$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10/checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999"
mkdir -p "$CHECKPOINT_DIR/assets/NathanWu7"
ln -sfn "$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10/norm_stats/pi0_lora_tacforce_tabero_enc/NathanWu7/tabero" \
  "$CHECKPOINT_DIR/assets/NathanWu7/tabero"
```

Start the OpenPI service:

```bash
cd "$T2_VLA_ROOT"

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 18019 \
  policy:checkpoint \
  --policy.config=pi0_lora_tacforce_tabero_enc \
  --policy.dir="$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10/checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999"
```

Run the firm-force evaluation:

```bash
cd "$TABERO_ROOT"
source scripts/tools/set_replay_env.sh inference

conda run --no-capture-output -n tabero python -u scripts/tools/run_task_evaluations.py \
  --policy-model openpi \
  --control-mode tactile \
  --server-host 127.0.1.1 \
  --server-port 18019 \
  --task-suites libero_object \
  --use-tabero-tasks \
  --num-total-experiments 50 \
  --hdf5-folder benchmarks/datasets/libero/assembled_hdf5 \
  --require-hdf5 \
  --prompt-adverbs firmly tightly \
  --output-dir evaluation_results/table3_force_e_fs_enc10_firm \
  --output-format both \
  --headless
```

Run the gentle-force evaluation:

```bash
cd "$TABERO_ROOT"
source scripts/tools/set_replay_env.sh inference

conda run --no-capture-output -n tabero python -u scripts/tools/run_task_evaluations.py \
  --policy-model openpi \
  --control-mode tactile \
  --server-host 127.0.1.1 \
  --server-port 18019 \
  --task-suites libero_object \
  --use-tabero-tasks \
  --num-total-experiments 50 \
  --hdf5-folder benchmarks/datasets/libero/assembled_hdf5 \
  --require-hdf5 \
  --prompt-adverbs gently softly \
  --output-dir evaluation_results/table3_force_e_fs_enc10_gentle \
  --output-format both \
  --headless
```

## Img+FS

### Model

| Item | Value |
| --- | --- |
| HF repo | [`NathanWu7/pi0_lora_tacimg_tabero`](https://huggingface.co/NathanWu7/pi0_lora_tacimg_tabero) |
| T2-VLA config | `pi0_lora_tacimg_tabero` |
| Checkpoint step | `49999` |
| Checkpoint dir | `$MODEL_ROOT/pi0_lora_tacimg_tabero/checkpoints/pi0_lora_tacimg_tabero/pi0_lora_tacimg_tabero/49999` |

Download the weights:

```bash
hf download NathanWu7/pi0_lora_tacimg_tabero \
  --local-dir "$MODEL_ROOT/pi0_lora_tacimg_tabero" \
  --include 'checkpoints/pi0_lora_tacimg_tabero/pi0_lora_tacimg_tabero/49999/params/**' \
  --include 'checkpoints/pi0_lora_tacimg_tabero/pi0_lora_tacimg_tabero/49999/assets/**' \
  --include 'norm_stats/**'
```

Start the OpenPI service:

```bash
cd "$T2_VLA_ROOT"

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 18017 \
  policy:checkpoint \
  --policy.config=pi0_lora_tacimg_tabero \
  --policy.dir="$MODEL_ROOT/pi0_lora_tacimg_tabero/checkpoints/pi0_lora_tacimg_tabero/pi0_lora_tacimg_tabero/49999"
```

Run the firm-force evaluation:

```bash
cd "$TABERO_ROOT"
source scripts/tools/set_replay_env.sh inference

conda run --no-capture-output -n tabero python -u scripts/tools/run_task_evaluations.py \
  --policy-model openpi \
  --control-mode tactile \
  --server-host 127.0.1.1 \
  --server-port 18017 \
  --task-suites libero_object \
  --use-tabero-tasks \
  --num-total-experiments 50 \
  --hdf5-folder benchmarks/datasets/libero/assembled_hdf5 \
  --require-hdf5 \
  --prompt-adverbs firmly \
  --output-dir evaluation_results/table3_img_fs_firm \
  --output-format both \
  --headless
```

Run the gentle-force evaluation:

```bash
cd "$TABERO_ROOT"
source scripts/tools/set_replay_env.sh inference

conda run --no-capture-output -n tabero python -u scripts/tools/run_task_evaluations.py \
  --policy-model openpi \
  --control-mode tactile \
  --server-host 127.0.1.1 \
  --server-port 18017 \
  --task-suites libero_object \
  --use-tabero-tasks \
  --num-total-experiments 50 \
  --hdf5-folder benchmarks/datasets/libero/assembled_hdf5 \
  --require-hdf5 \
  --prompt-adverbs gently \
  --output-dir evaluation_results/table3_img_fs_gentle \
  --output-format both \
  --headless
```

## Field+FS

### Model

| Item | Value |
| --- | --- |
| HF repo | [`NathanWu7/pi0_lora_tacfield_tabero`](https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero) |
| T2-VLA config | `pi0_lora_tacfield_tabero` |
| Checkpoint step | `49999` |
| Checkpoint dir | `$MODEL_ROOT/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999` |

Download the weights:

```bash
hf download NathanWu7/pi0_lora_tacfield_tabero \
  --local-dir "$MODEL_ROOT/pi0_lora_tacfield_tabero" \
  --include 'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params/**' \
  --include 'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/assets/**' \
  --include 'norm_stats/**'
```

This T2-VLA config expects `assets/NathanWu7/tabero_object_25`. If the checkpoint
only contains `assets/NathanWu7/tabero`, create a local symlink inside the
checkpoint assets directory:

```bash
cd "$MODEL_ROOT/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/assets/NathanWu7"
ln -sfn tabero tabero_object_25
```

Start the OpenPI service:

```bash
cd "$T2_VLA_ROOT"

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 18018 \
  policy:checkpoint \
  --policy.config=pi0_lora_tacfield_tabero \
  --policy.dir="$MODEL_ROOT/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999"
```

Run the firm-force evaluation:

```bash
cd "$TABERO_ROOT"
source scripts/tools/set_replay_env.sh inference

conda run --no-capture-output -n tabero python -u scripts/tools/run_task_evaluations.py \
  --policy-model openpi \
  --control-mode tactile \
  --server-host 127.0.1.1 \
  --server-port 18018 \
  --task-suites libero_object \
  --use-tabero-tasks \
  --num-total-experiments 50 \
  --hdf5-folder benchmarks/datasets/libero/assembled_hdf5 \
  --require-hdf5 \
  --prompt-adverbs firmly \
  --output-dir evaluation_results/table3_field_fs_firm \
  --output-format both \
  --headless
```

Run the gentle-force evaluation:

```bash
cd "$TABERO_ROOT"
source scripts/tools/set_replay_env.sh inference

conda run --no-capture-output -n tabero python -u scripts/tools/run_task_evaluations.py \
  --policy-model openpi \
  --control-mode tactile \
  --server-host 127.0.1.1 \
  --server-port 18018 \
  --task-suites libero_object \
  --use-tabero-tasks \
  --num-total-experiments 50 \
  --hdf5-folder benchmarks/datasets/libero/assembled_hdf5 \
  --require-hdf5 \
  --prompt-adverbs gently \
  --output-dir evaluation_results/table3_field_fs_gentle \
  --output-format both \
  --headless
```

## Outputs

Each evaluation writes a JSON and text summary under the selected output
directory:

```text
evaluation_results/table3_<model>_<firm_or_gentle>/success_rates_*.json
evaluation_results/table3_<model>_<firm_or_gentle>/success_rates_*.txt
```

Read the task-level and overall success rates from these files. The evaluation
stdout also prints force metrics, including the hybrid contact metrics used for
grip-force analysis.
