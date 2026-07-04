# 复现表 3：触觉模态消融实验

本文记录表 3 的触觉模态消融实验结果，并给出后三行实验的复现命令：

- `Force E+FS`
- `Img+FS`
- `Field+FS`

OpenPI 侧模型服务请从修改版
[`NathanWu7/Tabero-VTLA`](https://github.com/NathanWu7/Tabero-VTLA) 仓库启动。Isaac 侧评测 client 从当前 Tabero 仓库运行。

## 表 3 结果

F/G 分别表示 firm/gentle 语言提示。SR 表示成功率，AG 表示论文中报告的平均抓取力指标。`None` 表示不使用触觉输入，`Img` 表示触觉图像输入，`Field` 表示力场输入，`Force E` 表示通过 MLP encoder 输入力信息，`Force D` 表示通过 decoder 输入力信息，`FS` 表示启用 force-supervision loss。

论文结果使用论文风格的运行设置：Isaac Lab 2.2 与 Isaac Sim 5.0，所有 `contact_gripper` 传感器绑定到 `panda_.*finger`，并设置 `squeeze_ff_k_load_z = 0.6`。

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

## 本地复现实验结果

下表记录本地 `minicase_k09` 重跑结果，使用 Isaac Lab 2.3 与 Isaac Sim 5.1。该设置下，所有 `contact_gripper` 传感器都绑定到 `gelsight_mini_case_.*`，`squeeze_ff_k_load_z = 0.9`，`squeeze_ff_contact_threshold = 1.0`。每个 firm 或 gentle 数值都在 Tabero LIBERO object 子集上汇总，共 9 个任务、450 次实验。

`AG pred` 是评测汇总中的模型侧预测抓取力指标。`AG meas` 是环境侧测得的接触力指标。

| Variant | Model | F SR | G SR | F AG pred | G AG pred | F AG meas | G AG meas |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| minicase_k09 | Force E+FS enc10 | 0.789 | 0.316 | 29.06 | 3.73 | 20.19 | 1.87 |
| minicase_k09 | Img+FS | 0.860 | 0.331 | 31.91 | 3.97 | 20.57 | 2.45 |
| minicase_k09 | Field+FS | 0.911 | 0.358 | 33.77 | 6.58 | 20.76 | 4.49 |

## 通用前置配置

先准备三个本地路径：

```bash
TABERO_ROOT=/path/to/Tabero
Tabero_VTLA_ROOT=/path/to/Tabero-VTLA
MODEL_ROOT=/path/to/models
```

Tabero client 会从 `benchmarks/datasets/libero/assembled_hdf5` 读取 LIBERO 初始状态 HDF5 文件。从 Tabero 仓库根目录执行：

```bash
cd "$TABERO_ROOT"
source scripts/tools/set_replay_env.sh inference
```

OpenPI service 和 Tabero client 必须使用一致的 host 和 port。下面命令使用：

```text
server_host = 127.0.1.1
```

每个模型都先从 `Tabero_VTLA_ROOT` 启动 server，等待日志出现 `server listening on 0.0.0.0:<PORT>`，再从 `TABERO_ROOT` 运行 firm 和 gentle 评测命令。

评测命令使用 Tabero task 子集和下载好的 LIBERO 初始状态：

```bash
--task-suites libero_object
--use-tabero-tasks
--hdf5-folder benchmarks/datasets/libero/assembled_hdf5
--require-hdf5
```

## Force E+FS

### 模型

| 项目 | 值 |
| --- | --- |
| HF 仓库 | [`NathanWu7/pi0_lora_tacforce_tabero_enc_10`](https://huggingface.co/NathanWu7/pi0_lora_tacforce_tabero_enc_10) |
| Tabero-VTLA config | `pi0_lora_tacforce_tabero_enc` |
| Checkpoint step | `49999` |
| Checkpoint dir | `$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10/checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999` |

下载权重：

```bash
hf download NathanWu7/pi0_lora_tacforce_tabero_enc_10 \
  --local-dir "$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10" \
  --include 'checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999/params/**' \
  --include 'checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999/assets/**' \
  --include 'norm_stats/**'
```

该 Tabero-VTLA config 期望 checkpoint step 下存在 `assets/NathanWu7/tabero`。如果下载后的 checkpoint 里没有该 assets 目录，将下载到的 norm stats 链接到 checkpoint assets 目录：

```bash
CHECKPOINT_DIR="$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10/checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999"
mkdir -p "$CHECKPOINT_DIR/assets/NathanWu7"
ln -sfn "$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10/norm_stats/pi0_lora_tacforce_tabero_enc/NathanWu7/tabero" \
  "$CHECKPOINT_DIR/assets/NathanWu7/tabero"
```

启动 OpenPI service：

```bash
cd "$Tabero_VTLA_ROOT"

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 18019 \
  policy:checkpoint \
  --policy.config=pi0_lora_tacforce_tabero_enc \
  --policy.dir="$MODEL_ROOT/pi0_lora_tacforce_tabero_enc_10/checkpoints/pi0_lora_tacforce_tabero_enc/pi0_lora_tacforce_tabero_enc_10/49999"
```

运行 firm-force 评测：

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

运行 gentle-force 评测：

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

### 模型

| 项目 | 值 |
| --- | --- |
| HF 仓库 | [`NathanWu7/pi0_lora_tacimg_tabero`](https://huggingface.co/NathanWu7/pi0_lora_tacimg_tabero) |
| Tabero-VTLA config | `pi0_lora_tacimg_tabero` |
| Checkpoint step | `49999` |
| Checkpoint dir | `$MODEL_ROOT/pi0_lora_tacimg_tabero/checkpoints/pi0_lora_tacimg_tabero/pi0_lora_tacimg_tabero/49999` |

下载权重：

```bash
hf download NathanWu7/pi0_lora_tacimg_tabero \
  --local-dir "$MODEL_ROOT/pi0_lora_tacimg_tabero" \
  --include 'checkpoints/pi0_lora_tacimg_tabero/pi0_lora_tacimg_tabero/49999/params/**' \
  --include 'checkpoints/pi0_lora_tacimg_tabero/pi0_lora_tacimg_tabero/49999/assets/**' \
  --include 'norm_stats/**'
```

启动 OpenPI service：

```bash
cd "$Tabero_VTLA_ROOT"

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 18017 \
  policy:checkpoint \
  --policy.config=pi0_lora_tacimg_tabero \
  --policy.dir="$MODEL_ROOT/pi0_lora_tacimg_tabero/checkpoints/pi0_lora_tacimg_tabero/pi0_lora_tacimg_tabero/49999"
```

运行 firm-force 评测：

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

运行 gentle-force 评测：

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

### 模型

| 项目 | 值 |
| --- | --- |
| HF 仓库 | [`NathanWu7/pi0_lora_tacfield_tabero`](https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero) |
| Tabero-VTLA config | `pi0_lora_tacfield_tabero` |
| Checkpoint step | `49999` |
| Checkpoint dir | `$MODEL_ROOT/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999` |

下载权重：

```bash
hf download NathanWu7/pi0_lora_tacfield_tabero \
  --local-dir "$MODEL_ROOT/pi0_lora_tacfield_tabero" \
  --include 'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params/**' \
  --include 'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/assets/**' \
  --include 'norm_stats/**'
```

该 Tabero-VTLA config 期望存在 `assets/NathanWu7/tabero_object_25`。如果 checkpoint 里只有 `assets/NathanWu7/tabero`，在 checkpoint 的 assets 目录下创建本地软链接：

```bash
cd "$MODEL_ROOT/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/assets/NathanWu7"
ln -sfn tabero tabero_object_25
```

启动 OpenPI service：

```bash
cd "$Tabero_VTLA_ROOT"

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 18018 \
  policy:checkpoint \
  --policy.config=pi0_lora_tacfield_tabero \
  --policy.dir="$MODEL_ROOT/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999"
```

运行 firm-force 评测：

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

运行 gentle-force 评测：

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

## 输出

每次评测都会在指定输出目录下写入 JSON 和文本汇总：

```text
evaluation_results/table3_<model>_<firm_or_gentle>/success_rates_*.json
evaluation_results/table3_<model>_<firm_or_gentle>/success_rates_*.txt
```

从这些文件中读取 task 级别和 overall 成功率。评测 stdout 也会打印力相关指标，包括用于抓取力分析的 hybrid contact metrics。
