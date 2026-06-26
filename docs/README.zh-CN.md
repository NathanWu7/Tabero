# TacManip

[English](../README.md) | 中文

TacManip 是基于 NVIDIA **Isaac Sim + Isaac Lab** 的操作数据采集、回放、转换与推理评估工程。它主要用于在 Isaac 中复现开源轨迹或人工遥操作轨迹，导出统一的 HDF5/视频数据，转换为 LeRobot/OpenPI 格式，并评测带视觉、力觉、触觉观测的 OpenPI 策略。

## 文档索引

详细流程都放在 `docs/` 下。中文入口如下：

- [Isaac-Libero 使用手册](LIBERO_WORKFLOW.md)
- [Tools 工具脚本文档](TOOLS.zh-CN.md)
- [Benchmarks 与数据转换文档](BENCHMARKS.zh-CN.md)
- [OpenPI 推理文档](OPENPI.zh-CN.md)

英文版文档可从各中文文档顶部切换。

## 仓库结构

- `source/tac_manip/`：TacManip Isaac Lab 扩展、任务、资产与环境注册。
- `scripts/tools/`：数据采集、回放、评估、可视化和上传脚本。
- `benchmarks/common/`：从 Isaac 侧 HDF5/视频转换到 LeRobot/OpenPI 数据集的脚本。
- `benchmarks/openpi/`：TacManip OpenPI 推理客户端和调试工具。
- `benchmarks/datasets/`：LIBERO、Tabero、Tabero-force 和转换后数据的默认本地布局。
- `docs/`：全部用户文档。

## 快速配置

安装 TacManip 扩展：

```bash
python -m pip install -e source/tac_manip
```

安装仓库侧 OpenPI 推理客户端：

```bash
python -m pip install -e benchmarks/openpi/openpi-client
```

从 [`NathanWu7/Isaaclab_Libero`](https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero) 下载 LIBERO 数据。本仓库至少需要其中的 `assembled_hdf5/` 和 `USD/`；如果直接训练或推理，也可以使用预处理好的 `replayed_demos/` 和 `video_datasets/`。

```bash
hf download NathanWu7/Isaaclab_Libero \
  --repo-type dataset \
  --local-dir /path/to/Isaaclab_Libero
```

准备默认 LIBERO 数据软链接。保留本仓库中的 `benchmarks/datasets/libero/config` 和 `benchmarks/datasets/libero/utils`；只把下载数据里的子目录链接进来。

```bash
LIBERO_DATA=/path/to/Isaaclab_Libero

ln -sfn "$LIBERO_DATA/assembled_hdf5" benchmarks/datasets/libero/assembled_hdf5
ln -sfn "$LIBERO_DATA/USD" benchmarks/datasets/libero/USD

# 可选：如果你直接使用预处理好的 replay/video 数据，可以一并链接。
ln -sfn "$LIBERO_DATA/replayed_demos" benchmarks/datasets/libero/replayed_demos
ln -sfn "$LIBERO_DATA/video_datasets" benchmarks/datasets/libero/video_datasets
```

检查 LIBERO 软链接：

```bash
ls -l benchmarks/datasets/libero
test -d benchmarks/datasets/libero/assembled_hdf5
test -d benchmarks/datasets/libero/USD
```

使用触觉环境时，从 [`china-sae-robotics/Tactile_Manipulation_Dataset`](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset) 下载触觉标定资源：

```bash
huggingface-cli download china-sae-robotics/Tactile_Manipulation_Dataset \
  --repo-type dataset \
  --local-dir /path/to/Tactile_manipulation_dataset
```

准备触觉标定资源软链接：

```bash
ln -sfn /path/to/Tactile_manipulation_dataset source/tac_manip/tac_manip/assets/data
```

如果要重新 replay 采集，请把输出目录单独指定到默认数据软链接之外。完整说明见 [Isaac-Libero 使用手册](LIBERO_WORKFLOW.md) 和 [Tools 工具脚本文档](TOOLS.zh-CN.md)。

## 模型代码与权重

Tabero 对应的 OpenPI 侧模型代码维护在 [`NathanWu7/T2-VLA`](https://github.com/NathanWu7/T2-VLA)。该仓库负责模型训练/推理服务侧；TacManip 负责 Isaac Lab 环境、数据转换工具和推理 client。

对应模型权重位于 [`NathanWu7/pi0_lora_tacfield_tabero`](https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero)：

```bash
hf download NathanWu7/pi0_lora_tacfield_tabero \
  --local-dir /path/to/pi0_lora_tacfield_tabero
```

闭环评测时，先在模型代码仓库侧启动模型服务，再运行 TacManip 中的 `benchmarks/openpi/openpi_inference_client.py` 或 `scripts/tools/run_task_evaluations.py` 作为 Isaac 侧 client。

## 主流程

可以选择跑 **Isaac-Libero** 或 **Tabero** 两条路线：

- **Isaac-Libero**：使用标准 LIBERO 数据和标准 Franka 环境。如果需要跑这条路线，请单独参考 [Isaac-Libero 使用手册](LIBERO_WORKFLOW.md)。
- **Tabero**：使用力觉或触觉数据路线，包括 ContactForce、GelSight 触觉环境、13D `7dpf` 动作、Tabero 转换脚本，以及带力觉/触觉观测的 OpenPI 推理。见 [Tools 工具脚本文档](TOOLS.zh-CN.md)、[Benchmarks 与数据转换文档](BENCHMARKS.zh-CN.md) 和 [OpenPI 推理文档](OPENPI.zh-CN.md)。

## 实验效果

完整复现命令见 [复现文档](REPRODUCTION.zh-CN.md)。

### 论文 Table 3

F/G 分别表示 firm/gentle 语言提示。SR 表示成功率，AG 表示平均抓取力指标。`None` 表示不使用触觉输入，`Img` 表示触觉图像输入，`Field` 表示力场输入，`Force E` 表示通过 MLP encoder 输入力信息，`Force D` 表示通过 decoder 输入力信息，`FS` 表示启用 force-supervision loss。

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

如果要在当前代码中复现论文风格的力传感器设置，需要：

- 将 [force_position_action.py](../source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py) 中的 `squeeze_ff_k_load_z` 改为 `0.6`。
- 将 [franka_tactile_libero_env_cfg.py](../source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_tactile_libero_env_cfg.py) 中所有 `contact_gripper.prim_path` 改为 `"{ENV_REGEX_NS}/Robot/panda_.*finger"`。

### 本地 minicase 重跑结果

下表记录本地 `minicase_k09` 重跑结果，使用 Isaac Lab 2.3 与 Isaac Sim 5.1。该设置下，所有 `contact_gripper` 传感器都绑定到 `gelsight_mini_case_.*`，`squeeze_ff_k_load_z = 0.9`，`squeeze_ff_contact_threshold = 1.0`。每个 firm 或 gentle 数值都在 Tabero LIBERO object 子集上汇总，共 9 个任务、450 次实验。

本地 `tabero` conda 环境的包快照作为参考放在 [environment-tabero-isaaclab23-isaacsim51.yml](../envs/environment-tabero-isaaclab23-isaacsim51.yml)。该文件对应 Isaac Lab 2.3 / Isaac Sim 5.1 复现环境，不用于替代正常的 Isaac Lab / Isaac Sim 安装流程。

`AG pred` 是评测汇总中的模型侧预测抓取力指标。`AG meas` 是环境侧测得的接触力指标。

| Variant | Model | F SR | G SR | F AG pred | G AG pred | F AG meas | G AG meas |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| minicase_k09 | Force E+FS enc10 | 0.789 | 0.316 | 29.06 | 3.73 | 20.19 | 1.87 |
| minicase_k09 | Img+FS | 0.860 | 0.331 | 31.91 | 3.97 | 20.57 | 2.45 |
| minicase_k09 | Field+FS | 0.911 | 0.358 | 33.77 | 6.58 | 20.76 | 4.49 |

## 常用环境 ID

- `Isaac-Libero-Franka-Replay-Camera-v0`：标准 Franka 带相机 replay。
- `Isaac-Libero-Franka-IK-v0`：标准 task-space DiffIK 环境。
- `Isaac-Libero-Franka-OscPose-v0`：OSC pose-control 环境。
- `Isaac-Libero-Franka-Replay-Camera-ContactForce-v0`：带接触力观测的 replay 环境。
- `Isaac-Libero-Franka-Hybrid-ContactForce-v0`：带接触力的混合力-位控制环境。
- `Isaac-Libero-Franka-Replay-Camera-Tactile-v0`：带 GelSight 触觉传感器的 replay 环境。
- `Isaac-Libero-Franka-Hybrid-Tactile-v0`：触觉混合控制环境。

## 数据与模型说明

- 标准 LIBERO 数据通常使用 7D/8D task-space 动作。
- ContactForce 和 tactile Tabero 数据在包含力时使用 13D `7dpf` 动作。
- OpenPI client 会向模型 server 发送 RGB 图像、腕部图像、task-space state、语言 prompt，以及可选的力觉/触觉字段。

精确命令模板和常见问题请以 `docs/` 下的专题文档为准。
