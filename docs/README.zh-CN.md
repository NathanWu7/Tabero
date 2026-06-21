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
