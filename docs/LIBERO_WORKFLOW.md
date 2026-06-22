# Isaac-Libero 使用手册

[English](LIBERO_WORKFLOW.en.md) | 中文

本文只介绍本仓库里的 Isaac-Libero 标准流程：标准 Franka、标准 LIBERO 数据、7D task-space 状态/动作，以及默认 `diffik` OpenPI 推理。

开始之前，请先按根目录 [README](../README.md) 的 `Quick Setup` 配好：TacManip 扩展、OpenPI client、`NathanWu7/Isaaclab_Libero` 数据软链接，以及需要时的触觉标定资源软链接。

## 1. 先搞清楚要用到的几个环境

Isaac-Libero 常用 3 个环境：

- `Isaac-Libero-Franka-Replay-Camera-v0`
  - 用途：把现有 LIBERO 轨迹在 Isaac 中回放，并重新导出带相机的新 HDF5。
- `Isaac-Libero-Franka-IK-v0`
  - 用途：标准 task-space 控制环境，OpenPI 推理默认推荐先用它。
- `Isaac-Libero-Franka-OscPose-v0`
  - 用途：可选的 task-space 控制环境，只有在明确要测试 `osc` 时再使用。

推荐顺序是：先直接使用下载数据做训练、推理和评测；只有在确实需要自己的 replay 输出时，再做数据重收集。

## 2. 设置 Isaac-Libero 环境变量

本节假设你已经在主 README 中完成所有数据下载和软链接。

**使用下载好的数据：训练、推理和评测。** 如果你使用下载数据中的 `assembled_hdf5/`、`replayed_demos/` 和 `video_datasets/`，通常不需要重新采集。直接执行：

```bash
source scripts/tools/set_replay_env.sh inference
```

这个 profile 会把 `HDF5_TRAJ_SOURCE_DIR` 指向默认 `benchmarks/datasets/libero/assembled_hdf5`，并清掉 replay 输出相关变量，适合：

- 直接转换/训练现成 Isaac-Libero 数据
- OpenPI 推理
- 批量评测

可以检查路径是否正确：

```bash
echo "$HDF5_TRAJ_SOURCE_DIR"
```

你应该能看到类似：

```text
.../benchmarks/datasets/libero/assembled_hdf5
```

并且目录下应包含类似文件：

```text
libero_goal_task1_..._demo.hdf5
libero_10_task0_..._demo.hdf5
```

**需要自己重新收集数据。** 如果你想重新 replay 采集自己的 Isaac-Libero 数据，再执行：

```bash
source scripts/tools/set_replay_env.sh libero
```

这个 profile 会把下面这些目录指向默认 `benchmarks/datasets/libero` 布局：

- `HDF5_TRAJ_SOURCE_DIR`
- `OUTPUT_REPLAYED_DEMOS_DIR`
- `OUTPUT_REPLAYED_VIDEOS_DIR`
- `REPLAYED_DEMOS_DIR`

注意：如果你不想覆盖下载数据里的 `replayed_demos/` 和 `video_datasets/`，请在重收集前手动指定单独输出目录。重收集流程放在本文后半部分。

## 3. 直接转换成 LeRobot / OpenPI 训练格式

LeRobot/OpenPI 转换请在隔离的 `tabero_lerobot` 环境中运行，不要在 Isaac 运行环境中运行。`lerobot` 会引入一组可能和 Isaac Sim / Isaac Lab 钉死版本冲突的依赖，因此不要把它装回 Isaac 运行环境。

### 3.1 配置 `tabero_lerobot` 环境

如果本机还没有 `tabero_lerobot`，从仓库根目录用导出的环境文件创建：

```bash
conda env create -f envs/environment-tabero-lerobot.yml
conda activate tabero_lerobot
```

如果环境已经存在，但缺少 `lerobot` / `tyro` 等转换依赖，可以在该环境中用 freeze 文件补齐：

```bash
conda activate tabero_lerobot
python -m pip install -r envs/requirements-tabero-lerobot.txt
```

创建或修复后，先做最小验证：

```bash
python -c "import lerobot, tyro; print('lerobot/tyro ok')"
python -m pip check
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py --help
```

这个环境只用于 LeRobot 数据转换和相关上传/检查工具；不需要 Isaac Sim / Isaac Lab，也不用于启动仿真。

### 3.2 运行转换

请从 `Tabero` 仓库根目录执行：

```bash
conda activate tabero_lerobot
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --data_root benchmarks/datasets/libero \
  --output_dir /tmp/tabero_lerobot_openpi
```

如果你更习惯用 `conda run`，使用：

```bash
conda run --no-capture-output -n tabero_lerobot python \
  benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --data_root benchmarks/datasets/libero \
  --output_dir /tmp/tabero_lerobot_openpi
```

转换脚本会读取 `--data_root` 下的 `replayed_demos/` 和 `video_datasets/`。它不会直接读取 `HDF5_TRAJ_SOURCE_DIR`，`assembled_hdf5/` 主要用于 replay 输入或推理初始状态，不是直接转换 LeRobot 的输入。

最小可用输入是：

- `benchmarks/datasets/libero/replayed_demos`
- `benchmarks/datasets/libero/video_datasets`

如果这些目录是软链接，统计文件数量时用 `find -L`：

```bash
find -L benchmarks/datasets/libero/replayed_demos -maxdepth 1 -name '*.hdf5' | wc -l
find -L benchmarks/datasets/libero/video_datasets -maxdepth 2 -name '*.mp4' | wc -l
```

这个转换脚本会把数据整理成：

- `state`: 7D
- `action`: 7D
- 相机图像序列

如果只是验证转换链路，建议先只转换少量 suite 或少量任务。

常见环境错误：

- `ModuleNotFoundError: No module named 'lerobot'`：命令跑在了错误环境里；切换到 `tabero_lerobot`。
- `ModuleNotFoundError: No module named 'tyro'`：转换环境不完整；用 `python -c "import lerobot, tyro"` 验证。

## 4. 直接跑 OpenPI 推理

`openpi_inference_client.py` 是本仓库的 client。真正的模型推理服务需要先在 OpenPI 仓库侧启动。

默认 server 配置是：

```text
server_host = 127.0.1.1
server_port = 8000
```

如果你的 server 不是这个地址，请在推理命令里显式指定。

### 4.1 推荐先跑 `diffik`

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0
```

这个命令会使用：

- `Isaac-Libero-Franka-IK-v0`

如果已经执行过：

```bash
source scripts/tools/set_replay_env.sh inference
```

client 会自动从 `HDF5_TRAJ_SOURCE_DIR` 中找到对应任务的 HDF5，并读取其中的初始状态。

### 4.2 可选：使用 `osc`

如果 `diffik` 已经跑通，可以额外测试 OSC 控制：

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode osc \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0
```

这个命令会使用：

- `Isaac-Libero-Franka-OscPose-v0`

### 4.3 推理输入字段

Isaac-Libero 推理里，client 会给 OpenPI 发送这些核心字段：

- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `prompt`

其中：

- `observation/state` 是 7D task-space state
- `prompt` 来自任务配置文件里的语言指令

## 5. 直接做批量测试 / 评测

确认单任务推理能跑通后，可以用批量评测脚本。

### 5.1 评一个任务

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --task_ids 1 \
  --num_total_experiments 5 \
  --headless
```

### 5.2 评一个 suite

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --num_total_experiments 5 \
  --headless
```

### 5.3 评多个 suite

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal libero_10 libero_spatial libero_object \
  --num_total_experiments 5 \
  --headless
```

评测脚本会逐个任务启动 `benchmarks/openpi/openpi_inference_client.py`，统计成功率，并把结果写到 `evaluation_results/`。

## 6. Isaac-Libero 数据格式

对 Isaac-Libero 的 7D 流程来说，重点看下面几个字段：

- `data/demo_<k>/actions`
- `data/demo_<k>/obs/eef_pose`
- `data/demo_<k>/obs/gripper_pos`

其中：

- `eef_pose` 通常是 `(T, 7)`，表示 `pos(3) + quat(4)`
- 转换脚本会把姿态统一整理成 `axis-angle`
- 最终 state/action 都会整理成 7D：`[x, y, z, ax, ay, az, gripper]`

## 7. 需要时重新 replay 采集 Isaac-Libero 数据

如果你希望直接使用下载数据训练、推理和评测，可以跳过本节。只有在下面这些情况下才建议重新采集：

- 你想重新导出视频
- 你想生成自己的一套 `replayed_demos`
- 你想验证回放链路

### 7.1 推荐方式：回放已有 LIBERO 轨迹并重新采集

Isaac-Libero 的标准 7D 采集推荐使用：

- 环境：`Isaac-Libero-Franka-Replay-Camera-v0`
- `recorder_type`：`7dp`

其中 `7dp` 表示 `position(3) + axis-angle(3) + abs gripper(1)`，合起来是 7D。

### 7.2 先单独设置输出目录

如果不想覆盖下载数据，建议手动指定一个和默认软链接目录分开的输出根目录：

```bash
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/libero_replay/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/libero_replay/video_datasets
export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"
```

建议目录结构类似：

```text
/path/to/libero_replay/
  replayed_demos/
  video_datasets/
```

### 7.3 采集单个任务

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

这条命令会：

- 从 `HDF5_TRAJ_SOURCE_DIR` 找到 `libero_10` 的 `task 0` demo
- 在 Isaac 中回放
- 输出新的 HDF5 到 `OUTPUT_REPLAYED_DEMOS_DIR`
- 输出相机视频到 `OUTPUT_REPLAYED_VIDEOS_DIR`

### 7.4 采集整个 suite

```bash
python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-v0 \
  --task_suite libero_goal \
  --num_envs 1 \
  --video \
  --recorder_type 7dp \
  --dump_data
```

脚本会自动遍历该 suite 下的所有任务。

### 7.5 手动遥操作录制

如果不是回放开源轨迹，而是自己录数据，可以用：

```bash
python scripts/tools/record_demos.py \
  --task Isaac-Libero-Franka-IK-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --teleop_device spacemouse \
  --num_demos 5 \
  --dataset_file ./output/manual_demo.hdf5
```

这条路线适合快速录少量人工示例、验证环境和控制器是否正常，或做小规模数据补充。

## 8. 一条最推荐的完整流程

如果只是使用下载好的 Isaac-Libero 数据，推荐顺序是：

1. 按根 README 完成数据下载和软链接。
2. 执行 `source scripts/tools/set_replay_env.sh inference`。
3. 直接转换成 LeRobot/OpenPI 训练格式。
4. 启动 OpenPI server。
5. 跑单任务 `diffik` 推理。
6. 跑批量评测。

如果确实要重新采集数据，则在第 2 步改用 `source scripts/tools/set_replay_env.sh libero`，并参考第 7 节配置独立输出目录和 replay 命令。

## 9. 常见问题

### 9.1 为什么推理时没有从数据集初始状态 reset

通常是因为 `HDF5_TRAJ_SOURCE_DIR` 没设对，或者对应任务的 HDF5 文件不存在。

先检查：

```bash
echo "$HDF5_TRAJ_SOURCE_DIR"
```

再检查目录下是否有类似文件：

```text
libero_goal_task1_..._demo.hdf5
```

### 9.2 为什么转换脚本跳过了某些轨迹

先检查 `actions` 维度是否是标准 7D 或兼容的 8D。Isaac-Libero 这条转换脚本面向的是标准 7D/8D task-space 动作。

最稳妥的采集方式是：

- `Isaac-Libero-Franka-Replay-Camera-v0`
- `--recorder_type 7dp`

### 9.3 为什么批量评测跑了很多任务

因为 `run_task_evaluations.py` 默认会评估可用 suite 中的全部任务。要缩小范围，显式加：

- `--task_suites`
- `--task_ids`
