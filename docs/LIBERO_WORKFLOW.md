# 原版 LIBERO 使用手册

本文只介绍本仓库里原版 LIBERO 的标准使用流程：

- 标准 Franka
- 标准 LIBERO `assembled_hdf5`
- 7D task-space 状态与动作
- OpenPI 推理默认使用 `diffik`

如果你只想跑通原版 LIBERO，看这一份就够了。

## 1. 先搞清楚要用到的几个环境

本仓库里，原版 LIBERO 常用的是这 3 个环境：

- `Isaac-Libero-Franka-Replay-Camera-v0`
  - 用途：把现有 LIBERO 轨迹在 Isaac 中回放，并重新导出带相机的新 HDF5。
- `Isaac-Libero-Franka-IK-v0`
  - 用途：标准 task-space 控制环境，OpenPI 推理默认推荐先用它。
- `Isaac-Libero-Franka-OscPose-v0`
  - 用途：可选的 task-space 控制环境，只有在你明确想测试 `osc` 时再使用。

如果你的目标是“先把原版 LIBERO 跑通”，推荐默认路线是：

1. 先下载并软链接原版 LIBERO 数据和默认资产目录。
2. 直接使用现成 LIBERO 数据时，只需要做训练、推理或评测。
3. 只有在你确实要重新 replay 采集时，才单独指定新的输出目录。
4. 用 OpenPI server + `openpi_inference_client.py` 在 `diffik` 模式下推理。
5. 用 `run_task_evaluations.py` 做批量评测。

## 2. 环境准备

建议在仓库根目录执行以下命令。

### 2.1 Python 与仓库

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -m pip install -e benchmarks/openpi/openpi-client
```

### 2.2 下载并软链接默认数据目录

在 `qiwei` 分支下，先下载两套数据：

1. HuggingFace 数据集 [`NathanWu7/Isaaclab_Libero`](https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero)
   - 其中需要 `assembled_hdf5/` 和 `USD/`
2. `Tactile_manipulation_dataset`
   - 这里把它作为工程默认 `assets/data` 目录使用

推荐在仓库根目录执行：

```bash
ln -sfn /path/to/Isaaclab_Libero benchmarks/datasets/libero
ln -sfn /path/to/Tactile_manipulation_dataset source/tac_manip/tac_manip/assets/data
```

完成后，工程默认会从下面两个位置读取资源：

```text
benchmarks/datasets/libero/assembled_hdf5
source/tac_manip/tac_manip/assets/data
```

### 2.3 执行 `set_env`

软链接完成后，原版 LIBERO 一般不需要再手动设置额外的 LIBERO 全局路径变量。直接执行：

```bash
source scripts/tools/set_replay_env.sh inference
```

这会把原版 LIBERO 的 `assembled_hdf5` 指向仓库默认软链接目录，并清掉 replay 输出相关变量，适合：

- 直接使用现成 LIBERO 数据训练
- 推理
- 批量评测

建议确认一下默认数据是否就位：

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

### 2.4 可选：查看默认的 LIBERO replay 目录配置

如果你只是想看工程里原版 LIBERO 的整套默认目录配置，可以执行：

```bash
source scripts/tools/set_replay_env.sh libero
```

这个 profile 会把以下目录都指向默认的 `benchmarks/datasets/libero`：

- `HDF5_TRAJ_SOURCE_DIR`
- `OUTPUT_REPLAYED_DEMOS_DIR`
- `OUTPUT_REPLAYED_VIDEOS_DIR`
- `REPLAYED_DEMOS_DIR`

但要注意，这个默认目录通常就是你软链接过去的现成数据目录。

## 3. 什么时候需要 replay 采集

如果你希望**直接使用现成的 LIBERO 数据训练**，那么通常不需要 replay 再采集，直接跳到后面的训练、推理或评测步骤即可。

只有在下面这些情况下，才建议重新 replay 采集：

- 你想重新导出视频
- 你想生成自己的一套 `replayed_demos`
- 你想验证回放链路

最重要的一点：

- 如果你要重新采集自己的 LIBERO replay 数据，请**不要直接使用**默认的 `source scripts/tools/set_replay_env.sh libero`
- 因为它默认指向软链接后的 `benchmarks/datasets/libero`
- 这会让你新生成的 `replayed_demos/` 和 `video_datasets/` 落到默认数据目录下

## 4. 采集原版 LIBERO 7D 数据

### 4.1 推荐方式：回放已有 LIBERO 轨迹并重新采集

原版 LIBERO 的标准 7D 采集，推荐使用：

- 环境：`Isaac-Libero-Franka-Replay-Camera-v0`
- `recorder_type`：`7dp`

其中 `7dp` 表示：

- `position(3)`
- `axis-angle(3)`
- `abs gripper(1)`

合起来是 7D。

### 4.2 先单独设置输出目录

如果你确定要 replay 采集，请手动指定一个和默认软链接目录分开的输出根目录。

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

这里推荐手动 `export`，而不是直接使用默认的 `libero` profile。

### 4.3 采集单个任务

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

这条命令做的事情是：

- 从 `HDF5_TRAJ_SOURCE_DIR` 找到 `libero_10` 的 `task 0` 对应 demo
- 在 Isaac 中回放
- 输出新的 HDF5 到 `OUTPUT_REPLAYED_DEMOS_DIR`
- 输出相机视频到 `OUTPUT_REPLAYED_VIDEOS_DIR`

### 4.4 采集整个 suite

如果你希望批量重采一个 suite，可以只给 `task_suite`，不写 `task_id`：

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

### 4.5 手动遥操作录制

如果你不是回放开源轨迹，而是自己录数据，可以用：

```bash
python scripts/tools/record_demos.py \
  --task Isaac-Libero-Franka-IK-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --teleop_device spacemouse \
  --num_demos 5 \
  --dataset_file ./output/manual_demo.hdf5
```

这条路线适合：

- 快速录少量人工示例
- 验证环境和控制器是否正常
- 做小规模数据补充

## 5. 原版 LIBERO 数据应该长什么样

对原版 LIBERO 的 7D 流程来说，重点看下面几个字段：

- `data/demo_<k>/actions`
- `data/demo_<k>/obs/eef_pose`
- `data/demo_<k>/obs/gripper_pos`

其中：

- `eef_pose` 通常是 `(T, 7)`，表示 `pos(3) + quat(4)`
- 转换脚本会把姿态统一整理成 `axis-angle`
- 最终 state/action 都会整理成 7D：
  - `[x, y, z, ax, ay, az, gripper]`

如果你后面打算转 OpenPI/LeRobot，建议始终保持：

- 采集时用 `7dp`
- 推理时默认用 `diffik`

## 6. 转换成 LeRobot / OpenPI 训练格式

原版 LIBERO 的 7D 数据，使用下面这个脚本转换：

```bash
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --data_root /path/to/libero_replay
```

最小可用输入通常包括：

- `/path/to/libero_replay/replayed_demos`
- `/path/to/libero_replay/video_datasets`

这个转换脚本会把数据整理成：

- `state`: 7D
- `action`: 7D
- 相机图像序列

如果你只想先验证转换是否正常，推荐先只采一个任务、几条 demo，再执行转换。

## 7. 启动 OpenPI server

`openpi_inference_client.py` 只是客户端。真正的模型推理服务需要你先在 OpenPI 仓库侧启动。

你最终需要确认两件事：

- `server_host`
- `server_port`

默认值是：

```text
server_host = 127.0.1.1
server_port = 8000
```

如果你的 server 不是这个地址，记得在推理命令里显式改掉。

## 8. 单任务推理

### 7.1 推荐先跑 `diffik`

原版 LIBERO 的首选推理命令：

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0
```

这条命令会自动把环境设成：

- `Isaac-Libero-Franka-IK-v0`

如果你已经执行过：

```bash
source scripts/tools/set_replay_env.sh inference
```

那么 client 会自动从 `HDF5_TRAJ_SOURCE_DIR` 中找到对应任务的 HDF5，并读取其中的初始状态。

### 7.2 可选：使用 `osc`

如果你已经跑通 `diffik`，并且只是想额外测试 OSC 控制，可以改成：

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode osc \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0
```

这条命令会自动使用：

- `Isaac-Libero-Franka-OscPose-v0`

### 7.3 推理时 client 读入的核心输入

原版 LIBERO 推理里，client 会给 OpenPI 发送这些核心字段：

- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `prompt`

其中：

- `observation/state` 是 7D task-space state
- `prompt` 来自任务配置文件里的语言指令

## 9. 批量评测

如果你已经确认单任务能跑通，就可以用批量评测脚本。

### 8.1 评一个任务

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --task_ids 1 \
  --num_total_experiments 5 \
  --headless
```

### 8.2 评一个 suite

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --num_total_experiments 5 \
  --headless
```

### 8.3 评多个 suite

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal libero_10 libero_spatial libero_object \
  --num_total_experiments 5 \
  --headless
```

评测脚本会：

- 逐个任务启动 `benchmarks/openpi/openpi_inference_client.py`
- 统计成功率
- 把结果写到 `evaluation_results/`

## 10. 一条最推荐的完整流程

如果你只是想从零开始，最推荐按下面顺序跑。

### 第一步：下载并软链接默认数据目录

```bash
ln -sfn /path/to/Isaaclab_Libero benchmarks/datasets/libero
ln -sfn /path/to/Tactile_manipulation_dataset source/tac_manip/tac_manip/assets/data
```

### 第二步：执行 `set_env`

```bash
source scripts/tools/set_replay_env.sh inference
```

### 第三步：如果只用现成 LIBERO 数据，可直接跳过 replay 采集

如果你只是直接使用现成的 LIBERO 数据训练或评测，到这里就可以跳过 replay 数据收集。

### 第四步：如果你确实要 replay 采集，手动指定单独目录

```bash
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/libero_replay/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/libero_replay/video_datasets
export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"
```

### 第五步：采 7D replay 数据

```bash
python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --num_envs 1 \
  --video \
  --recorder_type 7dp \
  --dump_data
```

### 第六步：转 LeRobot

```bash
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --data_root /path/to/libero_replay
```

### 第七步：启动 OpenPI server

按 OpenPI 官方仓库的方式启动，并确认 `host` 和 `port`。

### 第八步：跑单任务推理

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --headless
```

### 第九步：做批量评测

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --num_total_experiments 5 \
  --headless
```

## 11. 常见问题

### 11.1 为什么推理时没有从数据集初始状态 reset

通常是因为 `HDF5_TRAJ_SOURCE_DIR` 没设对，或者对应任务的 HDF5 文件不存在。

先检查：

```bash
echo "$HDF5_TRAJ_SOURCE_DIR"
```

再检查目录下是否有类似文件：

```text
libero_goal_task1_..._demo.hdf5
```

### 11.2 为什么转换脚本跳过了某些轨迹

先检查 `actions` 维度是否是标准 7D 或兼容的 8D。原版 LIBERO 这条转换脚本面向的是标准 7D/8D task-space 动作。

最稳妥的采集方式是：

- `Isaac-Libero-Franka-Replay-Camera-v0`
- `--recorder_type 7dp`

### 11.3 为什么批量评测跑了很多任务

因为 `run_task_evaluations.py` 默认会评估可用 suite 中的全部任务。要缩小范围，显式加：

- `--task_suites`
- `--task_ids`

例如：

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --task_ids 1 \
  --num_total_experiments 5 \
  --headless
```

## 12. 最小命令清单

如果你只想抄命令，这里给一版最短路径。

### 环境

```bash
ln -sfn /path/to/Isaaclab_Libero benchmarks/datasets/libero
ln -sfn /path/to/Tactile_manipulation_dataset source/tac_manip/tac_manip/assets/data
source scripts/tools/set_replay_env.sh inference
```

### 采集

```bash
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/libero_replay/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/libero_replay/video_datasets
export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"

python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --num_envs 1 \
  --video \
  --recorder_type 7dp \
  --dump_data
```

### 转换

```bash
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --data_root /path/to/libero_replay
```

### 单任务推理

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --headless
```

### 批量评测

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode diffik \
  --task_suites libero_goal \
  --task_ids 1 \
  --num_total_experiments 5 \
  --headless
```
