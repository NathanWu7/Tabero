## TacManip

TacManip 是基于 NVIDIA **Isaac Sim + Isaac Lab** 的操作数据采集/回放工程，核心目标是把不同来源的轨迹（开源数据 / 手动遥操作）在 Isaac 中复现，并产出统一的数据格式以支持 **OpenPI** 等策略训练与推理评估。

本 README **只聚焦根目录总览**。更细脚本/评估细节请直接看链接（这里不展开）：
- `LIBERO_WORKFLOW.md`
- `scripts/tools/README.md`
- `scripts/tools/README_data_replay_eval.md`
- `benchmarks/openpi/README.md`

---

### 0）环境配置：Isaac Sim + Isaac Lab + 安装本仓库 + `set_replay_env`

#### 0.1 Isaac Sim + Isaac Lab

- **Isaac Sim**：请按 NVIDIA 官方流程安装，并确保版本与 Isaac Lab 兼容。
- **Isaac Lab**：请按官方仓库说明创建 Python 环境并安装依赖：[`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab)。

> 关键点：本仓库脚本通过 Isaac Lab 的 `AppLauncher` 启动仿真，因此运行脚本时务必使用你安装了 Isaac Lab 的 Python（通常是 conda 环境里的 python）。

#### 0.2 安装本仓库（TacManip 扩展）

在仓库根目录执行：

```bash
python -m pip install -e source/tac_manip
```

#### 0.21 模型推理客户端安装

如果你需要运行 OpenPI 推理客户端，请在仓库根目录继续执行：

```bash
python -m pip install -e benchmarks/openpi/openpi-client
```

#### 0.3 原版 LIBERO 资源准备

原版 LIBERO 的完整使用流程（下载、软链接、`set_env`、是否需要 replay、推理、评测）请优先参考：

- `LIBERO_WORKFLOW.md`

在 `qiwei` 分支下，先下载两套 HuggingFace 数据，然后软链接到工程默认目录。

原版 LIBERO 数据集 [`NathanWu7/Isaaclab_Libero`](https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero)，需要其中的：

- `assembled_hdf5/`
- `USD/`

assembled_hdf5为原始libero机械臂末端轨迹数据，不包括图像数据，用于在isaaclab中进行回放式重收集. USD为从原版libero迁移过来的物体assets. 如果你想直接训练和推理，可以直接用我们预先准备好的数据，相机数据在video_datasets目录下，机器人轨迹数据在replayed_demos目录下

- `replayed_demos/`
- `video_datasets/`

推荐做法：

```bash
# 1) 下载 Isaaclab_Libero 到任意本地目录
# 2) 软链接到工程默认位置
ln -sfn /path/to/Isaaclab_Libero benchmarks/datasets/libero
```
#### 0.4  触觉标定资源准备
标定资产仓库 [`Tactile_manipulation_dataset`](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset)，需要把它链接到 TacManip 的默认 assets 目录。

推荐做法：

```bash
# 1) 下载 Tactile_manipulation_dataset 到任意本地目录
# 2) 软链接到工程默认 assets 目录
ln -sfn /path/to/Tactile_manipulation_dataset source/tac_manip/tac_manip/assets/data
```

完成后，工程默认会通过下面两个路径使用数据：

- `benchmarks/datasets/libero/assembled_hdf5`
- `source/tac_manip/tac_manip/assets/data`

这样做之后，原版 LIBERO 流程一般不需要再手动设置额外的 LIBERO 全局路径变量。

#### 0.5 `set_replay_env.sh`：一条命令设置回放/再采集环境变量

如果你当前只关心原版 LIBERO，推荐直接看：

- `LIBERO_WORKFLOW.md`

`scripts/tools/set_replay_env.sh` 是一个“按 profile 设置环境变量”的脚本，必须用 `source` 执行，才能把变量写到当前 shell：

```bash
source scripts/tools/set_replay_env.sh <profile>
```

支持的 `profile`（脚本内定义）：
- `inference | infer`
- `libero`
- `tabero_gentle`
- `tabero_force_gentle`
- `tabero_firm`
- `tabero_force_firm`

它会设置并导出这些变量（非常重要，后续脚本会自动读这些变量）：
- **`HDF5_TRAJ_SOURCE_DIR`**：输入数据源目录（通常指 Libero `assembled_hdf5`）
- **`OUTPUT_REPLAYED_DEMOS_DIR`**：`replay_demos_with_camera.py` 写出的 HDF5 输出目录（`replayed_demos/`）
- **`OUTPUT_REPLAYED_VIDEOS_DIR`**：`replay_demos_with_camera.py` 写出的视频输出目录（`video_datasets/`）
- **`REPLAYED_DEMOS_DIR`**：评估/回放/推理等脚本读取的 HDF5 目录（通常设置为 `OUTPUT_REPLAYED_DEMOS_DIR`）
- **`USE_TABERO_TASKS`**：是否启用 `benchmarks/datasets/tabero/config/tabero_tasks.json` 里定义的 task 子集过滤（`replay_demos_with_camera.py` 会识别）

其中：

- `inference`：只设置原版 LIBERO 的 `assembled_hdf5`，适合直接做训练、推理、评测。
- `libero`：像 tabero 一样，把原版 LIBERO 的默认目录整套设好：
  - `HDF5_TRAJ_SOURCE_DIR -> benchmarks/datasets/libero/assembled_hdf5`
  - `OUTPUT_REPLAYED_DEMOS_DIR -> benchmarks/datasets/libero/replayed_demos/`
  - `OUTPUT_REPLAYED_VIDEOS_DIR -> benchmarks/datasets/libero/video_datasets/`
  - `REPLAYED_DEMOS_DIR -> OUTPUT_REPLAYED_DEMOS_DIR`
- `tabero_*`：逻辑不变。

> 原版 LIBERO 的重要提醒：
>
> - 如果你希望直接使用现成的 LIBERO 数据训练或评测，则**不需要**做 replay 再采集。
> - 如果你确实要重新采集自己的 LIBERO replay 数据，请**不要直接使用**默认的 `libero` profile 去写回软链接后的默认数据目录。
> - 请改为手动指定单独目录，否则现成数据下的 `replayed_demos/` 和 `video_datasets/` 可能被覆盖。

> 默认推荐流程：
>
> - 原版 LIBERO 训练 / 推理 / 评测：`source scripts/tools/set_replay_env.sh inference`
> - 原版 LIBERO 查看默认 replay 目录配置：`source scripts/tools/set_replay_env.sh libero`
> - 原版 LIBERO 重新采集自己的 replay 数据：不要用默认 `libero` profile，改用手动 `export`
>
> **手动 export 模板（用于单独回放采集自己的 LIBERO 数据）**
>
> ```bash
> export HDF5_TRAJ_SOURCE_DIR="$(pwd)/benchmarks/datasets/libero/assembled_hdf5"
> export OUTPUT_REPLAYED_DEMOS_DIR=<PATH_TO_OUTPUT>/replayed_demos
> export OUTPUT_REPLAYED_VIDEOS_DIR=<PATH_TO_OUTPUT>/video_datasets
> export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"
> export USE_TABERO_TASKS=0
> ```

---

### 1）环境介绍（Gym Env IDs）

所有下列环境都在 `source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/__init__.py` 注册。

#### 1.1 标准 Franka（无触觉）：joint / IK / OSC

- **`Isaac-Libero-Franka-Replay-Camera-v0`**
  - **用途**：标准 Franka 的 **joint-space** 数据收集/回放（JointPositionController），带相机
- **`Isaac-Libero-Franka-IK-v0`**
  - **用途**：标准 Franka 的 **task-space** 数据收集/回放（DiffIKController）
- **`Isaac-Libero-Franka-OscPose-v0`**
  - **用途**：标准 Franka 的 **task-space** 数据收集/回放（Operational Space / OSC Pose 控制）

#### 1.2 ContactSensor：标准 Franka + 指尖接触力（ContactForce）

- **`Isaac-Libero-Franka-Replay-Camera-ContactForce-v0`**
  - **用途**：joint-space 控制 + contact sensor 的标准 Franka 数据收集/回放
- **`Isaac-Libero-Franka-Hybrid-ContactForce-v0`**
  - **用途**：IK 控制（混合力-位 hybrid force-position）+ contact sensor 的标准 Franka 数据收集/回放
- **`Isaac-Libero-Franka-IK-Camera-ContactForce-v0`**
  - **用途**：IK 控制（纯位置）+ contact sensor 的标准 Franka 数据收集/回放

> 你提到的 `Isaac-Libero-Franka-IK-Camera-Contactsensor-v0` 在代码里实际注册名为 **`Isaac-Libero-Franka-IK-Camera-ContactForce-v0`**（语义一致）。

#### 1.3 TactileSensor：改装 Franka + GelSight Mini

- **`Isaac-Libero-Franka-Replay-Camera-Tactile-v0`**
  - **用途**：joint-space 控制 + tactile sensor 的改装 Franka 数据收集/回放
- **`Isaac-Libero-Franka-Hybrid-Tactile-v0`**
  - **用途**：IK 控制（混合力-位 hybrid force-position）+ tactile sensor 的改装 Franka 数据收集/回放
- **`Isaac-Libero-Franka-IK-Camera-Tactile-v0`**
  - **用途**：IK 控制（纯位置）+ tactile sensor 的改装 Franka 数据收集/回放

---

### 2）数据收集方式（两条主线）

如果你要跑的是原版 LIBERO，请直接跳到专门文档：

- `LIBERO_WORKFLOW.md`

#### 2.1 开源轨迹 replay 再采集（推荐）：`replay_demos_with_camera.py`

目标：把开源轨迹（例如 Libero `assembled_hdf5`）在 Isaac 中回放，并导出新的 `replayed_demos/*.hdf5`（可选导出视频/触觉视频）。

先设置输出目录（建议）：

```bash
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets
```

再设置输入目录（开源轨迹源）：

```bash
source scripts/tools/set_replay_env.sh inference
```

如果你要重新采集自己的 LIBERO replay 数据，建议把输出目录单独指定到一个新位置，不要直接写回默认软链接目录。

示例 A：标准 Franka joint-space 回放再采集（无触觉/力）

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

示例 B：ContactForce 回放再采集（建议 13D：`7dpf`）

```bash
python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-ContactForce-v0 \
  --task_suite libero_goal \
  --task_id 5 \
  --num_envs 1 \
  --recorder_type 7dpf \
  --dump_data
```

示例 C：Tactile 回放再采集（建议 13D：`7dpf`）

```bash
python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
  --task_suite libero_10 \
  --task_id 0 \
  --num_envs 1 \
  --video \
  --tactile_sensor_list gsmini_left gsmini_right \
  --tactile_output_type tactile_rgb \
  --recorder_type 7dpf \
  --dump_data
```

#### 2.2 手动遥操作录制：`record_demos.py`

目标：通过键盘 / SpaceMouse 等遥操作设备手动录制轨迹，输出一个 HDF5 文件。

示例：SpaceMouse + IK 环境

```bash
python scripts/tools/record_demos.py \
  --task Isaac-Libero-Franka-IK-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --teleop_device spacemouse \
  --num_demos 5 \
  --dataset_file ./output/manual_demo.hdf5
```

---

### 3）数据格式与动作格式（`recorder_type`）

#### 3.1 HDF5 的关键字段（转换脚本会用到）

典型 HDF5 字段（只列关键）：
- **动作**：`data/demo_<k>/actions`，形状 `(T, D)`
- **末端位姿**：`data/demo_<k>/obs/eef_pose`，形状 `(T, 7)` = `pos(3) + quat(wxyz, 4)`
- **夹爪开合**：`data/demo_<k>/obs/gripper_pos`（脚本通常取第 0 维作为标量）
- **接触力（可选）**：`data/demo_<k>/obs/gripper_net_force`，形状通常 `(T, H_sensor, 2, 3)`
- **触觉 marker motion（可选）**：`data/demo_<k>/obs/gripper_marker_motion`，形状 `(T, 2, 2, M, 2)`

#### 3.2 `replay_demos_with_camera.py` 的 `recorder_type`（动作维度）

`replay_demos_with_camera.py` 支持：
- **`7d2`**：axis-angle(3) + position(3) + binary gripper(1) = **7D**
- **`8d2`**：quat(4) + position(3) + binary gripper(1) = **8D**
- **`8dp`**：quat(4) + position(3) + abs gripper(1) = **8D**
- **`7dp`**：axis-angle(3) + position(3) + abs gripper(1) = **7D**
- **`7dpf`**：axis-angle(3) + position(3) + abs gripper(1) + Force(6) = **13D**

经验法则：
- **标准 Franka（无触觉/力）**：用 `7dp`（或 `8dp/7d2/8d2`）
- **ContactForce / Tactile**：如果要把力写进 action（13D），用 **`7dpf`**

#### 3.3 三类数据内容对比

- **标准 Franka（无触觉/力）**
  - **动作**：7D/8D（不含 force）
  - **obs**：`eef_pose`、`gripper_pos`
  - **视频（可选）**：agentview / eye_in_hand
- **标准 Franka + ContactSensor（tabero_force）**
  - **obs**：额外 `gripper_net_force`（录制阶段通常只保留当前帧；历史窗口在转换阶段离线构造）
  - **动作（推荐）**：13D（`7dpf`）= `[pos(3), axis-angle(3), gripper_abs(1), fL(3), fR(3)]`
- **改装 Franka + TactileSensor（tabero）**
  - **触觉图像**：`tactile_rgb` / `markers_rgb`（视频形式写到 `tactile_outputs/`）
  - **触觉力场（marker motion field）**：`obs/gripper_marker_motion`
  - **夹爪力**：`obs/gripper_net_force`（可用于 action 的 13D 或离线构造力历史）

补充：力坐标系一致性  
标准 Franka 与改装 Franka 的 `gripper_net_force` 都会被旋转到统一的夹爪局部坐标系，因此在相同接触条件下可直接对比分量语义。

---

### 4）convert 数据转换 + 上传 HuggingFace

#### 4.1 标准 Libero → LeRobot（无 force 动作）

```bash
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --data_root /path/to/your_dataset_root
```

> 注意：该脚本 **不支持 13D（带 force）动作**。如果你的 `actions` 是 13D，请使用下面两个脚本。

#### 4.2 标准 Franka + ContactSensor（tabero_force）→ LeRobot

```bash
python benchmarks/common/convert_all_libero_to_tabero_force.py \
  --data_root /path/to/tabero_force_root \
  --output_dir /path/to/output_dir \
  --repo_name tabero_force_all_libero_suites
```

#### 4.3 改装 Franka + TactileSensor（tabero）→ LeRobot

```bash
python benchmarks/common/convert_all_libero_to_tabero.py \
  --data_root /path/to/tabero_root \
  --output_dir /path/to/output_dir \
  --repo_name tabero_all_libero_suites
```

#### 4.4 上传到 HuggingFace（推荐只上传 `data/` 与 `meta/`）

```bash
python scripts/tools/upload_lerobot_to_hf.py \
  --local-path /path/to/lerobot_dataset_root \
  --repo-id your_username/your_dataset \
  --include-subdirs data meta
```

---

### 5）OpenPI：训练与推理（openpi inference）

如果你只需要原版 LIBERO 的训练、推理和评测路径，推荐优先阅读：

- `LIBERO_WORKFLOW.md`

OpenPI 在本仓库的主要使用方式：
- **训练**：先用第 4 部分把数据转换成 LeRobot 格式，再按 OpenPI 官方仓库训练/微调：[`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi)
- **推理**：用本仓库的 client 连接外部 OpenPI server，在 Isaac 环境里闭环执行：见 `benchmarks/openpi/README.md`

#### 5.1 推理示例命令（diffik / hybrid / tactile）

可选：如果你希望每次 reset 都从默认 LIBERO 数据里的初始状态开始（更稳定更可复现），推荐直接执行：

```bash
source scripts/tools/set_replay_env.sh inference
```

diffik：

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0
```

hybrid（ContactForce）：

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode hybrid \
  --task_suite libero_10 \
  --task_id 0 \
  --num_total_experiments 5
```

tactile（GelSight）：

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode tactile \
  --task_suite libero_10 \
  --task_id 0 \
  --tactile_output_type tactile_rgb \
  --num_total_experiments 5
```

#### 5.2 批量评测
```bash
python scripts/tools/run_task_evaluations.py   --policy_model openpi --control_mode tactile   --prompt_adverbs firmly tightly   --prompt_seed 0 --headless
```