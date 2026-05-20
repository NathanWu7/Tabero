## OpenPI（TacManip）推理指南（傻瓜版）

这个目录包含 OpenPI 的 **推理客户端** 与相关工具，用于在 Isaac Lab 环境里跑闭环推理评估。

你需要搞清楚两件事：
- **OpenPI Server**：模型推理服务（在 TacManip 外部运行，提供 `host:port`）
- **TacManip Client**：本仓库的 `benchmarks/openpi/openpi_inference_client.py`，负责：
  - 启动 Isaac Sim / Isaac Lab 环境
  - 采集相机/触觉/力等观测，打包成 OpenPI 输入字典
  - 调用 OpenPI server 推理得到动作，并执行到环境里

---

## 一、10 分钟 Quickstart（复制粘贴就能跑）

### 1）准备终端环境

在 TacManip 仓库根目录执行（确保你用的是装了 IsaacLab 的 Python 环境）：

```bash
export PYTHONPATH=/home/wqw/git_pkgs/TacManip:$PYTHONPATH
python -m pip install -e benchmarks/openpi/openpi-client
```

### 2）准备数据（可选但强烈推荐）

如果你希望每次 reset 都从数据集里的初始状态开始（更稳定、更可复现），设置（与 `task_configs` / `set_replay_env.sh inference` 一致）：

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
# 或: source scripts/tools/set_replay_env.sh inference
```

说明：
- 该目录下应包含按任务命名的文件：`{task_suite}_task{task_id}_*_demo.hdf5`
- client 会从这个目录里自动找对应 task 的 HDF5，并读取 episode 的 `initial_state`
- 也可用 CLI：`--hdf5-folder /path/to/...`（会写回 `HDF5_TRAJ_SOURCE_DIR`）

### 3）启动 OpenPI server（必须）

请按 OpenPI 官方仓库的方式启动 server，并确保能从本机访问：
- 参考：[`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi)

你只需要最终确认两点：
- server 的地址：`--server_host`（默认 `127.0.1.1`）
- server 的端口：`--server_port`（默认 `8000`）

### 4）跑一次推理（推荐从 diffik 开始）

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0
```

---

## 二、推理流程（看懂这张就不迷路）

### 1）整体链路

- **(A) Client 启动 Isaac Sim**：创建环境（diffik/osc/hybrid/tactile）
- **(B) Client 读取数据（可选）**：从 `HDF5_TRAJ_SOURCE_DIR`（或 `--hdf5-folder`）找到对应 HDF5，加载 episode 初始状态
- **(C) 循环推理**：
  - 从环境里读观测（相机、状态、力、触觉/marker motion 等）
  - 打包成 OpenPI 输入字典 `element`
  - 调用 `client.infer(element)` 拿到动作序列（server 返回 32D padded action chunk）
  - client 根据控制模式切片动作维度并执行到环境（7D 或 13D）
- **(D) 成功判定**：连续成功 `num_success_steps` 认为该 experiment 成功

### 2）client 输入给 OpenPI 的字段（非常重要）

所有模式都会发送：
- **`observation/image`**：主相机 RGB，`uint8`，形状 `(224,224,3)`
- **`observation/wrist_image`**：腕相机 RGB，`uint8`，形状 `(224,224,3)`
- **`observation/state`**：7D task-space state：`[x,y,z, ax,ay,az, gripper_abs]`（axis-angle + gripper），`float32`
- **`prompt`**：语言指令（从任务配置读取）

不同模式额外字段：
- **`control_mode=hybrid`（ContactForce）**
  - **`observation/gripper_force`**：力历史 `(H,6)`，`float32`，按 `[fL(3), fR(3)]`
- **`control_mode=tactile`（Tabero tactile）**
  - **`observation/tactile_image`**：触觉图 mosaic，`uint8`，`(224,224,3)`
  - **`observation/tactile_gripper_force`**：力历史 `(H,6)`，`float32`
  - **`observation/tactile_marker_motion`**：marker motion 历史 `(1+H, 2*M, 2)`，`float32`（第 0 帧为 init）

这些字段与 Tabero 数据转换脚本保持一致（`convert_all_libero_to_tabero.py`）。

---

## 三、三种常用模式：怎么选、怎么跑

### 1）`diffik`（最推荐先跑通）

- **用途**：纯视觉 + task-space 控制（7D）
- **动作**：OpenPI 输出 32D，client 取前 7D，再转成 8D quaternion 送进环境控制器

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal --task_id 1 \
  --num_total_experiments 5
```

### 2）`hybrid`（Tabero-force / ContactForce）

- **用途**：力-位混合控制（13D），额外喂 `gripper_force` 历史

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode hybrid \
  --task_suite libero_10 --task_id 0 \
  --num_total_experiments 5
```

### 3）`tactile`（Tabero tactile）

- **用途**：触觉 + marker motion + 力历史（13D）
- **依赖**：环境中必须存在触觉传感器（默认 `gsmini_left/gsmini_right`）

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode tactile \
  --task_suite libero_10 --task_id 0 \
  --tactile_output_type tactile_rgb \
  --num_total_experiments 5
```

可选参数（一般不用改）：
- `--tactile_sensor_names gsmini_left gsmini_right`
- `--force_history_len 8`
- `--marker_history_len 8`

---

## 四、常用参数（只写你会用到的）

- **`--server_host / --server_port`**：OpenPI server 地址（默认 `127.0.1.1:8000`）
- **`--task_suite / --task_id`**：选择任务（会用于加载语言指令 + HDF5 初始状态）
- **`--num_total_experiments`**：跑多少条独立尝试
- **`--max_inference_steps`**：每条最多推理多少个 action chunk
- **`--replan_steps`**：每个 chunk 执行多少步（默认 10）
- **`--debug_mode`**：0 最干净；2/3 会保存图片/动作（文件多）
- **`--replay_mode`**：对照模式（执行 GT actions，同时也跑推理做对比）

---

## 五、常见问题（按这个查基本都能解决）

- **找不到 HDF5 / 不加载初始状态**
  - 确认已 `export HDF5_TRAJ_SOURCE_DIR=...`（通常为 `assembled_hdf5`），或传入 `--hdf5-folder`
  - 确认目录里存在：`{task_suite}_task{task_id}_*_demo.hdf5`

- **tactile 模式报传感器不存在 / output key 不存在**
  - 确认环境是 tactile 环境（`control_mode=tactile` 会自动映射到 Hybrid-Tactile env）
  - 确认触觉传感器名是 `gsmini_left/gsmini_right`（或用 `--tactile_sensor_names` 改成你的）
  - `--tactile_output_type` 必须是该传感器实际输出的 key（常用：`tactile_rgb` / `markers_rgb`）

- **OpenPI server 连不上**
  - 先用最简单的方式确认 server 在跑：检查 host/port、防火墙、容器端口映射
  - 把 `--server_host` 改成 server 实际 IP（不要只用 127.0.0.1）

---

## 六、（可选）训练/微调说明

训练/微调属于 OpenPI 官方仓库侧内容，这里不再复述。请参考：
- [`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi)
