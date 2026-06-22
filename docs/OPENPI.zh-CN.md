# OpenPI（TacManip）推理指南（傻瓜版）

[English](OPENPI.md) | 中文

这个文档说明如何用 TacManip 的 Isaac Lab 环境连接外部 OpenPI 模型服务，进行闭环推理评测。

你需要区分两件事：

- **OpenPI Server**：模型推理服务，在 TacManip 外部运行，提供 `host:port`。
- **TacManip Client**：本仓库的 `benchmarks/openpi/openpi_inference_client.py`，负责启动 Isaac Sim / Isaac Lab 环境、采集观测、调用 server 推理，并把动作执行到环境里。

Tabero 当前推荐的 OpenPI-side service 是修改版仓库 [`NathanWu7/T2-VLA`](https://github.com/NathanWu7/T2-VLA)。[`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi) 是上游参考，不是本文推荐直接用于 Tabero 的服务端。

---

## 一、10 分钟 Quickstart（复制粘贴就能跑）

### 1）准备 TacManip 侧环境

在 TacManip 仓库根目录执行，确保当前 Python 是装了 Isaac Sim / Isaac Lab 的运行环境：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -m pip install -e benchmarks/openpi/openpi-client
```

### 2）准备数据（可选但强烈推荐）

如果你希望每次 reset 都从数据集里的初始状态开始，设置：

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
# 或：
source scripts/tools/set_replay_env.sh inference
```

说明：

- 该目录下应包含按任务命名的文件：`{task_suite}_task{task_id}_*_demo.hdf5`
- client 会从这个目录里自动找对应 task 的 HDF5，并读取 episode 的 `initial_state`
- 也可用 CLI：`--hdf5-folder /path/to/...`，它会写回 `HDF5_TRAJ_SOURCE_DIR`

### 3）启动 T2-VLA OpenPI service（必须）

先下载 `diffik` / `osc` smoke test 使用的 no-tactile 模型：

```bash
hf download NathanWu7/pi0_lora_notac_tabero \
  --local-dir /path/to/models/pi0_lora_notac_tabero \
  --include 'checkpoints/pi0_lora_notac_tabero/pi0_lora_notac_tabero/49999/params/**' \
  --include 'checkpoints/pi0_lora_notac_tabero/pi0_lora_notac_tabero/49999/assets/**' \
  --include 'norm_stats/**'
```

然后在 T2-VLA 仓库中启动服务：

```bash
cd /path/to/T2-VLA

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi0_lora_notac_tabero \
  --policy.dir=/path/to/models/pi0_lora_notac_tabero/checkpoints/pi0_lora_notac_tabero/pi0_lora_notac_tabero/49999
```

T2-VLA 的 `serve_policy.py` 会监听 `0.0.0.0`。TacManip client 默认连接：

```text
server_host = 127.0.1.1
server_port = 8000
```

如果服务端端口不是 `8000`，client 侧也必须同步改 `--server_port`。如果服务端不在本机，把 `--server_host` 改成服务端实际 IP。

### 4）跑一次 `diffik` 推理

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0 \
  --server_host 127.0.1.1 \
  --server_port 8000
```

正常启动后，终端会看到 prompt、单次实验结果，以及 `Success rate` 等汇总信息。单次 smoke test 失败不一定代表链路错误，重点先确认 client 能连上 server 并完成一轮推理。

---

## 二、OpenPI Service 与模型选择

### 1）通用启动模板

在 T2-VLA 仓库中，任意 checkpoint 都可以按下面模板启动：

```bash
cd /path/to/T2-VLA

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=<config_name> \
  --policy.dir=/path/to/checkpoint_step
```

`--policy.config` 和 `--policy.dir` 必须匹配同一个模型。`--policy.dir` 必须指向具体 checkpoint step 目录，该目录下应有 `params/` 和 `assets/`。

### 2）模型与 `control_mode` 对应关系

| TacManip client `control_mode` | 推荐服务端模型 | 说明 |
| --- | --- | --- |
| `diffik` | [`NathanWu7/pi0_lora_notac_tabero`](https://huggingface.co/NathanWu7/pi0_lora_notac_tabero) | 纯视觉 / 7D 动作，不发送触觉字段 |
| `osc` | [`NathanWu7/pi0_lora_notac_tabero`](https://huggingface.co/NathanWu7/pi0_lora_notac_tabero) | 纯视觉 / 7D 动作，和 `diffik` 复用同一服务端 |
| `tactile` | [`NathanWu7/pi0_lora_tacfield_tabero`](https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero) | 使用 `tactile_marker_motion`、触觉图和力历史 |
| `hybrid` | force-compatible checkpoint | 需要能读取 `gripper_force` 的模型，不要误用 tacfield/no-tactile checkpoint |

如果用 `diffik` / `osc` client 去连 `pi0_lora_tacfield_tabero`，server 会因为缺少 `tactile_marker_motion` 报错。反过来，如果用 `tactile` client 去连 no-tactile 模型，触觉输入会被模型忽略。

### 3）`tactile` 服务端示例

先下载 tacfield 权重：

```bash
hf download NathanWu7/pi0_lora_tacfield_tabero \
  --local-dir /path/to/models/pi0_lora_tacfield_tabero \
  --include 'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params/**' \
  --include 'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/assets/**' \
  --include 'norm_stats/**'
```

启动 tacfield service：

```bash
cd /path/to/T2-VLA

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi0_lora_tacfield_tabero \
  --policy.dir=/path/to/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999
```

---

## 三、推理流程

### 1）整体链路

- **(A) Client 启动 Isaac Sim**：创建环境（`diffik` / `osc` / `hybrid` / `tactile`）
- **(B) Client 读取数据（可选）**：从 `HDF5_TRAJ_SOURCE_DIR` 或 `--hdf5-folder` 找到对应 HDF5，加载 episode 初始状态
- **(C) 循环推理**：
  - 从环境里读观测（相机、状态、力、触觉 / marker motion 等）
  - 打包成 OpenPI 输入字典 `element`
  - 调用 `client.infer(element)` 拿到动作序列（server 返回 32D padded action chunk）
  - client 根据控制模式切片动作维度并执行到环境（7D 或 13D）
- **(D) 成功判定**：连续成功 `num_success_steps` 认为该 experiment 成功

### 2）client 输入给 OpenPI 的字段

TacManip client 会同时发送顶层 key 和 `observation/...` 兼容 key。T2-VLA 当前读取顶层 key。

所有模式都会发送：

- **`image` / `observation/image`**：主相机 RGB，`uint8`，形状 `(224,224,3)`
- **`wrist_image` / `observation/wrist_image`**：腕相机 RGB，`uint8`，形状 `(224,224,3)`
- **`state` / `observation/state`**：7D task-space state：`[x,y,z, ax,ay,az, gripper_abs]`，`float32`
- **`prompt`**：语言指令（从任务配置读取）

不同模式额外字段：

- **`control_mode=hybrid`**
  - **`gripper_force` / `observation/gripper_force`**：力历史 `(H,6)`，`float32`，按 `[fL(3), fR(3)]`
- **`control_mode=tactile`**
  - **`tactile_image` / `observation/tactile_image`**：触觉图 mosaic，`uint8`，`(224,224,3)`
  - **`tactile_gripper_force` / `observation/tactile_gripper_force`**：力历史 `(H,6)`，`float32`
  - **`tactile_marker_motion` / `observation/tactile_marker_motion`**：marker motion 历史 `(1+H, 2*M, 2)`，`float32`

---

## 四、常用模式：怎么选、怎么跑

### 1）`diffik`（推荐先跑通）

- **服务端模型**：`pi0_lora_notac_tabero`
- **用途**：纯视觉 + task-space 控制（7D）
- **动作**：OpenPI 输出 32D，client 取前 7D，再转成 8D quaternion 送进环境控制器

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0 \
  --server_host 127.0.1.1 \
  --server_port 8000
```

### 2）`osc`

- **服务端模型**：`pi0_lora_notac_tabero`
- **用途**：纯视觉 + OSC task-space 控制（7D）

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode osc \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0 \
  --server_host 127.0.1.1 \
  --server_port 8000
```

### 3）`hybrid`（Tabero-force / ContactForce）

- **服务端模型**：需要 force-compatible checkpoint
- **用途**：力-位混合控制（13D），额外喂 `gripper_force` 历史

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode hybrid \
  --task_suite libero_10 \
  --task_id 0 \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0 \
  --server_host 127.0.1.1 \
  --server_port 8000
```

### 4）`tactile`（Tabero tactile）

- **服务端模型**：`pi0_lora_tacfield_tabero`
- **用途**：触觉 + marker motion + 力历史（13D）
- **依赖**：环境中必须存在触觉传感器（默认 `gsmini_left/gsmini_right`）

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode tactile \
  --task_suite libero_10 \
  --task_id 0 \
  --tactile_output_type tactile_rgb \
  --num_total_experiments 1 \
  --max_inference_steps 30 \
  --debug_mode 0 \
  --server_host 127.0.1.1 \
  --server_port 8000
```

可选参数（一般不用改）：

- `--tactile_sensor_names gsmini_left gsmini_right`
- `--force_history_len 8`
- `--marker_history_len 8`

---

## 五、常用参数

- **`--server_host / --server_port`**：OpenPI server 地址（默认 `127.0.1.1:8000`）
- **`--task_suite / --task_id`**：选择任务（会用于加载语言指令 + HDF5 初始状态）
- **`--num_total_experiments`**：跑多少条独立尝试
- **`--max_inference_steps`**：每条最多推理多少个 action chunk
- **`--replan_steps`**：每个 chunk 执行多少步（默认 10）
- **`--debug_mode`**：0 最干净；2/3 会保存图片/动作（文件多）
- **`--replay_mode`**：对照模式（执行 GT actions，同时也跑推理做对比）

---

## 六、常见问题

- **找不到 HDF5 / 不加载初始状态**
  - 确认已 `export HDF5_TRAJ_SOURCE_DIR=...`（通常为 `assembled_hdf5`），或传入 `--hdf5-folder`
  - 确认目录里存在：`{task_suite}_task{task_id}_*_demo.hdf5`

- **`KeyError: "TaberoTacFieldInputs expects 'tactile_marker_motion' in data."`**
  - 原因：server 使用了 `pi0_lora_tacfield_tabero`，但 client 没有用 `--control_mode tactile`
  - 解决：如果跑 `diffik` / `osc`，服务端改用 `pi0_lora_notac_tabero`；如果要用 tacfield 模型，client 改用 `--control_mode tactile`

- **tactile 模式报传感器不存在 / output key 不存在**
  - 确认环境是 tactile 环境（`control_mode=tactile` 会自动映射到 Hybrid-Tactile env）
  - 确认触觉传感器名是 `gsmini_left/gsmini_right`（或用 `--tactile_sensor_names` 改成你的）
  - `--tactile_output_type` 必须是该传感器实际输出的 key（常用：`tactile_rgb` / `markers_rgb`）

- **checkpoint 路径错误**
  - `--policy.dir` 必须指向具体 step 目录，例如 `.../49999`
  - 该目录下应有 `params/` 和 `assets/`

- **OpenPI server 连不上**
  - 检查 T2-VLA service 是否还在运行
  - 检查服务端 `--port` 和 client `--server_port` 是否一致
  - 跨机器运行时，把 `--server_host` 改成 server 实际 IP（不要只用 `127.0.1.1`）

---

## 七、（可选）训练/微调说明

Tabero 模型训练、微调和 OpenPI service 侧代码请参考：

- [`NathanWu7/T2-VLA`](https://github.com/NathanWu7/T2-VLA)
- 上游参考：[`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi)
