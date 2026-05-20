## Benchmarks（数据介绍 / 转换 / OpenPI）

本目录主要包含三件事：

- **数据（datasets）**：Libero 原始数据、Tabero/Tabero-force 数据组织、以及转换后的 LeRobot（pi0/OpenPI）格式数据
- **转换（common）**：把 Isaac 侧的 HDF5 + 视频整理成 LeRobot 数据集（parquet + images/meta）
- **OpenPI（openpi）**：推理客户端（client）与调试可视化工具

更细的“采集/回放/评估脚本逻辑”请看：
- `scripts/tools/README.md`

---

### 1）数据介绍（benchmarks/datasets）

#### 1.1 `benchmarks/datasets/libero/`（Libero 数据与资产）

常见子目录含义：
- **`config/`**：任务配置（`libero_goal.json` 等）
- **`assembled_hdf5/`**：开源轨迹源（通常作为 replay 再采集的输入）
- **`USD/`**：场景/物体资产（USD）
- **`copy/`**：打包好的示例数据（zip、视频等）

> 注意：本仓库的数据采集主线是 “`assembled_hdf5` → Isaac 回放再采集 → `replayed_demos` + `video_datasets`”。  
> 这一步在 `scripts/tools/` 完成（见 `scripts/tools/README.md`），benchmarks 侧主要负责**转换**和 **OpenPI**。

#### 1.2 `benchmarks/datasets/tabero/` 与 `benchmarks/datasets/tabero_force/`

这两类目录用于组织“带力/触觉”的再采集输出（具体目录结构因数据源而异，但通常都包含）：
- `replayed_demos/`：HDF5（只包含成功 demo）
- `video_datasets/`：视频（RGB / wrist / tactile_outputs）

#### 1.3 `benchmarks/datasets/tabero_pi0/`（LeRobot/pi0 风格输出）

转换脚本的输出一般长这样（LeRobot 目录）：
- `data/`：parquet（chunk/file）
- `images/`：图片（按 LeRobot 写入格式）
- `meta/`：episodes/tasks 等元信息

---

### 2）数据格式转换（HDF5 + 视频 → LeRobot）

转换脚本位于 `benchmarks/common/`，它们都是 **tyro dataclass CLI**：最简单的方式是先跑 `--help` 看参数名。

```bash
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py --help
```

#### 2.1 `convert_all_libero_to_lerobot_openpi.py`（标准：无 force 动作）

- **适用**：标准 Franka（动作 7D/8D，不含 force）
- **不适用**：如果你的 `actions` 是 13D（例如 `recorder_type=7dpf`），请用 Tabero/Tabero-force 转换脚本

精简指令（默认参数，跑全部 suite）：

```bash
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py
```

全指令（显式指定输入/输出与 suite）：

```bash
python benchmarks/common/convert_all_libero_to_lerobot_openpi.py \
  --task_suites libero_goal libero_10 \
  --data_root benchmarks/datasets/libero \
  --output_dir benchmarks/datasets/tabero_pi0 \
  --repo_name lerobot_all_libero_suites
```

#### 2.2 `convert_all_libero_to_tabero_force.py`（ContactForce：tabero_force）

- **适用**：ContactForce 数据（tabero_force）
- **动作要求**：**必须 13D**（`7dpf`），否则会跳过
- **额外输出**：会离线构造 `observation/gripper_force` 的历史窗口 `(H,6)`

精简指令：

```bash
python benchmarks/common/convert_all_libero_to_tabero_force.py
```

全指令（单数据源）：

```bash
python benchmarks/common/convert_all_libero_to_tabero_force.py \
  --data_root benchmarks/datasets/tabero_force \
  --output_dir benchmarks/datasets/tabero_pi0 \
  --repo_name tabero_force_all_libero_suites \
  --force_history_len 8
```

全指令（双数据源合并：strong/soft，脚本支持）：

```bash
python benchmarks/common/convert_all_libero_to_tabero_force.py \
  --strong_data_root /path/to/strong_root \
  --soft_data_root /path/to/soft_root \
  --output_dir /path/to/output_dir \
  --repo_name tabero_force_all_libero_suites \
  --strong_adverbs firmly tightly \
  --soft_adverbs gently softly \
  --prompt_seed 0
```

#### 2.3 `convert_all_libero_to_tabero.py`（Tactile：tabero）

- **适用**：触觉数据（tabero）
- **动作要求**：**必须 13D**（`7dpf`）
- **额外输出**：
  - `tactile_image`：触觉图像 mosaic（4×4）
  - `tactile_gripper_force`：力历史 `(H,6)`
  - `tactile_marker_motion`：marker motion 历史 `(1+H, 2*M, 2)`

精简指令：

```bash
python benchmarks/common/convert_all_libero_to_tabero.py
```

全指令（单数据源 + 明确触觉类型）：

```bash
python benchmarks/common/convert_all_libero_to_tabero.py \
  --data_root benchmarks/datasets/tabero \
  --output_dir benchmarks/datasets/tabero_pi0 \
  --repo_name tabero_all_libero_suites \
  --tactile_output_type tactile_rgb \
  --force_history_len 8 \
  --marker_history_len 8
```

全指令（双数据源合并：strong/soft）：

```bash
python benchmarks/common/convert_all_libero_to_tabero.py \
  --strong_data_root /path/to/strong_root \
  --soft_data_root /path/to/soft_root \
  --output_dir /path/to/output_dir \
  --repo_name tabero_all_libero_suites \
  --strong_adverbs firmly tightly \
  --soft_adverbs gently softly \
  --prompt_seed 0
```

---

### 3）OpenPI：依赖安装（openpi client）与应用

OpenPI 分两部分：
- **OpenPI server**：外部服务（TacManip 仓库外启动）
- **TacManip client**：本仓库的 `benchmarks/openpi/openpi_inference_client.py`

详细推理说明见：`benchmarks/openpi/README.md`（此处只写安装与最常用入口）。

#### 3.1 安装 openpi client（本仓库侧）

精简指令（在仓库根目录执行）：

```bash
python -m pip install -e benchmarks/openpi/openpi-client
```

全指令（更稳：显式加上 `PYTHONPATH`）：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -m pip install -e benchmarks/openpi/openpi-client
```

#### 3.2 推理：`openpi_inference_client.py`

精简指令（diffik，跑一次）：

```bash
python benchmarks/openpi/openpi_inference_client.py \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 1
```

全指令（显式 server、headless、推理步数与可复现 reset 数据源）：

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5

python benchmarks/openpi/openpi_inference_client.py \
  --server_host 127.0.0.1 \
  --server_port 8000 \
  --control_mode diffik \
  --task_suite libero_goal \
  --task_id 1 \
  --num_total_experiments 50 \
  --max_inference_steps 30 \
  --replan_steps 10 \
  --debug_mode 0 \
  --headless
```

#### 3.3 调试可视化：`visualize_openpi_debug.py`

精简指令：

```bash
python benchmarks/openpi/visualize_openpi_debug.py
```

---

### 4）常见坑（benchmarks 侧）

- **转换脚本跳过很多轨迹**
  - 先检查 `video_datasets` 是否齐全；很多脚本会对齐视频帧与 action 长度，缺视频会直接判 invalid
- **动作维度不匹配**
  - `convert_all_libero_to_lerobot_openpi.py` 不支持 13D（带 force）
  - `convert_all_libero_to_tabero(_force).py` 需要 13D（`7dpf`）
- **OpenPI 不加载 reset 初始状态**
  - 检查 `HDF5_TRAJ_SOURCE_DIR`（或 `--hdf5-folder`）指向的目录是否包含 `{task_suite}_task{task_id}_*_demo.hdf5`


