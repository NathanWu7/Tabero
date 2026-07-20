<div align="center">

<img src="docs/assets/poster_preview.png" alt="Tabero Overview" width="800"/>

# 🤖 Tabero: Learning Gentle Manipulation with Closed-Loop Force Feedback from Vision, Touch, and Language

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue?style=flat-square&logo=google-scholar)](https://arxiv.org/abs/2605.27886)
[![arXiv](https://img.shields.io/badge/arXiv-2605.27886-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2605.27886)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENCE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=flat-square&logo=python)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.0%2B-orange?style=flat-square&logo=nvidia)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.2%2B-red?style=flat-square)](https://isaac-sim.github.io/IsaacLab/)
[![Code](https://img.shields.io/badge/Code-GitHub-black?style=flat-square&logo=github)](https://github.com/NathanWu7/Tabero-VTLA)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero)

[📄 Paper](https://arxiv.org/abs/2605.27886) •
[📦 Assets](https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero) •
[🖐️ Tactile Assets](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset) •
[🤗 Model Weights](https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero)

</div>

---

## 📖 Abstract

Tactile sensing is essential for robots to achieve human-like gentle manipulation capabilities. However, existing Vision-Language-Action (VLA) models struggle to exploit tactile feedback for gentle manipulation due to the scarcity of aligned vision-tactile-language data and the lack of effective closed-loop force feedback mechanisms.

**Tabero** addresses these challenges with:

- **Tabero Benchmark** — A data-efficient pipeline that repurposes open-source robot manipulation trajectories to generate diverse vision-tactile-language tasks in a high-fidelity tactile simulator (Isaac Lab + Taxim/FOTS), paired with a multidimensional evaluation protocol.
- **Tabero-VTLA** — A Vision-Tactile-Language-Action architecture featuring a **decoupled force-position command interface** executed by a fixed hybrid controller for real-time, force-aware manipulation.

> 🔬 **Key Result:** Our model maintains high task success while **reducing average grip force by over 70%** under gentle instructions, demonstrating its ability to modulate interaction forces based on multimodal experience.

---

## 🎬 Demo: Force-Aware Language-Conditioned Manipulation

Tabero-VTLA modulates grip force according to natural language instructions. Watch how the same task is executed differently under "gentle" vs "firm" language commands.

### 🧀 Task 1: Pick up the cream cheese → Place in basket

<table>
<tr>
<td width="50%" align="center">
  <img src="docs/assets/task1_gentle_success.gif" width="100%" alt="Gentle Success"/><br/>
  <sub>🟢 <b>Gentle</b> — "Gently pick up the cream cheese and place it in the basket."<br/>Significantly reduced grip force ✓</sub>
</td>
<td width="50%" align="center">
  <img src="docs/assets/task1_gentle_failure.gif" width="100%" alt="Gentle Failure"/><br/>
  <sub>🟡 <b>Failure Case</b> Illustrating the gentleness-reliability trade-off</sub>
</td>
</tr>
</table>

### 🍮 Task 7: Pick up the milk → Place in basket

<table>
<tr>
<td width="50%" align="center">
  <img src="docs/assets/task7_firm_success.gif" width="100%" alt="Firm Success"/><br/>
  <sub>🔴 <b>Firm</b> — "Tightly pick up the milk and place it in the basket."<br/>Standard force level applied ✓</sub>
</td>
<td width="50%" align="center">
  <img src="docs/assets/task7_gentle_success.gif" width="100%" alt="Gentle Success"/><br/>
  <sub>🟢 <b>Gentle</b> — "Softly pick up the milk and place it in the basket."<br/>Force reduced by ~70% while maintaining stable grasping ✓</sub>
</td>
</tr>
</table>

---

## ✨ Key Contributions

<div align="center">

| 🏷️ | Contribution | Description |
|:---:|:---|:---|
| **1** | **Tabero Benchmark** | A scalable pipeline repurposing open-source robot trajectories in a high-fidelity tactile simulator (Isaac Lab + Taxim/FOTS) to generate diverse vision-tactile-language datasets, with the **first standardized protocol for quantifying gentleness** in language-conditioned manipulation. |
| **2** | **Tabero-VTLA** | A suite of force-aware VLA models introducing a **decoupled force-position command interface**, enabling substantially reduced contact forces while preserving high task success through closed-loop tactile feedback. |
| **3** | **Comprehensive Evaluation** | New process-aware metrics — **Average/Maximum Grip Force**, **Average/Maximum Applied Force** — going beyond binary success rates to assess the quality of physical interaction. |

</div>

---

## 📦 Installation

### Prerequisites

- **Isaac Sim 5.0+** with **Isaac Lab 2.2+**
- **Python 3.10+**
- **CUDA 12.0+**

### Install Tabero Extension

```bash
# Clone the repository
git clone https://github.com/NathanWu7/TacManip.git
cd TacManip

# Install the Isaac Lab extension
python -m pip install -e source/tac_manip

# Install the OpenPI inference client
python -m pip install -e benchmarks/openpi/openpi-client
```

### Download Datasets

```bash
# LIBERO data (required for all workflows)
huggingface-cli download NathanWu7/Isaaclab_Libero \
  --repo-type dataset \
  --local-dir /path/to/Isaaclab_Libero

# Tactile calibration assets (for tactile environments)
huggingface-cli download china-sae-robotics/Tactile_Manipulation_Dataset \
  --repo-type dataset \
  --local-dir /path/to/Tactile_manipulation_dataset
```

### Setup Symlinks

```bash
LIBERO_DATA=/path/to/Isaaclab_Libero

ln -sfn "$LIBERO_DATA/assembled_hdf5" benchmarks/datasets/libero/assembled_hdf5
ln -sfn "$LIBERO_DATA/USD" benchmarks/datasets/libero/USD
ln -sfn "$LIBERO_DATA/replayed_demos" benchmarks/datasets/libero/replayed_demos
ln -sfn "$LIBERO_DATA/video_datasets" benchmarks/datasets/libero/video_datasets

# Tactile calibration assets
ln -sfn /path/to/Tactile_manipulation_dataset source/tac_manip/tac_manip/assets/data
```

---

## 🚀 Quick Start

### Evaluation with Pre-trained Models

```bash
# Download model weights
huggingface-cli download NathanWu7/pi0_lora_tacfield_tabero \
  --local-dir /path/to/pi0_lora_tacfield_tabero

# Start the model server (from the Tabero-VTLA repository)
# bash server.sh pi0_lora_tacfield_tabero 49999    #49999 denotes to the training step

# Then run evaluation:
python benchmarks/openpi/openpi_inference_client.py

# or batch evaluation:
python scripts/tools/run_task_evaluations.py
```

### Training

Model training code is maintained in the companion repository **[NathanWu7/Tabero-VTLA](https://github.com/NathanWu7/Tabero-VTLA)**. This repository provides the Isaac Lab environments, data conversion tools, and inference client.

---

## 📊 Experiment Results

### ⚙️ Reproduction Notes

> **Paper results (Table 3)** were obtained with **Isaac Lab 2.2 + Isaac Sim 5.0**, with the contact force sensor bound to the **gelpad** (the sensor contact surface).

> In **Isaac Lab 2.3 + Isaac Sim 5.1**, binding the force sensor to the gelpad fails due to an **unknown bug**. A viable workaround is to bind the sensor to the **minicase** (the sensor housing) instead — this introduces some performance degradation but produces functional results (see the Minicase Rerun table below).

> ⚠️ **If you encounter or solve this force sensor binding issue, please help by opening a GitHub Issue — contributions are greatly appreciated!**

---

<details open>
<summary><b>Paper Table 3 — Main Results (Isaac Lab 2.2 + Isaac Sim 5.0, gelpad binding)</b></summary>
<br/>

*F/G = Firm/Gentle prompts. SR = Success Rate. AG = Average Grip-force. See the paper for full details.*

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
| **Field+FS** | **0.86** | **0.52** | **32.4** | **3.7** |

</details>

<details>
<summary><b>Local Minicase Rerun (Isaac Lab 2.3 + Isaac Sim 5.1, minicase binding workaround)</b></summary>
<br/>

*9 tasks, 450 total trials. `minicase` refers to the sensor housing. `AG pred` = model-predicted, `AG meas` = contact-force measured.*

| Variant | Model | F SR | G SR | F AG pred | G AG pred | F AG meas | G AG meas |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| minicase_k09 | Force E+FS enc10 | 0.789 | 0.316 | 29.06 | 3.73 | 20.19 | 1.87 |
| minicase_k09 | Img+FS | 0.860 | 0.331 | 31.91 | 3.97 | 20.57 | 2.45 |
| minicase_k09 | **Field+FS** | **0.911** | **0.358** | **33.77** | **6.58** | **20.76** | **4.49** |

</details>

---

## 📂 Repository Map

| Directory | Description |
|:---|:---|
| `source/tac_manip/` | Tabero Isaac Lab extension — tasks, assets, environment registration |
| `scripts/tools/` | Collection, replay, evaluation, visualization, and upload scripts |
| `benchmarks/common/` | Converters from Isaac-side HDF5/video to LeRobot/OpenPI datasets |
| `benchmarks/openpi/` | Tabero OpenPI inference client and debug utilities |
| `benchmarks/datasets/` | Expected local data layout for LIBERO, Tabero, and converted datasets |
| `docs/` | All user-facing documentation |
| `envs/` | Reference conda environment snapshots |
| `docker/` | Docker build files |
| `tests/` | Unit tests |

---

## 🎮 Environment IDs

| Environment ID | Description |
|:---|:---|
| `Isaac-Libero-Franka-Replay-Camera-v0` | Standard Franka replay with cameras |
| `Isaac-Libero-Franka-IK-v0` | Standard task-space DiffIK environment |
| `Isaac-Libero-Franka-OscPose-v0` | OSC pose-control environment |
| `Isaac-Libero-Franka-Replay-Camera-ContactForce-v0` | Replay with contact-force observations |
| `Isaac-Libero-Franka-Hybrid-ContactForce-v0` | Hybrid force-position control with contact force |
| `Isaac-Libero-Franka-Replay-Camera-Tactile-v0` | Replay with GelSight tactile sensors |
| `Isaac-Libero-Franka-Hybrid-Tactile-v0` | Hybrid tactile environment |

---

## 📚 Documentation

Detailed workflow documentation is organized under [`docs/`](docs/):

| Document | Language |
|:---|:---|
| [Isaac-Libero Workflow](docs/LIBERO_WORKFLOW.en.md) | English |
| [Tools Guide](docs/TOOLS.md) | English |
| [Benchmarks & Data Conversion](docs/BENCHMARKS.md) | English / [中文](docs/BENCHMARKS.zh-CN.md) |
| [OpenPI Inference Guide](docs/OPENPI.md) | English / [中文](docs/OPENPI.zh-CN.md) |
| [Reproduction Guide](docs/REPRODUCTION.md) | English / [中文](docs/REPRODUCTION.zh-CN.md) |

---

## 🔗 Related Repositories

| Repository | Description |
|:---|:---|
| [NathanWu7/Tabero-VTLA](https://github.com/NathanWu7/Tabero-VTLA) | Tabero VTLA model training and serving code |
| [NathanWu7/pi0_lora_tacfield_tabero](https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero) | Pre-trained LoRA model weights |
| [NathanWu7/Isaaclab_Libero](https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero) | LIBERO benchmark data for Isaac Lab |
| [china-sae-robotics/Tactile_Manipulation_Dataset](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset) | Tactile calibration dataset |

---

## 📝 Citation

If you find Tabero useful in your research, please cite:

```bibtex
@misc{wu2026taberolearninggentlemanipulation,
      title={Tabero: Learning Gentle Manipulation with Closed-Loop Force Feedback from Vision, Touch, and Language}, 
      author={Qiwei Wu and Rui Zhang and Xin Xiang and Tao Li and Weihua Zhang and Junjie Lai and Renjing Xu},
      year={2026},
      eprint={2605.27886},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.27886}, 
}
```

<div align="center">
  <sub>If you use the tactile simulation pipeline, please also cite Taxim and FOTS_Tactile.</sub>
</div>

---

## 📄 License

This project is released under the [Apache License 2.0](LICENCE).

---

<div align="center">
  <sub>🤖 Built with <a href="https://developer.nvidia.com/isaac-sim">Isaac Sim</a> · <a href="https://isaac-sim.github.io/IsaacLab/">Isaac Lab</a> · <a href="https://github.com/NathanWu7/Tabero-VTLA">Tabero-VTLA</a></sub>
  <br/>
  <sub>English | <a href="docs/README.zh-CN.md">中文</a></sub>
</div>
