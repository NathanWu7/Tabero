#!/usr/bin/env bash
#
# 用法（必须 source 才能把环境变量设置到当前 shell）：
#   source scripts/tools/set_replay_env_test.sh <profile>
#
# profile 取值（示例）：
#   - tabero_gentle
#   - tabero_force_gentle
#   - tabero_firm
#   - tabero_force_firm
#
# 会设置并导出：
#   - HDF5_TRAJ_SOURCE_DIR      : 原始 libero assembled_hdf5 数据源目录
#   - OUTPUT_REPLAYED_DEMOS_DIR : 回放再采集输出 HDF5 目录（replayed_demos）
#   - OUTPUT_REPLAYED_VIDEOS_DIR: 回放视频输出目录（video_datasets）
#   - REPLAYED_DEMOS_DIR        : 读取回放后数据的目录（默认等于 OUTPUT_REPLAYED_DEMOS_DIR）
#   - USE_TABERO_TASKS          : 启用 tabero_tasks.json 子集（replay_demos_with_camera.py 自动识别）
#

_set_replay_env_test_restore_opts() {
  if [[ "${_SET_REPLAY_ENV_TEST_HAD_ERRExit:-0}" -eq 1 ]]; then set -e; else set +e; fi
  if [[ "${_SET_REPLAY_ENV_TEST_HAD_NOUNSET:-0}" -eq 1 ]]; then set -u; else set +u; fi
  if [[ "${_SET_REPLAY_ENV_TEST_HAD_PIPEFAIL:-0}" -eq 1 ]]; then set -o pipefail; else set +o pipefail; fi
}
[[ -o errexit ]] && _SET_REPLAY_ENV_TEST_HAD_ERRExit=1 || _SET_REPLAY_ENV_TEST_HAD_ERRExit=0
[[ -o nounset ]] && _SET_REPLAY_ENV_TEST_HAD_NOUNSET=1 || _SET_REPLAY_ENV_TEST_HAD_NOUNSET=0
[[ -o pipefail ]] && _SET_REPLAY_ENV_TEST_HAD_PIPEFAIL=1 || _SET_REPLAY_ENV_TEST_HAD_PIPEFAIL=0
trap '_set_replay_env_test_restore_opts' RETURN
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[set_replay_env_test] 请使用 source 执行，以便环境变量写入当前 shell："
  echo "  source scripts/tools/set_replay_env_test.sh <profile>"
  exit 2
fi

PROFILE="${1:-}"
if [[ -z "${PROFILE}" ]]; then
  echo "[set_replay_env_test] 缺少 profile 参数。示例：tabero_gentle / tabero_force_firm"
  return 2
fi

# 固定数据源（所有 profile 共用）
export HDF5_TRAJ_SOURCE_DIR="/home/wqw/git_pkgs/TacManip/benchmarks/datasets/libero/assembled_hdf5/"

DATASET_KIND=""
FORCE_MODE=""

case "${PROFILE}" in
  tabero_gentle) DATASET_KIND="tabero"; FORCE_MODE="gentle_force" ;;
  tabero_force_gentle) DATASET_KIND="tabero_force"; FORCE_MODE="gentle_force" ;;
  tabero_firm) DATASET_KIND="tabero"; FORCE_MODE="firm_force" ;;
  tabero_force_firm) DATASET_KIND="tabero_force"; FORCE_MODE="firm_force" ;;
  *)
    echo "[set_replay_env_test] 未知 profile: ${PROFILE}"
    echo "  允许值：tabero_gentle | tabero_force_gentle | tabero_firm | tabero_force_firm"
    return 2
    ;;
esac

ROOT="/home/wqw/git_pkgs/TacManip/benchmarks/datasets/${DATASET_KIND}/${FORCE_MODE}"

export OUTPUT_REPLAYED_DEMOS_DIR="${ROOT}/replayed_demos/"
export OUTPUT_REPLAYED_VIDEOS_DIR="${ROOT}/video_datasets/"

# 回放后的数据读取目录（评估/推理脚本读取）
export REPLAYED_DEMOS_DIR="${OUTPUT_REPLAYED_DEMOS_DIR}"

# 启用 Tabero 子集（tabero_tasks.json）
export USE_TABERO_TASKS="0"

echo "[set_replay_env_test] 已设置："
echo "  PROFILE=${PROFILE}"
echo "  HDF5_TRAJ_SOURCE_DIR=${HDF5_TRAJ_SOURCE_DIR}"
echo "  OUTPUT_REPLAYED_DEMOS_DIR=${OUTPUT_REPLAYED_DEMOS_DIR}"
echo "  OUTPUT_REPLAYED_VIDEOS_DIR=${OUTPUT_REPLAYED_VIDEOS_DIR}"
echo "  REPLAYED_DEMOS_DIR=${REPLAYED_DEMOS_DIR}"
echo "  USE_TABERO_TASKS=${USE_TABERO_TASKS}"


