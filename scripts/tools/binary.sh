#!/usr/bin/env bash
#
# 目的：
#   一键分别收集 tabero_binary 的 gentle/firm 两套 replay 再采集数据（动作 7D + binary gripper，recorder_type=7d2）。
#
# 依赖：
#   - scripts/tools/set_replay_env.sh（本仓库已有，负责设置 HDF5_TRAJ_SOURCE_DIR / OUTPUT_REPLAYED_* 等）
#   - scripts/tools/replay_demos_with_camera.py
#
# 用法：
#   bash scripts/tools/binary.sh --task <env_name> --task_suite <suite...> [--task_id <ids...>] [其它 replay_demos_with_camera.py 参数...]
#
# 示例（跑全部 suite，自动子进程遍历 task_id）：
#   bash scripts/tools/binary.sh \
#     --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
#     --task_suite libero_goal libero_spatial libero_object libero_10
#
# 只跑 tabero_tasks.json 子集（需要 env var USE_TABERO_TASKS=1，本脚本提供开关）：
#   bash scripts/tools/binary.sh --use-tabero-tasks \
#     --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
#     --task_suite libero_object
#
# 仅跑 gentle 或 firm：
#   bash scripts/tools/binary.sh --only gentle  --task ... --task_suite ...
#   bash scripts/tools/binary.sh --only firm    --task ... --task_suite ...
#
# 自定义输出根目录（会自动在其下创建 gentle_force/firm_force 两套目录）：
#   bash scripts/tools/binary.sh --output-root-base /data/tabero_binary --task ... --task_suite ...
#
# 注意：
#   - 本脚本默认强制加：--dump_data --video --headless --recorder_type 7d2
#   - 如果你想关闭视频，可以在调用时加：--no-video（会被脚本识别并不传 --video）
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ONLY="all"              # all|gentle|firm
OUTPUT_ROOT_BASE=""     # optional
NO_VIDEO="0"
USE_TABERO_TASKS="0"

print_help() {
  cat <<'EOF'
用法：
  bash scripts/tools/binary.sh [--only gentle|firm] [--output-root-base /abs/path] [--no-video] [--use-tabero-tasks] --task <env> --task_suite <suite...> [其它 replay 参数...]

说明：
  - 默认会跑 gentle + firm 两次（分别写到 tabero_binary/gentle_force 与 tabero_binary/firm_force）。
  - 默认会传：--dump_data --video --headless --recorder_type 7d2
  - --use-tabero-tasks：只跑 benchmarks/datasets/tabero/config/tabero_tasks.json 中列出的 task_id 子集
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    --only)
      ONLY="${2:-}"
      shift 2
      ;;
    --output-root-base)
      OUTPUT_ROOT_BASE="${2:-}"
      shift 2
      ;;
    --no-video)
      NO_VIDEO="1"
      shift 1
      ;;
    --use-tabero-tasks)
      USE_TABERO_TASKS="1"
      shift 1
      ;;
    *)
      ARGS+=("$1")
      shift 1
      ;;
  esac
done

if [[ "${ONLY}" != "all" && "${ONLY}" != "gentle" && "${ONLY}" != "firm" ]]; then
  echo "[binary.sh] --only 只支持: all|gentle|firm (收到: ${ONLY})" >&2
  exit 2
fi

# Ensure required args exist (best-effort)
if ! printf '%s\n' "${ARGS[@]}" | grep -q -- '^--task$'; then
  echo "[binary.sh] 缺少 --task 参数（例如 Isaac-Libero-Franka-Replay-Camera-Tactile-v0）" >&2
  echo "  你传入的参数为：${ARGS[*]-<empty>}" >&2
  exit 2
fi
if ! printf '%s\n' "${ARGS[@]}" | grep -q -- '^--task_suite$'; then
  echo "[binary.sh] 缺少 --task_suite 参数（例如 libero_goal libero_spatial ...）" >&2
  echo "  你传入的参数为：${ARGS[*]-<empty>}" >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
REPLAY_PY="${REPO_ROOT}/scripts/tools/replay_demos_with_camera.py"
SETENV_SH="${REPO_ROOT}/scripts/tools/set_replay_env.sh"

if [[ ! -f "${REPLAY_PY}" ]]; then
  echo "[binary.sh] 找不到 ${REPLAY_PY}" >&2
  exit 2
fi
if [[ ! -f "${SETENV_SH}" ]]; then
  echo "[binary.sh] 找不到 ${SETENV_SH}" >&2
  exit 2
fi

run_one() {
  local kind="$1"        # gentle|firm
  local profile="tabero_binary_${kind}"
  local custom_root=""
  if [[ -n "${OUTPUT_ROOT_BASE}" ]]; then
    custom_root="${OUTPUT_ROOT_BASE}/${kind}_force"
    mkdir -p "${custom_root}/replayed_demos" "${custom_root}/video_datasets"
  fi

  # Enable optional task subset (tabero_tasks.json) used by replay_demos_with_camera.py.
  # set_replay_env.sh respects pre-exported USE_TABERO_TASKS.
  export USE_TABERO_TASKS="${USE_TABERO_TASKS}"

  # shellcheck source=/dev/null
  if [[ -n "${custom_root}" ]]; then
    source "${SETENV_SH}" "${profile}" "${custom_root}"
  else
    source "${SETENV_SH}" "${profile}"
    mkdir -p "${OUTPUT_REPLAYED_DEMOS_DIR}" "${OUTPUT_REPLAYED_VIDEOS_DIR}"
  fi

  echo ""
  echo "[binary.sh] ===== Running profile=${profile} ====="
  echo "[binary.sh] OUTPUT_REPLAYED_DEMOS_DIR=${OUTPUT_REPLAYED_DEMOS_DIR}"
  echo "[binary.sh] OUTPUT_REPLAYED_VIDEOS_DIR=${OUTPUT_REPLAYED_VIDEOS_DIR}"
  echo "[binary.sh] USE_TABERO_TASKS=${USE_TABERO_TASKS}"

  local cmd=("${PYTHON}" "${REPLAY_PY}")
  cmd+=("${ARGS[@]}")
  cmd+=("--dump_data" "--headless" "--recorder_type" "7d2")
  if [[ "${NO_VIDEO}" == "0" ]]; then
    cmd+=("--video")
  fi

  echo "[binary.sh] CMD: ${cmd[*]}"
  "${cmd[@]}"
}

case "${ONLY}" in
  all)
    run_one gentle
    run_one firm
    ;;
  gentle)
    run_one gentle
    ;;
  firm)
    run_one firm
    ;;
esac

echo ""
echo "[binary.sh] Done."


