conda activate isaac5
# set -euo pipefail

# # 用法：
# #   SEEDS="0 1 2" PROMPT_SEED=0 bash eval.sh
# # 目标：不同 --seed（环境/随机初始化等），但 prompt 相关保持固定（--prompt_seed 固定）。
# SEEDS="${SEEDS:-11}"
# PROMPT_SEED="${PROMPT_SEED:-0}"

# for SEED in ${SEEDS}; do
#   echo "=== Running eval with seed=${SEED}, prompt_seed=${PROMPT_SEED} ==="
#   python scripts/tools/run_task_evaluations.py \
#     --policy_model openpi \
#     --control_mode tactile \
#     --seed "${SEED}" \
#     --prompt_seed "${PROMPT_SEED}" \
#     --prompt_adverbs firmly tightly \
#     --headless

#   python scripts/tools/run_task_evaluations.py \
#     --policy_model openpi \
#     --control_mode tactile \
#     --seed "${SEED}" \
#     --prompt_seed "${PROMPT_SEED}" \
#     --prompt_adverbs gently softly \
#     --headless
# done
python scripts/tools/run_task_evaluations.py   --policy_model openpi --control_mode tactile   --prompt_adverbs firmly tightly --prompt_seed 0 --headless
python scripts/tools/run_task_evaluations.py   --policy_model openpi --control_mode tactile   --prompt_adverbs gently softly  --prompt_seed 0 --headless
#python scripts/tools/run_task_evaluations.py   --policy_model openpi --control_mode tactile   --prompt_adverbs ""    --prompt_seed 0 --headless