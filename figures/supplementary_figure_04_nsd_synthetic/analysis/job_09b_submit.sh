#!/bin/bash
# ============================================================
# 一键提交 analysis_09b 全部并行 jobs
# 用法：bash job_09b_submit.sh
# ============================================================
set -e

SCRIPT_DIR=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses

# 1. Group A+B（快速，< 5 min）
JOB_AB=$(sbatch --parsable "$SCRIPT_DIR/job_09b_AB.sh")
echo "Submitted groupAB:  job $JOB_AB"

# 2. Bootstrap × 10 并行（依赖 AB 完成，确保 OUTPUT_DIR 已存在）
JOB_BS=$(sbatch --parsable \
    --dependency=afterok:$JOB_AB \
    "$SCRIPT_DIR/job_09b_bootstrap.sh")
echo "Submitted bootstrap array:  job $JOB_BS (array 0-9, depends on $JOB_AB)"

# 3. Merge（依赖所有 bootstrap array jobs 完成）
JOB_MERGE=$(sbatch --parsable \
    --dependency=afterok:$JOB_BS \
    "$SCRIPT_DIR/job_09b_merge.sh")
echo "Submitted merge:  job $JOB_MERGE (depends on $JOB_BS)"

echo ""
echo "Pipeline:"
echo "  $JOB_AB (groupAB) → $JOB_BS (bootstrap ×10) → $JOB_MERGE (merge+plot)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    $SCRIPT_DIR/logs/job_09b_*.out"
