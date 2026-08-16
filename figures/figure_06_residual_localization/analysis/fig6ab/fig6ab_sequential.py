"""
fig6ab_sequential.py — 顺序版本的 fig6ab.py

问题：fig6ab.py 将 16 个 NSD image set 作为一个 job array 同时提交（--array=1-16），
      与 fig4.py 等同时运行时会产生大量并发 job 互相抢占资源。

解决方案：每个 set 单独提交（--array=1-1），等待完成后再提交下一个，
          16 个 set 全部完成后统一跑 nsd_allSetMeanRandomise。

用法（在 localize_paper 目录下运行）：
    source env_setup_analysis.sh
    python3 -u fig6ab/fig6ab_sequential.py

    # 从某个 set 开始断点续跑（例如前 8 个已完成）：
    python3 -u fig6ab/fig6ab_sequential.py --start_set 8
"""
import os
import sys
import argparse

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")

current_dir = os.getcwd()
assert current_dir.endswith('localize_paper'), f"Must run from localize_paper dir, got: {current_dir}"
sys.path.append(current_dir)

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, setupPaths, check_jobIDs

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

parser = argparse.ArgumentParser()
parser.add_argument('--start_set', type=int, default=0, help='从哪个 set 开始（用于断点续跑，默认 0）')
parser.add_argument('--end_set', type=int, default=15, help='跑到哪个 set 结束（包含，默认 15）')
args = parser.parse_args()

sets_to_run = list(range(args.start_set, args.end_set + 1))
print(f"\n将顺序运行 sets: {sets_to_run}\n")

# ── Step 1: 逐 set 顺序提交 ──────────────────────────────────────────────────
for set_id in sets_to_run:
    print(f"\n{'='*60}")
    print(f"  提交 set {set_id} / {args.end_set}")
    print(f"{'='*60}")

    jobarrayDict = {1: [set_id, False]}
    save_obj(jobarrayDict, "nsd_residualModel/nsd_oneSet_jobdict")

    cmd = "sbatch --requeue --array=1-1 nsd_residualModel/nsd_oneSet.sh"
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    print(f"  jobID={jobID}, 等待完成...")
    waitForEnd(jobID)
    completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    print(f"  set {set_id} 完成: {completed}")

# ── Step 2: 所有 set 完成后，跑 allSetMeanRandomise ─────────────────────────
print(f"\n{'='*60}")
print(f"  所有 set 完成，提交 nsd_allSetMeanRandomise (max_set={args.end_set})")
print(f"{'='*60}")

# 注意：原始 fig6ab.py 第80行有 bug（缺少空格），此处已修正
cmd = (f"sbatch --requeue "
       f"nsd_residualModel/nsd_allSetMeanRandomise/nsd_allSetMeanRandomise.sh {args.end_set}")
sbatch_response = custom_run(cmd)
jobID = getjobID_num(sbatch_response)
waitForEnd(jobID)
check_jobIDs([jobID])

print("\n=== fig6ab_sequential.py 完成 ===")
