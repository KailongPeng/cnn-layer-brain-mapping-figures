import os
import sys

current_dir = os.getcwd()
if not current_dir.endswith(("localize_paper", "cnn-layer-brain-mapping-figures")):
    raise RuntimeError(f"run this script from a repository root, not {current_dir}")

sys.path.append(current_dir)
sys.path.append(
    os.path.join(current_dir, "figures/figure_03_synthesized_localization/analysis")
)
from utils import custom_run


AllSubs = sys.argv[1]
fileList = sys.argv[2]


def execute_till_done(cmd):
    while True:
        try:
            custom_run(cmd)
            return
        except Exception:
            pass


files_without_nans = []
for filename in fileList.split():
    clean_filename = f"{filename.split('.nii.gz')[0]}_noNAN.nii.gz"
    execute_till_done(f"fslmaths {filename} -nan {clean_filename}")
    files_without_nans.append(clean_filename)

merge_command = f"fslmerge -t {AllSubs} {' '.join(files_without_nans)}"
print(merge_command)
execute_till_done(merge_command)
