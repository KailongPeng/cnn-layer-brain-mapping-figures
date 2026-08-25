import sys, os
import numpy as np

current_dir = os.getcwd()

assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures'))

sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "figures/figure_03_synthesized_localization/analysis"))

from utils import setupPaths

[subjectDir, scratchResultFolder] = setupPaths()

jobarrayDict = np.load(f"figures/figure_03_synthesized_localization/analysis/stability/randomise_repStability_RSA_searchlight_sl_jobArray.npy",
                       allow_pickle=True)
jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]

jobarrayID = int(float(sys.argv[1]))
[radius] = jobarrayDict[jobarrayID]
print(f"{radius} {subjectDir} {scratchResultFolder}")