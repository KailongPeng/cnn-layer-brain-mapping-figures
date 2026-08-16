import sys, os
import numpy as np

current_dir = os.getcwd()

assert current_dir.endswith('localize_paper')

sys.path.append(current_dir)

from utils import setupPaths

[subjectDir, scratchResultFolder] = setupPaths()

jobarrayDict = np.load(f"Sfig3/Sfig3b/nsd_repStability_RSA_sl/"
                       f"randomise__repStability_RSA_searchlight_sl_jobArray_whichSet_1_{int(float(sys.argv[2]))}.npy",
                       allow_pickle=True)

jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]

jobarrayID = int(float(sys.argv[1]))
[radius, whichSet] = jobarrayDict[jobarrayID]
print(f"{radius} {whichSet} {scratchResultFolder}")