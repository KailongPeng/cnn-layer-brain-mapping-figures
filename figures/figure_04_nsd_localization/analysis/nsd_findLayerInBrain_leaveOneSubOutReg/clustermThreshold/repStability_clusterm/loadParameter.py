import sys, os
import numpy as np

current_dir = os.getcwd()

assert current_dir.endswith('localize_paper')

sys.path.append(current_dir)

from utils import setupPaths

[subjectDir, scratchResultFolder] = setupPaths()
jobarrayDict = np.load(
    f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/repStability_clusterm/randomise_repStability_allSets_jobArray.npy",
    allow_pickle=True)
jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]

jobarrayID = int(float(sys.argv[1]))
[radius, thresholdName, threshold, testMode] = jobarrayDict[jobarrayID]
print(f"{radius} {thresholdName} {threshold} {testMode} {subjectDir} {scratchResultFolder}")