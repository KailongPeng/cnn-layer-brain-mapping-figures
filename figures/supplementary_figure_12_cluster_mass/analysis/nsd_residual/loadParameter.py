import sys, os
import numpy as np

current_dir = os.getcwd()

assert current_dir.endswith('localize_paper')

sys.path.append(current_dir)

from utils import setupPaths

[subjectDir, scratchResultFolder] = setupPaths()

jobarrayDict = np.load(
    f"nsd_residualModel/clustermThreshold/randomise_findLayerInBrain_allSets_jobArray.npy",
    allow_pickle=True)
jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]


jobarrayID = int(float(sys.argv[1]))
[radius, method, layer, negativeCorrelationFlag, thresholdName, threshold, testMode] = jobarrayDict[jobarrayID]
print(f"{radius} {method} {layer} {negativeCorrelationFlag} {thresholdName} {threshold} {testMode} {subjectDir} {scratchResultFolder}")