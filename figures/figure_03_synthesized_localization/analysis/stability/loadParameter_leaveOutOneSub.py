import sys
import numpy as np

jobarrayDict = np.load(f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/"
                       f"randomise_leaveOutOneSub_repStability_RSA_searchlight_sl_jobArray.npy", allow_pickle=True)
jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]

jobarrayID = int(float(sys.argv[1]))
[leaveOutSub, radius] = jobarrayDict[jobarrayID]
print(f"{leaveOutSub} {radius}")