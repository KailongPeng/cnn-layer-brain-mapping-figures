import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import urllib.request
from tqdm import tqdm
from pprint import pprint
from os.path import join, exists, split
import sys

print(sys.executable)
print(sys.version)
print(sys.version_info)
import pandas as pd
import argparse
import random
from glob import glob
import subprocess
from tqdm import tqdm
import pickle5 as pickle
import time
import warnings

print(f"numpy version = {np.__version__}")
print(f"pandas version = {pd.__version__}")

current_dir = os.getcwd()

assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures')), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "figures/figure_04_nsd_localization/analysis"))

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, getSubs, setupPaths

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")


whichSet_1 = int(float(sys.argv[1]))
testMode = sys.argv[2] == "True"
print(f"whichSet_1={whichSet_1} testMode={testMode}")


repeated3imgIDs = [3050, 3158, 3172, 3435, 3627, 3688, 3810, 3848, 3857, 4052, 4250, 4424, 4668, 4691, 4787, 4836, 4893,
                   4931, 5035, 5543, 5584, 5603, 6432, 6445, 6490, 6522, 7121, 7367, 7655, 7841, 7860, 8007, 8110, 8510,
                   8647, 8808, 9028, 9049, 9435, 9918, 10047, 10065, 10108, 10245, 10472, 11160, 11522, 11618, 11690,
                   11797, 11838, 11845, 11933, 11943, 11953, 12309, 12797, 12923, 13139, 13313, 13315, 14529, 14595,
                   14794, 14821, 14899, 15365, 15507, 15820, 15940, 16253, 16345, 16703, 16824, 16918, 17049, 17375,
                   17486, 17550, 19075, 19182, 19200, 19934, 20055, 20081, 20207, 20266, 20634, 20651, 20821, 21193,
                   21195, 21219, 21602, 21951, 22264, 22516, 22531, 22588, 22795, 22880, 22958, 22968, 22994, 23037,
                   23716, 23730, 24203, 24215, 24531, 25112, 25269, 25285, 25589, 25703, 25742, 26164, 26352, 26436,
                   26459, 26721, 26896, 26910, 26991, 27243, 27276, 27327, 27436, 27569, 27879, 28069, 28097, 28160,
                   28190, 28320, 28350, 28525, 28596, 28752, 29382, 29712, 29887, 30082, 30396, 30408, 30673, 30857,
                   31029, 31234, 31660, 31748, 31802, 31838, 31937, 31965, 32054, 32233, 32626, 32773, 32892, 32911,
                   33172, 33246, 33522, 33544, 34127, 34239, 34830, 34851, 35744, 35753, 35799, 35987, 36577, 36624,
                   36732, 36911, 36946, 36975, 36979, 37060, 37225, 37495, 38023, 38026, 38298, 38487, 38642, 38830,
                   38854, 38979, 39290, 39370, 39403, 40424, 40576, 40722, 40771, 40841, 40847, 40925, 40936, 41117,
                   41163, 41567, 41654, 41711, 41928, 42167, 42172, 42215, 42225, 42564, 42698, 42782, 42852, 42913,
                   43160, 43165, 43429, 43446, 43620, 43676, 43819, 43951, 44053, 44721, 44845, 44981, 45130, 45214,
                   45596, 45751, 45844, 45946, 46151, 46373, 46461, 46463, 47291, 47294, 47409, 47688, 48158, 48261,
                   48375, 48423, 48509, 48680, 49467, 49732, 49945, 49956, 50115, 50654, 50735, 50756, 50812, 51078,
                   51149, 51522, 51789, 51984, 52071, 52555, 52932, 53053, 53371, 53728, 53729, 53882, 54079, 54258,
                   54362, 55108, 55406, 55409, 55670, 55679, 55969, 56155, 56682, 56724, 56785, 56851, 56868, 56912,
                   57047, 57436, 57554, 57651, 57839, 59047, 59195, 59285, 59586, 59700, 59995, 60095, 60187, 60252,
                   60506, 60554, 61134, 61511, 61678, 61739, 61810, 62016, 62229, 62303, 62684, 64296, 64868, 64881,
                   65149, 65254, 65268, 65446, 65770, 65800, 65873, 65944, 66005, 66217, 66279, 66331, 66465, 66490,
                   66774, 66837, 66947, 67238, 67296, 67743, 67803, 67830, 68279, 68340, 68815, 68843, 68859, 69008,
                   69241, 69840, 69855, 70096, 70194, 70233, 70428, 71233, 71242, 71411, 71895, 72016, 72210, 72313,
                   72949]

from utils import get_nsd_Subs
subs = get_nsd_Subs()


def sl(testMode=False, subs=None, whichSet_1=None):
    jobarrayDict = {}
    jobarrayID = 1
    setOf16Images = repeated3imgIDs[16 * whichSet_1: 16 * (whichSet_1 + 1)]
    print(f"currSet = {whichSet_1}: {setOf16Images}")
    for sub in subs:
        for radius in [3]:
            for method in ['reg']:
                def prepareModelFingerprints(whichSet):
                    if not os.path.exists(
                            f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/"
                            f"nsd/modelData/data/ActualFingerprints_whichSet_{whichSet}.npy"):
                        [meanMax_dict] = load_obj(
                            f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/nsd/"
                            f"modelData/data/ActualFingerprints_whichSet_{whichSet}")
                        ActualFingerprints = meanMax_dict['ActualFingerprints'][0]
                        np.save(f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/nsd/"
                                f"modelData/data/ActualFingerprints_whichSet_{whichSet}.npy", ActualFingerprints)

                prepareModelFingerprints(whichSet_1 + 1)
                for layer in range(1, 13):
                    jobarrayDict[jobarrayID] = [sub, radius, method, layer, whichSet_1 + 1, setOf16Images, testMode]
                    jobarrayID += 1
    np.save(f"figures/figure_04_nsd_localization/analysis/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/sl_jobArray_whichSet_1_{whichSet_1}.npy",
            jobarrayDict)
    if testMode:
        cmd = (f"sbatch --requeue --array=1-1 "
               f"figures/figure_04_nsd_localization/analysis/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/sl.sh {whichSet_1}")
    else:
        cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
               f"figures/figure_04_nsd_localization/analysis/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/sl.sh {whichSet_1}")
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


sl(testMode=testMode, subs=subs, whichSet_1=whichSet_1)

print("done")