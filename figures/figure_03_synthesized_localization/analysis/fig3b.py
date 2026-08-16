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

assert current_dir.endswith('localize_paper'), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, getSubs, setupPaths, check_jobIDs
subs = getSubs()
[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")


def display_surface_plot():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    plotMaximalLayer_pdf(resultType="synthesized: maximally fit layer",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap)

display_surface_plot()

print("done")