import os
os.chdir("/gpfs/milgram/project/turk-browne/projects/localize_paper")


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

from nilearn import datasets
from nilearn import plotting
from nilearn import surface
import numpy as np
import nibabel as nib

subs = getSubs()
[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")



def Sfig3a():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_repStabilityMask_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    plot_surface_repStabilityMask_pdf(resultType="repStabilityMask-synthesized-tfce-masked-tstat",
                                      views=['lateral', 'medial', 'posterior'], cmap=continuous_Cmap,
                                      vmax=14,
                                      figureName="Sfig3a_"
                                      )


Sfig3a()


def Sfig3b():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_repStabilityMask_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)

    plot_surface_repStabilityMask_pdf(resultType="repStabilityMask-nsd-tfce-masked-tstat",
                                      views=['lateral', 'medial', 'posterior'], cmap=continuous_Cmap,
                                      vmax=14,
                                      figureName="Sfig3b_"
                                      )


Sfig3b()


def fig3a():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    valueRange = plot_surface_saveImageReload_pdf(resultType="synthesized: repStabilityMasked TFCE masked tstat",
                                                  views=['posterior'],
                                                  cmap=continuousCmap,
                                                  colorBar=colorBar,
                                                  vmax=8,
                                                  forceRun=True,
                                                  figureName="fig3a_"
                                                  )

fig3a()


def fig3b():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    plotMaximalLayer_pdf(resultType="synthesized: maximally fit layer",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap,
                         figureName="fig3b_")

fig3b()


def fig4a():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    forceRun = True
    valueRange = plot_surface_saveImageReload_pdf(
        resultType="NSD nonresidual mean+randomise: repStabilityMasked TFCE masked tstat",
        views=['posterior'], cmap=continuousCmap, colorBar=colorBar,
        vmax=22,
        forceRun=forceRun,
        figureName="fig4a_"
    )


fig4a()


def fig4b():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    plotMaximalLayer_pdf(resultType="NSD nonresidual: maximally fit layer",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap,
                         figureName="fig4b_")


fig4b()


def fig7a():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    forceRun = True
    valueRange = plot_surface_saveImageReload_pdf(
        resultType="NSD residual mean+randomise: repStabilityMasked TFCE masked tstat",
        views=['posterior'], cmap=continuousCmap, colorBar=colorBar,
        vmax=21,
        forceRun=forceRun,
        figureName="fig7a_"
    )


fig7a()


def fig7b():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    plotMaximalLayer_pdf(resultType="NSD residual: maximally fit layer",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap,
                         figureName="fig7b_")

fig7b()


def Sfig4():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    colorBar = True
    forceRun = True
    colorBarColorNum = 7
    Cmap_16 = designCmap(cmap='gist_rainbow', colors=colorBarColorNum)
    valueRange = plot_surface_saveImageReload_pdf(resultType="NSD nonresidual",
                                                  views=['posterior'],
                                                  cmap=Cmap_16, colorBar=colorBar,
                                                  vmax=7,
                                                  colorBarColorNum=colorBarColorNum,
                                                  forceRun=forceRun,
                                                  figureName="Sfig4_"
                                                  )


Sfig4()


def Sfig5():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    colorBar = True
    forceRun = True
    colorBarColorNum = 7
    Cmap_16 = designCmap(cmap='gist_rainbow', colors=colorBarColorNum)
    valueRange = plot_surface_saveImageReload_pdf(resultType="NSD residual",
                                                  views=['posterior'], cmap=Cmap_16, colorBar=colorBar,
                                                  vmax=7,
                                                  colorBarColorNum=colorBarColorNum,
                                                  forceRun=forceRun,
                                                  figureName="Sfig5_"
                                                  )



Sfig5()


def Sfig6a_v1():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    forceRun = True
    thresholdName=0.001
    valueRange = plot_surface_saveImageReload_pdf(
        resultType=f"synthesized-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
        views=['posterior'], cmap=continuousCmap, colorBar=colorBar,
        vmax=8,
        thresholdName=thresholdName,
        forceRun=forceRun,
        figureName="Sfig6a_v1_"
    )


Sfig6a_v1()


def Sfig6a_v2():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    thresholdName = 0.001
    plotMaximalLayer_pdf(resultType=f"synthesized-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap,
                         thresholdName=thresholdName,
                         figureName="Sfig6a_v2_")

Sfig6a_v2()


def Sfig6b_v1():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    forceRun = True
    thresholdName=0.001
    valueRange = plot_surface_saveImageReload_pdf(
        resultType=f"nsd-nonresidual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
        views=['posterior'], cmap=continuousCmap, colorBar=colorBar,
        vmax=22,
        thresholdName=thresholdName,
        forceRun=forceRun,
        figureName="Sfig6b_v1_"
    )


Sfig6b_v1()


def Sfig6b_v2():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    thresholdName = 0.001
    plotMaximalLayer_pdf(resultType=f"nsd-nonresidual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap,
                         thresholdName=thresholdName,
                         figureName="Sfig6b_v2_")

Sfig6b_v2()


def Sfig6c_v1():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    forceRun = True
    thresholdName=0.001
    valueRange = plot_surface_saveImageReload_pdf(
        resultType=f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
        views=['posterior'], cmap=continuousCmap, colorBar=colorBar,
        vmax=21,
        thresholdName=thresholdName,
        forceRun=forceRun,
        figureName="Sfig6c_v1_"
    )


Sfig6c_v1()


def Sfig6c_v2():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    thresholdName = 0.001
    plotMaximalLayer_pdf(resultType=f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap,
                         thresholdName=thresholdName,
                         figureName="Sfig6c_v2_")

Sfig6c_v2()