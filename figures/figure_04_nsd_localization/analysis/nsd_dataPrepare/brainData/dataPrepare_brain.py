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
from utils import custom_and

subs = getSubs()
[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

jobarrayDict = load_obj(
    f"figures/figure_04_nsd_localization/analysis/nsd_dataPrepare/brainData/nsd_dataPrepare_brain_whichSet_1_{int(float(sys.argv[2]))}")

jobarrayID = int(float(sys.argv[1]))
[imgID, sub_i] = jobarrayDict[jobarrayID]

print(f"imgID={imgID} sub_i={sub_i}")

nsd_result_folder = f"{scratchResultFolder}/nsd_result_folder/"
two_mm = f"{nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}_2mm.nii.gz"

if not os.path.exists(two_mm):
    def dataPrepare_brain(imgID, sub_i):
        if os.path.exists(f"{nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}.npy"):
            print(f"already exists: {nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}.npy")
            return
        sharedImageIndex = pd.read_csv(
            f"figures/figure_04_nsd_localization/analysis/nsd_dataPrepare/sharedImageIndex.csv")
        sharedImageIndex_t = sharedImageIndex[
            custom_and([
                sharedImageIndex['imgID'] == imgID,
                sharedImageIndex['sub_i'] == sub_i])].copy()
        sharedImageIndex_t = sharedImageIndex_t.reset_index()
        print(f"sharedImageIndex_t={sharedImageIndex_t}")

        sub_img_activations = np.zeros((3, 182, 218, 182))
        affine_mat = []
        session_ts = []
        brain = None
        for rep in tqdm(range(len(sharedImageIndex_t))):
            session_t = int(sharedImageIndex_t.loc[rep, 'session_i'])
            session_ts.append(session_t)
            index_t = int(sharedImageIndex_t.loc[rep, 'index'])
            nsd_data_dir = "/gpfs/milgram/data/nsd/nsddata_betas/ppdata/"
            if len(session_ts) >= 2:
                if session_ts[-2] == session_t:
                    pass
                else:
                    print(f"loading {nsd_data_dir}/subj0{sub_i}/MNI/betas_fithrf/"
                          f"betas_session{f'{session_t}'.zfill(2)}.nii.gz")
                    brain_ = nib.load(f"{nsd_data_dir}subj0{sub_i}/MNI/betas_fithrf/"
                                      f"betas_session{f'{session_t}'.zfill(2)}.nii.gz")
                    brain = brain_.get_fdata()
                    affine_mat = brain_.affine
                    print(f"affine_mat={affine_mat}")
            else:
                print(f"loading {nsd_data_dir}subj0{sub_i}/MNI/betas_fithrf/"
                      f"betas_session{f'{session_t}'.zfill(2)}.nii.gz")
                brain_ = nib.load(f"{nsd_data_dir}subj0{sub_i}/MNI/betas_fithrf/"
                                  f"betas_session{f'{session_t}'.zfill(2)}.nii.gz")
                brain = brain_.get_fdata()
                affine_mat = brain_.affine
                print(f"affine_mat={affine_mat}")
            assert brain.shape == (182, 218, 182,
                                   750)
            sub_img_activations[rep, :, :, :] = brain[:, :, :, index_t]

        np.save(f"{nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}.npy", sub_img_activations)
        np.save(f"{nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}_affine_mat.npy", affine_mat)
        print(f"affine_mat={affine_mat}")
        print(f"saved to {nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}.npy")


    dataPrepare_brain(imgID, sub_i)


    def downSample(imgID, sub_i):
        toFill = np.load(f"{nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}.npy")
        toFill = np.transpose(toFill, (1, 2, 3, 0))
        assert toFill.shape == (182, 218, 182, 3)
        affine_mat_path = f"{nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}_affine_mat.npy"
        affine_mat = np.load(affine_mat_path)

        sl_nii = nib.Nifti1Image(toFill.astype('double'), affine_mat)
        nib.save(sl_nii, f"{nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}_1mm.nii.gz")
        print(f"saving to {nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}_1mm.nii.gz")

        one_mm = f"{nsd_result_folder}/brainData/data/sub_i{sub_i}_imgID{imgID}_1mm.nii.gz"

        cmd = f"flirt -in {one_mm} " \
              f"-interp nearestneighbour " \
              f"-ref /gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz " \
              f"-out {two_mm} " \
              f"-applyisoxfm 2"
        custom_run(cmd)

    downSample(imgID, sub_i)

print("done")