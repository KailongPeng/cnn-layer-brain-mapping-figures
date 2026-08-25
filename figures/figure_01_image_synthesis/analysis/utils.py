import os
import sys
import re
import pandas as pd
import subprocess
import numpy as np
import time





def save_obj(obj, name):
    import pickle5 as pickle
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    import pickle5 as pickle
    if name[-3:] == 'pkl':
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


def save_nib(_nii, fileName, affine_mat=None):
    import nibabel as nib
    _nii = _nii.astype('double')
    _nii[np.isnan(_nii)] = 0
    sl_nii = nib.Nifti1Image(_nii.astype('double'), affine_mat)
    nib.save(sl_nii, fileName)
    print(f"{fileName} saved")


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def getjobID_num(sbatch_response):
    import re
    jobID = re.findall(r'\d+', sbatch_response)[0]
    return jobID


def custom_and(l):
    t = True
    for i in l:
        t = t * np.asarray(i)
    return t


def custom_or(l):
    t = np.asarray(l[0])
    for i in l:
        t = t + np.asarray(i)
    return t


def custom_rename(file1, file2):
    cmd = f"mv {file1} {file2}";
    print(cmd);
    sbatch_response = subprocess.getoutput(cmd);
    print(sbatch_response)
    return 1


def custom_copy(file1, file2):
    cmd = f"cp {file1} {file2}";
    print(cmd);
    sbatch_response = subprocess.getoutput(cmd);
    print(sbatch_response)
    return 1


def custom_run(cmd):
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    check(sbatch_response)
    return sbatch_response


def custom_remove(fileName):
    cmd = f"rm {fileName}"
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    print(sbatch_response)


def wait(tmpFile, waitFor=0.1):
    while not os.path.exists(tmpFile):
        time.sleep(waitFor)
    return 1


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response or 'Unrecognised' in sbatch_response:
        raise Exception(sbatch_response)


def checkEndwithDone(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                last_line = f.readlines()[-1]
            EndwithDone = last_line == "done\n"
        except UnicodeDecodeError:
            with open(filename, 'br') as f:
                last_line = f.readlines()[-1]
            EndwithDone = last_line == b"done\n"
    else:
        EndwithDone = False
    return EndwithDone


def checkDone(jobIDs):
    from tqdm import tqdm
    from glob import glob
    completed = {}
    for jobID in tqdm(jobIDs):
        filename = f"./logs/*{jobID}*.out"
        filename = glob(filename)
        if len(filename) == 0:
            completed[jobID] = False
        else:
            filename = filename[0]
        completed[jobID] = checkEndwithDone(filename)
    if np.mean(list(completed.values())) == 1:
        status = True
    else:
        status = False
    return completed, status


def check_jobIDs(jobIDs):
    completed, status = checkDone(jobIDs)
    if status == True:
        pass
    else:
        print(completed)
        assert status == True
    return completed


def check_jobArray(jobID='', jobarrayNumber=10):
    arrayIDrange = np.arange(1, 1 + jobarrayNumber)
    jobIDs = []
    for arrayID in arrayIDrange:
        jobIDs.append(f"{jobID}_{arrayID}")
    completed = check_jobIDs(jobIDs)
    return completed


def waitForEnd(jobID):
    while jobID_running_myjobs(jobID):
        print(f"waiting for {jobID} to end")
        time.sleep(5)
    print(f"{jobID} finished")


def jobID_running_myjobs(jobID):
    jobID = str(jobID)
    cmd = f"squeue -j {jobID} -h"
    sbatch_response = subprocess.getoutput(cmd)
    return jobID in sbatch_response


def readtxt(file):
    f = open(file, "r", encoding="utf8")
    return f.read()


def writetxt(file, content):
    with open(file, 'w', encoding="utf8") as f:
        f.write(content)

def getMonDate():
    from datetime import date
    import datetime
    today = date.today()
    month_num = str(today.month)
    datetime_object = datetime.datetime.strptime(month_num, "%m")
    month_name = datetime_object.strftime("%b")
    if len(str(today.day)) == 1:
        return f"{month_name}  {today.day}"
    else:
        return f"{month_name} {today.day}"


def checkDate(csvPath, assertTrue=True):
    cmd = f"ls -al {csvPath}";
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd);
    print(sbatch_response)
    print(f"getMonDate = {getMonDate()}")
    if assertTrue:
        assert (("Oct 10" in sbatch_response) or
                ("Oct 11" in sbatch_response) or
                ("Oct 12" in sbatch_response) or
                ("Oct 13" in sbatch_response))
        newFlag = (("Oct 10" in sbatch_response) or
                   ("Oct 11" in sbatch_response) or
                   ("Oct 12" in sbatch_response) or
                   ("Oct 13" in sbatch_response))
    else:
        newFlag = (("Oct 10" in sbatch_response) or
                   ("Oct 11" in sbatch_response) or
                   ("Oct 12" in sbatch_response) or
                   ("Oct 13" in sbatch_response))
    return newFlag


def cal_resample(data=None, times=5000, return_all=False):
    if data is None:
        raise Exception
    if type(data) == list:
        data = np.asarray(data)
    iter_mean = []
    for _ in range(times):
        iter_distri = data[np.random.choice(len(data), len(data), replace=True)]
        iter_mean.append(np.nanmean(iter_distri))
    _mean = np.mean(iter_mean)
    _5 = np.percentile(iter_mean, 5)
    _95 = np.percentile(iter_mean, 95)
    if return_all:
        return _mean, _5, _95, iter_mean
    else:
        return _mean, _5, _95


def bar(means=None, upper=None, lower=None, ROINames=None, title=None, xLabel="", yLabel="", fontsize=50):
    import matplotlib.pyplot as plt
    if type(means) == list:
        means = np.asarray(means)
    if type(upper) == list:
        upper = np.asarray(upper)
    if type(means) == list:
        lower = np.asarray(lower)

    plt.figure(figsize=(4, 3), dpi=70)
    positions = list(np.arange(len(means)))

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.bar(positions, means, yerr=[means - lower, upper - means], align='center', alpha=0.5, ecolor='black',
           capsize=10)
    ax.set_ylabel(yLabel, fontsize=fontsize)
    ax.set_xlabel(xLabel, fontsize=fontsize)
    ax.set_xticks(positions)
    if ROINames is not None:
        xtick = ROINames
        ax.set_xticklabels(xtick, fontsize=fontsize, rotation=45, ha='right')
    ax.set_title(title, fontsize=fontsize)
    ax.yaxis.grid(True)
    _ = plt.tight_layout()
    _ = plt.show()


def getSubs(minusSub=None):
    subjects = ['sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub007', 'sub008', 'sub009', 'sub010', 'sub011',
                'sub012', 'sub013', 'sub014', 'sub015', 'sub016', 'sub017', 'sub019', 'sub020', 'sub021', 'sub022',
                'sub023', 'sub024', 'sub025', 'sub027', 'sub028', 'sub029', 'sub030', 'sub031', 'sub032', 'sub033']
    subjects = [value for value in subjects if value != minusSub]
    return subjects


def get_nsd_Subs(minusSub=None):
    subjects = [f"sub_i{_sub_i_}" for _sub_i_ in range(1, 9)]
    subjects = [value for value in subjects if value != minusSub]
    return subjects


def setupPaths():
    scratchResultFolder = f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/"
    subjectDir = "/gpfs/milgram/project/turk-browne/projects/localize_paper/subjects/"
    return [subjectDir, scratchResultFolder]


def print_nifti_info(file_path, check_assertions=True):
    import nibabel as nib
    img = nib.load(file_path)

    header = img.header

    print(f"File path: {file_path}")
    print(f"Data type: {header.get_data_dtype()}")
    print(f"Image dimensions: {header.get_data_shape()}")
    print(f"Voxel size: {header.get_zooms()}")

    affine = img.affine
    print(f"Affine matrix: \n{affine}")

    orientation = nib.aff2axcodes(affine)
    print(f"Orientation: {orientation}")

    if check_assertions:
        expected_dtype = 'float32'
        expected_shape = (91, 109, 91)
        expected_zooms = (2.0, 2.0, 2.0)
        expected_affine = [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
        expected_orientation = ('L', 'A', 'S')

        assert header.get_data_shape() == expected_shape, f"Image dimensions mismatch: {header.get_data_shape()} != {expected_shape}"
        assert header.get_zooms() == expected_zooms, f"Voxel size mismatch: {header.get_zooms()} != {expected_zooms}"
        assert all([all(a == b) for a, b in
                    zip(affine, expected_affine)]), f"Affine matrix mismatch: \n{affine} != \n{expected_affine}"
        assert orientation == expected_orientation, f"Orientation mismatch: {orientation} != {expected_orientation}"


def cfg_loading(pathname):
    import toml

    class StructDict(dict):
        '''Class that adds a structure type syntax to dictionaries,
           i.e. 'dict.field' will invoke dict['field']
        '''

        def __getattr__(self, key):
            '''Implement getattr to support syntax "data.field"'''
            try:
                val = self[key]
            except KeyError:
                val = None
            return val

        def __setattr__(self, key, val):
            '''Implement setattr to support syntax "data.field=x"'''
            self[key] = val

        def __delattr__(self, key):
            '''Implement delattr to support syntax "del data.field"'''
            if key in self:
                del self[key]

        def __getstate__(self):
            '''Needed for pickling, return the underlying dictionary'''
            return dict(self)

        def __setstate__(self, dict_entries):
            '''Needed for pickling, set the underlying dictionary'''
            self.update(dict_entries)

        def copy(self):
            return StructDict(super().copy())

    def recurseCreateStructDict(data):
        '''
        Given a recursive dictionary, i.e. a dictionary that has
        child dictionaries or lists of dictionaries,
        convert each child dictionary to a StructDict.
        '''
        if isinstance(data, dict):
            tmpDict = StructDict()
            for key, value in data.items():
                tmpDict[key] = recurseCreateStructDict(value)
            return tmpDict
        elif isinstance(data, list):
            tmpList = []
            for value in data:
                tmpList.append(recurseCreateStructDict(value))
            return tmpList
        return data

    cfg_dict = toml.load(pathname)
    cfg_Struct = recurseCreateStructDict(cfg_dict)
    return cfg_Struct