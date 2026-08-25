"""
修正置换检验的 searchlight worker（MPI 版本）
回应审稿意见 R1-P3

基于 fig3a/searchlight_score_linearModel/sl.py，仅修改打乱逻辑：
  原始（错误）：np.random.shuffle(上三角120个元素)
  修正（正确）：随机打乱16个刺激标签，同时重排RSM的行和列（Mantel test）

用法：由 sl.sh 通过 srun --mpi=pmi2 调用
  srun --mpi=pmi2 python3 -u .../sl.py ${SLURM_ARRAY_TASK_ID}
"""
import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")

testMode = False
print(f"testMode={testMode}")

import numpy as np

print(f"np.__version__={np.__version__}")
import sys

print(f"sys.executable={sys.executable}")
print(f"sys.path={sys.path}")

import nibabel as nib
import time
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from sklearn.metrics import r2_score

current_dir = os.getcwd()

assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures')), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "figures/figure_03_synthesized_localization/analysis"))

from utils import save_nib, setupPaths, getSubs
[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

jobarrayDict = np.load(
    f"figures/figure_03_synthesized_localization/analysis/corrected_permutation/sl_jobArray.npy",
    allow_pickle=True)

jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]

jobarrayID = int(float(sys.argv[1]))
[subject, radius, method, layer, _] = jobarrayDict[jobarrayID]
layer_1 = layer - 1
if testMode:
    repetitionNumber = 5
else:
    repetitionNumber = 500
print(f"subject={subject}, radius={radius}, method={method}, layer={layer}, layer_1={layer_1}, "
      f"repetitionNumber={repetitionNumber}")

# 保存到 rebuttal 专用目录，不覆盖原始结果
saveToFolder = (f'{scratchResultFolder}/rebuttal_analyses/corrected_permutation/'
                f'{subject}/rad{radius}/')

if not os.path.exists(saveToFolder):
    try:
        os.makedirs(saveToFolder)
    except:
        pass

finalSavingFile = f'{saveToFolder}/{subject}_layer{layer}_{method}_all_R2.npy'

otherSubs = getSubs(minusSub=subject)

if not os.path.exists(finalSavingFile):

    # ============================================================
    # 修正的打乱函数（核心改动）
    # ============================================================
    def shuffle_rsm_stimulus_labels(rsm_vec, n_stimuli=16):
        """
        正确的RSM置换方法：打乱刺激标签（行列协调打乱）。
        等价于 Mantel test。

        1. 将上三角向量还原为完整对称矩阵
        2. 生成随机刺激标签置换
        3. 同时打乱行和列
        4. 提取新的上三角
        """
        triu_idx = np.triu_indices(n_stimuli, k=1)

        # 还原为完整对称矩阵
        rsm_full = np.zeros((n_stimuli, n_stimuli))
        rsm_full[triu_idx] = rsm_vec
        rsm_full = rsm_full + rsm_full.T
        np.fill_diagonal(rsm_full, 0)

        # 随机置换刺激标签
        perm = np.random.permutation(n_stimuli)

        # 同时打乱行和列（等价于 P @ RSM @ P.T）
        rsm_permuted = rsm_full[np.ix_(perm, perm)]

        # 提取上三角
        return rsm_permuted[triu_idx]

    # ============================================================
    # Searchlight 回调函数（与原始 ModelReg 结构一致）
    # ============================================================
    def ModelReg(_data, sl_mask, myrad, bcvar):
        features = bcvar
        triu = np.triu_indices(16, 1)
        assert features.shape == (12, 16, 16)
        assert sl_mask.shape == (7, 7, 7)
        data4d = _data[0]
        assert data4d.shape == (7, 7, 7, 18)

        # 从 data4d 中心体素取回归系数（与原始代码一致）
        _coef_ = data4d[3, 3, 3, 0]
        _intercept_ = data4d[3, 3, 3, 1]

        # 提取 BOLD 数据并计算脑 RSM
        bolddata_sl = data4d[:, :, :, 2:]
        assert bolddata_sl.shape == (7, 7, 7, 16)

        bolddata_sl = bolddata_sl.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2],
                                          bolddata_sl.shape[3]).T
        assert bolddata_sl.shape == (16, 7 * 7 * 7)

        voxelCubeFingerprint = np.corrcoef(bolddata_sl)
        assert voxelCubeFingerprint.shape == (16, 16)
        voxelCubeFingerprint = voxelCubeFingerprint[triu]

        CNNlayerFingerprint = np.copy(features[layer_1])
        CNNlayerFingerprint = CNNlayerFingerprint[triu]

        assert CNNlayerFingerprint.shape == (120,)
        assert voxelCubeFingerprint.shape == (120,)

        def custom_linear_score(_x_, _y_, _coef, _intercept):
            _x_ = _x_.reshape(-1, 1)
            _y_ = _y_.reshape(-1, 1)
            y_pred = _coef * _x_ + _intercept
            _R2 = r2_score(_y_.reshape(-1, 1), y_pred.reshape(-1, 1))
            return _R2

        # 真实 R²（与原始一致）
        true_R2_ = custom_linear_score(CNNlayerFingerprint, voxelCubeFingerprint, _coef_, _intercept_)

        # *** 修正的置换检验 ***
        shuffled_R2 = np.zeros(repetitionNumber)
        for iii in range(repetitionNumber):
            # 原始（错误）：np.random.shuffle(CNNlayerFingerprint_shuffled)
            # 修正（正确）：打乱刺激标签，行列协调重排
            CNNlayerFingerprint_shuffled = shuffle_rsm_stimulus_labels(CNNlayerFingerprint)
            random_R2 = custom_linear_score(CNNlayerFingerprint_shuffled, voxelCubeFingerprint,
                                            _coef_, _intercept_)
            shuffled_R2[iii] = random_R2

        All_R2_ = np.zeros(repetitionNumber + 1)
        All_R2_[0] = true_R2_
        All_R2_[1:] = shuffled_R2
        return np.asarray(All_R2_)

    # ============================================================
    # 数据加载（与原始完全一致）
    # ============================================================
    copeIm = []
    affine_mat = []
    copes = []
    for cope in range(1, 17):
        print(cope, end='--')
        thisCope = f'{subjectDir}/{subject}/preprocess/allruns.gfeat/cope{cope}.feat/stats/cope1.nii.gz'
        copeIm = nib.load(thisCope)
        copes.append(copeIm)
        affine_mat = copeIm.affine
    nii = nib.funcs.concat_images(copes, check_affines=True)

    data = nii.get_fdata()
    print(f"\ndata.shape={data.shape}")
    assert data.shape == (91, 109, 91, 16)

    # 加载其他被试训练的回归系数并取平均
    coefs = np.zeros((91, 109, 91, 29))
    intercepts = np.zeros((91, 109, 91, 29))
    for sub_i, _sub_ in enumerate(otherSubs):
        if _sub_ != subject:
            loadFolder = (f'{scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/'
                          f'searchlight/{_sub_}/rad{radius}/')
            coef = nib.load(f"{loadFolder}/{_sub_}_layer{layer}_{method}_coef.nii.gz").get_fdata()
            intercept = nib.load(f"{loadFolder}/{_sub_}_layer{layer}_{method}_intercept.nii.gz").get_fdata()
            coefs[:, :, :, sub_i] = coef
            intercepts[:, :, :, sub_i] = intercept

    coefs = np.nanmean(coefs, axis=3)
    assert coefs.shape == (91, 109, 91)

    coefs = np.expand_dims(coefs, axis=3)
    intercepts = np.expand_dims(
        np.nanmean(intercepts, axis=3),
        axis=3)

    assert coefs.shape == (91, 109, 91, 1)
    assert intercepts.shape == (91, 109, 91, 1)
    weight = np.concatenate((coefs, intercepts), axis=3)
    assert weight.shape == (91, 109, 91, 2)

    data = np.concatenate((weight, data), axis=3)

    assert data.shape == (91, 109, 91, 18)

    mask = nib.load(
        f"{subjectDir}/{subject}/preprocess/allruns.gfeat/mask.nii.gz").get_fdata()
    if testMode:
        mask = np.zeros(mask.shape)
        mask[int(91 / 2 - 1), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2 - 1), int(91 / 2)] = 1
    print('mask dimensions: {}'.format(mask.shape))
    print('number of voxels in mask: {}'.format(np.sum(mask)))

    sl_rad = radius
    max_blk_edge = 5
    pool_size = 1
    bcvar = np.load(f'{subjectDir}/../data/inception/features.npy')

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    print(f"MPI.COMM_WORLD = {comm}")
    print(f"comm.rank = {rank}")
    print(f"comm.size = {size}")

    sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)

    sl.distribute([data], mask)
    sl.broadcast(bcvar)
    starttime = time.time()
    assert method == 'reg'

    sl_result = sl.run_searchlight(ModelReg)

    total = time.time() - starttime
    print('total time: {}s, estimated time for whole brain: {} h'.format(total,
                                                                         (total / np.sum(mask)) * 239565 / 60 / 60))

    if rank == 0:
        All_R2 = np.zeros((91, 109, 91, repetitionNumber + 1))
        print(f"loading All_R2")
        for x in range(sl_result.shape[0]):
            for y in range(sl_result.shape[1]):
                for z in range(sl_result.shape[2]):
                    if sl_result[x, y, z] is not None:
                        All_R2[x, y, z] = sl_result[x, y, z]
                        assert sl_result[x, y, z].shape == (repetitionNumber + 1,)
        print("epoch is:", time.asctime(time.gmtime()))
        print(f"saving {finalSavingFile}")
        print(f"All_R2.shape={All_R2.shape}")
        np.save(finalSavingFile, All_R2)
else:
    print(f'{finalSavingFile} already run')

print('done')
