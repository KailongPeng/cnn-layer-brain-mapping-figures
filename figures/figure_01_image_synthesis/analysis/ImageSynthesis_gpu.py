import tensorflow as tf
import numpy as np
import os
import math
import json
import shutil
from functools import reduce, partial
from PIL import Image
import imageio
from IPython.display import clear_output
import time
import sys
import pdb
import pickle
from tqdm import tqdm

print(f"CONDA_DEFAULT_ENV={os.environ['CONDA_DEFAULT_ENV']}")

current_dir = os.getcwd()

assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures')), \
    f"Run from the localize_paper or cnn-layer-brain-mapping-figures repository root, not {current_dir}"

print(f"Current directory is: {current_dir}")

figure_dir = os.path.join(current_dir, 'figures', 'figure_01_image_synthesis')
asset_dir = os.path.join(figure_dir, 'assets')

# --- 可复现性 ---
# 原代码有两处未播种的随机: np.random.shuffle(choose) 决定 mixed3a 层 16 个通道的
# 配对顺序(通道 i 配目标矩阵的第 i 个值), gen_crops 里的 np.random.normal 决定每轮
# foveated 裁剪位置。两者都没播种, 所以同样的种子图 + 同样的代码, 两次运行轨迹也不同,
# 结果无法复现、也无法把一次异常归因到某个具体输入。这里统一播种, 可用环境变量覆盖。
SYNTH_SEED = int(os.environ.get('SYNTH_SEED', '0'))
np.random.seed(SYNTH_SEED)
print(f"SYNTH_SEED={SYNTH_SEED}")

# RDM 行判别式 dim*sum(x^2)-(sum x)^2 低于此值即告警: 该值趋 0 表示某层的 16 个 RDM
# 取值趋于常数, 相关系数分母趋 0 —— 既是 NaN 的直接来源, 也可能说明优化正落进
# "把 RDM 压成常数" 这个退化解(常数向量与任何向量的相关都可视为 0, 是压低目标函数的捷径)。
DEGENERACY_WARN = float(os.environ.get('DEGENERACY_WARN', '1e-3'))

# 逐图日志(每轮 16 次 _LOSS2_ + 16 次 a/b/angle, 共 32 行)默认关掉: 刷屏且淹没
# [DEGENERACY]/[PROFILE]/[STAGE DONE] 这些真正要看的行。数值本身照常存进
# Vars.pkl 的 _LOSS2_Container / compareg1g2List, 关掉打印不丢数据。
# 需要逐图排查时 SYNTH_VERBOSE=1。
VERBOSE_PER_IMAGE = os.environ.get('SYNTH_VERBOSE', '0') == '1'

# --- 进度/ETA 记账 ---
JOB_START = time.time()
ITERS_BY_MAG = {0: 400, 1: 400, 2: 400, 3: 600}   # 与 tfLoop 里的 iters 分配一致, 合计 1800
TOTAL_ITERS = sum(ITERS_BY_MAG.values())


STAGE_AVG = {}      # mag -> 实测平均秒/轮, 每阶段结束时填入, 用于自我校正后续阶段的估计


def _typical(vals):
    """取中位数而非平均: 第 0 轮含 cuDNN autotune / XLA 编译, 比常态慢一个量级,
    用平均数会被这一个离群值拖歪整整一个窗口(实测前 20 轮的剩余时间估计会高 1.5~12 倍),
    中位数只要 3 个样本就能给出正确的常态速率, 对偶发的调度卡顿同样稳健。"""
    if len(vals) < 3:
        return float('nan')
    ordered = sorted(vals)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _fmt_dur(sec):
    if sec is None or sec != sec or sec < 0:
        return '?'
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return '{}h{:02d}m'.format(h, m)
    if m:
        return '{}m{:02d}s'.format(m, s)
    return '{}s'.format(s)


def load_seed_manifest(manifest_path, expected_n=16):
    with open(manifest_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    if len(paths) != expected_n:
        raise ValueError(f"Expected {expected_n} seeds, got {len(paths)}")
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    return paths


def loadimageList(imageList):
    image = []
    for image_file in imageList:
        im = Image.open(image_file)
        im = np.array(im.resize((539, 539))).astype(float)
        image.append(im)
    image = (image,)
    return image


def tfLoop(previousMag, previousPhase, continueFlag):
    print('previousMag=', previousMag)
    print('previousPhase=', previousPhase)

    model_fn = os.path.join(asset_dir, 'tensorflow_inception_graph.pb')

    graph = tf.Graph()
    with graph.as_default(), graph.device('/gpu:0'), tf.Session() as sess:
        with tf.gfile.FastGFile(model_fn, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        t_input = tf.placeholder(np.float32, name='input')
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input': t_preprocessed})
    layers = [op.name for op in graph.get_operations() if
              op.type == 'Conv2D' and 'import/' in op.name]
    layers1 = [l.split('/')[1] for l in layers if
               l.split('/')[1].split('2')[0] == 'conv']
    layers2 = [l.split('/')[1] for l in layers if
               'pool_reduce' in l]
    incept_layers = layers1 + layers2

    def gen_noise():
        return np.random.uniform(-10, 10, size=(imsize[1:])) + 100.0

    def shapegen(shape_array, layers, padding=1, stride=2, filterdim=3):
        init_dim = shape_array[1]
        init_array = shape_array
        n_filters = [int(graph.get_tensor_by_name('import/{}/conv:0'.format(name)).get_shape()[-1]) for name in layers]
        shapes_array = []
        for x, layer in enumerate(layers):
            if x == 0 or x == 1 or x == 3 or x == 5 or x == 10:
                new_dim = math.ceil(((init_dim + padding - filterdim) / stride) + 1)
                shapes_array.append([1, new_dim, new_dim, n_filters[x]])
                init_dim = new_dim
                init_array = [1, new_dim, new_dim, n_filters[x]]
            else:
                set_array = np.copy(init_array)
                set_array[3] = n_filters[x]
                shapes_array.append(set_array)
        shapes_array = np.array(shapes_array)
        return shapes_array

    def corrsgen(steps, lo=0, hi=1, match=0.25, nLayers=12, nMatch=8):
        corrs = np.arange(lo, (hi + .01), (hi - lo) / (steps - 1))
        layerCorrs = []
        layerWeights = []
        for step in range(steps):
            thisLayer = [match] * nMatch + [corrs[step]] * (nLayers - nMatch)
            layerCorrs.append(thisLayer)
            layerWeights.append([corrs[step], 1])
        return layerCorrs, layerWeights, np.around((corrs * 100), 0).astype(int)

    def enhance_dims(dims, increase_by):
        new_dims = [dims[0], math.ceil(dims[1] * increase_by), math.ceil(dims[2] * increase_by), dims[3]]
        return new_dims

    def mag_shapes(dims, fov_dims, layers, increase_by, increase_fov):
        new_dims = enhance_dims(dims, increase_by)
        new_fovs = enhance_dims(fov_dims, increase_fov)
        new_shapes = shapegen(new_fovs, layers)
        new_padshape = np.amax(new_shapes, axis=0)
        return new_dims, new_fovs, new_shapes

    def mag_images(images, shape):
        new_images = []
        for image in images:
            new_image = tf.image.resize_bilinear(np.expand_dims(image, 0), np.int32(shape[1:3])).eval()[0]
            new_images.append(new_image)
        return new_images

    def get_tensor(layers):
        activations = []
        for layer in layers:
            thisLayer = graph.get_tensor_by_name("import/{}:0".format(layer))
            activations.append(thisLayer)
        return activations

    def pad_activ(activations, shapes):
        padded_weights = []
        for dims, acts in zip(shapes, activations):
            padded = np.zeros(np.amax(shapes, 0))
            padded[:dims[0], :dims[1], :dims[2], :dims[3]] = acts
            padded_weights.append(padded)
        return padded_weights

    def saveArr(arr, name, shape, post='clip'):
        if post == 'clip':
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(shape[1:])
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = (arr - arr.mean()) / arr.std() * 0.1 + 0.5
            arr = np.uint8(np.clip(arr, 0, 1) * 255)
        imageio.imsave(name, arr)

    def gen_fovea(shape, fovshape):
        fovsize = fovshape[1]
        h, w = shape[1:3]
        sx = w - fovsize
        cx = sx / 2.
        sy = h - fovsize
        cy = sy / 2.
        return sx, sy, cx, cy

    def gen_crops(images, fovparams, fovshape):
        fovsize = fovshape[1]
        sx, sy, cx, cy = fovparams
        cropx = int(np.clip(np.random.normal(cx, sx * 0.3, 1), 0, sx))
        cropy = int(np.clip(np.random.normal(cy, sy * 0.3, 1), 0, sy))
        new_images = []
        for image in images:
            cropped = image[cropx:cropx + fovsize, cropy:cropy + fovsize]
            new_images.append(cropped)
        return new_images, cropx, cropy

    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def tffunc(*argtypes):
        """
        Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        """
        placeholders = list(map(tf.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)

    k = np.float32([1, 4, 6, 4, 1])
    k = np.outer(k, k)
    k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)

    def lap_split(img):
        """Split the image into lo and hi frequency components"""
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
            hi = img - lo2
        return lo, hi

    def lap_split_n(img, n):
        """Build Laplacian pyramid with n splits"""
        levels = []
        for i in range(n):
            img, hi = lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    def lap_merge(levels):
        """Merge Laplacian pyramid"""
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
        return img

    def normalize_std(img, eps=1e-10):
        """Normalize image by making its standard deviation = 1.0"""
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img / tf.maximum(std, eps)

    def lap_normalize(img, scale_n=4):
        """Perform the Laplacian pyramid normalization."""
        img = tf.expand_dims(img, 0)
        tlevels = lap_split_n(img, scale_n)
        tlevels = list(map(normalize_std, tlevels))
        out = lap_merge(tlevels)
        return out[0, :, :, :]

    imsize = [1, 196, 196, 3]

    step = 1.2
    magby = 1.4

    lapnorm = True
    lap_n = 4

    fov = True
    fov_grow = 1.35
    fov_init = 0.75
    _thisWeightContainer = []
    _LossContainer = []
    _LOSS2_Container = []
    decay = True
    decRate = 0.999

    post = 'clip'

    numImages = 16

    bestchansfile = os.path.join(asset_dir, 'bestchans{}.pkl'.format(numImages))
    corrsvalsfile = os.path.join(asset_dir, 'corrsvals{}.pkl'.format(numImages))
    if not os.path.exists(bestchansfile):
        bestchandict = {}
        corrvalsdict = {}
        iters = 3000000
        for layer in incept_layers[:]:
            print(layer)
            channels = [im for im in os.listdir(layer) if im.split('.')[-1] == 'png']
            channels.sort()
            imgs = []
            for channel in range(len(channels)):
                print('{}    {}/{}'.format(layer, channel + 1, len(channels)))
                clear_output(wait=True)
                img0 = Image.open('{}/{}.png'.format(layer, channel))
                img0 = img0.convert("RGB")
                img0 = np.float32(img0)
                act = sess.run(get_tensor([layer]), {t_input: img0})[0]
                ACT = act if channel == 0 else np.concatenate((ACT, act))
            ACT = np.reshape(ACT, (len(channels), -1))
            CORRS = np.corrcoef(ACT)
            lowavg = 1
            lowtop = 1
            for x in range(iters):
                newsamp = np.random.choice(len(channels), numImages, replace=False)
                corrs = CORRS[newsamp[:, None], newsamp]
                corrs[corrs > 0.99] = 0
                thisavg = np.mean(np.absolute(corrs))
                thistop = np.amax(np.absolute(corrs))
                if ((lowavg - thisavg) + (lowtop - thistop)) / (lowavg + lowtop) > .01:
                    lowavg = thisavg
                    lowtop = thistop
                    bestchans = newsamp
                    print('{}    {} {} {}'.format(layer, x, lowavg, lowtop))
            bestchandict[layer] = bestchans
            corrvalsdict[layer] = np.array([lowavg, lowtop])
            clear_output(wait=True)
    else:
        bestchandict = load_obj(bestchansfile.split('.pkl')[0])
        corrvalsdict = load_obj(corrsvalsfile.split('.pkl')[0])
        """
            What is bestchandict?
            It can be seen that for all 12 model layers, each layer has selected 16 channels as the target for
            maximization. The purpose of this is to make the synthesized images look very different.
            The execution is located at `dGrad = tf.gradients(thisWeight, t_input)[0]

            len(bestchandict) == 12
            bestchandict =
            {'conv2d0_pre_relu': array([11, 38, 37,  8, 48,  7, 59, 62, 50,  4, 33, 47, 45, 25, 15,  3]),
             'conv2d1_pre_relu': array([20, 12, 56, 10, 19,  3, 47, 40,  5, 23, 27, 45, 52, 46, 53,  1]),
             'conv2d2_pre_relu': array([ 88, 176, 107,  70,  30,  56, 128,  69,  47, 169,  29,  74,  33,
                     98, 143,  80]),
             'mixed3a_pool_reduce_pre_relu': array([ 9, 11, 27,  2, 30,  4, 25, 29,  5, 10, 22, 20, 19, 16, 12, 21]),
             'mixed3b_pool_reduce_pre_relu': array([44,  8, 29, 12, 54, 34, 17, 60,  7,  9, 61,  6, 32, 43, 53,  1]),
             'mixed4a_pool_reduce_pre_relu': array([ 3, 14, 13, 22, 37, 19, 12,  8, 36, 44, 34, 43, 30, 17, 10, 52]),
             'mixed4b_pool_reduce_pre_relu': array([22, 18, 56, 13, 53,  7, 24, 25,  3,  1, 12, 27, 35, 63, 44, 29]),
             'mixed4c_pool_reduce_pre_relu': array([54, 33,  7, 26,  2, 43, 12, 32, 27, 63, 61, 44,  1, 36, 19, 56]),
             'mixed4d_pool_reduce_pre_relu': array([45, 25, 16, 50, 18, 12, 37, 19,  0,  7, 63, 40, 61, 38, 43, 23]),
             'mixed4e_pool_reduce_pre_relu': array([ 80,  68, 116,  11, 110,  72,  49,  18, 125,  40,  14,  67, 109,
                     34,  24,  56]),
             'mixed5a_pool_reduce_pre_relu': array([112,  68,  65, 114,  87,  45,  71, 125,  12,  81,  31,  89,  11,
                     26,   1,  15]),
             'mixed5b_pool_reduce_pre_relu': array([107,  31,  78,   4,  32,  37,  77,  33, 106,  23,  35, 113,  46,
                     28,  95, 120])}
        """
    for i, layer in enumerate(incept_layers[:]):
        choose = bestchandict[layer]
        choose.sort()
        if layer == 'mixed3a_pool_reduce_pre_relu':
            seq = True
            while seq:
                tempseq = False
                np.random.shuffle(choose)
                for i in range(0, choose.shape[0] - 1, 2):
                    if choose[i] == choose[i + 1]:
                        tempseq = True
                        break
                if not tempseq:
                    seq = False
            print(choose)

        CHOOSE = choose if i == 0 else np.vstack((CHOOSE, choose))
    CHOOSE = np.expand_dims(CHOOSE, 0)
    print(CHOOSE.shape)


    version = 'GradOrth'
    matchval = 0.4
    structlayers = [8, 9, 10, 11]
    corlo = 0.5
    corhi = 1

    seed_mode = os.environ.get('SEED_MODE', 'noise')
    output_tag = os.environ.get('SYNTH_TAG', 'ImageSynthesis')
    seed_manifest = os.environ.get(
        'SEED_MANIFEST',
        os.path.join(asset_dir, 'structured_seed_16.txt')
    )
    if seed_mode not in ['noise', 'structured']:
        raise ValueError(f"Unsupported seed_mode: {seed_mode}")
    smoke = os.environ.get('SMOKE_TEST', '0') == '1'

    if version == 'GradOrth':
        CORRMAT = np.load(os.path.join(asset_dir, '{}corr_GradOrth.npy'.format(numImages)))
    else:
        raise Exception

    def saveVar(PathName, *var):
        with open(PathName, 'wb') as f:
            pickle.dump(var, f)

    tag = output_tag
    print(f"seed_mode={seed_mode}")
    print(f"tag={tag}")
    print(f"seed_manifest={seed_manifest}")
    print(f"smoke_test={smoke}")
    decay_flag = False
    clipDuringUpdateFlag = True
    run_root = f"{current_dir}/images/{tag}"
    if not os.path.isdir(run_root):
        os.makedirs(run_root)

    seed_metadata = {
        'seed_mode': seed_mode,
        'seed_manifest': seed_manifest,
        'tag': tag,
        'num_images': numImages,
        'image_loader': 'loadimageList_resize_539',
    }
    if seed_mode == 'structured':
        shutil.copyfile(seed_manifest, f"{run_root}/seed_manifest_used.txt")
    with open(f"{run_root}/seed_metadata.json", 'w') as f:
        json.dump(seed_metadata, f, indent=2)

    def truefn(multipliers, numOfthisOne, thisLayer, chans, currLayer, cell, i, ims):
        thisOne = tf.multiply(thisLayer[:, :, :, chans[currLayer][i]], multiplier[currLayer, cell, i])
        multipliers = tf.add(multipliers, tf.multiply(thisOne, len(ims) - 1))
        return multipliers, (numOfthisOne + len(ims) - 1)

    def falsefn(multipliers, numOfthisOne, thisLayer, chans, currLayer, cell, i, ims):
        thisOne = tf.multiply(thisLayer[:, :, :, chans[currLayer][i]], multiplier[currLayer, cell, i])
        multipliers = tf.add(multipliers, thisOne)
        return multipliers, (numOfthisOne + 1)

    def _stable_pearson(thisLayer, otherLayer, dim):
        """数值稳健版 Pearson 相关系数。

        原写法用 sqrt((dim*sum(x^2) - (sum x)^2) * (...y...)) 作分母。判别式数学上恒 >= 0,
        但在 float32 下它是两个近似相等的大数相减, 会灾难性抵消: 实测当 16 个取值高度接近时,
        约 14% 的情形判别式会落到 0 的负侧(误差约 1 ulp), sqrt(负数) 即 NaN; 方差恰为 0 时
        则是 0/0。此处只在这两种本来就必然产生 NaN 的退化情形返回 0, 其余情况与原式逐位相同,
        不引入 epsilon, 因此不影响任何健康运行已有的结果。

        必须用 double-where 写法(先把 sqrt 和除法的输入换成安全值, 再选分支): tf.where 会对
        两个分支都求导, 若直接对退化值做 sqrt/除法, NaN 梯度仍会从未选中的分支漏回来。
        """
        sum_x = tf.reduce_sum(thisLayer)
        sum_y = tf.reduce_sum(otherLayer)
        numerator = (dim * tf.reduce_sum(tf.multiply(thisLayer, otherLayer))) - (sum_x * sum_y)
        disc_x = dim * tf.reduce_sum(tf.square(thisLayer)) - tf.square(sum_x)
        disc_y = dim * tf.reduce_sum(tf.square(otherLayer)) - tf.square(sum_y)
        disc = tf.maximum(disc_x, 0.0) * tf.maximum(disc_y, 0.0)
        ok = disc > 0.0
        denominator = tf.sqrt(tf.where(ok, disc, tf.ones_like(disc)))
        return tf.where(ok, numerator / denominator, tf.zeros_like(numerator))

    def TFCorr(thisLayer, otherLayer, multiplier, currLayer, cell, i, loss, numOfthisLoss):
        thisCorr = _stable_pearson(thisLayer, otherLayer, dim)
        thisLoss = tf.square(thisCorr - multiplier[currLayer, cell, i])
        loss = tf.add(loss, thisLoss)
        return loss, (numOfthisLoss + 1), thisCorr

    def DoNothing(loss, numOfthisLoss):
        thisCorr = 0.0
        return loss, numOfthisLoss, thisCorr

    def ListCorrSquare(thisLayer, otherLayer):
        dim = len(thisLayer)
        return tf.square(_stable_pearson(thisLayer, otherLayer, dim))

    def calImageRDMlayerCorrelation(currImageRDM):
        corrSum = []
        for currLayerI in range(12):
            for currLayerJ in range(currLayerI):
                corrSum.append(ListCorrSquare(currImageRDM[currLayerI * numImages:(currLayerI + 1) * numImages],
                                              currImageRDM[currLayerJ * numImages:(currLayerJ + 1) * numImages]))
        return corrSum

    def calculateCorrBetweenLayers(ActualCorrelation):
        M = 0.0
        numImages = ActualCorrelation.shape[1]
        for currLayerI in range(ActualCorrelation.shape[0]):
            for currLayerJ in range(currLayerI):
                t = np.corrcoef(ActualCorrelation[currLayerI][np.triu_indices(numImages, k=1)],
                                ActualCorrelation[currLayerJ][np.triu_indices(numImages, k=1)])[0, 1]
                M = M + t ** 2
        return M

    def breaklink(a):
        a = a + 1
        a = a - 1
        return a

    class custom_AdamOptimizer():
        def __init__(self, images, learning_rate=1.2, beta1=0.9, beta2=0.999, epsilon=1e-8, name="Adam"):
            self._lr = learning_rate
            self._beta1 = beta1
            self._beta2 = beta2
            self._epsilon = epsilon
            self.m = np.zeros(images.shape)
            self.v = np.zeros(images.shape)
            self.CostList = []

        def update(self, g, currIter,
                   pairn):
            self.t = currIter + 1
            self.m[pairn] = self._beta1 * self.m[pairn] + (1 - self._beta1) * g
            self.v[pairn] = self._beta2 * self.v[pairn] + (1 - self._beta2) * np.multiply(g, g)
            M = self.m[pairn] / (1 - self._beta1 ** self.t)
            V = self.v[pairn] / (1 - self._beta2 ** self.t)
            return (self._lr * M / (np.sqrt(V) + self._epsilon))

    import math
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / math.pi * 180

    def compare(a, b, currIter=None, pairn=None):
        a = a.flatten()
        b = b.flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        # 带上 iter/pairn: 原来 16 张图每轮各打印一次这三行且互相无法区分,
        # 出现异常时无法判断是哪一轮的哪张图触发的。
        if VERBOSE_PER_IMAGE:
            print("iter={} pairn={} | a is {} long | b is {} long | angle is {}".format(
                currIter, pairn, na, nb, angle_between(a, b)))
        return (na / nb) if nb != 0 else float('nan')

    def plot_vector(m, t):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(range(len(m)), m)
        ax.set_title(t)
        plt.show()

    total_start = time.time()
    lapnorm = True
    lap_n = 4
    mags = 4
    decRate = 0.999
    fov_grow = 1.4
    fov_init = 1.0

    chanPair = CHOOSE
    layerChans = [i for i in np.arange(chanPair.shape[1]) if not all(chanPair[0, i] == np.array([0] * numImages))]

    thisCorrContainer = []
    ActualCorrelationContainer = []
    CorrBetweenLayersContainer = []
    compareg1g2List = []
    for pair in range(chanPair.shape[0]):
        theseChans = chanPair[pair]
        corr = 0.2
        if continueFlag:
            mag = previousMag
        else:
            mag = previousMag + 1
        if mag == 0:
            if seed_mode == 'noise':
                ims = [gen_noise() for noise in range(numImages)]
            else:
                seed_paths = load_seed_manifest(seed_manifest, expected_n=numImages)
                ims = list(loadimageList(seed_paths)[0])
        else:
            PreviousImgFile = '{}/images/{}/mag{}phase{}/ims.pkl'.format(current_dir, tag, previousMag, previousPhase)
            with open(PreviousImgFile, 'rb') as f:
                [ims] = pickle.load(f)
        assert len(ims) == numImages
        if mag == 0 and seed_mode == 'structured':
            assert ims[0].shape == (539, 539, 3)
        elif mag == 0 and seed_mode == 'noise':
            assert ims[0].shape == tuple(imsize[1:])
        else:
            assert ims[0].shape[-1] == 3

        print(f"np.mean(ims)={np.mean(ims)}")
        print(f"np.std(ims)={np.std(ims)}")

        if smoke:
            print('smoke shortcut: saving seed snapshots without optimization')
            saveFolder = '{}/images/{}/mag{}phase{}'.format(current_dir, tag, mag, 0)
            if not os.path.isdir(saveFolder):
                os.makedirs(saveFolder)
            curr_iter_folder = '{}/currIter0'.format(saveFolder)
            if not os.path.isdir(curr_iter_folder):
                os.makedirs(curr_iter_folder)

            smoke_shape = [1] + list(ims[0].shape)
            for numbered, thisOne in enumerate(ims):
                saveArr(thisOne, '{}/im{}of{}.png'.format(curr_iter_folder, numbered + 1, len(ims)), smoke_shape, post=post)
                saveArr(thisOne, '{}/im{}of{}.png'.format(saveFolder, numbered + 1, len(ims)), smoke_shape, post=post)

            saveVar('{}/ims.pkl'.format(curr_iter_folder), ims)
            saveVar('{}/ims.pkl'.format(saveFolder), ims)

            if len(ActualCorrelationContainer) == 0:
                ActualCorrelationContainer.append(np.zeros((len(incept_layers), len(ims), len(ims))))
            if len(CorrBetweenLayersContainer) == 0:
                CorrBetweenLayersContainer.append(np.nan)

            PathName = '{}/Vars.pkl'.format(curr_iter_folder)
            saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer,
                    CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)

            PathName = '{}/images/{}/Vars.pkl'.format(current_dir, tag)
            saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer,
                    CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)
            seed_metadata['run_complete'] = True
            with open(f"{run_root}/seed_metadata.json", 'w') as f:
                json.dump(seed_metadata, f, indent=2)
            return mag, 0

        shape = imsize
        fshape = enhance_dims(shape, fov_init) if fov else shape
        scalefov = fov_grow if fov else magby

        shapes = shapegen(fshape, incept_layers)
        padshape = np.amax(shapes, axis=0)

        with graph.as_default(), graph.device('/gpu:0'), tf.Session() as sess:
            lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

            cell = tf.placeholder(tf.int32, shape=(None))
            toMatch = tf.placeholder(tf.float32, shape=(None))
            benchmark = tf.placeholder(tf.float32, shape=(None))
            chans = tf.placeholder(tf.int32, shape=(None))
            cutAt = tf.placeholder(tf.int32, shape=(None))
            multiplier = tf.placeholder(tf.float32, shape=(None))

            currImageRDM = []
            loss = tf.constant(0, tf.float32)
            numOfthisLoss = tf.constant(0, tf.float32)
            _weights = []

            matchActivations = get_tensor(incept_layers)
            for currLayer, layer in enumerate(incept_layers):
                thisLayer = matchActivations[currLayer]
                c = cutAt[currLayer]

                dim = tf.cast(tf.reduce_prod(c), tf.float32)

                if currLayer in layerChans:
                    multipliers = tf.constant(0, tf.float32)
                    numOfthisOne = tf.constant(0, tf.float32)
                    for i, im in enumerate(ims):
                        multipliers, numOfthisOne = tf.cond(tf.math.equal(i, cell),
                                                            lambda: truefn(multipliers, numOfthisOne, thisLayer, chans,
                                                                           currLayer, cell, i, ims),
                                                            lambda: falsefn(multipliers, numOfthisOne, thisLayer, chans,
                                                                            currLayer, cell, i, ims))

                        other = toMatch[i, currLayer]
                        otherLayer = other[:c[0], :c[1], :c[2], :c[3]]

                        loss, numOfthisLoss, thisCorr = tf.cond(~tf.math.equal(i, cell),
                                                                lambda: TFCorr(thisLayer, otherLayer, multiplier,
                                                                               currLayer, cell, i, loss, numOfthisLoss),
                                                                lambda: DoNothing(loss, numOfthisLoss))

                        currImageRDM.append(thisCorr)

                    _thisWeight = (tf.reduce_mean(tf.truediv(multipliers, numOfthisOne))) * -1
                    _weights.append(_thisWeight)

            corrSum = reduce(tf.add, calImageRDMlayerCorrelation(
                currImageRDM))
            # 退化监控张量: 每层 RDM 行(当前图与全部 16 张图在该层的相关)的判别式。
            # 它趋于 0 就意味着该行趋于常数, 相关系数分母趋 0。只读, 不参与梯度。
            rdmRowDisc = tf.stack([
                (numImages * tf.reduce_sum(tf.square(currImageRDM[L * numImages:(L + 1) * numImages]))
                 - tf.square(tf.reduce_sum(currImageRDM[L * numImages:(L + 1) * numImages])))
                for L in range(12)])
            Loss = tf.reduce_sum(loss)
            thisWeight = reduce(tf.add, _weights)

            Grad = tf.gradients(Loss, t_input)[0]
            dGrad = tf.gradients(thisWeight, t_input)[0]
            CorrGrad = tf.gradients(corrSum, t_input)[0]
            Grad3 = dGrad + CorrGrad * 3000
            last = time.time()
            if mag == 4:
                exit()
                raise Exception
            if mag > 0:
                for currMag in range(mag):
                    shape, fshape, shapes = mag_shapes(shape, fshape, incept_layers, magby, fov_grow)
                if not continueFlag:
                    ims = mag_images(ims, shape)
            fovparams = gen_fovea(shape, fshape)

            phases = ['Grad3']
            for phase, func in enumerate(phases):
                if func == 'Grad3':
                    AdamOpt1 = custom_AdamOptimizer(np.asarray(ims), learning_rate=1.2)
                    AdamOpt2 = custom_AdamOptimizer(np.asarray(ims), learning_rate=1.2)
                    if mag == 0:
                        iters = 400
                    elif mag == 1:
                        iters = 400
                    elif mag == 2:
                        iters = 400
                    elif mag == 3:
                        iters = 600
                    else:
                        raise Exception
                    # 400+400+400+600=1800
                else:
                    AdamOpt = custom_AdamOptimizer(np.asarray(ims), learning_rate=1.2)
                    iters = 200

                if smoke:
                    iters = min(iters, 1)
                _iterTimes = []
                _stageStart = time.time()
                # 分段计时: 每轮实际有 64 次 sess.run(16 主梯度 + 16 PADDED 前向 +
                # 32 次 lap_norm_func), 加上 numpy 侧的 Adam。要判断该往哪优化,
                # 必须先知道 26s 到底花在哪一段, 否则就是瞎猜。
                _PROF = {}
                _profIters = 0
                for currIter in range(iters):
                    if currIter in [250, 300, 350, 400, 450, 500, 550]:
                        if decay_flag:
                            AdamOpt1._lr = AdamOpt1._lr / 2
                            AdamOpt2._lr = AdamOpt2._lr / 2
                    _now = time.time()
                    _dt = _now - last
                    # 第 0 轮含 cuDNN autotune / XLA 编译, 比常态慢一个量级, 不计入均值
                    if currIter > 0:
                        _iterTimes.append(_dt)
                    _avg = _typical(_iterTimes[-20:])
                    _stageLeft = (iters - currIter) * _avg
                    _doneIters = sum(ITERS_BY_MAG.get(_m, 0) for _m in range(mag)) + currIter
                    # 后续 mag 阶段每轮更慢。优先用已完成阶段的实测比值外推; 还没有两个已完成
                    # 阶段可比时, 退回"计算量∝面积(magby^2=1.96)"。该假设已由四月那次运行的
                    # checkpoint 时间戳验证: mag0=36.0 s/iter, mag2=145.1(两阶段共4.03倍,
                    # 即每阶段2.01), mag3=300.9(对mag2为2.07), 与1.96高度吻合。
                    # (注意别用目录 mtime 反推速率: 阶段完成后还会重写目录内文件, mtime 会偏晚。)
                    _done = sorted(STAGE_AVG)
                    if len(_done) >= 2:
                        _ratio = STAGE_AVG[_done[-1]] / STAGE_AVG[_done[-2]]
                        _basis = '实测比值{:.2f}'.format(_ratio)
                    else:
                        _ratio = magby ** 2
                        _basis = '面积假设{:.2f}未验证'.format(_ratio)
                    _future, _rate = 0.0, _avg
                    for _m in range(mag + 1, len(ITERS_BY_MAG)):
                        _rate = _rate * _ratio
                        _future += ITERS_BY_MAG[_m] * _rate
                    _recentL2 = _LOSS2_Container[-numImages:]
                    _l2 = ('L2 {:.1f}~{:.1f}'.format(min(_recentL2), max(_recentL2))
                           if _recentL2 else 'L2 -')
                    print('{} imgs | mag {}/{} iter {}/{} ({:.1f}%) | iter {:.1f}s 常态 {} | {} '
                          '| 本阶段剩余 {} | 已运行 {} | 全程 {}/{} | 全程还需 {} ({})'.format(
                              numImages, mag + 1, mags, currIter + 1, iters,
                              (currIter + 1) / iters * 100, _dt,
                              ('{:.1f}s'.format(_avg) if _avg == _avg else '?'), _l2,
                              _fmt_dur(_stageLeft), _fmt_dur(_now - JOB_START),
                              _doneIters, TOTAL_ITERS, _fmt_dur(_stageLeft + _future), _basis),
                          flush=True)
                    _profIters += 1
                    if _profIters >= 20 and _PROF:
                        _tot = sum(_PROF.values())
                        _parts = '  '.join('{}={:.2f}s/轮({:.0f}%)'.format(_k, _v / _profIters, 100.0 * _v / _tot)
                                           for _k, _v in sorted(_PROF.items(), key=lambda kv: -kv[1]))
                        print('[PROFILE] 近{}轮 已归因{:.2f}s/轮(占实测{:.0f}%) | {}'.format(
                            _profIters, _tot / _profIters,
                            (100.0 * _tot / _profIters / _avg) if _avg == _avg and _avg else float('nan'),
                            _parts), flush=True)
                        _PROF.clear()
                        _profIters = 0
                    last = _now
                    _p0 = time.time()
                    cropped, xx, yy = gen_crops(ims, fovparams, fshape)
                    _PROF['crop'] = _PROF.get('crop', 0.0) + (time.time() - _p0)
                    ActualCorrelation = np.zeros(CORRMAT.shape) if phase == 0 and currIter == 0 else ActualCorrelation
                    for pairn, (thiscrop, pairmate) in enumerate(zip(cropped, ims)):
                        if smoke and pairn > 0:
                            continue
                        if pairn == 0 and phase == 0:
                            _p0 = time.time()
                            for anum, thisone in enumerate(cropped):
                                thisact = sess.run(matchActivations, feed_dict={t_input: thisone})
                                padthisact = pad_activ(thisact, shapes)
                                PADDED = np.expand_dims(padthisact, 0) if anum == 0 else np.concatenate(
                                    (PADDED, np.expand_dims(padthisact, 0)))
                            _PROF['padded_fwd'] = _PROF.get('padded_fwd', 0.0) + (time.time() - _p0)

                        assert PADDED.shape[0] == len(cropped)
                        if func == 'Grad3':
                            g1 = np.zeros_like(ims[0])
                            g2 = np.zeros_like(ims[0])
                        else:
                            g = np.zeros_like(ims[0])

                        _p0 = time.time()
                        if func == 'Grad3':
                            grad1, grad2, _LOSS1_, _LOSS2_, this, _thisWeight, _Loss, _currImageRDM, _rdmDisc = sess.run(
                                [dGrad, CorrGrad, thisWeight, corrSum, matchActivations, thisWeight, Loss,
                                 currImageRDM, rdmRowDisc],
                                feed_dict={t_input: thiscrop, cutAt: shapes, toMatch: PADDED, cell: pairn,
                                           chans: theseChans, multiplier: CORRMAT})

                        elif func == 'dGrad':
                            grad, _LOSS_, this, _thisWeight, _Loss, _currImageRDM = sess.run(
                                [dGrad, Loss, matchActivations, thisWeight, Loss, currImageRDM],
                                feed_dict={t_input: thiscrop, cutAt: shapes, toMatch: PADDED, cell: pairn,
                                           chans: theseChans, multiplier: CORRMAT})

                        elif func == 'Grad':
                            grad, _LOSS_, this, _thisWeight, _Loss, _currImageRDM = sess.run(
                                [Grad, thisWeight, matchActivations, thisWeight, Loss, currImageRDM],
                                feed_dict={t_input: thiscrop, cutAt: shapes, toMatch: PADDED, cell: pairn,
                                           chans: theseChans, multiplier: CORRMAT})

                        elif func == 'CorrGrad':
                            grad, _LOSS_, this, _thisWeight, _Loss, _currImageRDM = sess.run(
                                [CorrGrad, corrSum, matchActivations, thisWeight, Loss, currImageRDM],
                                feed_dict={t_input: thiscrop, cutAt: shapes, toMatch: PADDED, cell: pairn,
                                           chans: theseChans, multiplier: CORRMAT})

                        else:
                            raise Exception
                        _PROF['sessrun_grad'] = _PROF.get('sessrun_grad', 0.0) + (time.time() - _p0)
                        if func == 'Grad3':
                            if VERBOSE_PER_IMAGE:
                                print('_LOSS2_', _LOSS2_)
                            _LOSS2_Container.append(_LOSS2_)
                            _worstDisc = float(np.min(_rdmDisc))
                            if _worstDisc < DEGENERACY_WARN:
                                print('[DEGENERACY] mag={} iter={} pairn={} layer={} min_rdm_disc={:.3e}'.format(
                                    mag, currIter, pairn, int(np.argmin(_rdmDisc)), _worstDisc), flush=True)
                            if not np.isfinite(_LOSS2_):
                                # 加了数值防护后这里不该再出现非有限值; 真出现说明还有别的来源,
                                # 立刻中止, 而不是继续跑几小时往图里写废数据。
                                raise RuntimeError(
                                    'corrSum={} 非有限值 (mag={} iter={} pairn={} min_rdm_disc={:.3e})'.format(
                                        _LOSS2_, mag, currIter, pairn, _worstDisc))
                        _thisWeightContainer.append(_thisWeight)
                        _LossContainer.append(_Loss)

                        for currLayer in range(12):
                            for currIm in range(numImages):
                                ActualCorrelation[currLayer, pairn, currIm] = _currImageRDM[
                                    currLayer * numImages + currIm]
                        padded = pad_activ(this, shapes)
                        PADDED[pairn] = padded

                        if func == 'Grad3':
                            g1[xx:xx + fshape[1], yy:yy + fshape[1]] = grad1
                            g2[xx:xx + fshape[1], yy:yy + fshape[1]] = grad2
                            _p0 = time.time()
                            g1 = g1 / g1.std() + 1e-8 if not lapnorm else lap_norm_func(g1)
                            g2 = g2 / g2.std() + 1e-8 if not lapnorm else lap_norm_func(g2)
                            _p1 = time.time()
                            _PROF['lapnorm'] = _PROF.get('lapnorm', 0.0) + (_p1 - _p0)
                            g1 = AdamOpt1.update(g1, currIter, pairn)
                            g2 = AdamOpt2.update(g2, currIter, pairn)
                            _PROF['adam'] = _PROF.get('adam', 0.0) + (time.time() - _p1)

                            compareg1g2List.append(compare(g1, g2, currIter=currIter, pairn=pairn))

                            def getIDrange(start_num):
                                IDList = []
                                for i in start_num:
                                    IDList = IDList + list(range(i, i + 10))
                                return IDList

                            if decay_flag:
                                start_num = [0, 250, 300, 350, 400, 450, 500, 550]
                            else:
                                start_num = [0]
                            IDList = getIDrange(start_num)
                            i = len(compareg1g2List) - 1
                            if currIter in IDList:
                                compareg1g2 = np.mean(compareg1g2List[(i - i % 50):i + 1])
                            g2 = breaklink(g2 * compareg1g2)
                            ims[pairn] = ims[pairn] - g1 - g2

                            # 始终生效的 NaN 哨兵(不受 SYNTH_VERBOSE 影响)。
                            # corrSum 那条检查挡不住这条路径: compareg1g2 来自 compare() 里的
                            # 范数相除, 分母为 0 时为 nan, 乘进 g2 就让整张图变 nan, 而它只在
                            # 前 10 轮计算一次之后就冻结, 一旦取到 nan 会静默毁掉后面全部迭代。
                            # 用 .sum() 做哨兵: 单次遍历、无额外分配, 相对 26s/轮 可忽略。
                            _nanSrc = []
                            if not np.isfinite(_LOSS2_):
                                _nanSrc.append('corrSum={}'.format(_LOSS2_))
                            if not np.isfinite(compareg1g2):
                                _nanSrc.append('compareg1g2={}'.format(compareg1g2))
                            if not np.isfinite(np.asarray(g1).sum()):
                                _nanSrc.append('g1(dGrad侧)')
                            if not np.isfinite(np.asarray(g2).sum()):
                                _nanSrc.append('g2(CorrGrad侧)')
                            if not np.isfinite(np.asarray(ims[pairn]).sum()):
                                _nanSrc.append('ims[{}]'.format(pairn))
                            if _nanSrc:
                                raise RuntimeError(
                                    '[NAN] mag={} iter={} pairn={} 非有限值来源: {} | '
                                    'min_rdm_disc={:.3e} | 立即中止, 不继续污染后续迭代。'
                                    '复现用: SYNTH_SEED={} SEED_MODE={} SYNTH_TAG={}'.format(
                                        mag, currIter, pairn, ', '.join(_nanSrc),
                                        float(np.min(_rdmDisc)), SYNTH_SEED, seed_mode, tag))

                        else:
                            g[xx:xx + fshape[1], yy:yy + fshape[1]] = grad
                            g = g / g.std() + 1e-8 if not lapnorm else lap_norm_func(g)
                            g = AdamOpt.update(g, currIter, pairn)
                            ims[pairn] = ims[pairn] - g

                        if clipDuringUpdateFlag:
                            def Altered_NewCondense(ims):
                                _ = np.asarray(ims)
                                low = np.percentile(_, 0.25)
                                high = np.percentile(_, 99.75)
                                _ = (_ - low) / (high - low) * 255
                                __ = np.clip(_, 0, 255)
                                image = PNGimageMatrix2List(__)
                                return image

                            def PNGimageMatrix2List(__):
                                PNGimage = []
                                for currLayer in range(__.shape[0]):
                                    PNGimage.append(__[currLayer])
                                return PNGimage

                            ims = Altered_NewCondense(ims)

                        if decay:
                            ims[pairn] = decRate * ims[pairn]
                    if currIter == iters - 1 or currIter % 50 == 0:
                        saveFolder = '{}/images/{}/mag{}phase{}'.format(current_dir, tag, mag, phase, currIter)
                        if not os.path.isdir(saveFolder):
                            os.makedirs(saveFolder)
                        saveFolder = '{}/images/{}/mag{}phase{}/currIter{}'.format(current_dir, tag, mag, phase,
                                                                                   currIter)
                        if not os.path.isdir(saveFolder):
                            os.makedirs(saveFolder)
                        for numbered, thisOne in enumerate(ims):
                            saveArr(thisOne, '{}/im{}of{}.png'.format(saveFolder, numbered + 1,
                                                                      len(ims)), shape, post=post)
                        saveVar('{}/ims.pkl'.format(saveFolder), ims)
                        PathName = '{}/Vars.pkl'.format(saveFolder)
                        saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer,
                                CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)

                    if currIter == iters - 1:
                        saveFolder = '{}/images/{}/mag{}phase{}'.format(current_dir, tag, mag, phase, currIter)
                        if not os.path.isdir(saveFolder):
                            os.makedirs(saveFolder)
                        for numbered, thisOne in enumerate(ims):
                            saveArr(thisOne, '{}/im{}of{}.png'.format(saveFolder, numbered + 1,
                                                                      len(ims)), shape, post=post)
                        saveVar('{}/ims.pkl'.format(saveFolder), ims)
                        PathName = '{}/Vars.pkl'.format(saveFolder)
                        saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer,
                                CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)

                    ActualCorrelationContainer.append(breaklink(ActualCorrelation))
                    CorrBetweenLayersContainer.append(calculateCorrBetweenLayers(ActualCorrelation))

    PathName = '{}/images/{}/Vars.pkl'.format(current_dir, tag)
    saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer, CorrBetweenLayersContainer,
            compareg1g2List, _LOSS2_Container)
    seed_metadata['run_complete'] = True
    with open(f"{run_root}/seed_metadata.json", 'w') as f:
        json.dump(seed_metadata, f, indent=2)
    total_end = time.time()
    _stageAvg = _typical(_iterTimes)
    if _stageAvg == _stageAvg:
        STAGE_AVG[mag] = _stageAvg
    print('[STAGE DONE] mag={} iters={} 本阶段耗时={} 常态={:.1f}s/iter 累计已运行={}'.format(
        mag, iters, _fmt_dur(total_end - _stageStart), _stageAvg,
        _fmt_dur(total_end - JOB_START)), flush=True)
    print('in total %.02f' % ((total_end - total_start) / 60 / 60), 'h has passed')
    print('tag={}\n'.format(tag))
    previousMag = mag
    previousPhase = phase

    tf.reset_default_graph()
    return previousMag, previousPhase


mags = 1 if os.environ.get('SMOKE_TEST', '0') == '1' else 5
previousPhase = 0
continueFlag = False

# 断点续跑: 每个 mag 阶段本来就是从磁盘读上一阶段的 mag{N-1}phase0/ims.pkl 开始的,
# 所以只要让循环从指定阶段起步即可续上; 又因为 Adam 优化器本来就每阶段重建,
# 在 mag 边界续跑不丢任何优化器状态。作业被时限杀掉后, 用 START_MAG 指定重启点。
# 注意: 固定 SYNTH_SEED 后, 续跑会从头重放随机序列, 所以裁剪抖动与不间断跑不完全一致,
# 是另一条同样合法的轨迹, 但不是逐位复现。
START_MAG = int(os.environ.get('START_MAG', '0'))
if START_MAG > 0:
    _need = '{}/images/{}/mag{}phase{}/ims.pkl'.format(current_dir, tag, START_MAG - 1, previousPhase)
    assert os.path.exists(_need), (
        'START_MAG={} 需要上一阶段的图像状态, 但找不到 {}。'
        '请确认该阶段确实跑完了(完成的阶段目录里应同时有 ims.pkl 和 16 张 imNof16.png)。'.format(START_MAG, _need))
    print('[RESUME] 从 mag{} 续跑, 读取 {}'.format(START_MAG, _need), flush=True)

for mag in tqdm(range(START_MAG, mags)):
    if continueFlag:
        previousMag = mag
    else:
        previousMag = mag - 1
    tfLoop(previousMag, previousPhase, continueFlag)
