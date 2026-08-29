#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import math
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from functools import reduce, partial
from PIL import Image
import imageio
from IPython.display import clear_output
import time
import sys
import pdb
'''
    这个函数是对于一次图片的扩大，进行一次处理，具体的逻辑是：
        从深度梦想开始，不断做拉普拉斯金字塔正则化，打开或关闭一些参数，包括低值、低规范、低贡献修剪、图像中的foveating、剪掉未更新的像素
        最开始的初始化的图片是如何初始化的？
            def gen_noise():
                return np.random.uniform(-10, 10, size=(imsize[1:])) + 100.0
            ims = [gen_noise() for noise in range(numImages)]
        保存的图片的位置在哪里？
            saveArr(thisOne, '{}/im{}of{}.png'.format(saveFolder, numbered+1, len(ims)), shape, post=post)
        从哪里加载图片？
'''

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.environ.get(
    'FIG1D_PROJECT_ROOT', os.path.join(SCRIPT_DIR, '..')))
ASSET_ROOT = os.path.abspath(os.environ.get(
    'FIG1D_ASSET_ROOT',
    '/gpfs/milgram/project/turk-browne/projects/localize'))
MODEL_GRAPH = os.path.abspath(os.environ.get(
    'FIG1D_MODEL_GRAPH',
    '/gpfs/milgram/project/turk-browne/users/kp578/localize/utils/'
    'tensorflow_inception_graph.pb'))
OUTPUT_ROOT = os.path.abspath(os.environ.get(
    'FIG1D_OUTPUT_ROOT', os.path.join(PROJECT_ROOT, 'images')))
SYNTH_TAG = os.environ.get('SYNTH_TAG', '')
SYNTH_SEED = int(os.environ.get('SYNTH_SEED', '0'))
N_MAGS = int(os.environ.get('FIG1D_N_MAGS', '4'))
ITER_LIMIT = int(os.environ.get('FIG1D_ITER_LIMIT', '0'))
SEED_MODE = os.environ.get('FIG1D_SEED_MODE', 'noise')
SEED_MANIFEST = os.path.abspath(os.environ.get(
    'FIG1D_SEED_MANIFEST',
    os.path.join(PROJECT_ROOT,
                 'ImageSynthesis/seed_manifests/r2q1_structured_seed_16.txt')))
SEED_LICENSE_RECORD = os.path.abspath(os.environ.get(
    'FIG1D_SEED_LICENSE_RECORD',
    os.path.join(PROJECT_ROOT,
                 'NSD_similarity/FINAL_fig1_sfig17_fig2_nsd_source_license_record.csv')))

if SEED_MODE not in ('noise', 'nsd', 'ccby4'):
    raise ValueError('FIG1D_SEED_MODE must be noise, nsd, or ccby4.')
expected_tag_prefix = 'fig1d_replay_{}_'.format(SEED_MODE)
if not SYNTH_TAG.startswith(expected_tag_prefix):
    raise ValueError(
        'SYNTH_TAG must be explicit and start with {}; this prevents seed-mode '
        'mixups and writes into historical output directories.'.format(
            expected_tag_prefix))
if os.path.basename(SYNTH_TAG) != SYNTH_TAG:
    raise ValueError('SYNTH_TAG must be a directory name, not a path.')
if not 1 <= N_MAGS <= 4:
    raise ValueError('FIG1D_N_MAGS must be between 1 and 4.')
if ITER_LIMIT < 0:
    raise ValueError('FIG1D_ITER_LIMIT must be 0 or a positive integer.')

REQUIRED_ASSETS = {
    'model_graph': MODEL_GRAPH,
    'target_matrix': os.path.join(ASSET_ROOT, 'utils', '16corr_GradOrth.npy'),
    'best_channels': os.path.join(ASSET_ROOT, 'utils', 'bestchans16.pkl'),
    'channel_correlations': os.path.join(ASSET_ROOT, 'utils', 'corrsvals16.pkl'),
}
missing_assets = [path for path in REQUIRED_ASSETS.values() if not os.path.isfile(path)]
if missing_assets:
    raise FileNotFoundError('Missing Figure 1D assets: {}'.format(', '.join(missing_assets)))


def read_seed_manifest(path):
    with open(path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    if len(image_paths) != 16:
        raise ValueError('Image seed manifest must contain exactly 16 non-empty lines.')
    image_paths = [os.path.abspath(p) for p in image_paths]
    if len(set(image_paths)) != 16:
        raise ValueError('Image seed manifest contains duplicate image paths.')
    missing = [p for p in image_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError('Missing seed images: {}'.format(', '.join(missing)))
    return image_paths


SEED_IMAGE_PATHS = read_seed_manifest(SEED_MANIFEST) if SEED_MODE != 'noise' else []
if SEED_MODE != 'noise' and not os.path.isfile(SEED_LICENSE_RECORD):
    raise FileNotFoundError('Missing seed license record: {}'.format(
        SEED_LICENSE_RECORD))

RUN_DIR = os.path.join(OUTPUT_ROOT, SYNTH_TAG)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.mkdir(RUN_DIR)  # Atomic no-overwrite gate.

np.random.seed(SYNTH_SEED)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


try:
    git_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
        universal_newlines=True).strip()
except Exception:
    git_commit = 'unavailable'

RUN_METADATA_PATH = os.path.join(RUN_DIR, 'run_metadata.json')
PROGRESS_PATH = os.path.join(RUN_DIR, 'progress.json')
REPLAY_START_TIME = time.time()
latest_progress = {
    'status': 'initializing',
    'seed_mode': SEED_MODE,
    'tag': SYNTH_TAG,
}
run_metadata = {
    'status': 'running',
    'run_complete': False,
    'tag': SYNTH_TAG,
    'project_root': PROJECT_ROOT,
    'asset_root': ASSET_ROOT,
    'model_graph': MODEL_GRAPH,
    'output_root': OUTPUT_ROOT,
    'git_commit': git_commit,
    'synth_seed': SYNTH_SEED,
    'seed_mode': SEED_MODE,
    'seed_manifest': SEED_MANIFEST if SEED_MODE != 'noise' else None,
    'seed_manifest_sha256': (sha256_file(SEED_MANIFEST)
                             if SEED_MODE != 'noise' else None),
    'seed_license_record': (SEED_LICENSE_RECORD
                            if SEED_MODE != 'noise' else None),
    'seed_license_record_sha256': (sha256_file(SEED_LICENSE_RECORD)
                                   if SEED_MODE != 'noise' else None),
    'seed_preprocessing': ('PIL RGB resize to 196x196 using bilinear resampling'
                           if SEED_MODE != 'noise' else 'historical uniform noise'),
    'downstream_rng_aligned_to_noise': True,
    'n_mags': N_MAGS,
    'iter_limit': ITER_LIMIT,
    'diagnostic_run': N_MAGS < 4 or ITER_LIMIT > 0,
    'completed_mags': [],
    'started_at_utc': datetime.now(timezone.utc).isoformat(),
    'asset_sha256': {name: sha256_file(path) for name, path in REQUIRED_ASSETS.items()},
    'seed_image_sha256': ([sha256_file(path) for path in SEED_IMAGE_PATHS]
                          if SEED_MODE != 'noise' else []),
}


def write_run_metadata():
    with open(RUN_METADATA_PATH, 'w') as f:
        json.dump(run_metadata, f, indent=2, sort_keys=True)


def write_progress(progress):
    temporary_path = PROGRESS_PATH + '.tmp'
    with open(temporary_path, 'w') as f:
        json.dump(progress, f, indent=2, sort_keys=True)
    os.replace(temporary_path, PROGRESS_PATH)


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return '{}d {:02d}h {:02d}m'.format(days, hours, minutes)
    if hours:
        return '{}h {:02d}m'.format(hours, minutes)
    return '{}m {:02d}s'.format(minutes, seconds)


def side_length_for_mag(mag):
    side = 196
    for _ in range(mag):
        side = int(math.ceil(side * 1.4))
    return side


write_run_metadata()
write_progress(latest_progress)
if SEED_MODE != 'noise':
    with open(os.path.join(RUN_DIR, 'seed_manifest_used.txt'), 'w') as f:
        f.write('\n'.join(SEED_IMAGE_PATHS) + '\n')


def tfLoop(previousMag,previousPhase,continueFlag):
    # try:
    #     previousMag = int(sys.argv[1])
    #     previousPhase = int(sys.argv[2])
    # except:
    #     print('no sys.argv')
    print('previousMag=',previousMag)
    print('previousPhase=',previousPhase)


    main_dir = ASSET_ROOT
    model_fn = MODEL_GRAPH

    # creating TensorFlow session and loading the model
    graph = tf.Graph()
    with graph.as_default(), graph.device('/cpu:0'), tf.Session() as sess:
        with tf.gfile.FastGFile(model_fn, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        t_input = tf.placeholder(np.float32, name='input') # define the input tensor
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input':t_preprocessed})

    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]

    layers1 = [l.split('/')[1] for l in layers if l.split('/')[1].split('2')[0] == 'conv']
    layers2 = [l.split('/')[1] for l in layers if 'pool_reduce' in l]
    #layers3 = ['softmax1', 'softmax2']
    incept_layers = layers1 + layers2 # + layers3

    def gen_noise():
        return np.random.uniform(-10, 10, size=(imsize[1:])) + 100.0

    def load_image_seeds():
        images = []
        for image_path in SEED_IMAGE_PATHS:
            with Image.open(image_path) as image:
                image = image.convert('RGB').resize((196, 196), Image.BILINEAR)
                images.append(np.asarray(image, dtype=np.float32))
        return images


    def shapegen(shape_array, layers, padding=1, stride=2, filterdim=3):
        init_dim = shape_array[1]
        init_array = shape_array
        n_filters = [int(graph.get_tensor_by_name('import/{}/conv:0'.format(name)).get_shape()[-1]) for name in layers]
        shapes_array = []
        for x, layer in enumerate(layers):
            if x == 0 or x == 1 or x == 3 or x == 5 or x == 10:
                new_dim = math.ceil(((init_dim+padding-filterdim)/stride) + 1)
                shapes_array.append([1,new_dim,new_dim,n_filters[x]])
                init_dim = new_dim
                init_array = [1,new_dim,new_dim,n_filters[x]]
            else:
                set_array = np.copy(init_array)
                set_array[3] = n_filters[x]
                shapes_array.append(set_array)
        shapes_array = np.array(shapes_array)
        return shapes_array

    def corrsgen(steps, lo=0, hi=1, match=0.25, nLayers=12, nMatch=8):
        corrs = np.arange(lo,(hi+.01), (hi-lo)/(steps-1))
        layerCorrs = []
        layerWeights = []
        for step in range(steps):
            thisLayer = [match] * nMatch + [corrs[step]] * (nLayers-nMatch)
            layerCorrs.append(thisLayer)
            layerWeights.append([corrs[step], 1])
        return layerCorrs, layerWeights, np.around((corrs*100),0).astype(int)

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
            new_image = tf.image.resize_bilinear(np.expand_dims(image,0), np.int32(shape[1:3])).eval()[0]
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
            padded = np.zeros(np.amax(shapes,0))
            padded[:dims[0],:dims[1],:dims[2],:dims[3]] = acts
            padded_weights.append(padded)
        return padded_weights

    def saveArr(arr, name, shape, post='clip'):
        if post == 'clip':
            arr = np.clip((arr.reshape(shape[1:])), 0, 255).astype(np.uint8)
        else:
            arr = (arr-arr.mean())/arr.std()*0.1 + 0.5
            arr = np.uint8(np.clip(arr, 0, 1)*255)
        #arr = Image.fromarray(arr)
        #arr.save(name)
        imageio.imsave(name, arr)

    def gen_fovea(shape, fovshape):
        fovsize = fovshape[1]
        h,w = shape[1:3]
        sx = w-fovsize
        cx = sx/2.
        sy = h-fovsize
        cy = sy/2.
        return sx, sy, cx, cy

    def gen_crops(images, fovparams, fovshape):
        fovsize = fovshape[1]
        sx, sy, cx, cy = fovparams
        cropx = int(np.clip(np.random.normal(cx, sx*0.3, 1), 0, sx))
        cropy = int(np.clip(np.random.normal(cy, sy*0.3, 1), 0, sy))
        new_images = []
        for image in images:
            cropped = image[cropx:cropx+fovsize, cropy:cropy+fovsize]
            new_images.append(cropped)
        return new_images, cropx, cropy

    import pickle

    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    # THESE FUNCTIONS TAKEN DIRECTLY FROM GOOGLE'S DEEPDREAM NOTEBOOK
    def tffunc(*argtypes):
        '''将TF-graph生成函数转换为普通函数的助手。 Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    # 使用TF来调整图像大小的辅助函数 Helper function that uses TF to resize an image
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    resize = tffunc(np.float32, np.int32)(resize)

    k = np.float32([1,4,6,4,1])
    k = np.outer(k, k)
    k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

    def lap_split(img):
        '''将图像分成低频和高频成分 Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
            hi = img-lo2
        return lo, hi

    def lap_split_n(img, n):
        '''构建有n个分割点的拉普拉斯金字塔 Build Laplacian pyramid with n splits'''
        levels = []
        for i in range(n):
            img, hi = lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    def lap_merge(levels):
        '''合并拉普拉斯金字塔 Merge Laplacian pyramid'''
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
        return img

    def normalize_std(img, eps=1e-10):
        '''通过使其标准偏差=1.0来归一化图像 Normalize image by making its standard deviation = 1.0'''
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img/tf.maximum(std, eps)

    def lap_normalize(img, scale_n=4):
        '''进行拉普拉斯金字塔归一化处理 Perform the Laplacian pyramid normalization.'''
        img = tf.expand_dims(img,0)
        tlevels = lap_split_n(img, scale_n)
        tlevels = list(map(normalize_std, tlevels))
        out = lap_merge(tlevels)
        return out[0,:,:,:]


    imsize = [1,196,196,3]

    step = 1.2
    magby=1.4

    '从深度梦想开始，不断做拉普拉斯金字塔正则化，打开或关闭一些参数，包括低值、低规范、低贡献修剪、图像中的foveating、剪掉未更新的像素'
    # Start with deep dream, continually do laplacian pyramid regularization, turn on or off a number of parameters
    # including low value, low norm, low contribution trimming, foveating in the image, clipping unupdated pixels

    lapnorm = True
    lap_n = 4

    fov = True
    fov_grow = 1.35  # typically 1.35
    fov_init = 0.75  # typically 0.75
    _thisWeightContainer = []
    _LossContainer =[]
    _LOSS2_Container=[]
    decay = True
    decRate = 0.999

    #low_val = False
    #clip_norm = False
    #low_cont = False

    #trim_focus = False
    #alpha_bg = False

    #low_norm_n = 15
    #low_val_n = 15
    #low_cont_n = 5

    # deprocess by clipping? If not, then it will condense and scale pixels into allowable values, rather than
    # just clipping impossible pixels
    post = 'clip'

    "用于寻找最不一样的通道"
    ########################################################
    ##### FOR FINDING MOST DIFFERENTIABLE CHANNELS #########
    ################ 用于寻找最不一样的通道 ####################
    ########################################################

    # 到目前为止，只有几个数字在这里可以使用，也可能只有16个。 so far, only a few numbers will be funtional here, or maybe just 16.
    numImages = 16

    bestchansfile = '{}/utils/bestchans{}.pkl'.format(main_dir, numImages)
    corrsvalsfile = '{}/utils/corrsvals{}.pkl'.format(main_dir, numImages)
    if not os.path.exists(bestchansfile):

        bestchandict={}
        corrvalsdict={}

        iters = 3000000

        for layer in incept_layers[:]:
            print(layer)
            channels = [im for im in os.listdir(layer) if im.split('.')[-1] == 'png']
            channels.sort()
            imgs=[]
            for channel in range(len(channels)):
                print('{}    {}/{}'.format(layer, channel+1, len(channels)))
                clear_output(wait=True)
                img0 = Image.open('{}/{}.png'.format(layer, channel))
                img0 = img0.convert("RGB")
                img0 = np.float32(img0)
                act = sess.run(get_tensor([layer]), {t_input:img0})[0]
                ACT = act if channel == 0 else np.concatenate((ACT, act))
            ACT = np.reshape(ACT, (len(channels), -1))
            CORRS = np.corrcoef(ACT)

            lowavg = 1
            lowtop = 1
            for x in range(iters):
                newsamp = np.random.choice(len(channels), numImages, replace=False)
                corrs = CORRS[newsamp[:,None], newsamp]
                corrs[corrs>0.99] = 0
                thisavg = np.mean(np.absolute(corrs))
                thistop = np.amax(np.absolute(corrs))
                if ((lowavg-thisavg)+(lowtop-thistop))/(lowavg+lowtop) > .01:
                    lowavg = thisavg
                    lowtop = thistop
                    bestchans = newsamp
                    print('{}    {} {} {}'.format(layer, x, lowavg, lowtop))
            bestchandict[layer]=bestchans
            corrvalsdict[layer]=np.array([lowavg, lowtop])
            clear_output(wait = True)
        save_obj(bestchandict, bestchansfile)
        save_obj(corrvalsdict, corrsvalsfile)
    else:
        bestchandict = load_obj(bestchansfile.split('.pkl')[0])
        corrvalsdict = load_obj(corrsvalsfile.split('.pkl')[0])


    for i, layer in enumerate(incept_layers[:]):
        choose = bestchandict[layer]
        choose.sort()
        if layer == 'mixed3a_pool_reduce_pre_relu':
            seq = True
            while seq:
                tempseq = False
                np.random.shuffle(choose)
                for i in range(0,choose.shape[0]-1,2):
                    if choose[i] == choose[i+1]:
                        tempseq = True
                        break
                if not tempseq:
                    seq = False
            print(choose)

        CHOOSE = choose if i == 0 else np.vstack((CHOOSE, choose))
    CHOOSE = np.expand_dims(CHOOSE, 0)
    print(CHOOSE.shape)

    '构建相关性结构'
    ### BUILDING THE CORRELATION STRUCTURE

    version = 'GradOrth'  # orth（正交相关矩阵）或rand（在某些层产生随机）。  orth (orthogonal correlation matrices) or rand (generate random in certain layers)
    matchval = 0.4  # value to match all layers except for structured on
    structlayers = [8,9,10,11]  # layers to assign an actual random correlation structure to
    corlo = 0.5  # lowest correlation value to assign in random correlation structure
    corhi = 1  # highest correlation value to assign in random correlation structure

    if version == 'orth':
        CORRMAT = np.load('{}/utils/{}corr.npy'.format(main_dir, numImages))
        CORRMAT = np.swapaxes(CORRMAT,0,2)
    elif version == 'GradOrth':
            CORRMAT = np.load('{}/utils/{}corr_GradOrth.npy'.format(main_dir, numImages))
    else:
        CORRMAT = np.ones((12,numImages,numImages))
        CORRMAT[:,:,:] = matchval
        for i in range(12):
            np.fill_diagonal(CORRMAT[i], 1)
        test_array = np.random.uniform(corlo, corhi, size=(numImages,numImages))
        test_array[1,:] = 0.05
        np.fill_diagonal(test_array, 1)
        for x in range(numImages):
            for y in range(numImages):
                test_array[y,x] = test_array[x,y]
        for layer in structlayers:
            CORRMAT[layer] = test_array

    #save variables:
    def saveVar(PathName,*var):
        with open(PathName, 'wb') as f:
            pickle.dump(var, f)

    "用于复用优化 MULTIPLEXED OPTIMIZATION"
    #############################################
    ###### FOR MULTIPLEXED OPTIMIZATION #########
    #############################################
    tag=SYNTH_TAG
    decay_flag=False
    clipDuringUpdateFlag=True
    assert os.path.isdir(RUN_DIR)

    def truefn(multipliers,numOfthisOne,thisLayer,chans,currLayer,cell,i,ims):
        thisOne = tf.multiply(thisLayer[:,:,:,chans[currLayer][i]],multiplier[currLayer,cell,i])
        multipliers=tf.add(multipliers,tf.multiply(thisOne,len(ims) - 1))
        return multipliers,(numOfthisOne+len(ims) - 1)
    def falsefn(multipliers,numOfthisOne,thisLayer,chans,currLayer,cell,i,ims):
        thisOne = tf.multiply(thisLayer[:,:,:,chans[currLayer][i]],multiplier[currLayer,cell,i])
        multipliers=tf.add(multipliers,thisOne)
        return multipliers,(numOfthisOne+1)
    def TFCorr(thisLayer,otherLayer,multiplier,currLayer,cell,i,loss,numOfthisLoss):
        numerator = (dim *
                     tf.reduce_sum(tf.multiply(thisLayer, otherLayer))) - (tf.reduce_sum(thisLayer) *
                     tf.reduce_sum(otherLayer))
        denominator = tf.sqrt((dim * tf.reduce_sum(tf.square(thisLayer)) - tf.square(tf.reduce_sum(thisLayer))) *
                              (dim * tf.reduce_sum(tf.square(otherLayer)) - tf.square(tf.reduce_sum(otherLayer))))
        thisCorr = tf.truediv(numerator, denominator)
        # Pull the desired correlation value, and get squared difference, and append
        thisLoss = tf.square(thisCorr - multiplier[currLayer,cell,i])
        loss=tf.add(loss,thisLoss)
        return loss,(numOfthisLoss+1),thisCorr
    def DoNothing(loss,numOfthisLoss):
        thisCorr=0.0
        return loss,numOfthisLoss,thisCorr
    def ListCorrSquare(thisLayer,otherLayer):
        dim = len(thisLayer)#dim = tf.cast(tf.reduce_prod(c), tf.float32)
        numerator = (dim *
                     tf.reduce_sum(tf.multiply(thisLayer, otherLayer))) - (tf.reduce_sum(thisLayer) *
                     tf.reduce_sum(otherLayer))
        denominator = tf.sqrt((dim * tf.reduce_sum(tf.square(thisLayer)) - tf.square(tf.reduce_sum(thisLayer))) *
                              (dim * tf.reduce_sum(tf.square(otherLayer)) - tf.square(tf.reduce_sum(otherLayer))))
        thisCorr = tf.truediv(numerator, denominator)
        return tf.square(thisCorr)
    def calImageRDMlayerCorrelation(currImageRDM):
        corrSum = []#tf.constant(0.0,shape=[11])
        for currLayerI in range(12):
            for currLayerJ in range(currLayerI):
                corrSum.append(ListCorrSquare(currImageRDM[currLayerI*numImages:(currLayerI+1)*numImages], currImageRDM[currLayerJ*numImages:(currLayerJ+1)*numImages]))
        return corrSum
    def calculateCorrBetweenLayers(ActualCorrelation):
        M=0.0
        numImages=ActualCorrelation.shape[1]
        for currLayerI in range(ActualCorrelation.shape[0]):
            for currLayerJ in range(currLayerI):
                #t=np.corrcoef(ActualCorrelation[currLayerI,:].flatten(), ActualCorrelation[currLayerJ,:].flatten())[0,1]
                t=np.corrcoef(ActualCorrelation[currLayerI][np.triu_indices(numImages,k = 1)],
                              ActualCorrelation[currLayerJ][np.triu_indices(numImages,k = 1)])[0,1]
                M=M+t**2
        return M
    def breaklink(a):
        a=a+1
        a=a-1
        return a
    "定义AdamOptimizer和梯度测量方法 define AdamOptimizer and gradient measuring method"
    class kpAdamOptimizer():
        def __init__(self, images, learning_rate=1.2, beta1=0.9, beta2=0.999, epsilon=1e-8, name="Adam"):
            self._lr = learning_rate
            self._beta1 = beta1
            self._beta2 = beta2
            self._epsilon = epsilon
            # self.t = 0
            self.m = np.zeros(images.shape)
            self.v = np.zeros(images.shape)
            # self.img=image
            self.CostList=[]
        def update(self,g,currIter,pairn):#raw_g is original g, raw_g.shape is the shape of image, while g.shape is the shape of np.asarray(ims)
            # only self.m and self.v is accumulated inside the Adam object, the other things are given or calculated every update time
            self.t = currIter+1
            self.m[pairn] = self._beta1 * self.m[pairn] + (1 - self._beta1)*g
            self.v[pairn]=self._beta2*self.v[pairn]+(1-self._beta2)*np.multiply(g,g)
            M=self.m[pairn]/(1-self._beta1**self.t)
            V=self.v[pairn]/(1-self._beta2**self.t)
            return (self._lr*M/(np.sqrt(V)+self._epsilon))


    import math
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)
    def angle_between(v1, v2):
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/math.pi*180
    def compare(a,b):
        a=a.flatten()
        b=b.flatten()
        print("{} is {} long\n{} is {} long. \n angle is {}".format('a',np.linalg.norm(a),'b',np.linalg.norm(b),angle_between(a, b)))
        return (np.linalg.norm(a)/np.linalg.norm(b))
    def plot_vector(m,t):
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,1, figsize=(4,4))
        ax.plot(range(len(m)),m)
        ax.set_title(t)
        plt.show()

    total_start = time.time()
    lapnorm = True
    lap_n = 4
    mags = 4
    decRate = 0.999
    fov_grow = 1.4  # typically 1.35
    fov_init = 1.0  # typically 0.75

    chanPair = CHOOSE   # a 1 X number of Layers X number of Images array, containing relatively dissimilar channels
    "一个我们要设置通道的层的列表 a list of layers wherein we want to set a channel"
    layerChans = [i for i in np.arange(chanPair.shape[1]) if not all(chanPair[0,i] == np.array([0] * numImages))]

    thisCorrContainer=[]
    ActualCorrelationContainer=[]
    CorrBetweenLayersContainer=[]
    compareg1g2List=[]
    for pair in range(chanPair.shape[0]):
        theseChans = chanPair[pair]
        # pull dream weight, r value suffix, and correlation value for this point
        corr = 0.2
        # initialize appropriate shapes and images
        if continueFlag:
            mag = previousMag
        else:
            mag = previousMag+1
        if mag == 0:
            noise_ims = [gen_noise() for noise in range(numImages)]
            ims = noise_ims if SEED_MODE == 'noise' else load_image_seeds()
            assert len(ims) == numImages
            assert all(image.shape == tuple(imsize[1:]) for image in ims)
        else:
            PreviousImgFile='{}/mag{}phase{}/ims.pkl'.format(RUN_DIR,previousMag,previousPhase)
            with open(PreviousImgFile,'rb') as f:
               [ims] = pickle.load(f)
        # PreviousImgFile='{}/images/MultiSynth_46Image_trial1/mag0phase1/ims.pkl'
        # with open(PreviousImgFile,'rb') as f:
        #    [ims] = pickle.load(f)
        # if fovea is set, change the dimensions to match the fovea size
        shape = imsize
        fshape = enhance_dims(shape, fov_init) if fov else shape
        scalefov = fov_grow if fov else magby

        "设置窗口大小，找到剩余的宽度，以及窗口居中的位置 set window size, find leftover width, and position where window is centered"
        shapes = shapegen(fshape, incept_layers)
        padshape = np.amax(shapes, axis=0)

        "设置图形，并添加一些占位符和操作 set up the graph, and add a number of placeholders and operations"
        with graph.as_default(), graph.device('/cpu:0'), tf.Session() as sess:
            "这个函数实现了拉普拉斯金字塔梯度的归一化 This function implements laplacian pyramid gradient normalization"
            lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

            "要匹配的特征（即其他图像），要达到的r值，通道和相对通道权重) features to match (i.e. other image), r values to reach, channels, and relative channel weights)"
            cell = tf.placeholder(tf.int32, shape=(None))
            toMatch = tf.placeholder(tf.float32, shape=(None))
            benchmark = tf.placeholder(tf.float32, shape=(None))
            chans = tf.placeholder(tf.int32, shape=(None))
            cutAt = tf.placeholder(tf.int32, shape=(None))
            multiplier = tf.placeholder(tf.float32, shape=(None))

            currImageRDM=[]
            loss = tf.constant(0,tf.float32)
            numOfthisLoss=tf.constant(0,tf.float32)
            _weights = []#tf.convert_to_tensor(0,tf.float32)

            # Get activations from all layers for a given image
            matchActivations = get_tensor(incept_layers)
            # lossContainer=[]
            for currLayer, layer in enumerate(incept_layers):
                # pull activations for this image. From feed dict, get other activations, where to cut, and cut.
                thisLayer = matchActivations[currLayer]
                c = cutAt[currLayer]

                # produce int for number of observations from all of the dimensions of this layer
                dim = tf.cast(tf.reduce_prod(c), tf.float32)


                # also try to maximize the provided channels
                if currLayer in layerChans:
                    multipliers = tf.constant(0,tf.float32)
                    numOfthisOne = tf.constant(0,tf.float32)
                    for i, im in enumerate(ims):
                        multipliers,numOfthisOne=tf.cond(tf.math.equal(i,cell), lambda:truefn(multipliers,numOfthisOne,thisLayer,chans,currLayer,cell,i,ims), lambda:falsefn(multipliers,numOfthisOne,thisLayer,chans,currLayer,cell,i,ims))

                        other = toMatch[i,currLayer]
                        otherLayer = other[:c[0],:c[1],:c[2],:c[3]]

                        loss,numOfthisLoss,thisCorr=tf.cond(~tf.math.equal(i,cell), lambda:TFCorr(thisLayer,otherLayer,multiplier,currLayer,cell,i,loss,numOfthisLoss), lambda:DoNothing(loss,numOfthisLoss))

                        currImageRDM.append(thisCorr)
                        #lossContainer.append(loss)

                    _thisWeight = (tf.reduce_mean(tf.truediv(multipliers,numOfthisOne))) * -1
                    _weights.append(_thisWeight)

            corrSum=reduce(tf.add, calImageRDMlayerCorrelation(currImageRDM))
            Loss = tf.reduce_sum(loss)
            thisWeight = reduce(tf.add, _weights)

            "函数来计算成本函数相对于像素值的梯度（输入）。 function to compute the gradient of the cost function with respect to the pixel values (the input)"
            Grad = tf.gradients(Loss, t_input)[0]
            dGrad = tf.gradients(thisWeight, t_input)[0]
            CorrGrad = tf.gradients(corrSum, t_input)[0]
            Grad3=dGrad+CorrGrad*3000#2970
            last = time.time()
            # for mag in range(mags):
            if mag == 4:
                exit()
                raise Exception
            if mag > 0:
                for currMag in range(mag):
                    shape, fshape, shapes = mag_shapes(shape, fshape, incept_layers, magby, fov_grow)
                if not continueFlag:
                    ims = mag_images(ims, shape)
            fovparams = gen_fovea(shape, fshape)
            # phases = [dGrad,Grad]
            # phases = [dGrad,Grad,Grad3] #image results stored in localize/images/multipleObjectiveTrial1 folder, text analize stored in `lab records in Turk-Browne Lab` Jul8 2019 `actual correlation between layers along optimization process`. used `ActualCorrelation` as `multiplier` at phase 2(the third phase)
            phases = ['Grad3']
            for phase, func in enumerate(phases):
                if func == 'Grad3':
                    AdamOpt1=kpAdamOptimizer(np.asarray(ims),learning_rate=1.2)
                    AdamOpt2=kpAdamOptimizer(np.asarray(ims),learning_rate=1.2)
                    if mag == 0:
                        iters=400
                    elif mag == 1:
                        iters=400
                    elif mag == 2:
                        iters=400
                    elif mag == 3:
                        iters=600
                    else:
                        raise Exception
                    if ITER_LIMIT:
                        iters = min(iters, ITER_LIMIT)
                else:
                    AdamOpt=kpAdamOptimizer(np.asarray(ims),learning_rate=1.2)
                    iters=200
                iteration_durations = []
                for currIter in range(iters):
                    iteration_started_at = time.time()
                    if currIter in [250,300,350,400,450,500,550]:
                        if decay_flag:
                            AdamOpt1._lr=AdamOpt1._lr/2
                            AdamOpt2._lr=AdamOpt2._lr/2
                    print('{} images, size={}/{}, phase={}/{}, {}%, iter time={}s'.format(numImages, mag+1,
                                                                                         mags, phase+1, len(phases),
                                                                                         np.around(((currIter+1)/iters)*100, 2),
                                                                                         int(time.time() - last)))
                    last = time.time()
                    #clear_output(wait=True)
                    cropped, xx, yy = gen_crops(ims, fovparams, fshape)
                    # ActualCorrelation=np.zeros(CORRMAT.shape)
                    ActualCorrelation=np.zeros(CORRMAT.shape) if phase==0 and currIter==0 else ActualCorrelation
                    for pairn, (thiscrop, pairmate) in enumerate(zip(cropped, ims)):
                        if pairn == 0 and phase == 0:
                            for anum, thisone in enumerate(cropped):
                                thisact = sess.run(matchActivations, feed_dict={t_input:thisone})
                                padthisact = pad_activ(thisact, shapes)
                                PADDED = np.expand_dims(padthisact, 0) if anum == 0 else np.concatenate((PADDED, np.expand_dims(padthisact, 0)))

                        assert PADDED.shape[0] == len(cropped)
                        if func == 'Grad3':
                            g1 = np.zeros_like(ims[0])
                            g2 = np.zeros_like(ims[0])
                        else:
                            g = np.zeros_like(ims[0])

                        "多目标梯度下降法 multiple objective gradient descent"
                        if func == 'Grad3':
                            grad1, grad2, _LOSS1_, _LOSS2_, this, _thisWeight, _Loss, _currImageRDM = sess.run([dGrad, CorrGrad, thisWeight, corrSum, matchActivations, thisWeight, Loss, currImageRDM],
                                                  feed_dict={t_input:thiscrop, cutAt:shapes, toMatch:PADDED, cell:pairn,
                                                            chans:theseChans, multiplier:CORRMAT}) #benchmark:corr,


                        elif func == 'dGrad':#dGrad
                            grad, _LOSS_, this, _thisWeight, _Loss, _currImageRDM = sess.run([dGrad, Loss, matchActivations, thisWeight, Loss, currImageRDM],
                                                  feed_dict = {t_input:thiscrop, cutAt:shapes, toMatch:PADDED, cell:pairn,
                                                            chans:theseChans, multiplier:CORRMAT}) #benchmark:corr,

                        elif func == 'Grad':#Grad
                            grad, _LOSS_, this, _thisWeight, _Loss, _currImageRDM = sess.run([Grad, thisWeight, matchActivations, thisWeight, Loss, currImageRDM],
                                                  feed_dict={t_input:thiscrop, cutAt:shapes, toMatch:PADDED, cell:pairn,
                                                            chans:theseChans, multiplier:CORRMAT}) #benchmark:corr,

                        elif func == 'CorrGrad':#CorrGrad
                            grad, _LOSS_, this, _thisWeight, _Loss, _currImageRDM = sess.run([CorrGrad, corrSum, matchActivations, thisWeight, Loss, currImageRDM],
                                                  feed_dict={t_input:thiscrop, cutAt:shapes, toMatch:PADDED, cell:pairn,
                                                            chans:theseChans, multiplier:CORRMAT}) #benchmark:corr,

                        else:
                            raise Exception(f"func={func}")
                        if func == 'Grad3':
                            print('_LOSS2_',_LOSS2_)
                            _LOSS2_Container.append(_LOSS2_)
                        _thisWeightContainer.append(_thisWeight)
                        _LossContainer.append(_Loss)

                        for currLayer in range(12):
                            for currIm in range(numImages):
                                ActualCorrelation[currLayer,pairn,currIm]=_currImageRDM[currLayer*numImages+currIm]
                        padded = pad_activ(this, shapes)
                        PADDED[pairn] = padded


                        if func == 'Grad3':#dGrad
                            g1[xx:xx+fshape[1],yy:yy+fshape[1]] = grad1
                            g2[xx:xx+fshape[1],yy:yy+fshape[1]] = grad2
                            g1 = g1/g1.std()+1e-8 if not lapnorm else lap_norm_func(g1)
                            g2 = g2/g2.std()+1e-8 if not lapnorm else lap_norm_func(g2)
                            g1=AdamOpt1.update(g1,currIter,pairn)
                            g2=AdamOpt2.update(g2,currIter,pairn)

                            compareg1g2List.append(compare(g1,g2))

                            def getIDrange(start_num):
                                IDList=[]
                                for i in start_num:
                                    IDList=IDList+list(range(i,i+10))
                                return IDList
                            if decay_flag:
                                start_num=[0,250,300,350,400,450,500,550]
                            else:
                                start_num=[0]
                            IDList=getIDrange(start_num)
                            i=len(compareg1g2List)-1
                            if currIter in IDList:
                                compareg1g2=np.mean(compareg1g2List[(i-i%50):i+1])
                            g2=breaklink(g2*compareg1g2)
                            ims[pairn] = ims[pairn] - g1 - g2

                        else:
                            g[xx:xx+fshape[1],yy:yy+fshape[1]] = grad
                            g = g/g.std()+1e-8 if not lapnorm else lap_norm_func(g)
                            g = AdamOpt.update(g,currIter,pairn)
                            ims[pairn] = ims[pairn] - g

                        if clipDuringUpdateFlag:
                            def Altered_NewCondense(ims):
                                _=np.asarray(ims)
                                low=np.percentile(_, 0.25)
                                high=np.percentile(_,99.75)
                                _=(_-low)/(high-low)*255
                                __=np.clip(_, 0,255)
                                image=PNGimageMatrix2List(__)
                                return image
                            def PNGimageMatrix2List(__):
                                PNGimage=[]
                                for currLayer in range(__.shape[0]):
                                    PNGimage.append(__[currLayer])
                                return PNGimage

                            ims=Altered_NewCondense(ims)

                        if decay:
                            ims[pairn] = decRate*ims[pairn]
                    if currIter == iters-1 or currIter%50==0:
                        saveFolder='{}/mag{}phase{}'.format(RUN_DIR,mag,phase)
                        if not os.path.isdir(saveFolder):
                            os.mkdir(saveFolder)
                        saveFolder='{}/mag{}phase{}/currIter{}'.format(RUN_DIR,mag,phase,currIter)
                        if not os.path.isdir(saveFolder):
                            os.mkdir(saveFolder)
                        for numbered, thisOne in enumerate(ims):
                            saveArr(thisOne, '{}/im{}of{}.png'.format(saveFolder, numbered+1, len(ims)), shape, post=post) # /gpfs/milgram/project/turk-browne/projects/localize/images/{tag}/mag{}phase{}/im{}of{}.png
                        saveVar('{}/ims.pkl'.format(saveFolder),ims)
                        PathName = '{}/Vars.pkl'.format(saveFolder)
                        saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer, CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)

                    if currIter == iters-1:
                        saveFolder='{}/mag{}phase{}'.format(RUN_DIR,mag,phase)
                        if not os.path.isdir(saveFolder):
                            os.mkdir(saveFolder)
                        for numbered, thisOne in enumerate(ims):
                            saveArr(thisOne, '{}/im{}of{}.png'.format(saveFolder, numbered+1,
                                                                                 len(ims)), shape, post=post)
                        saveVar('{}/ims.pkl'.format(saveFolder),ims)
                        PathName = '{}/Vars.pkl'.format(saveFolder)
                        saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer, CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)


                    ActualCorrelationContainer.append(breaklink(ActualCorrelation))
                    CorrBetweenLayersContainer.append(calculateCorrBetweenLayers(ActualCorrelation))

                    iteration_durations.append(time.time() - iteration_started_at)
                    seconds_per_iteration = float(np.mean(iteration_durations[-10:]))
                    remaining_seconds = (iters - currIter - 1) * seconds_per_iteration
                    current_area = float(side_length_for_mag(mag) ** 2)
                    historical_iters = (400, 400, 400, 600)
                    for future_mag in range(mag + 1, N_MAGS):
                        future_iters = historical_iters[future_mag]
                        if ITER_LIMIT:
                            future_iters = min(future_iters, ITER_LIMIT)
                        future_area = float(side_length_for_mag(future_mag) ** 2)
                        remaining_seconds += (seconds_per_iteration *
                                              future_area / current_area *
                                              future_iters)
                    estimated_finish = (datetime.now(timezone.utc) +
                                        timedelta(seconds=remaining_seconds))
                    completed_iterations = currIter + 1
                    latest_progress.update({
                        'status': 'running',
                        'mag_zero_based': mag,
                        'mag_one_based': mag + 1,
                        'n_mags': N_MAGS,
                        'iteration': completed_iterations,
                        'iterations_in_mag': iters,
                        'mag_percent': 100.0 * completed_iterations / iters,
                        'seconds_per_iteration_recent_mean': seconds_per_iteration,
                        'elapsed_seconds': time.time() - REPLAY_START_TIME,
                        'eta_seconds': remaining_seconds,
                        'eta_human': format_duration(remaining_seconds),
                        'estimated_finish_utc': estimated_finish.isoformat(),
                        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                        'eta_method': ('mean of up to 10 latest iterations; future '
                                       'magnifications scaled by image area'),
                    })
                    write_progress(latest_progress)
                    print(
                        'PROGRESS mag={}/{}, iter={}/{}, recent={:.1f}s/iter, '
                        'ETA={}, finish={}'.format(
                            mag + 1, N_MAGS, completed_iterations, iters,
                            seconds_per_iteration,
                            latest_progress['eta_human'],
                            latest_progress['estimated_finish_utc']))

    PathName = '{}/Vars.pkl'.format(RUN_DIR)
    saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer, CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)
    total_end = time.time()
    print('in total %.02f'%((total_end-total_start)/60/60), 'h has passed')
    print('tag={}\n'.format(tag))
    previousMag = mag
    previousPhase = phase

    tf.reset_default_graph()
    return previousMag,previousPhase
# filename='../../python/MultiSynth/MultiSynth_16Image_SinglePhase_StepDecay_seperateMag.py'
# print('excuting ',filename)
# from math import *
# exec(open("filename").read(), {"previousMag":mag,"previousPhase":phase })


previousPhase=0
continueFlag=False

try:
    for mag in range(0, N_MAGS):
        if continueFlag:
            previousMag=mag
        else:
            previousMag=mag-1
        completed_mag, previousPhase = tfLoop(previousMag,previousPhase,continueFlag)
        run_metadata['completed_mags'].append(completed_mag)
        write_run_metadata()
except BaseException as exc:
    run_metadata['status'] = 'failed'
    run_metadata['error_type'] = type(exc).__name__
    run_metadata['error_message'] = str(exc)
    run_metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_run_metadata()
    latest_progress.update({
        'status': 'failed',
        'error_type': type(exc).__name__,
        'error_message': str(exc),
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    write_progress(latest_progress)
    raise
else:
    run_metadata['status'] = ('diagnostic_complete'
                              if run_metadata['diagnostic_run'] else 'complete')
    run_metadata['run_complete'] = not run_metadata['diagnostic_run']
    run_metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_run_metadata()
    latest_progress.update({
        'status': run_metadata['status'],
        'eta_seconds': 0,
        'eta_human': '0m 00s',
        'estimated_finish_utc': run_metadata['finished_at_utc'],
        'updated_at_utc': run_metadata['finished_at_utc'],
    })
    write_progress(latest_progress)
