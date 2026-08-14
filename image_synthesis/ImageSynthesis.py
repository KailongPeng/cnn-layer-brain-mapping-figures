import tensorflow as tf
import numpy as np
import os
import math
import json
import shutil
from functools import reduce, partial
from pathlib import Path
from PIL import Image
import imageio
import time
import pickle
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = Path(
    os.environ.get('SYNTH_ASSET_DIR', PROJECT_DIR / 'image_synthesis' / 'assets')
).resolve()
OUTPUT_DIR = Path(
    os.environ.get('SYNTH_OUTPUT_DIR', PROJECT_DIR / 'outputs' / 'image_synthesis')
).resolve()
SYNTH_DEVICE = os.environ.get('SYNTH_DEVICE', '/cpu:0')


def load_selected_channels(path, expected_layers, expected_n=16):
    with open(path, 'r') as f:
        raw = json.load(f)
    if set(raw) != set(expected_layers):
        missing = sorted(set(expected_layers) - set(raw))
        extra = sorted(set(raw) - set(expected_layers))
        raise ValueError(f"Selected-channel keys do not match model layers; missing={missing}, extra={extra}")
    channels = {}
    for layer in expected_layers:
        values = np.asarray(raw[layer], dtype=np.int32)
        if values.shape != (expected_n,) or np.any(values < 0):
            raise ValueError(f"Expected {expected_n} nonnegative channels for {layer}, got {values}")
        channels[layer] = values
    return channels


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

    model_fn = ASSET_DIR / 'tensorflow_inception_graph.pb'
    if not model_fn.is_file():
        raise FileNotFoundError(f"Missing Inception graph: {model_fn}")

    graph = tf.Graph()
    with graph.as_default(), graph.device(SYNTH_DEVICE), tf.Session() as sess:
        with tf.gfile.FastGFile(str(model_fn), 'rb') as f:
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

    selected_channels_file = ASSET_DIR / 'selected_channels_16.json'
    if not selected_channels_file.is_file():
        raise FileNotFoundError(f"Missing selected-channel file: {selected_channels_file}")
    bestchandict = load_selected_channels(selected_channels_file, incept_layers, numImages)
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
    seed_manifest = os.environ.get('SEED_MANIFEST')
    if seed_mode not in ['noise', 'structured']:
        raise ValueError(f"Unsupported seed_mode: {seed_mode}")
    if seed_mode == 'structured' and not seed_manifest:
        raise ValueError("SEED_MANIFEST is required when SEED_MODE=structured")
    smoke = os.environ.get('SMOKE_TEST', '0') == '1'

    if version == 'GradOrth':
        target_matrix_file = ASSET_DIR / '{}corr_GradOrth.npy'.format(numImages)
        if not target_matrix_file.is_file():
            raise FileNotFoundError(f"Missing synthesis target matrix: {target_matrix_file}")
        CORRMAT = np.load(str(target_matrix_file))
        expected_shape = (len(incept_layers), numImages, numImages)
        if CORRMAT.shape != expected_shape or not np.all(np.isfinite(CORRMAT)):
            raise ValueError(f"Expected finite target matrix with shape {expected_shape}, got {CORRMAT.shape}")
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
    run_root = OUTPUT_DIR / tag
    run_root.mkdir(parents=True, exist_ok=True)

    seed_metadata = {
        'seed_mode': seed_mode,
        'seed_manifest': seed_manifest,
        'tag': tag,
        'num_images': numImages,
        'image_loader': 'loadimageList_resize_539',
    }
    if seed_mode == 'structured':
        shutil.copyfile(seed_manifest, run_root / 'seed_manifest_used.txt')
    with open(run_root / 'seed_metadata.json', 'w') as f:
        json.dump(seed_metadata, f, indent=2)

    def truefn(multipliers, numOfthisOne, thisLayer, chans, currLayer, cell, i, ims):
        thisOne = tf.multiply(thisLayer[:, :, :, chans[currLayer][i]], multiplier[currLayer, cell, i])
        multipliers = tf.add(multipliers, tf.multiply(thisOne, len(ims) - 1))
        return multipliers, (numOfthisOne + len(ims) - 1)

    def falsefn(multipliers, numOfthisOne, thisLayer, chans, currLayer, cell, i, ims):
        thisOne = tf.multiply(thisLayer[:, :, :, chans[currLayer][i]], multiplier[currLayer, cell, i])
        multipliers = tf.add(multipliers, thisOne)
        return multipliers, (numOfthisOne + 1)

    def TFCorr(thisLayer, otherLayer, multiplier, currLayer, cell, i, loss, numOfthisLoss):
        numerator = (dim *
                     tf.reduce_sum(tf.multiply(thisLayer, otherLayer))) - (tf.reduce_sum(thisLayer) *
                                                                           tf.reduce_sum(otherLayer))
        denominator = tf.sqrt((dim * tf.reduce_sum(tf.square(thisLayer)) - tf.square(tf.reduce_sum(thisLayer))) *
                              (dim * tf.reduce_sum(tf.square(otherLayer)) - tf.square(tf.reduce_sum(otherLayer))))
        thisCorr = tf.truediv(numerator, denominator)
        thisLoss = tf.square(thisCorr - multiplier[currLayer, cell, i])
        loss = tf.add(loss, thisLoss)
        return loss, (numOfthisLoss + 1), thisCorr

    def DoNothing(loss, numOfthisLoss):
        thisCorr = 0.0
        return loss, numOfthisLoss, thisCorr

    def ListCorrSquare(thisLayer, otherLayer):
        dim = len(thisLayer)
        numerator = (dim *
                     tf.reduce_sum(tf.multiply(thisLayer, otherLayer))) - (tf.reduce_sum(thisLayer) *
                                                                           tf.reduce_sum(otherLayer))
        denominator = tf.sqrt((dim * tf.reduce_sum(tf.square(thisLayer)) - tf.square(tf.reduce_sum(thisLayer))) *
                              (dim * tf.reduce_sum(tf.square(otherLayer)) - tf.square(tf.reduce_sum(otherLayer))))
        thisCorr = tf.truediv(numerator, denominator)
        return tf.square(thisCorr)

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

    def compare(a, b):
        a = a.flatten()
        b = b.flatten()
        print("{} is {} long\n{} is {} long. \n angle is {}".format('a', np.linalg.norm(a), 'b', np.linalg.norm(b),
                                                                    angle_between(a, b)))
        return (np.linalg.norm(a) / np.linalg.norm(b))

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
            PreviousImgFile = run_root / 'mag{}phase{}'.format(previousMag, previousPhase) / 'ims.pkl'
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
            saveFolder = run_root / 'mag{}phase0'.format(mag)
            saveFolder.mkdir(parents=True, exist_ok=True)
            curr_iter_folder = saveFolder / 'currIter0'
            curr_iter_folder.mkdir(parents=True, exist_ok=True)

            smoke_shape = [1] + list(ims[0].shape)
            for numbered, thisOne in enumerate(ims):
                saveArr(thisOne, curr_iter_folder / 'im{}of{}.png'.format(numbered + 1, len(ims)), smoke_shape, post=post)
                saveArr(thisOne, saveFolder / 'im{}of{}.png'.format(numbered + 1, len(ims)), smoke_shape, post=post)

            saveVar(curr_iter_folder / 'ims.pkl', ims)
            saveVar(saveFolder / 'ims.pkl', ims)

            if len(ActualCorrelationContainer) == 0:
                ActualCorrelationContainer.append(np.zeros((len(incept_layers), len(ims), len(ims))))
            if len(CorrBetweenLayersContainer) == 0:
                CorrBetweenLayersContainer.append(np.nan)

            PathName = curr_iter_folder / 'Vars.pkl'
            saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer,
                    CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)

            PathName = run_root / 'Vars.pkl'
            saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer,
                    CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)
            seed_metadata['run_complete'] = True
            with open(run_root / 'seed_metadata.json', 'w') as f:
                json.dump(seed_metadata, f, indent=2)
            return mag, 0

        shape = imsize
        fshape = enhance_dims(shape, fov_init) if fov else shape
        scalefov = fov_grow if fov else magby

        shapes = shapegen(fshape, incept_layers)
        padshape = np.amax(shapes, axis=0)

        with graph.as_default(), graph.device(SYNTH_DEVICE), tf.Session() as sess:
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
                for currIter in range(iters):
                    if currIter in [250, 300, 350, 400, 450, 500, 550]:
                        if decay_flag:
                            AdamOpt1._lr = AdamOpt1._lr / 2
                            AdamOpt2._lr = AdamOpt2._lr / 2
                    print('{} images, size={}/{}, phase={}/{}, {}%, iter time={}s'.format(numImages, mag + 1,
                                                                                          mags, phase + 1, len(phases),
                                                                                          np.around(((
                                                                                                             currIter + 1) / iters) * 100,
                                                                                                    2),
                                                                                          int(time.time() - last)))
                    last = time.time()
                    cropped, xx, yy = gen_crops(ims, fovparams, fshape)
                    ActualCorrelation = np.zeros(CORRMAT.shape) if phase == 0 and currIter == 0 else ActualCorrelation
                    for pairn, (thiscrop, pairmate) in enumerate(zip(cropped, ims)):
                        if smoke and pairn > 0:
                            continue
                        if pairn == 0 and phase == 0:
                            for anum, thisone in enumerate(cropped):
                                thisact = sess.run(matchActivations, feed_dict={t_input: thisone})
                                padthisact = pad_activ(thisact, shapes)
                                PADDED = np.expand_dims(padthisact, 0) if anum == 0 else np.concatenate(
                                    (PADDED, np.expand_dims(padthisact, 0)))

                        assert PADDED.shape[0] == len(cropped)
                        if func == 'Grad3':
                            g1 = np.zeros_like(ims[0])
                            g2 = np.zeros_like(ims[0])
                        else:
                            g = np.zeros_like(ims[0])

                        if func == 'Grad3':
                            grad1, grad2, _LOSS1_, _LOSS2_, this, _thisWeight, _Loss, _currImageRDM = sess.run(
                                [dGrad, CorrGrad, thisWeight, corrSum, matchActivations, thisWeight, Loss,
                                 currImageRDM],
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
                        if func == 'Grad3':
                            print('_LOSS2_', _LOSS2_)
                            _LOSS2_Container.append(_LOSS2_)
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
                            g1 = g1 / g1.std() + 1e-8 if not lapnorm else lap_norm_func(g1)
                            g2 = g2 / g2.std() + 1e-8 if not lapnorm else lap_norm_func(g2)
                            g1 = AdamOpt1.update(g1, currIter, pairn)
                            g2 = AdamOpt2.update(g2, currIter, pairn)

                            compareg1g2List.append(compare(g1, g2))

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
                        base_folder = run_root / 'mag{}phase{}'.format(mag, phase)
                        base_folder.mkdir(parents=True, exist_ok=True)
                        saveFolder = base_folder / 'currIter{}'.format(currIter)
                        saveFolder.mkdir(parents=True, exist_ok=True)
                        for numbered, thisOne in enumerate(ims):
                            saveArr(thisOne, saveFolder / 'im{}of{}.png'.format(numbered + 1,
                                                                               len(ims)), shape, post=post)
                        saveVar(saveFolder / 'ims.pkl', ims)
                        PathName = saveFolder / 'Vars.pkl'
                        saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer,
                                CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)

                    if currIter == iters - 1:
                        saveFolder = run_root / 'mag{}phase{}'.format(mag, phase)
                        saveFolder.mkdir(parents=True, exist_ok=True)
                        for numbered, thisOne in enumerate(ims):
                            saveArr(thisOne, saveFolder / 'im{}of{}.png'.format(numbered + 1,
                                                                               len(ims)), shape, post=post)
                        saveVar(saveFolder / 'ims.pkl', ims)
                        PathName = saveFolder / 'Vars.pkl'
                        saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer,
                                CorrBetweenLayersContainer, compareg1g2List, _LOSS2_Container)

                    ActualCorrelationContainer.append(breaklink(ActualCorrelation))
                    CorrBetweenLayersContainer.append(calculateCorrBetweenLayers(ActualCorrelation))

    PathName = run_root / 'Vars.pkl'
    saveVar(PathName, _thisWeightContainer, _LossContainer, ActualCorrelationContainer, CorrBetweenLayersContainer,
            compareg1g2List, _LOSS2_Container)
    seed_metadata['run_complete'] = True
    with open(run_root / 'seed_metadata.json', 'w') as f:
        json.dump(seed_metadata, f, indent=2)
    total_end = time.time()
    print('in total %.02f' % ((total_end - total_start) / 60 / 60), 'h has passed')
    print('tag={}\n'.format(tag))
    previousMag = mag
    previousPhase = phase

    tf.reset_default_graph()
    return previousMag, previousPhase


mags = 1 if os.environ.get('SMOKE_TEST', '0') == '1' else 5
previousPhase = 0
continueFlag = False

for mag in tqdm(range(0, mags)):
    if continueFlag:
        previousMag = mag
    else:
        previousMag = mag - 1
    tfLoop(previousMag, previousPhase, continueFlag)
