import tensorflow as tf
import numpy as np
import os
import math
from functools import reduce, partial
from PIL import Image
import imageio
from IPython.display import clear_output
import time
import sys
import pdb
import pickle
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import math

print(f"CONDA_DEFAULT_ENV={os.environ['CONDA_DEFAULT_ENV']}")

current_dir = os.getcwd()

assert current_dir.endswith('localize_paper'), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, getSubs, setupPaths
from utils import custom_and

subs = getSubs()
[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

nsd_result_folder = f"{scratchResultFolder}/nsd_result_folder/"

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def breaklink(a):
    b = a.copy()
    return b


def loadPng2ims(
        pathName):
    files = glob(os.path.join(pathName, '*.png'))
    print(f"len(files)={len(files)}")
    files.sort()
    image = []
    for image_file in files:
        image.append(
            imageio.imread(image_file).astype(float)
        )
    image = (image,)
    return image


def loadimageList(
        imageList):
    from PIL import Image
    image = []
    for image_file in imageList:
        im = Image.open(image_file)
        im = np.array(im.resize((539, 539))).astype(float)
        image.append(im)
    image = (image,)
    return image


def plot_ims_matrix(image, size):
    fig, ax = plt.subplots(size[0], size[1], figsize=(40, 40))
    for currIms in range(size[0] * size[1]):
        nrow = size[0]
        try:
            ax[int(currIms % nrow), int(currIms / nrow)].imshow(image[currIms] / 255)
        except:
            ax[int(currIms % nrow), int(currIms / nrow)].imshow(np.ones(image[0].shape))
        ax[int(currIms % nrow), int(currIms / nrow)].set_xticks([])
        ax[int(currIms % nrow), int(currIms / nrow)].set_yticks([])
    fig.tight_layout()
    plt.show()


def getUpperTri_Flatten_benchmark_2(CORRMAT):
    CORRMAT_reshape = np.zeros((12, 120))
    for currLayer in range(12):
        _ = CORRMAT[currLayer, :, :]
        CORRMAT_reshape[currLayer] = _[np.triu_indices(16, k=1)]
    return CORRMAT_reshape


def png2RDM(pngPathOrIms,
            title):
    "6 RDM get the actual RDM from images."
    "6.1  png2RDM define function png2RDM"
    import tensorflow as tf
    from functools import reduce, partial
    from PIL import Image
    from IPython.display import clear_output
    import time
    import sys
    import pdb

    def CorrBetweenNeighboringLayers(CORRMAT_GradOrth):
        M = []
        for currLayer in range(CORRMAT_GradOrth.shape[0] - 1):
            t = np.corrcoef(CORRMAT_GradOrth[currLayer][np.triu_indices(numImages, k=1)],
                            CORRMAT_GradOrth[currLayer + 1][np.triu_indices(numImages, k=1)])[0, 1]
            M.append(t)
        return M

    def CorrBetweenALLLayers(CORRMAT_GradOrth):
        M = []
        numImages = CORRMAT_GradOrth.shape[1]
        for currLayerI in range(CORRMAT_GradOrth.shape[0]):
            for currLayerJ in range(currLayerI):
                t = np.corrcoef(CORRMAT_GradOrth[currLayerI][np.triu_indices(numImages, k=1)],
                                CORRMAT_GradOrth[currLayerJ][np.triu_indices(numImages, k=1)])[0, 1]
                M.append(t)
        return M

    def SumOfSquare_CorrBetweenALLLayers(CORRMAT_GradOrth):
        M = 0.0
        numImages = CORRMAT_GradOrth.shape[1]
        print('numImages', numImages)
        for currLayerI in range(CORRMAT_GradOrth.shape[0]):
            for currLayerJ in range(currLayerI):
                t = np.corrcoef(CORRMAT_GradOrth[currLayerI][np.triu_indices(numImages, k=1)],
                                CORRMAT_GradOrth[currLayerJ][np.triu_indices(numImages, k=1)])[0, 1]
                M = M + t ** 2
        return M

    def loadPng2ims(pathName):
        files = os.listdir(pathName)
        new_files = []
        for _ in files:
            if _.endswith('.png'):
                new_files.append(_)
        new_files.sort()
        image = []
        for image_file in new_files:
            image.append(
                imageio.imread(os.path.join(pathName, image_file)).astype(float)
            )
        image = (image,)
        return image

    "tf the following functions are in the tf loop"

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
            arr = np.clip((arr.reshape(shape[1:])), 0, 255).astype(np.uint8)
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

    "Deepdream THESE FUNCTIONS TAKEN DIRECTLY FROM GOOGLE'S DEEPDREAM NOTEBOOK"

    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    "TF Helper function that uses TF to resize an image"

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)
    k = np.float32([1, 4, 6, 4, 1])
    k = np.outer(k, k)
    k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)

    def lap_split(img):
        '''Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
            hi = img - lo2
        return lo, hi

    def lap_split_n(img, n):
        '''Build Laplacian pyramid with n splits'''
        levels = []
        for i in range(n):
            img, hi = lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    def lap_merge(levels):
        '''Merge Laplacian pyramid'''
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
        return img

    def normalize_std(img, eps=1e-10):
        '''Normalize image by making its standard deviation = 1.0'''
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img / tf.maximum(std, eps)

    def lap_normalize(img, scale_n=4):
        '''Perform the Laplacian pyramid normalization.'''
        img = tf.expand_dims(img, 0)
        tlevels = lap_split_n(img, scale_n)
        tlevels = list(map(normalize_std, tlevels))
        out = lap_merge(tlevels)
        return out[0, :, :, :]

    " save variables"

    def saveVar(PathName, *var):
        with open(PathName, 'wb') as f:
            pickle.dump(var, f)


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

    def calculateCorrBetweenLayers(ActualFingerprints):
        M = 0.0
        numImages = ActualFingerprints.shape[1]
        for currLayerI in range(ActualFingerprints.shape[0]):
            for currLayerJ in range(currLayerI):
                correlationBetweenFingerprint = np.corrcoef(
                    ActualFingerprints[currLayerI][np.triu_indices(numImages, k=1)],
                    ActualFingerprints[currLayerJ][np.triu_indices(numImages, k=1)]
                )[0, 1]
                M = M + correlationBetweenFingerprint ** 2
        return M

    "AdamOptimizer define AdamOptimizer and gradient measuring method"

    class kpAdamOptimizer():
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

    model_fn = f'/gpfs/milgram/project/turk-browne/users/kp578/localize/utils/tensorflow_inception_graph.pb'

    PNGimage = pngPathOrIms

    numImages = len(PNGimage[0])
    imsize = list((1,) + PNGimage[0][0].shape)
    pngVarPathName = '{}/pngVars.pkl'.format(pngPathOrIms)
    if not os.path.isfile(pngVarPathName):
        graph = tf.Graph()
        with graph.as_default(), graph.device('/cpu:0'), tf.Session() as sess:
            with tf.gfile.FastGFile(model_fn, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            t_input = tf.placeholder(np.float32, name='input')
            imagenet_mean = 117.0
            t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
            tf.import_graph_def(graph_def, {'input': t_preprocessed})

        layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]

        layers1 = [l.split('/')[1] for l in layers if l.split('/')[1].split('2')[0] == 'conv']
        layers2 = [l.split('/')[1] for l in layers if 'pool_reduce' in l]
        incept_layers = layers1 + layers2




        _thisWeightContainer = []
        _LossContainer = []







        bestchansfile = '{}/utils/bestchans{}.pkl'.format(main_dir, numImages)
        corrsvalsfile = '{}/utils/corrsvals{}.pkl'.format(main_dir, numImages)




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

        if version == 'orth':
            CORRMAT = np.load('{}/utils/{}corr.npy'.format(main_dir, numImages))
            CORRMAT = np.swapaxes(CORRMAT, 0, 2)
        elif version == 'GradOrth':
            CORRMAT = np.load('{}/utils/{}corr_GradOrth.npy'.format(main_dir, numImages))
        else:
            pass

        total_start = time.time()
        lapnorm = True
        lap_n = 4
        fov_init = 1.0

        chanPair = CHOOSE
        layerChans = [i for i in np.arange(chanPair.shape[1]) if not all(chanPair[0, i] == np.array([0] * numImages))]

        thisCorrContainer = []
        ActualFingerprintContainer = []
        CorrBetweenLayersContainer = []
        for pair in range(chanPair.shape[0]):
            theseChans = chanPair[pair]
            ims = PNGimage[0]
            shape = imsize
            fshape = shape

            shapes = shapegen(fshape, incept_layers)
            padshape = np.amax(shapes, axis=0)

            with graph.as_default(), graph.device('/cpu:0'), tf.Session() as sess:
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
                                                                lambda: truefn(multipliers, numOfthisOne, thisLayer,
                                                                               chans, currLayer, cell, i, ims),
                                                                lambda: falsefn(multipliers, numOfthisOne, thisLayer,
                                                                                chans, currLayer, cell, i, ims))

                            other = toMatch[i, currLayer]
                            otherLayer = other[:c[0], :c[1], :c[2], :c[3]]

                            loss, numOfthisLoss, thisCorr = tf.cond(~tf.math.equal(i, cell),
                                                                    lambda: TFCorr(thisLayer, otherLayer, multiplier,
                                                                                   currLayer, cell, i, loss,
                                                                                   numOfthisLoss),
                                                                    lambda: DoNothing(loss, numOfthisLoss))

                            currImageRDM.append(thisCorr)

                        _thisWeight = (tf.reduce_mean(tf.truediv(multipliers, numOfthisOne))) * -1
                        _weights.append(_thisWeight)

                corrSum = calImageRDMlayerCorrelation(currImageRDM)
                Loss = tf.reduce_sum(loss)
                thisWeight = reduce(tf.add, _weights)

                last = time.time()
                fovparams = gen_fovea(shape, fshape)

                last = time.time()
                cropped, xx, yy = gen_crops(ims, fovparams, fshape)
                ActualFingerprint = np.zeros(
                    CORRMAT.shape)
                for pairn, (thiscrop, pairmate) in enumerate(zip(cropped, ims)):
                    if pairn == 0:
                        for anum, thisone in enumerate(cropped):
                            thisact = sess.run(matchActivations, feed_dict={t_input: thisone})
                            padthisact = pad_activ(thisact, shapes)
                            PADDED = np.expand_dims(padthisact, 0) if anum == 0 else np.concatenate(
                                (PADDED, np.expand_dims(padthisact, 0)))

                    assert PADDED.shape[0] == len(cropped)

                    this, _thisWeight, _Loss, _currImageRDM = sess.run(
                        [matchActivations, thisWeight, Loss, currImageRDM],
                        feed_dict={t_input: thiscrop, cutAt: shapes, toMatch: PADDED, cell: pairn,
                                   chans: theseChans, multiplier: CORRMAT})

                    _thisWeightContainer.append(_thisWeight)
                    _LossContainer.append(_Loss)

                    for currLayer in range(12):
                        for currIm in range(numImages):
                            ActualFingerprint[currLayer, pairn, currIm] = _currImageRDM[currLayer * numImages + currIm]
                    padded = pad_activ(this, shapes)
                    PADDED[pairn] = padded
                ActualFingerprintContainer.append(breaklink(ActualFingerprint))
                CorrBetweenLayersContainer.append(calculateCorrBetweenLayers(ActualFingerprint))


        total_end = time.time()
        print('in total %.02f' % ((total_end - total_start) / 60), 'min has passed')
    else:
        with open(pngVarPathName, 'br') as f:
            _thisWeightContainer, _LossContainer, ActualFingerprintContainer, CorrBetweenLayersContainer = pickle.load(
                f)
    ID = 0
    print('{}-CorrBetweenLayers={}'.format(title, calculateCorrBetweenLayers(ActualFingerprintContainer[ID])))
    return ActualFingerprintContainer, PNGimage


jobarrayID = int(sys.argv[1])
print(f"jobarrayID={jobarrayID}")
whichSet_1 = int(float(sys.argv[2]))
whichSet = whichSet_1 + 1
print(f"whichSet_1={whichSet_1}")
jobarrayDict = load_obj(f"Sfig3/Sfig3b/nsd_dataPrepare/modelData/nsd_dataPrepare_model_whichSet_1_{whichSet_1}")
jobarrayID = int(float(sys.argv[1]))
[setOf16Images] = jobarrayDict[jobarrayID]
print(f"setOf16Images={setOf16Images}")

if not os.path.exists(f"{nsd_result_folder}/modelData/"
                      f"data/ActualFingerprints_whichSet_{whichSet}.pkl"):
    imgList = []
    for imgID in setOf16Images:
        imagePath = glob(f"{nsd_result_folder}/modelData/stimuli/{imgID}.png")[0]
        imgList.append(imagePath)
    print(f"imgList={imgList}")
    Ims = loadimageList(imgList)

    def get_meanOffDiagonal(M):
        dimension = M.shape[0]
        print(f"matrix shape={M.shape}")
        offDiagonalMask = np.ones((1, dimension - 1))[0]
        offDiagonalMask = np.diag(offDiagonalMask, 1)
        OffDiagonal_mean = np.sum(np.multiply(M, offDiagonalMask)) / (dimension - 1)
        OffDiagonal_max = np.max(np.multiply(M, offDiagonalMask))
        print(f"mean Off Diagonal={OffDiagonal_mean}")
        print(f"max Off Diagonal={OffDiagonal_max}")
        return float(OffDiagonal_mean), float(OffDiagonal_max)

    def get_upperTriMean(M):
        dimension = M.shape[0]
        upperTri_Mean = np.mean(M[np.triu_indices(dimension, k=1)])
        upperTri_Max = np.max(M[np.triu_indices(dimension, k=1)])
        print(f"mean upper triangle={upperTri_Mean}")
        print(f"max upper triangle={upperTri_Max}")
        return float(upperTri_Mean), float(upperTri_Max)


    title = ''
    for currImg in range(len(Ims[0])):
        print(Ims[0][currImg].shape)
        assert Ims[0][currImg].shape == (539, 539, 3)
    ActualFingerprints, PNGimage = png2RDM(Ims,
                                           title)
    correlationOfFingerprints = np.corrcoef(getUpperTri_Flatten_benchmark_2(ActualFingerprints[0]))
    OffDiagonal_mean, OffDiagonal_max = get_meanOffDiagonal(correlationOfFingerprints)
    upperTri_Mean, upperTri_Max = get_upperTriMean(correlationOfFingerprints)
    meanMax_dict = {}
    meanMax_dict['OffDiagonal_mean'] = OffDiagonal_mean
    meanMax_dict['OffDiagonal_max'] = OffDiagonal_max
    meanMax_dict['upperTri_Mean'] = upperTri_Mean
    meanMax_dict['upperTri_Max'] = upperTri_Max

    meanMax_dict['ActualFingerprints'] = ActualFingerprints
    meanMax_dict['correlationOfFingerprints'] = correlationOfFingerprints
    meanMax_dict['setOf16Images'] = setOf16Images
    save_obj([meanMax_dict],
             f"{nsd_result_folder}/modelData/"
             f"data/ActualFingerprints_whichSet_{whichSet}")

print("done")