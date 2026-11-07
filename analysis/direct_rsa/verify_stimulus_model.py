"""Independently compare ordered experimental PNG activations with saved RSMs.

Run on a compute node in the existing TensorFlow 1 environment. This reads
inputs only and never replaces the historical features.npy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.project)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(str(output))
    paths = [root / 'images/final' / ('im%02dof16.png' % i) for i in range(1, 17)]
    model_path = root / 'utils/tensorflow_inception_graph.pb'
    features_path = root / 'data/inception/features.npy'
    images = [np.asarray(Image.open(str(path)).convert('RGB'), dtype=np.float32) for path in paths]
    graph = tf.Graph()
    with graph.as_default():
        definition = tf.GraphDef()
        definition.ParseFromString(model_path.read_bytes())
        image_input = tf.placeholder(tf.float32, shape=[None, None, 3])
        tf.import_graph_def(definition, {'input': tf.expand_dims(image_input - 117.0, 0)})
    convs = [op.name.split('/')[1] for op in graph.get_operations()
             if op.type == 'Conv2D' and op.name.startswith('import/')]
    layers = [name for name in convs if name.split('2')[0] == 'conv']
    layers += [name for name in convs if 'pool_reduce' in name]
    if len(layers) != 12:
        raise ValueError('Expected 12 historical layers, got %r' % layers)
    tensors = [graph.get_tensor_by_name('import/%s/conv:0' % name) for name in layers]
    vectors = [[] for _ in layers]
    config = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=1)
    with tf.Session(graph=graph, config=config) as session:
        for i, image in enumerate(images):
            activations = session.run(tensors, feed_dict={image_input: image})
            for collection, activation in zip(vectors, activations):
                collection.append(activation.reshape(-1).astype(np.float64))
            print('image %d/16 complete' % (i + 1), flush=True)
    actual = np.asarray([np.corrcoef(layer) for layer in vectors])
    expected = np.load(str(features_path), allow_pickle=False)
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError('Invalid model RSM shape or values')
    tri = np.triu_indices(16, 1)
    a = actual[:, tri[0], tri[1]]
    b = expected[:, tri[0], tri[1]]
    errors = np.max(np.abs(a-b), axis=1)
    # The historical graph computed Pearson in float32; the independent
    # activation comparison uses float64, so exact byte equality is not expected.
    passed = bool(np.allclose(a, b, atol=1e-4, rtol=1e-4))
    report = {'passed': passed, 'tolerance_atol': 1e-4, 'tolerance_rtol': 1e-4,
              'tensorflow': tf.__version__, 'layers': layers,
              'image_shapes': [list(x.shape) for x in images],
              'model_sha256': digest(model_path), 'features_sha256': digest(features_path),
              'images': [{'index': i, 'condition': chr(65+i), 'filename': path.name,
                          'sha256': digest(path)} for i, path in enumerate(paths)],
              'max_absolute_error_per_layer': errors.tolist(),
              'comparison': 'strict upper triangle; diagonal excluded',
              'method': 'RGB PNG; subtract 117; full pre-ReLU convolution activation Pearson'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError('Saved model RSM does not match independent ordered PNG calculation')


if __name__ == '__main__':
    main()
