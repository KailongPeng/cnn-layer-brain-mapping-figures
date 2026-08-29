#!/usr/bin/env python3
"""Recompute Sfig5B/C exactly from a completed image-seeded synthesis run."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


N_IMAGES = 16
N_LAYERS = 12
FINAL_MAG = 3
FINAL_ITERATION = 599
PANEL_IMAGE_SIZE = 320
DEFAULT_GRAPH = Path(
    "/gpfs/milgram/project/turk-browne/users/kp578/localize/"
    "utils/tensorflow_inception_graph.pb"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "ImageSynthesis" / "seed_manifests" /
    "r2q1_structured_seed_16.txt"
)
DEFAULT_CACHE = (
    PROJECT_ROOT / "rebuttal" / "analyses" /
    "structured_seed_data" / "nsd_seed_final_synthesis_exact.npz"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError("cannot read {}: {}".format(path, exc))


def manifest_names(path):
    if not path.is_file():
        raise FileNotFoundError("seed manifest is missing: {}".format(path))
    names = [Path(line.strip()).name for line in path.read_text().splitlines()
             if line.strip()]
    if len(names) != N_IMAGES or len(set(names)) != N_IMAGES:
        raise RuntimeError("seed manifest must contain 16 unique entries: {}".format(path))
    return names


def manifest_paths(path):
    entries = [Path(line.strip()) for line in path.read_text().splitlines()
               if line.strip()]
    if len(entries) != N_IMAGES:
        raise RuntimeError("seed manifest must contain 16 entries: {}".format(path))
    return entries


def validate_completed_run(run_dir, current_manifest, expected_seed_mode,
                           current_license_record=None):
    metadata = read_json(run_dir / "run_metadata.json")
    progress = read_json(run_dir / "progress.json")
    failures = []
    expected = {
        "status": "complete",
        "run_complete": True,
        "seed_mode": expected_seed_mode,
        "completed_mags": [0, 1, 2, 3],
        "diagnostic_run": False,
        "iter_limit": 0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append("metadata {}={!r}, expected {!r}".format(
                key, metadata.get(key), value))
    if progress.get("status") != "complete":
        failures.append("progress status={!r}, expected 'complete'".format(
            progress.get("status")))
    recorded_seed_hashes = metadata.get("seed_image_sha256", [])
    if len(recorded_seed_hashes) != N_IMAGES:
        failures.append("metadata does not contain 16 seed-image hashes")
    else:
        current_seed_paths = manifest_paths(current_manifest)
        missing_seeds = [str(path) for path in current_seed_paths
                         if not path.is_file()]
        if missing_seeds:
            failures.append("current seed images are missing: {}".format(
                ", ".join(missing_seeds)))
        elif [sha256(path) for path in current_seed_paths] != recorded_seed_hashes:
            failures.append("run seed hashes do not match the current Panel A images")

    used_manifest = run_dir / "seed_manifest_used.txt"
    if not used_manifest.is_file():
        failures.append("missing seed_manifest_used.txt")
    elif manifest_names(used_manifest) != manifest_names(current_manifest):
        failures.append("run seed manifest does not match the current manifest order")
    if expected_seed_mode == "ccby4":
        recorded_license_hash = metadata.get("seed_license_record_sha256", "")
        if current_license_record is None or not current_license_record.is_file():
            failures.append("current CC BY 4.0 license record is missing")
        elif sha256(current_license_record) != recorded_license_hash:
            failures.append(
                "run license-record hash does not match the current CC BY 4.0 record")
    if failures:
        raise RuntimeError("completed-run gate failed:\n- " + "\n- ".join(failures))
    return metadata


def final_image_paths(run_dir):
    stage = run_dir / "mag{}phase0".format(FINAL_MAG)
    checkpoint = stage / "currIter{}".format(FINAL_ITERATION)
    stage_paths = [stage / "im{}of16.png".format(i)
                   for i in range(1, N_IMAGES + 1)]
    checkpoint_paths = [checkpoint / path.name for path in stage_paths]
    for stage_path, checkpoint_path in zip(stage_paths, checkpoint_paths):
        for path in (stage_path, checkpoint_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError("final image is missing: {}".format(path))
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (539, 539):
                    raise RuntimeError("unexpected final image {}: {} {}".format(
                        path, image.mode, image.size))
        if sha256(stage_path) != sha256(checkpoint_path):
            raise RuntimeError(
                "stage/checkpoint image mismatch: {} and {}".format(
                    stage_path, checkpoint_path))
    return stage_paths


def discover_layers(graph):
    operations = [op.name for op in graph.get_operations()
                  if op.type == "Conv2D" and "import/" in op.name]
    early = [name.split("/")[1] for name in operations
             if name.split("/")[1].split("2")[0] == "conv"]
    pool_reduce = [name.split("/")[1] for name in operations
                   if "pool_reduce" in name]
    layers = early + pool_reduce
    if len(layers) != N_LAYERS or len(set(layers)) != N_LAYERS:
        raise RuntimeError("model graph yielded unexpected layers: {}".format(layers))
    return layers


def pearson_rows(rows):
    matrix = np.asarray(rows, dtype=np.float64)
    matrix -= matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise RuntimeError("CNN activation vector is constant or non-finite")
    corr = np.dot(matrix, matrix.T) / np.outer(norms, norms)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def forward_final_images(model_graph, image_paths):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    graph = tf.Graph()
    with graph.as_default(), graph.device("/cpu:0"):
        with tf.gfile.FastGFile(str(model_graph), "rb") as handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(handle.read())
        tensor_input = tf.placeholder(np.float32, name="input")
        preprocessed = tf.expand_dims(tensor_input - 117.0, 0)
        tf.import_graph_def(graph_def, {"input": preprocessed})

    layers = discover_layers(graph)
    tensors = [graph.get_tensor_by_name("import/{}:0".format(layer))
               for layer in layers]
    activations = [[] for _ in layers]
    with tf.Session(graph=graph) as session:
        for index, path in enumerate(image_paths, 1):
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            outputs = session.run(tensors, feed_dict={tensor_input: pixels})
            for layer_index, output in enumerate(outputs):
                activations[layer_index].append(np.asarray(output).ravel())
            print("CNN forward {}/{}: {}".format(index, N_IMAGES, path.name),
                  flush=True)

    rsms = np.empty((N_LAYERS, N_IMAGES, N_IMAGES), dtype=np.float64)
    for layer_index, layer_rows in enumerate(activations):
        rsms[layer_index] = pearson_rows(layer_rows)
        activations[layer_index] = None
    upper = np.triu_indices(N_IMAGES, k=1)
    fingerprints = rsms[:, upper[0], upper[1]]
    corr = np.corrcoef(fingerprints)
    return layers, rsms, fingerprints, corr


def panel_images(image_paths):
    result = []
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for path in image_paths:
        with Image.open(path) as image:
            result.append(np.asarray(
                image.convert("RGB").resize(
                    (PANEL_IMAGE_SIZE, PANEL_IMAGE_SIZE), resampling),
                dtype=np.uint8))
    return np.stack(result)


def validate_products(rsms, fingerprints, corr, synthesized_images):
    expected_shapes = {
        "final_rsms": (N_LAYERS, N_IMAGES, N_IMAGES),
        "fingerprints": (N_LAYERS, 120),
        "correlation_fingerprints": (N_LAYERS, N_LAYERS),
        "synthesized_images": (N_IMAGES, PANEL_IMAGE_SIZE, PANEL_IMAGE_SIZE, 3),
    }
    products = {
        "final_rsms": rsms,
        "fingerprints": fingerprints,
        "correlation_fingerprints": corr,
        "synthesized_images": synthesized_images,
    }
    for name, array in products.items():
        if array.shape != expected_shapes[name]:
            raise RuntimeError("{} shape={}, expected {}".format(
                name, array.shape, expected_shapes[name]))
        if not np.all(np.isfinite(array)):
            raise RuntimeError("{} contains non-finite values".format(name))
    if not np.allclose(rsms, rsms.swapaxes(1, 2), atol=1e-10, rtol=0):
        raise RuntimeError("final RSMs are not symmetric")
    upper = np.triu_indices(N_IMAGES, k=1)
    rebuilt_fingerprints = rsms[:, upper[0], upper[1]]
    rebuilt_corr = np.corrcoef(rebuilt_fingerprints)
    if not np.allclose(fingerprints, rebuilt_fingerprints, atol=1e-12, rtol=0):
        raise RuntimeError("fingerprints cannot be rebuilt from final RSMs")
    if not np.allclose(corr, rebuilt_corr, atol=1e-12, rtol=0):
        raise RuntimeError("layer correlations cannot be rebuilt from fingerprints")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("layer-correlation diagonal is not one")


def metrics(corr):
    upper = corr[np.triu_indices(N_LAYERS, k=1)]
    adjacent = np.diag(corr, k=1)
    return {
        "mean_offdiagonal_r": float(np.mean(upper)),
        "mean_absolute_offdiagonal_r": float(np.mean(np.abs(upper))),
        "rms_offdiagonal_r": float(np.sqrt(np.mean(upper ** 2))),
        "max_absolute_offdiagonal_r": float(np.max(np.abs(upper))),
        "mean_adjacent_r": float(np.mean(adjacent)),
        "mean_absolute_adjacent_r": float(np.mean(np.abs(adjacent))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed-license-record", type=Path)
    parser.add_argument(
        "--expected-seed-mode", choices=("nsd", "ccby4"), default="nsd"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("no-overwrite gate: {}".format(output))
    if not args.model_graph.is_file() or args.model_graph.stat().st_size < 1_000_000:
        raise FileNotFoundError("model graph is missing or too small: {}".format(
            args.model_graph))

    license_record = (args.seed_license_record.resolve()
                      if args.seed_license_record else None)
    metadata = validate_completed_run(
        run_dir, args.seed_manifest.resolve(), args.expected_seed_mode,
        license_record
    )
    image_paths = final_image_paths(run_dir)
    print("COMPLETED {} RUN GATE PASSED".format(
        args.expected_seed_mode.upper()), flush=True)
    print("Exact source: mag3phase0 final PNGs (identical to currIter599)",
          flush=True)

    layers, rsms, fingerprints, corr = forward_final_images(
        args.model_graph.resolve(), image_paths)
    synthesized_images = panel_images(image_paths)
    validate_products(rsms, fingerprints, corr, synthesized_images)
    values = metrics(corr)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sfig5_final_",
                                     dir=str(output.parent)) as temporary:
        temporary_output = Path(temporary) / output.name
        np.savez_compressed(
            temporary_output,
            synthesized_images=synthesized_images,
            final_rsms=rsms,
            fingerprints=fingerprints,
            correlation_fingerprints=corr,
            n_layers=np.array(N_LAYERS),
            n_images=np.array(N_IMAGES),
            layer_names=np.asarray(layers),
            source_image_names=np.asarray([path.name for path in image_paths]),
            source_image_sha256=np.asarray([sha256(path) for path in image_paths]),
            source_run_tag=np.asarray(run_dir.name),
            source_run_commit=np.asarray(metadata.get("git_commit", "unavailable")),
            source_seed_mode=np.asarray(args.expected_seed_mode),
            source_seed_license_record_sha256=np.asarray(
                metadata.get("seed_license_record_sha256", "unavailable")
            ),
            source_checkpoint=np.asarray("mag3phase0/currIter599"),
            analysis_method=np.asarray(
                "exact CNN forward from final post-update PNGs"),
        )
        if temporary_output.stat().st_size > 12 * 1024 * 1024:
            raise RuntimeError("final cache unexpectedly exceeds 12 MiB")
        try:
            os.link(str(temporary_output), str(output))
        except FileExistsError:
            raise FileExistsError("no-overwrite gate: {}".format(output))

    print("SFIG5 FINAL EXACT ANALYSIS PASSED")
    for key, value in values.items():
        print("{}={:.6f}".format(key, value))
    print("Saved: {} ({:,} bytes)".format(output, output.stat().st_size))


if __name__ == "__main__":
    main()
