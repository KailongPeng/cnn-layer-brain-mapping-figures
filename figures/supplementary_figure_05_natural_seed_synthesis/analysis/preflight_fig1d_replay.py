#!/usr/bin/env python3
"""Read-only gates for the historical Figure 1D synthesis replay."""

import argparse
import csv
import hashlib
import importlib
import os
import pickle
import sys


ASSET_NAMES = (
    "utils/16corr_GradOrth.npy",
    "utils/bestchans16.pkl",
    "utils/corrsvals16.pkl",
)
IMAGE_SEED_MODES = ("nsd", "ccby4")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        default=os.environ.get(
            "FIG1D_ASSET_ROOT",
            "/gpfs/milgram/project/turk-browne/projects/localize",
        ),
    )
    parser.add_argument(
        "--model-graph",
        default=os.environ.get(
            "FIG1D_MODEL_GRAPH",
            "/gpfs/milgram/project/turk-browne/users/kp578/localize/utils/"
            "tensorflow_inception_graph.pb",
        ),
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get(
            "FIG1D_OUTPUT_ROOT",
            "/gpfs/milgram/scratch60/turk-browne/kp578/localize/fig1d_replay",
        ),
    )
    parser.add_argument("--tag", default=os.environ.get("SYNTH_TAG", ""))
    parser.add_argument(
        "--seed-mode", default=os.environ.get("FIG1D_SEED_MODE", "noise")
    )
    parser.add_argument(
        "--seed-manifest",
        default=os.environ.get(
            "FIG1D_SEED_MANIFEST",
            "ImageSynthesis/seed_manifests/r2q1_structured_seed_16.txt",
        ),
    )
    parser.add_argument(
        "--seed-license-record",
        default=os.environ.get(
            "FIG1D_SEED_LICENSE_RECORD",
            "NSD_similarity/FINAL_fig1_sfig17_fig2_nsd_source_license_record.csv",
        ),
    )
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args()

    failures = []
    print("Figure 1D synthesis replay preflight (read-only)")
    print("Asset root: ", os.path.abspath(args.asset_root))
    print("Model graph:", os.path.abspath(args.model_graph))
    print("Output root:", os.path.abspath(args.output_root))
    print("Run tag:    ", args.tag or "MISSING")
    print("Seed mode:  ", args.seed_mode)

    if args.seed_mode not in ("noise",) + IMAGE_SEED_MODES:
        failures.append("seed mode must be noise, nsd, or ccby4")
    expected_prefix = "fig1d_replay_{}_".format(args.seed_mode)
    if not args.tag.startswith(expected_prefix):
        failures.append("tag must start with {}".format(expected_prefix))
    if os.path.basename(args.tag) != args.tag:
        failures.append("tag must be a directory name, not a path")

    print("\nHistorical assets")
    if not os.path.isfile(args.model_graph):
        print("MISSING ", args.model_graph)
        failures.append("missing model graph: {}".format(args.model_graph))
    else:
        size = os.path.getsize(args.model_graph)
        if size <= 0:
            failures.append("empty model graph: {}".format(args.model_graph))
        print(
            "OK {:>12,} bytes  {}  {}".format(
                size, sha256_file(args.model_graph), args.model_graph
            )
        )
    for relative_path in ASSET_NAMES:
        path = os.path.join(args.asset_root, relative_path)
        if not os.path.isfile(path):
            print("MISSING ", path)
            failures.append("missing asset: {}".format(path))
            continue
        size = os.path.getsize(path)
        if size <= 0:
            failures.append("empty asset: {}".format(path))
        print("OK {:>12,} bytes  {}  {}".format(size, sha256_file(path), path))

    target = os.path.join(args.output_root, args.tag)
    print("\nNo-overwrite target")
    if os.path.lexists(target):
        print("EXISTS  ", target)
        failures.append("target already exists: {}".format(target))
    else:
        print("AVAILABLE", target)

    parent = args.output_root
    while not os.path.exists(parent):
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            break
        parent = next_parent
    if not os.path.isdir(parent) or not os.access(parent, os.W_OK | os.X_OK):
        failures.append("nearest output parent is not writable: {}".format(parent))

    seed_paths = []
    if args.seed_mode in IMAGE_SEED_MODES:
        print("\n{} seed images".format(args.seed_mode.upper()))
        manifest_path = os.path.abspath(args.seed_manifest)
        print("Manifest:   ", manifest_path)
        try:
            with open(manifest_path) as handle:
                seed_paths = [line.strip() for line in handle if line.strip()]
        except Exception as exc:
            failures.append(
                "cannot read seed manifest: {}: {}".format(type(exc).__name__, exc)
            )
        if len(seed_paths) != 16:
            failures.append(
                "seed manifest has {} paths, expected 16".format(len(seed_paths))
            )
        seed_paths = [os.path.abspath(path) for path in seed_paths]
        if len(set(seed_paths)) != len(seed_paths):
            failures.append("seed manifest contains duplicate paths")

        license_path = os.path.abspath(args.seed_license_record)
        print("Licenses:   ", license_path)
        try:
            with open(license_path, newline="") as handle:
                all_license_rows = list(csv.DictReader(handle))
            if args.seed_mode == "nsd":
                license_rows = [
                    row for row in all_license_rows
                    if row.get("figure_use") == "Sfig17_panel_A_seed"
                ]
                license_rows.sort(key=lambda row: int(row["image_order"]))
                if len(license_rows) != 16:
                    failures.append(
                        "license record has {} Sfig17 seed rows, expected 16".format(
                            len(license_rows)
                        )
                    )
                expected_names = [
                    "{}.png".format(row["local_filename_number"])
                    for row in license_rows
                ]
                for row in license_rows:
                    image_id = row.get("local_filename_number", "UNKNOWN")
                    if row.get("plos_cc_by_4_compatibility") != "YES":
                        failures.append(
                            "seed {} is not marked PLOS-compatible".format(image_id)
                        )
                    if row.get("current_flickr_license_name") != "CC BY 2.0":
                        failures.append(
                            "seed {} lacks a current CC BY 2.0 verification".format(
                                image_id
                            )
                        )
                    if row.get("has_person_coco_annotation", "").lower() != "no":
                        failures.append(
                            "seed {} has or may have a COCO person annotation".format(
                                image_id
                            )
                        )
                license_summary = (
                    "16/16 current CC BY 2.0, PLOS-compatible, no person"
                )
            else:
                license_rows = [
                    row for row in all_license_rows
                    if row.get("figure_use") == "Proposed Fig S5A seed"
                ]
                license_rows.sort(key=lambda row: int(row["image_order"]))
                if len(license_rows) != 16:
                    failures.append(
                        "license record has {} CC BY 4.0 seed rows, expected 16".format(
                            len(license_rows)
                        )
                    )
                expected_names = [
                    "{:02d}.jpg".format(int(row["image_order"]))
                    for row in license_rows
                ]
                for row in license_rows:
                    image_id = row.get("image_order", "UNKNOWN")
                    if row.get("license") != "CC BY 4.0":
                        failures.append(
                            "seed {} is not licensed CC BY 4.0".format(image_id)
                        )
                    if row.get("license_url", "").rstrip("/") != (
                            "https://creativecommons.org/licenses/by/4.0"):
                        failures.append(
                            "seed {} has the wrong license URL".format(image_id)
                        )
                    if not row.get("source_page_url", "").startswith(
                            "https://commons.wikimedia.org/wiki/File:"):
                        failures.append(
                            "seed {} lacks a Wikimedia Commons file page".format(
                                image_id
                            )
                        )
                    if row.get("nsd_image_number") != "N/A":
                        failures.append(
                            "seed {} is incorrectly labelled as an NSD image".format(
                                image_id
                            )
                        )
                checksum_path = os.path.join(
                    os.path.dirname(seed_paths[0]) if seed_paths else "",
                    "SHA256SUMS.txt",
                )
                try:
                    with open(checksum_path) as handle:
                        expected_hashes = {
                            line.split()[1]: line.split()[0]
                            for line in handle if line.strip()
                        }
                    if sorted(expected_hashes) != sorted(expected_names):
                        failures.append(
                            "CC BY 4.0 seed checksum inventory is not exactly 01.jpg-16.jpg"
                        )
                    for path in seed_paths:
                        name = os.path.basename(path)
                        if (os.path.isfile(path) and
                                expected_hashes.get(name) != sha256_file(path)):
                            failures.append(
                                "seed image fingerprint mismatch: {}".format(path)
                            )
                except Exception as exc:
                    failures.append(
                        "cannot validate CC BY 4.0 seed fingerprints: {}: {}".format(
                            type(exc).__name__, exc
                        )
                    )
                license_summary = "16/16 Wikimedia Commons CC BY 4.0"

            observed_names = [os.path.basename(path) for path in seed_paths]
            if observed_names != expected_names:
                failures.append(
                    "seed manifest order/names do not match the license record"
                )
            if not failures:
                print("License gate: {}".format(license_summary))
        except Exception as exc:
            failures.append(
                "cannot validate seed license record: {}: {}".format(
                    type(exc).__name__, exc
                )
            )
        for index, path in enumerate(seed_paths, 1):
            if not os.path.isfile(path) or not os.access(path, os.R_OK):
                print("MISSING/UNREADABLE {:02d} {}".format(index, path))
                failures.append("missing or unreadable seed image: {}".format(path))
            else:
                print(
                    "OK {:02d} {:>9,} bytes  {}  {}".format(
                        index, os.path.getsize(path), sha256_file(path), path
                    )
                )

    if args.check_runtime:
        print("\nPython runtime")
        for module_name in ("tensorflow", "numpy", "PIL", "imageio", "IPython"):
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, "__version__", "version unavailable")
                print("OK {:<12} {}".format(module_name, version))
            except Exception as exc:
                failures.append(
                    "cannot import {}: {}: {}".format(
                        module_name, type(exc).__name__, exc
                    )
                )
        try:
            tensorflow = importlib.import_module("tensorflow")
            for attribute in ("Session", "GraphDef", "placeholder", "reset_default_graph"):
                if not hasattr(tensorflow, attribute):
                    failures.append("TensorFlow runtime lacks tf.{}".format(attribute))
        except Exception:
            pass

        print("\nAsset contents")
        try:
            numpy = importlib.import_module("numpy")
            target_path = os.path.join(args.asset_root, "utils/16corr_GradOrth.npy")
            target = numpy.load(target_path)
            print("Target matrix shape:", target.shape)
            if target.shape != (12, 16, 16):
                failures.append(
                    "target matrix shape is {}, expected (12, 16, 16)".format(
                        target.shape
                    )
                )
            if not numpy.isfinite(target).all():
                failures.append("target matrix contains non-finite values")
        except Exception as exc:
            failures.append(
                "cannot validate target matrix: {}: {}".format(
                    type(exc).__name__, exc
                )
            )

        for filename in ("bestchans16.pkl", "corrsvals16.pkl"):
            path = os.path.join(args.asset_root, "utils", filename)
            try:
                with open(path, "rb") as handle:
                    value = pickle.load(handle)
                print("OK unpickled {:<20} type={}".format(filename, type(value).__name__))
                if not isinstance(value, dict):
                    failures.append("{} does not contain a dictionary".format(path))
            except Exception as exc:
                failures.append(
                    "cannot unpickle {}: {}: {}".format(
                        path, type(exc).__name__, exc
                    )
                )
        if args.seed_mode in IMAGE_SEED_MODES:
            try:
                image_module = importlib.import_module("PIL.Image")
                for path in seed_paths:
                    with image_module.open(path) as image:
                        image.verify()
                    with image_module.open(path) as image:
                        converted = image.convert("RGB").resize(
                            (196, 196), image_module.BILINEAR
                        )
                        if converted.size != (196, 196):
                            failures.append("failed to resize seed image: {}".format(path))
                print("Decoded and resized 16/16 seed images to 196x196 RGB")
            except Exception as exc:
                failures.append(
                    "cannot decode seed images: {}: {}".format(
                        type(exc).__name__, exc
                    )
                )
        print("Python:      ", sys.executable)

    if failures:
        print("\nPREFLIGHT FAILED")
        for failure in failures:
            print("-", failure)
        return 1

    print("\nFIG1D REPLAY PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
