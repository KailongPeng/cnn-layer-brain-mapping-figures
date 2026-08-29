#!/usr/bin/env python3
"""Validate Figure 1D replay products without loading TensorFlow."""

import argparse
import json
import os
import sys


def is_png(path):
    try:
        with open(path, "rb") as handle:
            return handle.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-status", default="diagnostic_complete")
    parser.add_argument("--expected-mags", type=int, nargs="+", default=[0])
    parser.add_argument("--expect-run-complete", action="store_true")
    parser.add_argument(
        "--expected-seed-mode", choices=("noise", "nsd", "ccby4")
    )
    args = parser.parse_args()

    run_dir = os.path.join(args.output_root, args.tag)
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    progress_path = os.path.join(run_dir, "progress.json")
    failures = []

    try:
        with open(metadata_path) as handle:
            metadata = json.load(handle)
    except Exception as exc:
        print("VALIDATION FAILED: cannot read {}: {}".format(metadata_path, exc))
        return 1

    try:
        with open(progress_path) as handle:
            progress = json.load(handle)
    except Exception as exc:
        print("VALIDATION FAILED: cannot read {}: {}".format(progress_path, exc))
        return 1

    if metadata.get("status") != args.expected_status:
        failures.append(
            "status is {!r}, expected {!r}".format(
                metadata.get("status"), args.expected_status
            )
        )
    if progress.get("status") != args.expected_status:
        failures.append(
            "progress status is {!r}, expected {!r}".format(
                progress.get("status"), args.expected_status
            )
        )
    expected_run_complete = args.expect_run_complete
    if metadata.get("run_complete") is not expected_run_complete:
        failures.append(
            "run_complete is {!r}, expected {!r}".format(
                metadata.get("run_complete"), expected_run_complete
            )
        )
    if metadata.get("completed_mags") != args.expected_mags:
        failures.append(
            "completed_mags is {!r}, expected {!r}".format(
                metadata.get("completed_mags"), args.expected_mags
            )
        )
    if (args.expected_seed_mode is not None and
            metadata.get("seed_mode") != args.expected_seed_mode):
        failures.append(
            "seed_mode is {!r}, expected {!r}".format(
                metadata.get("seed_mode"), args.expected_seed_mode
            )
        )
    if args.expected_seed_mode in ("nsd", "ccby4"):
        manifest_used = os.path.join(run_dir, "seed_manifest_used.txt")
        if not os.path.isfile(manifest_used) or os.path.getsize(manifest_used) == 0:
            failures.append("missing or empty seed manifest copy: {}".format(
                manifest_used
            ))
        if len(metadata.get("seed_image_sha256", [])) != 16:
            failures.append("metadata does not contain 16 seed-image hashes")
        if len(metadata.get("seed_license_record_sha256", "")) != 64:
            failures.append("metadata lacks a seed-license-record fingerprint")

    png_paths = []
    for mag in args.expected_mags:
        stage_dir = os.path.join(run_dir, "mag{}phase0".format(mag))
        for filename in ("ims.pkl", "Vars.pkl"):
            path = os.path.join(stage_dir, filename)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                failures.append("missing or empty stage product: {}".format(path))
        png_paths.extend(
            os.path.join(stage_dir, "im{}of16.png".format(index))
            for index in range(1, 17)
        )
    invalid_pngs = [path for path in png_paths if not is_png(path)]
    if invalid_pngs:
        failures.append("invalid/missing PNGs: {}".format(", ".join(invalid_pngs)))

    if failures:
        print("FIG1D REPLAY VALIDATION FAILED")
        for failure in failures:
            print("-", failure)
        return 1

    print("FIG1D REPLAY VALIDATION PASSED")
    print("Run directory:", run_dir)
    print(
        "Validated {} PNGs plus per-mag ims.pkl, Vars.pkl, and run metadata".format(
            len(png_paths)
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
