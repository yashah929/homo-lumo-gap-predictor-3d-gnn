#!/usr/bin/env python3
"""Create the immutable 80/20 split and four development folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qm9_gap.splits import create_fixed_splits, validate_fixed_splits
from qm9_gap.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=None,
        help="Explicit override, primarily for tests; otherwise read prepared-data metadata.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing split arrays.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(config["paths"]["splits_dir"])
    existing = output / "split_manifest.json"
    if args.num_molecules is None:
        metadata_path = Path(config["paths"]["data_root"]) / "processed" / "metadata_v1.json"
        if not metadata_path.exists():
            raise FileNotFoundError("Prepared-data metadata is absent; run scripts/prepare_data.py first")
        with metadata_path.open(encoding="utf-8") as handle:
            num_molecules = int(json.load(handle)["num_molecules"])
    else:
        num_molecules = args.num_molecules
    expected = int(config["dataset"]["expected_num_molecules"])
    if num_molecules != expected and args.num_molecules is None:
        raise RuntimeError(f"Prepared dataset has {num_molecules} molecules; expected {expected}")
    if existing.exists() and not args.force:
        validate_fixed_splits(num_molecules, output)
        with existing.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        print(json.dumps(manifest, indent=2))
        print("Existing split files validated; no files were replaced.")
        return
    manifest = create_fixed_splits(num_molecules, output, int(config["seed"]))
    validate_fixed_splits(num_molecules, output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
