#!/usr/bin/env python3
"""Download QM9 and create the deterministic chemically enriched graph cache."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qm9_gap.data import QM9GapDataset
from qm9_gap.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_yaml(args.config)
    dataset = QM9GapDataset(
        config["paths"]["data_root"],
        rbf_max=config["model"]["rbf_max"],
        domain_policy=config["dataset"]["rbf_domain_policy"],
        domain_tolerance=config["dataset"]["rbf_domain_tolerance"],
    )
    expected = int(config["dataset"]["expected_num_molecules"])
    if len(dataset) != expected:
        raise RuntimeError(f"Expected {expected} usable QM9 molecules, found {len(dataset)}")
    metadata_path = Path(config["paths"]["data_root"]) / "processed" / "metadata_v1.json"
    print(f"Prepared {len(dataset)} molecules; metadata: {metadata_path}")


if __name__ == "__main__":
    main()
