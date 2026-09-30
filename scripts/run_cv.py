#!/usr/bin/env python3
"""Run one development-only configuration/fold experiment."""

from __future__ import annotations

import argparse
import json

from qm9_gap.cross_validation import array_index_to_experiment, grid_configurations, run_cv_experiment
from qm9_gap.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--grid", default="configs/hyperparameter_grid.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--array-index", type=int)
    group.add_argument("--configuration-id")
    parser.add_argument("--fold", type=int, choices=range(4))
    args = parser.parse_args()
    base = load_yaml(args.config)
    grid = load_yaml(args.grid)
    if args.array_index is not None:
        if args.fold is not None:
            parser.error("--fold is encoded by --array-index and must not also be given")
        experiment, fold = array_index_to_experiment(args.array_index, grid)
    else:
        if args.fold is None:
            parser.error("--fold is required with --configuration-id")
        matches = [item for item in grid_configurations(grid) if item["configuration_id"] == args.configuration_id]
        if not matches:
            parser.error(f"Unknown configuration ID: {args.configuration_id}")
        experiment, fold = matches[0], args.fold
    result = run_cv_experiment(base, experiment, fold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
