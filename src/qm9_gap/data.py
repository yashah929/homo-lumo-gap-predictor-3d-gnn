"""QM9 acquisition, target verification, chemical enrichment, and split access."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import QM9

from .features import atom_features_from_rdkit
from .graph import chemical_edge_features, complete_directed_edge_index, pairwise_edge_distances

LOGGER = logging.getLogger(__name__)

# PyG QM9 target order, documented by torch_geometric.datasets.QM9. Indices
# 0--11 are molecular properties; 12--18 are atomization energies/frequencies.
QM9_TARGET_NAMES = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atomization",
    "U_atomization",
    "H_atomization",
    "G_atomization",
    "A",
    "B",
    "C",
)
HOMO_LUMO_GAP_TARGET_INDEX = 4


def select_homo_lumo_gap(targets: Tensor, target_index: int = 4) -> Tensor:
    """Select the direct QM9 HOMO-LUMO gap target, already expressed in eV.

    The assertions intentionally make the PyG convention explicit: target 2 is
    HOMO, target 3 is LUMO, and target 4 is their reported gap. The model uses
    target 4 directly; it does not train separate orbital-energy predictors.
    """
    assert HOMO_LUMO_GAP_TARGET_INDEX == 4
    assert QM9_TARGET_NAMES[HOMO_LUMO_GAP_TARGET_INDEX] == "gap"
    if target_index != HOMO_LUMO_GAP_TARGET_INDEX:
        raise ValueError(
            f"QM9 target index must be 4 ('gap'), received {target_index}. "
            "Changing this would violate the scientific objective."
        )
    if targets.shape[-1] <= target_index:
        raise ValueError(f"QM9 target tensor has only {targets.shape[-1]} columns")
    gap = targets[..., target_index]
    # Independent semantic sanity check: in the PyG convention columns 2 and
    # 3 are HOMO and LUMO. CSV rounding permits a few meV of discrepancy.
    derived_gap = targets[..., 3] - targets[..., 2]
    if not torch.allclose(gap, derived_gap, atol=5.0e-3, rtol=5.0e-4):
        raise ValueError(
            "QM9 target index 4 is inconsistent with LUMO minus HOMO; "
            "the upstream target convention may have changed"
        )
    return gap


def _sdf_molecule_for_data(supplier: Any, base_data: Data) -> Any:
    if not hasattr(base_data, "idx"):
        raise ValueError("PyG QM9 data does not expose its original SDF index")
    raw_index = int(torch.as_tensor(base_data.idx).reshape(-1)[0])
    molecule = supplier[raw_index]
    if molecule is None:
        raise ValueError(f"RDKit could not read QM9 SDF molecule {raw_index}")
    observed = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    expected = torch.as_tensor(base_data.z).tolist()
    if observed != expected:
        raise ValueError(
            f"SDF/PyG atom-order mismatch at raw molecule {raw_index}: "
            f"SDF={observed}, PyG={expected}"
        )
    return molecule


class QM9GapDataset(InMemoryDataset):
    """Deterministically cached, hydrogen-explicit complete graphs for QM9."""

    CACHE_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        rbf_max: float = 10.0,
        domain_policy: str = "error",
        domain_tolerance: float = 0.1,
        transform: Any = None,
    ) -> None:
        self.rbf_max = float(rbf_max)
        self.domain_policy = domain_policy
        self.domain_tolerance = float(domain_tolerance)
        if domain_policy not in {"error", "warn"}:
            raise ValueError("domain_policy must be 'error' or 'warn'")
        super().__init__(str(root), transform=transform)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:  # PyTorch < 2.6
            self.data, self.slices = torch.load(self.processed_paths[0])
        with Path(self.processed_paths[1]).open(encoding="utf-8") as handle:
            self.preprocessing_metadata = json.load(handle)
        self._validate_cached_distance_domain()

    def _validate_cached_distance_domain(self) -> None:
        maximum_distance = float(self.preprocessing_metadata["maximum_pairwise_distance_angstrom"])
        if maximum_distance <= self.rbf_max + self.domain_tolerance:
            return
        message = (
            f"Cached maximum QM9 pairwise distance is {maximum_distance:.6f} A; "
            f"configured RBF maximum is {self.rbf_max:.3f} A."
        )
        if self.domain_policy == "error":
            raise ValueError(message + " Increase model.rbf_max or set dataset.rbf_domain_policy=warn.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return [f"qm9_gap_graphs_v{self.CACHE_VERSION}.pt", f"metadata_v{self.CACHE_VERSION}.json"]

    def download(self) -> None:
        """Download is delegated to the source PyG QM9 dataset."""

    def process(self) -> None:
        source_root = Path(self.root) / "pyg_source"
        source = QM9(root=str(source_root))
        sdf_path = Path(source.raw_dir) / "gdb9.sdf"
        if not sdf_path.exists():
            raise FileNotFoundError(f"Expected hydrogen-explicit QM9 SDF at {sdf_path}")
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)

        processed: list[Data] = []
        maximum_distance = 0.0
        maximum_index = -1
        for dataset_index, base_data in enumerate(source):
            molecule = _sdf_molecule_for_data(supplier, base_data)
            z, x_cat, atomic_mass = atom_features_from_rdkit(molecule)
            edge_index = complete_directed_edge_index(z.numel())
            distance = pairwise_edge_distances(base_data.pos, edge_index)
            edge_chem = chemical_edge_features(molecule, edge_index)
            local_maximum = float(distance.max()) if distance.numel() else 0.0
            if local_maximum > maximum_distance:
                maximum_distance = local_maximum
                maximum_index = dataset_index
            gap = select_homo_lumo_gap(base_data.y).reshape(1)
            processed.append(
                Data(
                    z=z,
                    x_cat=x_cat,
                    atomic_mass=atomic_mass,
                    pos=base_data.pos.to(torch.float32),
                    edge_index=edge_index,
                    edge_distance=distance.to(torch.float32),
                    edge_chem=edge_chem,
                    y=gap.to(torch.float32),
                    molecule_index=torch.tensor([dataset_index], dtype=torch.long),
                    raw_qm9_index=torch.as_tensor(base_data.idx, dtype=torch.long).reshape(1),
                    smiles=getattr(base_data, "smiles", ""),
                )
            )
            if (dataset_index + 1) % 10000 == 0:
                LOGGER.info("Processed %d/%d QM9 molecules", dataset_index + 1, len(source))

        message = (
            f"Maximum QM9 pairwise distance is {maximum_distance:.6f} A "
            f"(dataset index {maximum_index}); configured RBF maximum is {self.rbf_max:.3f} A."
        )
        LOGGER.info(message)
        if maximum_distance > self.rbf_max + self.domain_tolerance:
            if self.domain_policy == "error":
                raise ValueError(message + " Increase model.rbf_max or set dataset.rbf_domain_policy=warn.")
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        torch.save(self.collate(processed), self.processed_paths[0])
        metadata = {
            "cache_version": self.CACHE_VERSION,
            "num_molecules": len(processed),
            "target_index": HOMO_LUMO_GAP_TARGET_INDEX,
            "target_name": "gap",
            "target_unit": "eV",
            "explicit_hydrogens": True,
            "complete_directed_graph": True,
            "maximum_pairwise_distance_angstrom": maximum_distance,
            "maximum_distance_molecule_index": maximum_index,
            "rbf_max_angstrom": self.rbf_max,
            "domain_policy": self.domain_policy,
        }
        with Path(self.processed_paths[1]).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")


def load_split_indices(splits_dir: str | Path, name: str) -> np.ndarray:
    path = Path(splits_dir) / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run scripts/create_splits.py first.")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Invalid one-dimensional integer split file: {path}")
    return values.astype(np.int64, copy=False)


def load_cv_fold(splits_dir: str | Path, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one development-only fold without reading final-test indices."""
    if fold not in range(4):
        raise ValueError("fold must be 0, 1, 2, or 3")
    development = load_split_indices(splits_dir, "development_indices")
    validation = load_split_indices(splits_dir, f"cv_fold_{fold}_validation_indices")
    training = np.setdiff1d(development, validation, assume_unique=False)
    return training, validation


def target_statistics(dataset: Any, indices: np.ndarray | list[int]) -> tuple[float, float]:
    """Compute mean and population SD using only explicitly supplied indices."""
    values = torch.tensor([float(dataset[int(i)].y.reshape(-1)[0]) for i in indices], dtype=torch.float64)
    if values.numel() < 2:
        raise ValueError("At least two training targets are required for standardization")
    standard_deviation = float(values.std(unbiased=False))
    if standard_deviation <= 0.0:
        raise ValueError("Training-target standard deviation must be positive")
    return float(values.mean()), standard_deviation
