import torch
from rdkit import Chem

from qm9_gap.features import CATEGORICAL_FEATURE_NAMES, atom_features_from_rdkit
from qm9_gap.graph import chemical_edge_features, complete_directed_edge_index


def test_required_features_and_explicit_hydrogens_are_retained() -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)F"))
    z, categorical, mass = atom_features_from_rdkit(molecule)
    assert z.numel() == molecule.GetNumAtoms()
    assert int((z == 1).sum()) > 0
    assert categorical.shape == (molecule.GetNumAtoms(), len(CATEGORICAL_FEATURE_NAMES))
    assert mass.shape == (molecule.GetNumAtoms(), 1)
    assert torch.all(mass > 0)


def test_nonbonded_edge_chemistry_is_zero() -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CO"))
    edge_index = complete_directed_edge_index(molecule.GetNumAtoms())
    features = chemical_edge_features(molecule, edge_index)
    for edge, values in zip(edge_index.t().tolist(), features, strict=True):
        bond = molecule.GetBondBetweenAtoms(*edge)
        if bond is None:
            assert torch.count_nonzero(values) == 0
        else:
            assert values[0] == 1
