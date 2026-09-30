import pytest
import torch

from qm9_gap.data import HOMO_LUMO_GAP_TARGET_INDEX, QM9_TARGET_NAMES, select_homo_lumo_gap


def test_direct_gap_target_is_explicitly_index_four() -> None:
    targets = torch.arange(19, dtype=torch.float32).reshape(1, -1)
    targets[0, 4] = targets[0, 3] - targets[0, 2]
    assert HOMO_LUMO_GAP_TARGET_INDEX == 4
    assert QM9_TARGET_NAMES[4] == "gap"
    assert float(select_homo_lumo_gap(targets)) == 1.0
    with pytest.raises(ValueError, match="must be 4"):
        select_homo_lumo_gap(targets, target_index=3)


def test_gap_semantic_check_detects_upstream_target_reordering() -> None:
    targets = torch.zeros((1, 19))
    targets[0, 2] = -6.0
    targets[0, 3] = 1.0
    targets[0, 4] = 2.0
    with pytest.raises(ValueError, match="inconsistent"):
        select_homo_lumo_gap(targets)
