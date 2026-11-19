"""Optional CPU PyTorch experiment; skipped if torch is unavailable."""
import pytest

torch = pytest.importorskip('torch')
from recon.neural_pilot import one_seed


def test_trainable_modular_learner_mask_is_not_unlearning():
    torch.set_num_threads(1)
    r = one_seed(0, steps=90)
    assert r['in_domain_both_score'] > .95
    assert r['reversible_mask_score'] < .75
    assert abs(r['mask_restored_zero_update_score'] -
               r['in_domain_both_score']) < 1e-9
    assert r['retrained_after_reset_score'] > .95
    assert r['destructive_output_projection_reset_score'] < .75