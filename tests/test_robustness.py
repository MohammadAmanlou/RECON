"""Regression tests for adversarial and independent RECON validations."""
import numpy as np
import pytest

from recon.adaptive import propose_adaptively
from recon.core import (B, B4, FOURTH_BASE, SIGNS, World, feature_matrix,
                        hyperedges, make_world, sample_contrasts, select_packets)
from recon.robustness import (SCENARIOS, fourth_action_falsification,
                              experiment, scenario_world, selected_confirmation)


def test_correct_transpose_for_fourth_projection_regression():
    # B and B.T are not equal. The original implementation incorrectly used
    # solve(B, B4), silently corrupting held-out fourth-action comparisons.
    w = make_world(17)
    packets = np.arange(1 << w.p)
    x = feature_matrix(packets, w.edges)
    observed_contrasts = x @ w.w @ B.T
    correct_weights = np.linalg.solve(B.T, B4)
    predicted = observed_contrasts @ correct_weights
    truth = w.probabilities(packets, fourth=True) - FOURTH_BASE
    assert np.allclose(predicted, truth, atol=1e-12)
    assert not np.allclose(np.linalg.solve(B, B4), correct_weights)


def test_fourth_assay_is_valid_bernoulli_in_every_base_world():
    for seed in range(64):
        w = make_world(seed)
        values = w.probabilities(np.arange(1 << w.p), fourth=True)
        assert np.min(values) >= 0.01
        assert np.max(values) <= 0.99


def test_order_four_misspecification_is_real_and_hidden_from_dictionary():
    w, shift, rep = scenario_world(5, 'unmodeled_four_way')
    assert shift is None and rep == 1024
    assert len([edge for edge in w.shared_support() if len(edge) == 4]) == 1
    assert all(len(edge) <= 3 for edge in hyperedges(6, 3))


def test_off_target_relearning_changes_observations_without_changing_theta():
    w, shift, _ = scenario_world(4, 'recovery_interference')
    assert shift is not None and shift[1] == -0.06
    edge, _ = shift
    mask = sum(1 << i for i in edge)
    basic = sample_contrasts(w, np.asarray([mask]), 2000, 12)
    changed = sample_contrasts(w, np.asarray([mask]), 2000, 12,
                               recovery_shift=shift)
    # Because sampling uses an identical RNG state and differs only in the
    # recovery treated probability, this difference isolates the perturbation.
    assert abs(changed[0, 2] - basic[0, 2]) > .035
    assert w.w[w.edges.index(edge), 0] == 0.


def test_adaptive_proposals_use_only_previous_observations():
    w = make_world(2)
    edges = hyperedges(6, 3)
    first = propose_adaptively(w, edges, budget=20, replicates=200, seed=17,
                                pilot=8)
    second = propose_adaptively(w, edges, budget=20, replicates=200, seed=17,
                                 pilot=8)
    assert np.array_equal(first.packets, second.packets)
    assert np.array_equal(first.contrasts, second.contrasts)
    assert len(np.unique(first.packets)) == 20
    assert first.acquisition_trials == 6 * 20 * 200
    assert first.contrasts.shape == (20, 3)
    with pytest.raises(ValueError):
        propose_adaptively(w, edges, budget=15, replicates=200,
                           seed=17, pilot=16)


def test_all_stress_scenarios_can_execute_without_invalid_probabilities():
    for scenario in SCENARIOS:
        result = experiment(0, scenario, 'd_optimal', packet_budget=20)
        assert result['scenario'] == scenario
        assert 0 <= result['support_f1'] <= 1
        assert 0 <= result['minimal_packet_f1'] <= 1
        assert result['heldout_packet_count'] > 0
        assert np.isfinite(result['heldout_4th_action_mae'])


def test_confirmation_is_independent_and_not_oracle_selected():
    r = selected_confirmation(1, budget=25, replicates=1000,
                              confirmation_replicates=20000, max_targets=2)
    assert r['selected_without_oracle']
    assert 0 <= r['certified'] <= r['selected'] <= 2
    assert r['true_theta_covered'] <= r['total_tested']
    assert r['confirmation_raw_trials'] >= 0
    assert r['negative_controls'] == 1


def test_fourth_falsification_exact_oracle_calibration():
    null = fourth_action_falsification(3, draws=4000, misspecification=0.)
    bad = fourth_action_falsification(3, draws=4000, misspecification=.08)
    assert null['actual'] == pytest.approx(null['predicted'])
    assert bad['actual'] - bad['predicted'] == pytest.approx(.08)
    assert bad['reject']
    with pytest.raises(ValueError):
        fourth_action_falsification(3, draws=4000, misspecification=.9)


def test_weak_signal_is_below_fixed_decision_threshold():
    # Deliberate failure mode: every planted common effect is below the
    # prespecified .03 cutoff, even with infinite data / perfect estimation.
    world, _, _ = scenario_world(0, 'weak_signal')
    active = world.w[:, 0][world.w[:, 0] > 0]
    assert active.size > 0
    assert float(active.max()) < .03