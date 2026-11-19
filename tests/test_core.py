import numpy as np
import pytest

from recon import (B, B4, make_world, feature_matrix, hyperedges, select_packets,
                   sample_contrasts, reconstruct_theta, group_lasso,
                   exact_identification_witness, confirm_interaction)


def test_two_channel_ambiguity_and_three_channel_rank():
    w = exact_identification_witness()
    assert w["rank_full"] == 3
    assert all(value == 2 for value in w["rank_two_channel"].values())
    for menu, h in w["two_channel_null_vectors"].items():
        index = {"AL": [0, 1], "AR": [0, 2], "LR": [1, 2]}[menu]
        assert np.allclose(B[index] @ np.asarray(h), 0.)
    assert np.allclose(np.array([1., 1., -1.]) @ B, [1., 0., 0.])
    assert np.allclose(B4, np.array([1., 2., 1.]))


def test_mobius_basis_and_exact_rank():
    for p in range(2, 6):
        edges = hyperedges(p, min(3, p))
        packets = np.arange(1 << p)
        phi = feature_matrix(packets, edges)
        assert np.linalg.matrix_rank(phi) == len(edges)
        assert np.linalg.matrix_rank(np.kron(phi, B)) == 3 * len(edges)


def test_synthetic_world_valid_probability_and_order_three_support():
    for seed in range(20):
        w = make_world(seed)
        q = w.probabilities(np.arange(64))
        assert q.shape == (64, 3)
        assert (q > 0).all() and (q < 1).all()
        assert len([e for e in w.shared_support() if len(e) == 3]) >= 2


def test_reconstruction_exact_in_noise_free_world():
    world = make_world(5)
    masks = np.arange(1 << world.p)
    phi = feature_matrix(masks, world.edges)
    response = phi @ world.w @ B.T
    estimated = np.linalg.lstsq(phi, response, rcond=None)[0]
    assert np.allclose(reconstruct_theta(estimated), world.w[:, 0], atol=1e-10)
    assert np.allclose(2 * response[:, 2] - response[:, 1], phi @ world.w @ B4)


def test_d_optimal_is_seeded_and_budgeted():
    edges = hyperedges(6, 3)
    a = select_packets(6, edges, 40, 19, "d_optimal")
    b = select_packets(6, edges, 40, 19, "d_optimal")
    assert len(np.unique(a)) == 40
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        select_packets(6, edges, 64, 1, "random")


def test_group_lasso_and_three_channel_sample_shape():
    w = make_world(3)
    packets = select_packets(w.p, w.edges, 48, 5, "random")
    responses = sample_contrasts(w, packets, 512, 12)
    assert responses.shape == (48, 3)
    fit = group_lasso(feature_matrix(packets, w.edges), responses, .006)
    assert fit.shape == (41, 3)
    assert np.isfinite(fit).all()


def test_independent_confirm_certificate_and_cost():
    w = make_world(0)
    edge = sorted(w.shared_support(), key=lambda x: -len(x))[0]
    cert = confirm_interaction(w, edge, 100000, 17, n_targets=4)
    assert abs(cert["estimate"] - cert["true_theta"]) < cert["radius"]
    assert cert["total_simulated_bernoulli_trials"] == 3 * 2 ** len(edge) * 100000
    assert cert["certified_positive"]