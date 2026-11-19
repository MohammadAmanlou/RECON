import numpy as np
from recon.scientific import generate_dag, verify_scientific_tasks


def test_shared_intermediate_adjoint():
    p = generate_dag(1)
    x = np.array([.3, -.2, .4])
    assert np.allclose(p.reverse_gradient(x), p.forward_gradient(x))
    assert np.allclose(p.reverse_gradient(x), p.central_difference(x), atol=1e-5)


def test_100_generated_programs():
    audit = verify_scientific_tasks(100)
    assert audit["tasks_checked"] == 100
    assert audit["max_reverse_forward_abs_error"] < 1e-9
    assert audit["max_reverse_finite_diff_abs_error"] < 1e-4