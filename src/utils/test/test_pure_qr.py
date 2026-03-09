from ..transforms.pure_qr import pure_QR_eigenvalues, precond_example


import numpy as np
import pytest


def generate_A(m: int) -> np.ndarray:
    return (
        np.diag([2.0] * m)
        + np.diag([-1.0] * (m - 1), 1)
        + np.diag([-1.0] * (m - 1), -1)
    )


@pytest.mark.parametrize("m", [5, 10, 20])
def test_pure_QR_eigenvalues(m):
    A = generate_A(m)
    d_true = np.array(
        [2.0 + 2 * np.cos((k * np.pi) / (m + 1)) for k in range(1, m + 1)]
    )
    d = pure_QR_eigenvalues(A, tol=1e-10)
    assert np.linalg.norm(d - d_true) < 1e-4


# @pytest.mark.parametrize("m", [5, 10, 15])
# def test_preconditionned_example(m):
#     A = generate_A(m)
#     np.random.seed(1878 * m)
#     b = np.random.randn(m)
#     Ahat = np.triu(A)
#     assert np.linalg.norm(Ahat[np.tril_indices(m, k=-1)]) < 1e-6
#     precond = precond_example(A)
#     assert np.linalg.norm(Ahat @ precond(b) - np.array([b]).T) < 1e-6
