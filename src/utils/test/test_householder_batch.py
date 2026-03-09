import numpy as np
import pytest

from ..transforms.householder_batch import householder_batch, reverse_householder_batch


@pytest.mark.parametrize(
    "n, p, r", [(5, 3, 2), (15, 6, 4), (30, 20, 3), (100, 35, 20), (40, 30, 30)]
)
def test_householder_batches(n, p, r):
    """
    test the implementation of the householder algorithm in batches
    """
    np.random.seed(105 * p + 297 * n)
    A = np.random.randn(n, p)
    A0 = A.copy()

    bat_WY = householder_batch(A, r)

    # test that A is now upper triangular
    assert np.linalg.norm(A[np.tril_indices(len(A), m=len(A[0]), k=-1)]) < 1e-6

    Wx, Yx = bat_WY["W"], bat_WY["Y"]
    # test that all decompositions produce unitary matrices
    for W, Y in zip(Wx, Yx):
        m = len(W)
        I = np.eye(m)
        Q = I - W @ Y.T
        assert np.linalg.norm(I - Q.T @ Q) < 1e-6

    reverse_householder_batch(A, r, Wx, Yx)

    # test the reverse process
    assert np.linalg.norm(A0 - A) < 1e-6
