import numpy as np
import pytest

from ..transforms.householder import householder, householder_qr
from ..transforms.householder_extension import householder_decomposition, W_Y_from_beta_vx


@pytest.mark.parametrize("n, p", [(15, 5), (35, 10), (55, 25), (150, 55)])
def test_householder_decomposition(n, p):
    """
    Test the householder algorithm using the decomposition of the form Q* = I - W @ Y*
    """
    np.random.seed(105 * p + 297 * n)
    A = np.random.randn(n, p)
    Q, _ = householder_qr(A)

    W, Y = householder_decomposition(A)
    Q_dec = np.eye(n) - W @ Y.T

    assert np.linalg.norm(Q[:, :p] - Q_dec[:, :p]) < 1e-6


@pytest.mark.parametrize("n, p", [(15, 5), (35, 10), (55, 25), (150, 55)])
def test_householder_decomposition_inplace(n, p):
    """
    Test the householder algorithm using the decomposition of the form Q* = I - W @ Y*
    Use optional functionality in inplace householder implementation
    """
    np.random.seed(105 * p + 297 * n)
    A = np.random.randn(n, p)
    Q, _ = householder_qr(A)

    beta, vx = householder(A, decomposed=True)

    W, Y = W_Y_from_beta_vx(beta, vx)
    Q_dec = np.eye(n) - W @ Y.T

    assert np.linalg.norm(Q[:, :p] - Q_dec[:, :p]) < 1e-6
