import pyLHD
import numpy as np
import pytest


@pytest.mark.parametrize("p,k",[(7,1), (11,1),(3,3), (19,1)])
def test_paley(p,k):
  H = pyLHD.paley(p=p,k=k)
  np.testing.assert_allclose(H + H.T,2*np.eye(H.shape[0]),rtol=1e-06)


@pytest.mark.parametrize("p, k", [(7,1), (11,1),(3,3), (19,1)])
def test_jacobsthal(p,k):
  Q = pyLHD.jacobsthal_matrix(p=p,k=k)
  pyLHD.is_cyclic(Q)
  np.testing.assert_allclose(Q@np.ones_like(Q), np.ones_like(Q)@Q,rtol=1e-06)
  

