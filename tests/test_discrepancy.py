from scipy.stats import qmc
from pyLHD.criteria import discrepancy, UniformProCriterion
import pytest
from pytest import approx
import numpy as np

def test_design():
  sampler = qmc.LatinHypercube(d=2, strength=2, seed=88)
  sample = sampler.random(n=9)
  design = np.array([[0.20157202, 0.20157202],
                     [0.53490535, 0.05646416],
                     [0.86823868, 0.22523203],
                     [0.05646416, 0.53490535],
                     [0.38979749, 0.38979749],
                     [0.72313082, 0.55856536],
                     [0.22523203, 0.86823868],
                     [0.55856536, 0.72313082],
                     [0.8918987,  0.8918987]])
  np.testing.assert_allclose(sample, design, rtol=1e-06)


def test_design_2():
  sampler = qmc.LatinHypercube(d=4, strength=2, seed=88)
  sample = sampler.random(n=25)
  design = np.array([[0.07256593, 0.07256593, 0.07256593, 0.07256593],
                     [0.27256593, 0.1403271, 0.27256593, 0.27256593],
                     [0.47256593, 0.00108353, 0.47256593, 0.47256593],
                     [0.67256593, 0.1712639, 0.67256593, 0.67256593],
                     [0.87256593, 0.11741126, 0.87256593, 0.87256593],
                     [0.1403271, 0.27256593, 0.3403271, 0.5403271],
                     [0.3403271, 0.3403271, 0.5403271, 0.7403271],
                     [0.5403271, 0.20108353, 0.7403271, 0.9403271],
                     [0.7403271, 0.3712639, 0.9403271, 0.1403271],
                     [0.9403271, 0.31741126, 0.1403271, 0.3403271],
                     [0.00108353, 0.47256593, 0.40108353, 0.80108353],
                     [0.20108353, 0.5403271, 0.60108353, 0.00108353],
                     [0.40108353, 0.40108353, 0.80108353, 0.20108353],
                     [0.60108353, 0.5712639, 0.00108353, 0.40108353],
                     [0.80108353, 0.51741126, 0.20108353, 0.60108353],
                     [0.1712639, 0.67256593, 0.7712639, 0.3712639],
                     [0.3712639, 0.7403271, 0.9712639, 0.5712639],
                     [0.5712639, 0.60108353, 0.1712639, 0.7712639],
                     [0.7712639, 0.7712639, 0.3712639, 0.9712639],
                     [0.9712639, 0.71741126, 0.5712639, 0.1712639],
                     [0.11741126, 0.87256593, 0.91741126, 0.71741126],
                     [0.31741126, 0.9403271, 0.11741126, 0.91741126],
                     [0.51741126, 0.80108353, 0.31741126, 0.11741126],
                     [0.71741126, 0.9712639, 0.51741126, 0.31741126],
                     [0.91741126, 0.91741126, 0.71741126, 0.51741126]])
  np.testing.assert_allclose(sample, design, rtol=1e-05)


@pytest.mark.parametrize("n, d, method, expected", 
[(9, 2, "centered_L2",0.07903468), 
(9, 2, "L2",0.02732294),
(9, 2, "L2_star",0.05094798),
(9, 2, "modified_L2",0.08547994),
(9, 2, "symmetric_L2",0.1983027),
(9, 2, "wrap_around_L2",0.120021),
(9, 2, "mixture_L2",0.1105645),
(25, 4, "centered_L2",0.07397585), 
(25, 4, "L2",0.004238196),
(25, 4, "L2_star",0.0327091),
(25, 4, "modified_L2",0.094341),
(25, 4, "symmetric_L2",0.4241608),
(25, 4, "wrap_around_L2",0.1137553),
(25, 4, "mixture_L2",0.127585)
])
def test_discrepancy(n, d, expected,method):
    sampler = qmc.LatinHypercube(d=d, strength=2, seed=88)
    sample = sampler.random(n=n)
    result = discrepancy(sample, method=method)
    assert result == approx(expected)


def test_balanced_centered_L2():
  balanced_design = np.array([
      [2, 3, 2], [4, 5, 13], [0, 11, 9], [3, 16, 17],
      [1, 22, 22], [8, 0, 7], [6, 8, 19], [9, 14, 24],
      [5, 18, 4], [7, 20, 11], [12, 2, 21], [10, 9, 0],
      [14, 12, 14], [13, 15, 6], [11, 24, 15], [17, 4, 10],
      [15, 7, 5], [19, 10, 18], [16, 19, 20],
      [18, 23, 1], [21, 1, 16], [23, 6, 23],
      [22, 13, 3], [20, 17, 12], [24, 21, 8]
  ])
  result = discrepancy(balanced_design, method='balanced_centered_L2')
  expected = np.sqrt(0.0015340707624760253)
  assert result == approx(expected)


def test_UniformProjection():
  upd = np.array([
      [2, 3, 2], [4, 5, 13], [0, 11, 9], [3, 16, 17],
      [1, 22, 22], [8, 0, 7], [6, 8, 19], [9, 14, 24],
      [5, 18, 4], [7, 20, 11], [12, 2, 21], [10, 9, 0],
      [14, 12, 14], [13, 15, 6], [11, 24, 15], [17, 4, 10],
      [15, 7, 5], [19, 10, 18], [16, 19, 20],
      [18, 23, 1], [21, 1, 16], [23, 6, 23],
      [22, 13, 3], [20, 17, 12], [24, 21, 8]
  ])
  result = UniformProCriterion(upd)
  expected = 0.000527906844443852
  assert result == approx(expected)

