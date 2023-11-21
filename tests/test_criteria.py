import numpy as np
from scipy.stats import qmc
import pytest
from pytest import approx
from pyLHD.criteria import MaxAbsCor, MaxProCriterion, phi_p, AvgAbsCor
from pyLHD.OLHD import OLHD_Ye98

sampler = qmc.LatinHypercube(d=2, strength=2,seed=88)
sample = sampler.random(n=9)

def test_initial_design():
  lhd_strength2 = np.array([[0.20157202, 0.20157202],
                            [0.53490535, 0.05646416],
                            [0.86823868, 0.22523203],
                            [0.05646416, 0.53490535],
                            [0.38979749, 0.38979749],
                            [0.72313082, 0.55856536],
                            [0.22523203, 0.86823868],
                            [0.55856536, 0.72313082],
                            [0.8918987,  0.8918987]])
  np.testing.assert_allclose(sample,lhd_strength2,rtol=1e-06)                          

def test_MaxAbsCor():
  assert MaxAbsCor(sample) == approx(0.06601886)

def test_MaxProCriterion():
  assert MaxProCriterion(sample) == approx(29.59171)


@pytest.mark.parametrize("p,q,expected", 
[(15,1,3.069592), (10,1,3.163822), (15,2,4.335965),(17,2,4.322557)])
class TestPhiP:
    def test_phi_p(self, p,q,expected):
        assert phi_p(sample,p=p,q=q) == approx(expected)


@pytest.mark.parametrize("m,expected", 
[(2,0.0), (4,0.0), (8,0.0),(10,0.0)])
class TestZeroMaxAbsCor:
  def test_zero_MaxAbsCor(self, m, expected):
    olhd = OLHD_Ye98(m=m)
    assert MaxAbsCor(olhd) == approx(expected)



@pytest.mark.parametrize("m,expected", 
[(2,0.0), (4,0.0), (8,0.0),(10,0.0)])
class TestZeroAvgAbsCor:
  def test_zero_AvgAbsCor(self, m, expected):
    olhd = OLHD_Ye98(m=m)
    assert AvgAbsCor(olhd) == approx(expected)

