import numpy as np
from scipy.stats import qmc
import pytest
from pytest import approx
from pyLHD.criteria import MaxAbsCor, MaxProCriterion, phi_p, AvgAbsCor
from pyLHD.criteria import LqDistance, MeshRatio, coverage

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

  
@pytest.mark.parametrize("n, d, expected", [(9, 2, 0.06601886), (25, 4, 0.11555915088)])
def test_MaxAbsCor(n, d, expected):
    sampler = qmc.LatinHypercube(d=d, strength=2, seed=88)
    sample = sampler.random(n=n)
    result = MaxAbsCor(sample)
    assert result == approx(expected)


@pytest.mark.parametrize("n, d, expected", [(9, 2, 0.06601886), (25, 4, 0.05952789)])
def test_AvgAbsCor(n, d, expected):
    sampler = qmc.LatinHypercube(d=d, strength=2, seed=88)
    sample = sampler.random(n=n)
    result = AvgAbsCor(sample)
    assert result == approx(expected)


@pytest.mark.parametrize("n, d, expected", [(9, 2, 29.59171), (25, 4, 42.30633)])
def test_MaxProCriterion(n, d, expected):
    sampler = qmc.LatinHypercube(d=d, strength=2, seed=88)
    sample = sampler.random(n=n)
    result = MaxProCriterion(sample)
    assert result == approx(expected)


@pytest.mark.parametrize("n, d, expected", [(9, 2, 0.1874005), (25, 4, 0.09345244)])
def test_coverage(n, d, expected):
    sampler = qmc.LatinHypercube(d=d, strength=2, seed=88)
    sample = sampler.random(n=n)
    result = coverage(sample)
    assert result == approx(expected)


@pytest.mark.parametrize("n, d, expected", [(9, 2, 1.562099), (25, 4, 1.507535)])
def test_MeshRatio(n, d, expected):
    sampler = qmc.LatinHypercube(d=d, strength=2, seed=88)
    sample = sampler.random(n=n)
    result = MeshRatio(sample)
    assert result == approx(expected)


@pytest.mark.parametrize("n,d,p,q,expected", [
    (9, 2, 15, 1, 3.069592), (9, 2, 10, 1, 3.163822), (9, 2, 15, 2, 4.335965),
    (9, 2, 17, 2, 4.322557), (25, 4, 15, 1, 2.1165579886), (25, 4, 10, 1, 2.2764186),
    (25, 4, 15, 2, 3.6045218), (25, 4, 17, 2, 3.540896)])
def test_phi_p(n, d, p, q, expected):
  sampler = qmc.LatinHypercube(d=d, strength=2, seed=88)
  sample = sampler.random(n=n)
  assert phi_p(sample, p=p, q=q) == approx(expected)   
   
        
@pytest.mark.parametrize("n,d,i,j,q,expected",
                         [(9, 2, 0, 1, 1, 0.4784412), (9, 2, 2, 1, 1, 0.5021012), 
                          (9, 2, 2, 1, 2, 0.3736224), (9, 2, 4, 1, 2, 0.3635483),
                          (25, 4, 0, 1, 1, 0.6677612), (25, 4, 2, 1, 1, 0.7392436), 
                          (25, 4, 9, 4, 2, 0.9295394), (25, 4, 14, 8, 2, 0.8853406)])
def test_inter_site(n, d, i, j, q, expected):
  sampler = qmc.LatinHypercube(d=d, strength=2, seed=88)
  sample = sampler.random(n=n)
  lq = LqDistance(sample,q=q)
  assert lq.index(i=i, j=j) == approx(expected)