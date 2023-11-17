from scipy.stats import qmc
from pyLHD.criteria import discrepancy
from pytest import approx
import numpy as np

x = qmc.LatinHypercube(d=3,seed=10)
space = x.random(n=10)


def test_initial_design():
  lhd = np.array([[0.40439983, 0.57923182, 0.61715551],
                  [0.78507179, 0.24871954, 0.38640804],
                  [0.83109635, 0.91582523, 0.1574491],
                  [0.9043074,  0.81746671, 0.56617847],
                  [0.24242395, 0.42466981, 0.71728961],
                  [0.60665615, 0.08550053, 0.92544198],
                  [0.18606486, 0.10934712, 0.27738856],
                  [0.01467603, 0.76936821, 0.80301696],
                  [0.34821658, 0.66775254, 0.07175665],
                  [0.5394135,  0.36662355, 0.43213512]])
  np.testing.assert_allclose(space, lhd,rtol=1e-6)
    

def test_discrepancy_C2():
  assert discrepancy(space) == approx(0.1119206)


def test_discrepancy_L2():
  assert discrepancy(space, method = 'L2') == approx(0.017661)


def test_discrepancy_L2star():
  assert discrepancy(space, method = 'L2_star') == approx(0.04971855)


def test_discrepancy_M2():
  assert discrepancy(space, method = 'modified_L2') == approx(0.1298479)


def test_discrepancy_S2():
  assert discrepancy(space, method = 'symmetric_L2') == approx(0.4455809)


def test_discrepancy_W2():
  assert discrepancy(space, method = 'wrap_around_L2') == approx(0.1700446)


def test_discrepancy_Mix2():
  assert discrepancy(space, method = 'mixture_L2') == approx(0.1754307)








