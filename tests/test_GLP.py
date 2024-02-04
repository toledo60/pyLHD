import pyLHD
import pytest
from pytest import approx

@pytest.mark.parametrize("p, t, expected", [(3, 2, 14), (5, 2, 126), (7,3,25212)])
def test_OddPrimePower(p, t, expected):
    N = p**t
    sample = pyLHD.GoodLatticePoint(size = (N, pyLHD.euler_phi(N)))
    result = pyLHD.LqDistance(sample,q=1).design()
    assert result == approx(expected)


@pytest.mark.parametrize("p, expected", [(11, 50), (7,18), (17,128),(23,242)])
def test_OddPrimeDouble(p, expected):
    N = 2*p
    sample = pyLHD.GoodLatticePoint(size = (N, pyLHD.euler_phi(N)))
    result = pyLHD.LqDistance(sample,q=1).design()
    assert result == approx(expected)


@pytest.mark.parametrize("p, q, expected", [(3, 7, 60), (3, 11, 160), (5,13,768)])
def test_MultiplyOddPrimes(p, q, expected):
    N = p*q
    sample = pyLHD.GoodLatticePoint(size = (N, pyLHD.euler_phi(N)))
    result = pyLHD.LqDistance(sample,q=1).design()
    assert result == approx(expected)


@pytest.mark.parametrize("t,expected", [(2,2), (5,128), (7,2048)])
def test_Power2(t, expected):
    N = 2**t
    sample = pyLHD.GoodLatticePoint(size = (N, pyLHD.euler_phi(N)))
    result = pyLHD.LqDistance(sample,q=1).design()
    assert result == approx(expected) 

