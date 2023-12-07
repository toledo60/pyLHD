import pytest
from pytest import approx
from pyLHD.criteria import AvgAbsCor, MaxAbsCor
from pyLHD.orthogonal import OLHD_Ye98

@pytest.mark.benchmark
@pytest.mark.parametrize("m", [4, 8, 10])
def test_AvgAbsCor(benchmark, m):
    sample = OLHD_Ye98(m=m)

    def run_test():
      return AvgAbsCor(sample)

    result = benchmark(run_test)
    # Adjust the expected result based on the specific test case
    if m == 4:
      assert result == approx(0.0)
    elif m == 8:
      assert result == approx(0.0)
    elif m == 10:
      assert result == approx(0.0)
      
      
      
@pytest.mark.benchmark
@pytest.mark.parametrize("m", [4, 8, 10])
def test_MaxAbsCor(benchmark, m):
    sample = OLHD_Ye98(m=m)

    def run_test():
      return MaxAbsCor(sample)

    result = benchmark(run_test)
    # Adjust the expected result based on the specific test case
    if m == 4:
      assert result == approx(0.0)
    elif m == 8:
      assert result == approx(0.0)
    elif m == 10:
      assert result == approx(0.0)
