import numpy as np
import pytest
from pytest import approx
from pyLHD.criteria import AvgAbsCor, MaxAbsCor
from pyLHD.OLHD import OLHD_Ye98


#@pytest.mark.parametrize("m", [2,3])
#def test_ye98(m):
#  olhd = OLHD_Ye98(m=m)
#  if m==2:
#    design = np.array([[2,-1],[1,2],[0,0],[-2,1],[-1,-2]])
#    np.testing.assert_allclose(design, olhd,rtol=1e-6)
#  if m == 3:
#    design = np.array([[3,-4,-2,1],[4,3,-1,-2],
#                       [1,-2,4,-3],[2,1,3,4],
#                       [0,0,0,0],[-3,4,2,-1],
#                       [-4,-3,1,2],[-1, 2,-4,3],
#                       [-2,-1,-3,-4]])
#    np.testing.assert_allclose(design, olhd,rtol=1e-6)
  

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
