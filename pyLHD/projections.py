import numpy as np
from pyLHD.helpers import is_prime, WilliamsTransform, level_permutation, is_balanced_design
from pyLHD.base import GoodLatticePoint
import numpy.typing as npt

def best_linear_permutation(N:int) -> int:
  """Optimal linear permutation value to minimize the uniform projection criterion

  Args:
      N (int): A prime integer

  Raises:
      ValueError:  If `N` is not a prime integer

  Returns:
      Optimal value of `b` to apply a linear level permutation and minimize the uniform projection criterion. That is $D_b = D + b (mod \\, N)$
  """
  if not is_prime(N):
    raise ValueError("'N' must be a prime number")
  
  c_zero = int(np.sqrt((N**2 - 1)/12))
  if c_zero >= (np.sqrt((N**2 - 4)/12) - 0.5):
    c =  c_zero
  else:
    c = c_zero + 1
      
  y = (N-1)/2 + c
  if (y % 2) == 0: 
    b = int(y/2)
  else :
    b = int((2*N - y - 1)/2)
  return b 


def UniformProLHD(N:int) -> np.ndarray:
  """Generate a Uniform Projection Design

  Args:
      N (int): An odd integer

  Raises:
      ValueError: If `N` is not an odd integer

  Returns:
      An $(N \\times N-1)$ Uniform projection design
  
  Example:
  ```{python}
  import pyLHD
  sample = pyLHD.UniformProLHD(N=11)
  sample
  ```
  ```{python}
  pyLHD.UniformProCriterion(sample)
  ```
  """
  if not is_prime(N):
    raise ValueError('N should be a prime number')
  D = GoodLatticePoint(size=(N,N-1))
  b = best_linear_permutation(N)
  return WilliamsTransform(level_permutation(D, b=b))