import numpy as np
from pyLHD.helpers import is_prime, WilliamsTransform, level_permutation, is_balanced_design
from pyLHD.base import GoodLatticePoint
import numpy.typing as npt

def UPC_lower_bound(arr: npt.ArrayLike) -> float:
  """Uniform Projection Criteria Lower Bound

  Args:
      arr (npt.ArrayLike): A numpy ndarray

  Returns:
      The lower bound of the Uniform projection criteria, whenever `arr` is a balanced $(N,s^m)$ design
  
  Notes:
      The lower bound is achieved if and only if the design $D$ is an equidistant design under the $L_1$-distance
  
  Example:
  ```{python}
  import pyLHD
  sample = pyLHD.EquidistantLHD(N=11)
  sample
  ```
  ```{python}
  pyLHD.UniformProCriterion(sample)
  ```
  ```{python}
  pyLHD.UPC_lower_bound(sample)
  ```
  """
  try:
    s = np.unique(arr).size
    is_balanced_design(arr, s = s)
  except ValueError as e:
    print(f"Error: {e}")
  n,m = arr.shape
  val1 = 5*m*(4*(s**4) + 2*(13*n -17)*(s**2) - n +5) - (n-1)*(8*(s**4)+150*(s**2)-33)
  val2 = 720*(m-1)*(n-1)*(s**4)
  val3 = (1+ ((-1)**s))/(64*(s**4))
  return (val1/val2) + val3


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