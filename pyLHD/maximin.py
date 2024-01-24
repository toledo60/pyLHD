import numpy as np
import numpy.typing as npt
from typing import Literal
from pyLHD.helpers import LevelPermutation, WilliamsTransform,is_prime
from pyLHD.base import GoodLatticePoint
from pyLHD.criteria import LqDistance
from typing import Union, Optional


def LeaveOneOut(arr: npt.ArrayLike, b: int,
                method: Literal['LP', 'WT'] = 'LP') -> npt.ArrayLike:
  """Apply the Leave-one-out Procedure to Generate a Maxmin LHD

  Args:
      arr (npt.ArrayLike): A numpy ndarry, with initial shape $(n \\times d)$
      b (int): Integer to apply either linear level permutation or William's transformation
      method (Literal[&#39;LP&#39;, &#39;WT&#39;], optional): Linear level permutation (LP) or William's transformation (WT). Defaults to 'LP'.

  Raises:
      TypeError: `b` must be an integer
      ValueError: Given an LHD with column permutations (0,1,...,n-1), `b` must be within (0,1,...,n-1)
      ValueError: If `method` is not 'LP' or 'WT'

  Returns:
      npt.ArrayLike: After removing the last constant row of initial LHD, an $(n-1) \\times d$ maximin LHD is returned
  Example:
  ```{python}
  import pyLHD
  n = 11
  d = pyLHD.euler_phi(n)
  x = pyLHD.GoodLatticePoint(size = (n,d))
  x
  ```
  The initial $L_1$-distance of `x` is
  ```{python}
  pyLHD.LqDistance(x, q=1)
  ```
  After applying the Leave-one-out method with a simple linear level permutation, we should obtain an $(n-1) \\times d$ LHD with higher $L_1$-distance
  ```{python}
  x_lp = pyLHD.LeaveOneOut(x, b = 1, method = 'LP')
  x_lp
  ```
  ```{python}
  pyLHD.LqDistance(x_lp,q=1)
  ```
  Leave-one-out method using William's transformation
  ```{python}
  x_wt = pyLHD.LeaveOneOut(x, b = 1, method = 'WT')
  x_wt
  ```
  ```{python}
  pyLHD.LqDistance(x_wt,q=1)
  ```
  """
  if not isinstance(b,int):
    raise TypeError("'b' should be an integer")

  if b < 0 or b > (arr.shape[0]-1):
    raise ValueError(f"'b' should be within the range 0 to {arr.shape[0]-1}")
  # Apply Level Permutation or Williams Transformation
  if method == 'LP':
    new_arr = LevelPermutation(arr, b=b)
  elif method == 'WT':
    y = LevelPermutation(arr, b=b)
    new_arr = WilliamsTransform(y)
  else:
    raise ValueError("Invalid 'method' specified. Choose 'LP' or 'WT'")
  
  new_arr = new_arr[:-1, :]
  # Decrease levels greater than the deleted level by one
  new_arr[new_arr > b] -= 1
  return new_arr

def BestLinearPermutation(N:int) -> int:
  """Optimal linear permutation value to achieve larger L1-distance for a LHD

  Args:
      N (int): A prime integer

  Raises:
      ValueError:  If `N` is not a prime integer

  Returns:
      int: Optimal value of `b` to apply a linear level permutation and achieve higher $L_1$-distance. That is $D_b = D + b (mod \\, N)$
  """
  if not is_prime(N):
    raise ValueError("'N' must be a prime number")
  c_zero = int(np.sqrt((N**2 - 1)/12))
  def_poly = c_zero**2 + 2*((c_zero + 1)**2)
  
  if def_poly >= (N**2 - 1)/4:
    c = c_zero
  else :
    c = c_zero + 1
      
  y = (N-1)/2 + c
  if (y % 2) == 0: 
    b = int(y/2)
  else :
    b = int((2*N - y - 1)/2)
  return b 


def maximinLHD(size: tuple[int, int], h: list[int] = None,
               method: Literal['LP','WT'] = 'LP',
               seed: Optional[Union[int, np.random.Generator]] = None) -> npt.ArrayLike:
  """Generate a maximin LHD based on the L1-distance

  Args:
      size (tuple of ints): Output shape of (n,d), where `n` and `d` are the number of rows and columns, respectively.      
      h (list of ints): A generator vector used to multiply each row of the design. Each element in `h` must be smaller than and coprime to `n`    
      method (Literal[&#39;LP&#39;, &#39;WT&#39;], optional): Linear level permutation (LP) or William's transformation (WT). Defaults to 'LP'.
      seed (Optional[Union[int, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.
  Raises:
      ValueError: If `method` is not 'LP' or 'WT'

  Returns:
      npt.ArrayLike: A maximin LHD based on the L1-distance. Construction is obtained by applying Williams transformation on linearly permuted good lattice point (GLP) designs
  Example:
  ```{python}
  import pyLHD
  x = pyLHD.GoodLatticePoint(size = (11,10))
  pyLHD.LqDistance(x)
  ```
  ```{python}
  y = pyLHD.maximinLHD(size = (11,10), method = 'LP')
  pyLHD.LqDistance(y)
  ```
  ```{python}
  w = pyLHD.maximinLHD(size = (11,10), method = 'WT')
  pyLHD.LqDistance(w)
  ```

  """

  n,_ = size
  D =  GoodLatticePoint(size=size, h = h, seed=seed)

  def find_best_b(transform_func):
    max_L1 = float('-inf')
    best_b = -1
    for it in range(n):
      current_L1 = LqDistance(transform_func(LevelPermutation(D, b=it)))
      if current_L1 > max_L1:
        max_L1 = current_L1
        best_b = it
    return best_b  
  
  if is_prime(n):
    b = BestLinearPermutation(n)
  else:
    transform_func = lambda x: x if method == 'LP' else WilliamsTransform
    b = find_best_b(transform_func)
  if method == 'LP':
    return LevelPermutation(D, b=b)
  elif method == 'WT':
    return WilliamsTransform(LevelPermutation(D, b=b))
  else:
    raise ValueError("'method' must be either 'LP' or 'WT'")


def EquidistantLHD(N:int) -> npt.ArrayLike:
  """Generate an Equidistant Latin Hypercube

  Args:
      N (int): An odd integer

  Returns:
      npt.ArrayLike: Given an odd integer $N=(2m+1)$, return an $(m \\times m)$ equidistant LHD. 
          This design, is a cyclic Latin square, with each level occuring once in each row and once in each column.
          It is also a maximin distance LHD in terms of $L_1$-distance
  Example:
  ```{python}
  import pyLHD
  N = 11
  x = pyLHD.EquidistantLHD(N = N)
  x
  ```
  ```{python}
  pyLHD.pairwise_InterSite(x, q=1)
  ```
  ```{python}
  pyLHD.LqDistance(x, q=1)
  ```
  """
  m = (N-1)//2
  D = GoodLatticePoint(size = (N,N-1))
  w = WilliamsTransform(D, modified=True)//2
  return w[:m,:m]