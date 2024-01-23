import numpy.typing as npt
from typing import Literal
from pyLHD.helpers import LevelPermutation, WilliamsTransform

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