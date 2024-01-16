import numpy as np
import numpy.typing as npt
from typing import Optional, Union
from numbers import Integral
from pyLHD.helpers import permute_columns, check_seed, totatives


def LatinSquare(size: tuple[int,int], baseline: int = 1, seed: Optional[Union[Integral, np.random.Generator]] = None) -> npt.ArrayLike:
  """ Generate a random (n x d) Latin square

  Args:
      size (tuple of ints): Output shape of (n,d), where `n` and `d` are the number of rows and columns, respectively.
      baseline (int, optional): A integer, which defines the minimum value for each column of the matrix. Defaults to 1.

      seed (Optional[Union[Integral, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.

  Returns:
      Generated random (n x d) Latin square, in which each column is a random permutation of {baseline,baseline+1, ..., baseline+(n-1)}
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.LatinSquare(size = (5,3),seed = 1)
  ```
  """
  rng = check_seed(seed)
  perms = np.tile(np.arange(start=baseline, stop= baseline + size[0]), (size[1], 1)).T
  perms = permute_columns(perms, seed=rng)

  return perms


def LatinHypercube(size: tuple[int, int], scramble: Optional[bool] = True,
                   seed: Optional[Union[Integral, np.random.Generator]] = None) -> npt.ArrayLike:
  """Generate a random Latin Hypercube Design

  Args:
      size (tuple of ints): Output shape of (n,d), where `n` and `d` are the number of rows and columns, respectively.      
      scramble (Optional[bool], optional): When False, center samples within cells of a multi-dimensional grid. 
          Otherwise, samples are randomly placed within cells of the grid. Defaults to True.
      seed (Optional[Union[Integral, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.
  Returns:
      A Latin hypercube sample of $n$ points generated in $[0,1)^d$ 
          Each univariate marginal distribution is stratisfied, placing exactly one point in 
          $[j/n,(j+1)/n)$ for $j=0,1,\dots,n-1$
  Examples:
  ```{python}
  import pyLHD
  pyLHD.LatinHypercube(size = (5,3),seed = 1)
  ```
  ```{python}
  pyLHD.LatinHypercube(size = (5,3), seed = 1, scramble = False)
  ```          
  """

  rng = check_seed(seed)
  perms = LatinSquare(size = size, seed=rng)
  samples = 0.5 if not scramble else rng.uniform()

  return (perms-samples)/size[0]


def GoodLatticePoint(N:int) -> npt.ArrayLike:
  """ Good Lattice Point (GLP) Design 

  Args:
      N (int): An integer specifying the number of rows

  Returns:
      A (N x euler_phi(N)) GLP design, where euler_phi(N) is the number of poositive integers that are less than and coprime to N
  
  Examples:
  ```{python}
  import pyLHD
  N = 5
  pyLHD.totatives(N)
  ```
  ```{python}
  pyLHD.GoodLatticePoint(N)
  ```
  ```{python}
  N = 11
  pyLHD.GoodLatticePoint(N)
  ```
  """  
  h = totatives(N)
  row_indices = np.arange(1, N + 1).reshape(-1, 1)
  return  (row_indices * h) % N