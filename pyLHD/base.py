import numpy as np
import numpy.typing as npt
from typing import Optional, Union
from pyLHD.helpers import permute_columns, check_seed, totatives, verify_generator

def LatinSquare(size: tuple[int,int], baseline: int = 1, seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
  """ Generate a random (n x d) Latin square

  Args:
      size (tuple of ints): Output shape of (n,d), where `n` and `d` are the number of rows and columns, respectively.
      baseline (int, optional): A integer, which defines the minimum value for each column of the matrix. Defaults to 1.

      seed (Optional[Union[int, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.

  Returns:
      Generated random (n x d) Latin square, in which each column is a random permutation of {baseline,baseline+1, ..., baseline+(n-1)}
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.LatinSquare(size = (5,5),seed = 1)
  ```
  """
  n,d = size
  if n != d:
    raise ValueError("'size' should be a square, i.e, n=d")
  rng = check_seed(seed)
  perms = np.tile(np.arange(start=baseline, stop= baseline + n), (d, 1)).T
  return permute_columns(perms, seed=rng)


def LatinHypercube(size: tuple[int, int], scramble: Optional[bool] = True,
                   seed: Optional[Union[int, np.random.Generator]] = None) -> npt.ArrayLike:
  """Generate a random Latin Hypercube Design

  Args:
      size (tuple of ints): Output shape of (n,d), where `n` and `d` are the number of rows and columns, respectively.      
      scramble (Optional[bool], optional): When False, center samples within cells of a multi-dimensional grid. 
          Otherwise, samples are randomly placed within cells of the grid. Defaults to True.
      seed (Optional[Union[int, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
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
  n,d = size
  rng = check_seed(seed)

  perms = np.tile(np.arange(start=1, stop= 1 + n), (d, 1)).T
  perms = permute_columns(perms, seed=rng)
  samples = 0.5 if not scramble else rng.uniform()
  return (perms-samples)/n


def GoodLatticePoint(size: tuple[int, int], h: list[int] = None,
                     seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
  """ Good Lattice Point (GLP) Design 

  Args:
      size (tuple of ints): Output shape of (n,d), where `n` and `d` are the number of rows and columns, respectively
      h (list of ints): A generator vector used to multiply each row of the design. Each element in `h` must be smaller than and coprime to `n`    
      seed (Optional[Union[int, np.random.Generator]]): If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.
  
  Returns:
      Generated random (n x d) Good lattice point set, where each column is a random permutation of {0,1,...,n-1}
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.GoodLatticePoint(size = (10,4))
  ```
  ```{python}
  pyLHD.GoodLatticePoint(size = (10,3),seed = 1)
  ```
  """  
  n,d = size
  if d >= n:
    raise ValueError('d must be less than n. Recall, size = (n,d)')
  
  if h is not None:
    h = verify_generator(h,n,d)
  else:
    h = totatives(n)  
    if len(h) != d:
      rng = check_seed(seed)
      h = rng.choice(h,d,replace = False)
  
  row_indices = np.arange(1, n + 1).reshape(-1, 1)
  return  (row_indices * h) % n