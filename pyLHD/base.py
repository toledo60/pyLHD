import numpy.typing as npt
from typing import Optional, Union
from numbers import Integral
from pyLHD.helpers import permute_columns, check_seed
import numpy as np


def LatinSquare(size: tuple[int,int], seed: Optional[Union[Integral, np.random.Generator]] = None) -> npt.ArrayLike:
  """ Generate a (n x d) Latin square, where each column is a random permutation of {1,2,...,n}

  Args:
      size (tuple of ints): Output shape of (n,d), where `n` and `d` are the number of rows and columns, respectively.
      seed (Optional[Union[Integral, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.

  Returns:
      Generate (n x d) matrix, in which each column is a random permutation of {1,2,...,n}
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.LatinSquare(size = (5,3),seed = 1)
  ```
  """
  rng = check_seed(seed)
  perms = np.tile(np.arange(start=1, stop=size[0]+1), (size[1], 1)).T
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


def GoodLatticePoint(size: tuple[int,int], h: list = None,
                     seed: Optional[Union[Integral, np.random.Generator]] = None) -> npt.ArrayLike:
  """ Good Lattice Point (GLP) Design 

  Args:
      size (tuple of ints): Output shape of (n,d), where `n` and `d` are the number of rows and columns, respectively.
      h (list, optional): A list whose length is same as `d`, with its elements that are smaller than and coprime to `n`. 
          Defaults to None. If None, a random sample of `d` elements between 1 and (`n`-1).
      seed (Optional[Union[Integral, np.random.Generator]]): If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.

  Returns:
      A (n x d) GLP design.
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.GoodLatticePoint(size = (5,3))
  ```
  ```{python}
  pyLHD.GoodLatticePoint(size = (8,4),h=[1,3,5,7])
  ```
  """
  rng = check_seed(seed)
  n_rows, n_columns = size
  if h is None:
    seq = np.arange(start=1, stop=n_rows)
    h_sample = rng.choice(seq, n_columns, replace=False)
  else:
    h_sample = rng.choice(h, n_columns, replace=False)

  mat = np.zeros((n_rows, n_columns))

  for i in range(n_rows):
    for j in range(n_columns):
      mat[i, j] = ((i+1)*h_sample[j]) % n_rows
  return mat.astype(int)
