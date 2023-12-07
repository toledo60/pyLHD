import numpy.typing as npt
from typing import Optional, Union
from numbers import Integral
from pyLHD.helpers import permute_columns, check_seed
import numpy as np


def permutations_matrix(n_rows: int, n_columns: int, seed: Optional[Union[Integral, np.random.Generator]] = None) -> npt.ArrayLike:
  """ Generate (n_rows x n_columns) matrix, in which each column is a random permutation of {1,2,...,n_rows}

  Args:
      n_rows (int): number of rows
      n_columns (int): number of columns
      seed (Optional[Union[Integral, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.

  Returns:
      Generate (n_rows x n_columns) matrix, in which each column is a random permutation of {1,2,...,n_rows}
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.permutations_matrix(n_rows = 6, n_columns = 3, seed = 1)
  ```
  """
  rng = check_seed(seed)
  perms = np.tile(np.arange(start=1, stop=n_rows+1), (n_columns, 1)).T
  perms = permute_columns(perms, seed=rng)

  return perms


def random_lhd(n_rows: int, n_columns: int, scramble: Optional[bool] = True,
               seed: Optional[Union[Integral, np.random.Generator]] = None) -> npt.ArrayLike:
  """Generate a random Latin Hypercube Design

  Args:
      n_rows (int): number of rows
      n_columns (int): number of columns (dimnesion of parameter space)
      scramble (Optional[bool], optional): When False, center samples within cells of a multi-dimensional grid. 
          Otherwise, samples are randomly placed within cells of the grid. Defaults to True.
      seed (Optional[Union[Integral, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.
  Returns:
      A Latin hypercube sample of $n =$ `n_rows` points generated in $[0,1)^d$, where $d$=`n_columns`. 
          Each univariate marginal distribution is stratisfied, placing exactly one point in 
          $[j/n,(j+1)/n)$ for $j=0,1,\dots,n-1$
  Examples:
  ```{python}
  import pyLHD
  pyLHD.random_lhd(n_rows = 5, n_columns = 3, seed = 1)
  ```
  ```{python}
  pyLHD.random_lhd(n_rows = 5, n_columns = 3, seed = 1, scramble = False)
  ```          
  """
             
  rng = check_seed(seed)
  perms = permutations_matrix(n_rows= n_rows, n_columns= n_columns, seed = rng)
  samples = 0.5 if not scramble else rng.uniform()

  return (perms-samples)/n_rows


def GLPdesign(n_rows: int, n_columns: int, h: list = None,
              seed: Optional[Union[Integral, np.random.Generator]] = None) -> npt.ArrayLike:
  """ Good Lattice Point (GLP) Design 

  Args:
      n_rows (int): A positive integer specifying the number of rows
      n_columns (int): A postive integer specifying the number of columns
          h (list, optional): A list whose length is same as (n_columns), with its elements that are smaller than and coprime to (n_rows). 
          Defaults to None. If None, a random sample of (n_columns) elements between 1 and (n_rows-1).
      seed (Optional[Union[Integral, np.random.Generator]]): If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.

  Returns:
      A (n_rows by n_columns) GLP design.
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.GLPdesign(n_rows=5,n_columns=3)
  ```
  ```{python}
  pyLHD.GLPdesign(n_rows=8,n_columns=4,h=[1,3,5,7])
  ```
  """
  rng = check_seed(seed)
  if h is None:
    seq = np.arange(start=1,stop=n_rows)
    h_sample = rng.choice(seq,n_columns,replace=False)
  else:
    h_sample = rng.choice(h,n_columns,replace=False)
  
  mat = np.zeros((n_rows,n_columns))

  for i in range(n_rows):
    for j in range(n_columns):
      mat[i,j] = ((i+1)*h_sample[j])% n_rows
  return mat.astype(int)