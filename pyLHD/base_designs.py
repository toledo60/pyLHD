import numpy as np
import numpy.typing as npt
from typing import Optional
from pyLHD.utils import permute_columns

# Generate a random Latin Hypercube Design (LHD)

def rLHD(n_rows: int, n_columns: int, unit_cube: bool = False) -> npt.ArrayLike:
  """ Generate a random Latin Hypercube Design (LHD)

  Args:
      n_rows (int): A positive integer specifying the number of rows
      n_columns (int): A postive integer specifying the number of columns
      unit_cube (bool): If True, design will be in the unit cube [0,1]^n_columns.

  Returns:
      return a random (n_rows by n_columns) LHD
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.rLHD(n_rows=5,n_columns = 4,unit_cube = False)
  ```
  ```{python}
  pyLHD.rLHD(n_rows=5,n_columns = 4, unit_cube = True)
  ```
  """
  rng = np.random.default_rng()
  rs = np.arange(start=1, stop=n_rows+1)
  space = []
  for _ in range(n_columns):
    space.append(rng.choice(rs, n_rows, replace=False))
  D = np.asarray(space).transpose()
  
  if unit_cube:
    return (D-0.5)/n_rows
  else:
    return D


def random_lhd(n_rows: int, n_columns: int, scramble: Optional[bool] = True,
               seed: Optional[int] = None) -> npt.ArrayLike:
  """Generate a random Latin Hypercube Design

  Args:
      n_rows (int): number of rows (number of samples)
      n_columns (int): number of columns (dimnesion of parameter space)
      scramble (Optional[bool], optional): When False, center samples within cells of a multi-dimensional grid. 
          Otherwise, samples are randomly placed within cells of the grid. Defaults to True.
      seed (Optional[int], optional): If seed is an int or None, a new numpy.random.Generator is created using np.random.default_rng(seed). Defaults to None.

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
             
  rng = np.random.default_rng(seed)
  perms = np.tile(np.arange(start=1, stop=n_rows+1), (n_columns, 1)).T
  perms = permute_columns(perms, seed=seed)

  samples = 0.5 if not scramble else rng.uniform()
  return (perms-samples)/n_rows


# Good Lattice Point Design

def GLPdesign(n_rows: int,n_columns: int, h: list = None) -> npt.ArrayLike:
  """ Good Lattice Point (GLP) Design 

  Args:
      n_rows (int): A positive integer specifying the number of rows
      n_columns (int): A postive integer specifying the number of columns
          h (list, optional): A list whose length is same as (n_columns), with its elements that are smaller than and coprime to (n_rows). 
          Defaults to None. If None, a random sample of (n_columns) elements between 1 and (n_rows-1).

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
  rng = np.random.default_rng()
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
