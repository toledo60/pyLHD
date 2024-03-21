import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Literal
from pyLHD.helpers import permute_columns, check_seed, totatives, verify_generator
from pyLHD.criteria import discrepancy


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
      size (tuple of ints): Output shape of $(n,d)$, where $n$ and $d$ are the number of rows and columns, respectively.
      scramble (Optional[bool], optional): When False, center samples within cells of a multi-dimensional grid. 
          Otherwise, samples are randomly placed within cells of the grid. Defaults to True.
      seed (Optional[Union[int, np.random.Generator]]) : If `seed` is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed).
          If `seed` is already a `Generator` instance, then the provided instance is used. Defaults to None.

  Returns:
      A Latin hypercube sample of $n$ points generated in $[0,1)^d$
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
      size (tuple of ints): Output shape of $(n,d)$, where `n` and `d` are the number of rows and columns, respectively
      h (list of ints): A generator vector used to multiply each row of the design. Each element in `h` must be smaller than and coprime to `n`    
      seed (Optional[Union[int, np.random.Generator]]): If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.
  
  Returns:
      Generated random $(n x d)$ Good lattice point set, where each column is a random permutation of $\\{0,1, \\dots ,n-1 \\}$
  
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



def UniformDesign(size:tuple[int,int], n_levels:int, n_iters:int = 100, 
                  criteria: Literal["L2", "L2_star", "centered_L2", "modified_L2","balanced_centered_L2",
                                    "mixture_L2", "symmetric_L2", "wrap_around_L2"] = "centered_L2") -> np.ndarray:
  """Generate a Uniform Design (U-type)

  Args:
      size (tuple of ints): Output shape of (n,d), where `n` and `d` are the number of rows and columns, respectively
      n_levels (int): number of levels in each column for the uniform design
      n_iters (int, optional): Maximium iterations to optimize specified criteria. Defaults to 100.
      criteria (str, optional): Type of discrepancy. Defaults to 'centered_L2'. Options include: 'L2', 'L2_star','centered_L2', 'modified_L2', 'mixture_L2', 'symmetric_L2', 'wrap_around_L2'

  Raises:
      ValueError: `n` should be a multiple of the `n_levels`

  Returns:
      np.ndarray: A U-type uniform design, $U(n,q^d)$ where $n$ is the number of rows, $q$ are the numbe of levels, and $d$ is the dimension of the uniform design.
  Examples:
  ```{python}
  import pyLHD
  pyLHD.UniformDesign(size = (10,3), n_levels = 2)
  ```
  ```{python}
  pyLHD.UniformDesign(size = (10,3), n_levels = 5, criteria = "mixture_L2")  
  ```
  """
  n, d = size
  if n % n_levels != 0:
    raise ValueError("n must be divisible by `n_levels` to ensure equal frequency of levels.")
  
  best_design = None
  best_score = np.inf
  
  for _ in range(n_iters):
    repetitions_per_level = n // n_levels
    base_design = np.tile(np.arange(0, n_levels), repetitions_per_level)
    design_matrix = np.zeros((n, d), dtype=int)
    
    for i in range(d):
      np.random.shuffle(base_design)
      design_matrix[:, i] = base_design
    
    score = discrepancy(design_matrix, method = criteria)
    if score < best_score:
      best_design = design_matrix
      best_score = score
  
  return best_design