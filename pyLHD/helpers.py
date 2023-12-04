import pyLHD
import math
import numpy as np
from collections import OrderedDict
import numpy.typing as npt
from typing import Optional, List


def permute_columns(arr: npt.ArrayLike, columns: Optional[List[int]] = None,
                    seed: Optional[int] = None) -> npt.ArrayLike:
  """Random permute columns in a numpy ndarray

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      columns (Optional[List[int]], optional): If columns is None all columns will be randomly permuted, otherwise provide a list of columns to permute. Defaults to None.
      seed (Optional[int], optional): If seed is an int or None, a new numpy.random.Generator is created using np.random.default_rng(seed). Defaults to None.

  Returns:
      numpy ndarray with columns of choice randomly permuted 
  
  Examples:
  ```{python}
  import pyLHD
  x = pyLHD.random_lhd(n_rows = 5, n_columns = 3, seed = 1)
  x
  ```
  Permute all columns
  ```{python}
  pyLHD.permute_columns(x)
  ```
  Permute columns [0,1] with `seed=1`
  ```{python}
  pyLHD.permute_columns(x, columns = [0,1], seed = 1)
  ```
  """

  rng = np.random.default_rng(seed)

  if columns is not None:
    for i in columns:
      rng.shuffle(arr[:, i])
  else:
    n_rows, n_columns = arr.shape
    ix_i = rng.random((n_rows, n_columns)).argsort(axis=0)
    ix_j = np.tile(np.arange(n_columns), (n_rows, 1))
    arr = arr[ix_i, ix_j]

  return arr


def swap_elements(arr: npt.ArrayLike, idx: int, type: str='col') -> npt.ArrayLike:
  """ Swap two random elements in a matrix

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      idx (int): A positive integer, which stands for the (idx) column or row of (arr) type (str, optional): 
          If type is 'col', two random elements will be exchanged within column (idx).
          If type is 'row', two random elements will be exchanged within row (idx). Defaults to 'col'.

  Returns:
      A new design matrix after the swap of elements
  
  Examples:
  Choose the first columns of `random_lhd` and swap two randomly selected elements
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows = 5, n_columns = 3)
  random_lhd
  ```
  Choose column 1 of random_lhd and swap two randomly selected elements
  ```{python}
  pyLHD.swap_elements(random_lhd,idx=1,type='col')
  ```
  Choose the first row of random_lhd and swap two randomly selected elements
  ```{python}
  pyLHD.swap_elements(random_lhd,idx=1,type='row')
  ```
  """
  n_rows = arr.shape[0]
  n_columns = arr.shape[1]
  rng = np.random.default_rng()

  if type == 'col':
    location = rng.choice(n_rows, 2, replace=False)

    arr[location[0], idx], arr[location[1],idx] = arr[location[1], idx], arr[location[0], idx]
  else:
    location = rng.choice(n_columns, 2, replace=False)
    arr[idx, location[0]], arr[idx, location[1]] = arr[idx, location[1]], arr[idx, location[0]]

  return arr


# Williams Transformation

def williams_transform(arr: npt.ArrayLike,baseline: int =1) -> npt.ArrayLike:
  """ Williams Transformation

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      baseline (int, optional): A integer, which defines the minimum value for each column of the matrix. Defaults to 1.

  Returns:
      After applying Williams transformation, a matrix whose sizes are the same as input matrix
  
  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=5,n_columns=3)
  random_lhd
  ```
  Change the baseline
  ```{python}
  pyLHD.williams_transform(random_lhd,baseline=3)
  ```
  """
  n = arr.shape[0]
  k = arr.shape[1]

  elements = np.unique(arr[:,0])
  min_elements = np.amin(elements)

  if min_elements != 0:
    arr = arr-min_elements
  
  wt = arr

  for i in range(n):
    for j in range(k):
      if arr[i,j] < (n/2):
        wt[i,j] = 2*arr[i,j]+1
      else:
        wt[i,j] = 2* (n-arr[i,j])
  
  if baseline !=1:
    wt = wt+ (baseline-1)
  return wt


# Convert an orthogonal array to LHD

def OA2LHD(arr: npt.ArrayLike) -> npt.ArrayLike:
  """ Transform an Orthogonal Array (OA) into an LHD

  Args:
      arr (numpy.ndarray): An orthogonal array matrix

  Returns:
      LHD whose sizes are the same as input OA. The assumption is that the elements of OAs must be positive
  
  Examples:
  First create an OA(9,2,3,2)
  ```{python}
  import numpy as np
  example_OA = np.array([[1,1],[1,2],[1,3],[2,1],
                         [2,2],[2,3],[3,1],[3,2],[3,3] ])
  ```
  Transform the "OA" above into a LHD according to Tang (1993)
  ```{python}
  import pyLHD
  pyLHD.OA2LHD(example_OA)      
  ```  
  """
  n = arr.shape[0]
  m = arr.shape[1]
  s = np.unique(arr[:,0]).size

  lhd = arr
  k = np.zeros((s,int(n/s),1))
  rng = np.random.default_rng()
  for j in range(m):
    for i in range(s): 
      k[i] = np.arange(start=i*int(n/s) + 1,stop=i*int(n/s)+int(n/s)+1).reshape(-1,1)
      k[i] = rng.choice(k[i],s,replace=False)*100
      np.place(lhd[:, j], lhd[:, j]== (i+1), k[i].flatten().tolist())
  lhd = lhd/100
  return lhd.astype(int)

# Evaluate design based on chosed criteria

def eval_design(arr: npt.ArrayLike, criteria: str = 'phi_p',p: int = 15,q: int = 1) -> float:
  """ Evaluate a design based on a chosen criteria, a simple wrapper for all `criteria` in `pyLHD`

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      criteria (str, optional): Criteria to choose from. Defaults to 'phi_p'. 
          Options include 'phi_p','MaxProCriterion','AvgAbsCor','AvgAbsCor'
          p (int): A positive integer, which is the parameter in the phi_p formula. The default is set to be 15
          q (int): If (q) is 1, (inter_site) is the Manhattan (rectangular) distance. If (q) is 2, (inter_site) is the Euclidean distance.

  Returns:
      Calculation of chosen criteria for any LHD

  Examples:
  By default `phi_p` with `p=15` and `q=1`
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=5,n_columns=3)
  pyLHD.eval_design(random_lhd)
  ```
  Evaluate design based on MaxProCriterion 
  ```{python}
  pyLHD.eval_design(random_lhd,criteria='MaxProCriterion')
  ``` 
  """
  criteria_functions = {
    'MaxProCriterion': pyLHD.MaxProCriterion,
    'AvgAbsCor': pyLHD.AvgAbsCor,
    'MaxAbsCor': pyLHD.MaxAbsCor
  }
    
  if criteria in criteria_functions:
    return criteria_functions[criteria](arr)
  elif criteria == 'phi_p':
    return pyLHD.phi_p(arr, p=p, q=q)
  else:
    raise ValueError(f"Invalid criteria: {criteria}")


def check_bounds(arr: npt.ArrayLike,
                 lower_bounds: npt.ArrayLike, 
                 upper_bounds: npt.ArrayLike) -> tuple[npt.ArrayLike, ...]:
  """ Check conditions for bounds input

  Args:
      arr (npt.ArrayLike): A numpy ndarray

      lower_bounds (npt.ArrayLike): Lower bounds of data
      upper_bounds (npt.ArrayLike): Upper bounds of data

  Raises:
      ValueError: If lower, upper bounds are not same dimension of sample `arr`
      ValueError: Whenver any of the lower bounds are greater than any of the upper bounds

  Returns:
      tuple[npt.ArrayLike, ...]: A tuple of numpy.ndarrays 
  """
  d = arr.shape[1]

  try:
    lower_bounds = np.broadcast_to(lower_bounds, d)
    upper_bounds = np.broadcast_to(upper_bounds, d)
  except ValueError as exc:
    msg = ("'lower_bounds' and 'upper_bounds' must be broadcastable and respect"
    " the sample dimension")
    raise ValueError(msg) from exc
  
  if not np.all(lower_bounds < upper_bounds):
    raise ValueError("Make sure all 'lower_bounds < upper_bounds'")
  
  return lower_bounds, upper_bounds


def scale(arr: npt.ArrayLike, lower_bounds: list, upper_bounds: list) -> npt.ArrayLike:
  """Sample scaling from unit hypercube to different bounds

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      lower_bounds (list): Lower bounds of transformed data
      upper_bounds (list): Upper bounds of transformed data

  Returns:
      npt.ArrayLike: Scaled numpy ndarray to [lower_bounds, upper_bounds]
  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(10,2, seed = 1)
  random_lhd
  ```
  ```{python}
  lower_bounds = [-3,2]
  upper_bounds = [10,4]
  pyLHD.scale(random_lhd,lower_bounds, upper_bounds)
  ```
  """
  lb, ub = check_bounds(arr, lower_bounds, upper_bounds)
  return lb + arr * (ub - lb)


# Distance Matrix Computation

def distance_matrix(arr: npt.ArrayLike, metric: str = 'euclidean', p: int = 2) -> npt.ArrayLike:
  """ Distance matrix based on specified distance measure

  Args:
      arr (numpy.ndarray): A design matrix
      metric (str, optional): Specifiy the following distance measure: 
          'euclidean': Usual distance between the two vectors (L_2 norm)
          'maximum': Maximum distance between two components of x and y (supremum norm)
          'manhattan': Absolute distance between the two vectors (L_1 norm)
          'minkowski': The p norm, the pth root of the sum of the pth powers of the differences of the components
      
      p (int, optional): The power of the Minkowski distance. Defaults to 2.

  Returns:
      The calculated distance matrix baed on specified distance measure
  
  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=5,n_columns=3)
  pyLHD.distance_matrix(random_lhd)
  ```
  ```{python}
  pyLHD.distance_matrix(random_lhd, metric = 'manhattan')
  ```
  ```{python}
  pyLHD.distance_matrix(random_lhd, metric = 'minkowski', p=5)
  ```
  """
  
  if metric == 'euclidean' and p ==2:
    dist = lambda p1, p2: np.sqrt(((p1-p2)**2).sum())
  elif metric == 'manhattan':
    dist = lambda p1,p2: np.abs(p1-p2).sum()
  elif metric == 'minkowski':
    dist = lambda p1, p2: ((np.abs(p1-p2)**(p)).sum())**(1/p)
  elif metric == 'maximum':
    dist = lambda p1,p2: np.max(np.abs(p1-p2)).sum()
  
  return np.asarray([[dist(p1, p2) for p2 in arr] for p1 in arr])


# Determine if a number is prime

def is_prime(n):
  """Determine if a number is prime

  Args:
      n (int): Any integer

  Returns:
      [logical]: True if n is prime, False if n is not prime 
  """
  if n % 2 == 0 and n > 2:
    return False
  return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


# Compute n choose k 

def comb(n,k):
  f = math.factorial
  return f(n) // f(k)// f(n-k)

# compute ecdf

def ecdf(x):
  ys = np.arange(1, len(x)+1)/float(len(x))
  d = dict(zip(np.argsort(x), ys.tolist()))
  l = OrderedDict(sorted(d.items(), key=lambda t: t[0])).values()
  return list(l)