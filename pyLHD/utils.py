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

# Exchange two random elements in a matrix

def exchange(arr: npt.ArrayLike, idx: int, type: str='col') -> npt.ArrayLike:
  """ Exchange two random elements in a matrix

  Args:
      arr (numpy.ndarray): A design matrix
      idx (int): A positive integer, which stands for the (idx) column or row of (arr) type (str, optional): 
          If type is 'col', two random elements will be exchanged within column (idx).
          If type is 'row', two random elements will be exchanged within row (idx). Defaults to 'col'.

  Returns:
      A new design matrix after the exchange
  
  Examples:
  Choose the first columns of `random_lhd` and exchange two randomly selected elements
  ```{python}
  import pyLHD
  random_lhd = pyLHD.rLHD(n_rows = 5, n_columns = 3)
  random_lhd
  ```
  Choose column 1 of random_lhd and exchange two randomly selected elements
  ```{python}
  pyLHD.exchange(random_lhd,idx=1,type='col')
  ```
  Choose the first row of random_lhd and exchange two randomly selected elements
  ```{python}
  pyLHD.exchange(random_lhd,idx=1,type='row')
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
      arr (numpy.ndarray): A design matrix
      baseline (int, optional): A integer, which defines the minimum value for each column of the matrix. Defaults to 1.

  Returns:
      After applying Williams transformation, a matrix whose sizes are the same as input matrix
  
  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.rLHD(n_rows=5,n_columns=3)
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
      arr (numpy.ndarray): A design matrix
      criteria (str, optional): Criteria to choose from. Defaults to 'phi_p'. 
          Options include 'phi_p','MaxProCriterion','AvgAbsCor','AvgAbsCor'
          p (int): A positive integer, which is the parameter in the phi_p formula. The default is set to be 15
          q (int): If (q) is 1, (dij) is the Manhattan (rectangular) distance. If (q) is 2, (dij) is the Euclidean distance.

  Returns:
      Calculation of chosen criteria for any LHD

  Examples:
  By default `phi_p` with `p=15` and `q=1`
  ```{python}
  import pyLHD
  random_lhd = pyLHD.rLHD(n_rows=5,n_columns=3)
  pyLHD.eval_design(random_lhd)
  ```
  Evaluate design based on MaxProCriterion 
  ```{python}
  pyLHD.eval_design(random_lhd,criteria='MaxProCriterion')
  ``` 
  """
  if criteria == 'phi_p':
    result = pyLHD.phi_p(arr,p=p,q=q)
  elif criteria == 'MaxProCriterion':
    result =  pyLHD.MaxProCriterion(arr)
  elif criteria == 'AvgAbsCor':
    result =  pyLHD.AvgAbsCor(arr)
  elif criteria == 'MaxAbsCor':
    result =  pyLHD.MaxAbsCor(arr)
  return result


# Adjust the range of a design to [min,max]

def adjust_range(arr: npt.ArrayLike, min: float, max: float, digits: int=None) -> npt.ArrayLike:
  """ Adjust the range of a design to [min,max]

  Args:
      arr (numpy.ndarray): A design matrix
      min (float): desired lower bound of design 
      max (float): desired upper bound of design 
      digits (int): number of digits to which the design is rounded
  Returns:
      Design with new range [min,max]

  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.rLHD(n_rows=5,n_columns=3,unit_cube=True)
  ```
  ```{python}
  pyLHD.adjust_range(random_lhd,-1, 1)
  ```
  ```{python}
  pyLHD.adjust_range(random_lhd, 5, 12, digits = 3)
  ```
  """  
  if (min == max):
    raise ValueError('min and max should be different values')
  if (min > max):
    raise ValueError('make sure min < max')
  
  arr_min = np.amin(arr)
  arr_max = np.amax(arr)
  
  range_diff = arr_max - arr_min
  result = (arr - arr_min)/range_diff * (max-min) + min

  if digits is not None:
    return np.around(result,digits)
  else:
    return result


def scale(arr: npt.ArrayLike,uniformize: bool =False) -> npt.ArrayLike:
  """ Scales design to be within [0,1]
  
  Args:
      arr (numpy.ndarray): A design matrix
      uniformize (bool): If True, Rosenblatt transformation is applied 
          (uniformize by applying the empirical cumulative distribution). 
          If False (default), scaling is done by using the minimum and maximum value
  Returns:
      The scaled design

  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.rLHD(n_rows=5,n_columns=3,unit_cube=True)
  pyLHD.scale(random_lhd)
  ```
  """  
  min = np.amin(arr,axis=0)
  range = np.ptp(arr, axis=0)
  
  if not uniformize:
    return (arr-min)/range
  else:
    return np.apply_along_axis(pyLHD.ecdf, 0, arr)


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
  random_lhd = pyLHD.rLHD(n_rows=5,n_columns=3,unit_cube=True)
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