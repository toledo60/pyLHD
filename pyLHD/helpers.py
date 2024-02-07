import math
import numpy as np
import numpy.typing as npt
from typing import Optional, List, Union, NoReturn
from itertools import combinations

def distance_matrix(arr: npt.ArrayLike, metric: str = 'euclidean', p: int = 2) -> np.ndarray:
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
  random_lhd = pyLHD.LatinHypercube(size = (5,3))
  pyLHD.distance_matrix(random_lhd)
  ```
  ```{python}
  pyLHD.distance_matrix(random_lhd, metric = 'manhattan')
  ```
  ```{python}
  pyLHD.distance_matrix(random_lhd, metric = 'minkowski', p=5)
  ```
  """
  p1 = arr[:, np.newaxis]
  p2 = arr[np.newaxis,:]

  metrics = {
      'euclidean': np.linalg.norm(p1 - p2, axis=-1),
      'manhattan': np.sum(np.abs(p1 - p2), axis=-1),
      'minkowski': np.sum(np.abs(p1 - p2)**p, axis=-1)**(1/p),
      'maximum': np.amax(np.abs(p1 - p2), axis=-1)
  }

  return metrics[metric]  


def replace_values(arr: npt.ArrayLike, mapping: dict) -> np.ndarray:
  """
  Replace values in a numpy array based on a provided mapping dictionary

  Args:
      arr (npt.ArrayLike): A numpy array with values to be replaced.
      mapping (dict): A dictionary where keys correspond to values in `arr` and values are the replacement values.

  Returns:
      A numpy array with replaced values.

  Raises:
      ValueError: If `mapping` does not contain the same unique values as in `arr`, or if the keys do not match.
  
  Examples:
  
  ```{python}
  import pyLHD
  random_ls = pyLHD.LatinSquare(size = (4,4), seed = 1)
  random_ls
  ```
  Consider the mapping $1 \\rightarrow 2, 2 \\rightarrow 11, 3 \\rightarrow 12, 4 \\rightarrow 13$
  ```{python}
  mapping = {1:10, 2:11, 3:12, 4:13}
  pyLHD.replace_values(random_ls, mapping = mapping)
  ```

  """  
  unique_values_set = set(np.unique(arr))
  mapping_keys_set = set(mapping.keys())

  if unique_values_set != mapping_keys_set:
    missing_keys = unique_values_set - mapping_keys_set
    extra_keys = mapping_keys_set - unique_values_set
    error_message = []
    if missing_keys:
      error_message.append(f"Missing keys in mapping: {missing_keys}")
    if extra_keys:
      error_message.append(f"Extra keys in mapping: {extra_keys}")
    raise ValueError(' '.join(error_message))

  return np.vectorize(mapping.get)(arr)


def permute_columns(arr: npt.ArrayLike, columns: Optional[List[int]] = None,
                    seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
  """Randomly permute columns in a numpy ndarray

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      columns (Optional[List[int]], optional): If columns is None all columns will be randomly permuted, otherwise provide a list of columns to permute. Defaults to None.
      seed (Optional[Union[int, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.
  Returns:
      numpy ndarray with columns of choice randomly permuted 
  
  Examples:
  ```{python}
  import pyLHD
  x = pyLHD.LatinHypercube(size = (5,3), seed = 1)
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

  rng = check_seed(seed)

  if columns is not None:
    for i in columns:
      rng.shuffle(arr[:, i])
  else:
    n_rows, n_columns = arr.shape
    ix_i = rng.random((n_rows, n_columns)).argsort(axis=0)
    ix_j = np.tile(np.arange(n_columns), (n_rows, 1))
    arr = arr[ix_i, ix_j]

  return arr


def permute_rows(arr: npt.ArrayLike, rows: Optional[List[int]] = None,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
  """Randomly permute rows in a numpy ndarray

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      rows (Optional[List[int]], optional): If `rows` is None all columns will be randomly permuted, otherwise provide a list of rows to permute. Defaults to None.
      seed (Optional[Union[int, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.
  Returns:
      numpy ndarray with rows of choice randomly permuted 
  
  Examples:
  ```{python}
  import pyLHD
  x = pyLHD.LatinHypercube(size = (5,3), seed = 1)
  x
  ```
  Permute all columns
  ```{python}
  pyLHD.permute_rows(x)
  ```
  Permute columns [0,1] with `seed=1`
  ```{python}
  pyLHD.permute_rows(x, rows = [0,1], seed = 1)
  ```
  """                 
  rng = check_seed(seed)

  if rows is not None:
    for i in rows:
      rng.shuffle(arr[i, :])
  else:
    n_rows, n_columns = arr.shape
    ix_i = np.tile(np.arange(n_rows), (n_columns, 1)).T
    ix_j = rng.random((n_rows, n_columns)).argsort(axis=1)
    arr = arr[ix_i, ix_j]

  return arr


def swap_elements(arr: npt.ArrayLike, idx: int, type: str = 'col',
                  seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
  """ Swap two random elements in a matrix

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      idx (int): A positive integer, which stands for the (idx) column or row of (arr) type (str, optional): 
          If type is 'col', two random elements will be exchanged within column (idx).
          If type is 'row', two random elements will be exchanged within row (idx). Defaults to 'col'.
      seed (Optional[Union[int, np.random.Generator]]) : If `seed`is an integer or None, a new numpy.random.Generator is created using np.random.default_rng(seed). 
          If `seed` is already a ``Generator` instance, then the provided instance is used. Defaults to None.

  Returns:
      A new design matrix after the swap of elements
  
  Examples:
  Choose the first columns of `random_lhd` and swap two randomly selected elements
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (5,3))
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
  n_rows, n_columns = arr.shape
  rng = check_seed(seed)

  if type == 'col':
    location = rng.choice(n_rows, 2, replace=False)
    arr[location[0], idx], arr[location[1],idx] = arr[location[1], idx], arr[location[0], idx]
  else:
    location = rng.choice(n_columns, 2, replace=False)
    arr[idx, location[0]], arr[idx, location[1]] = arr[idx, location[1]], arr[idx, location[0]]

  return arr


def column_combinations(arr: npt.ArrayLike, k:int) -> List[npt.ArrayLike]:
  """
  Generates all unique combinations of columns from the given array, selecting 'k' columns at a time.

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      k (int): The number of columns to include in each combination. Must be a positive integer and less than or equal to the number of columns in 'arr'.

  Returns:
      List[npt.ArrayLike]: A list of arrays, each being a combination of 'k' columns from the original array. The combinations are returned as slices of the original array, not copies.
  
  Examples:

  ```{python}
  import pyLHD
  random_ls = pyLHD.LatinSquare(size = (4,4),seed = 1)
  random_ls
  ```
  Obtain all 2 column combinations of `random_ls`
  ```{python}
  pyLHD.column_combinations(random_ls, k = 2)
  ```

  """
  n_columns = arr.shape[1]
  if k <=0 or k > n_columns:
    raise ValueError(" `k` must be a positive integer and less than or equal to the number of columns in 'arr'")
  column_combinations = combinations(range(n_columns),k)
  return [arr[:,[i,j]] for i,j in column_combinations]


def scale(arr: npt.ArrayLike, lower_bounds: list, upper_bounds: list, as_integers: bool = False) -> np.ndarray:
  """Sample scaling from unit hypercube to different bounds

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      lower_bounds (list): Lower bounds of transformed data
      upper_bounds (list): Upper bounds of transformed data
      as_integers (bool): Should scale design to integer values on specified bounds. Defaults to False.s

  Returns:
      Scaled numpy ndarray to [lower_bounds, upper_bounds]
  Examples:
  ```{python}
  import pyLHD
  sample = pyLHD.LatinHypercube(size = (10,2), seed = 1)
  sample
  ```
  ```{python}
  lower_bounds = [-3,2]
  upper_bounds = [10,4]
  pyLHD.scale(sample,lower_bounds, upper_bounds)
  ```
  ```{python}
  pyLHD.scale(sample,lower_bounds, upper_bounds, as_integers = True)
  ```
  """
  lb, ub = check_bounds(arr, lower_bounds, upper_bounds)
  scaled = lb + arr * (ub - lb)
  
  if not as_integers:
    return scaled
  else:
    return np.floor(scaled).astype(np.int64)
  

####################################
#### GoodLatticePonint Helpers #####
####################################

def zero_base(arr: npt.ArrayLike) -> np.ndarray:
  """ Normalize the columns by subtracting the minimum element of each column

  Args:
      arr (npt.ArrayLike): A numpy ndarray

  Returns:
      A normalized array such that the columns are subtracted by the minimum element of each column
  
  Example:
  ```{python}
  import pyLHD
  x = pyLHD.LatinSquare(size = (5,5), baseline = 3, seed = 1)
  x
  ```
  ```{python}
  pyLHD.zero_base(x)
  ```

  """
  x = arr.copy()
  min_elements = np.amin(x, axis=0)
  x -= min_elements
  return x


def level_permutation(arr: npt.ArrayLike, b: Union[int,list], modulus:int = None) -> np.ndarray:
  """Apply level permutations to a Good lattice point (GLP) design

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      b (Union[int,list]): Value by which each element in the array is to be level permuted. Can either be an integer or a list of integers
      modulus (int): Modulus used for the permutation. Defaults to None. If None, the number of rows is used as the modulus.

  Returns:
      npt.ArrayLike: A new array where each element is the result of `(arr + b) % modulus`
  Examples:
  ```{python}
  import pyLHD
  GLP = pyLHD.GoodLatticePoint(size = (10, pyLHD.euler_phi(10)))
  GLP
  ```
  Apply a simple linear level permutation in the form of $D = D+b (mod \,N)$
  ```{python}
  pyLHD.level_permutation(GLP,b = 2)
  ```
  ```{python}
  pyLHD.level_permutation(GLP, b = [1,4,3,2])
  ```
  """
  n_rows, n_columns = arr.shape
  if modulus is None:
    modulus = n_rows
  
  if isinstance(b,int):
    return (arr + b)%modulus
  elif isinstance(b, list):
    permuted_arr = arr.copy()
    for i in range(n_columns):
      permuted_arr[:,i] = (permuted_arr[:,i] + b[i]) % modulus
    return permuted_arr
  else:
    raise ValueError("'b' should either be an integer or list of integers")


def totatives(N:int) -> List[int]:
  """
  Generate all positive integers less than and coprime to N from [1,N)

  Args:
      N (int): The number to find coprimes for

  Returns:
      List[int]: A list of integers from [1,N) that are coprime to N
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.totatives(11)
  ```
  """
  return [i for i in range(1, N + 1) if are_coprime(i, N)]


def euler_phi(N:int) -> int:
  """
  Euler's Totient function

  Args:
      N (int): The number to find coprimes for

  Returns:
      (int): The number of positive integers from [1,N), less than and coprime to N 
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.totatives(11)
  ```
  ```{python}
  pyLHD.euler_phi(11)
  ```
  """
  if is_prime(N) and N>1:
    return N-1
  else:
    return len(totatives(N))


def WilliamsTransform(arr: npt.ArrayLike, baseline: int = 0, modified = False) -> np.ndarray:
  """ Williams Transformation

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      baseline (int, optional): A integer, which defines the minimum value for each column of the matrix. Defaults to 0.
      modified (bool,optional): Implement modifed version of Williams Transformation. Defaults to False.

  Returns:
      After applying Williams transformation, a matrix whose columns are permutations from {baseline,baseline+1, ..., baseline+(n-1)}.
          For the modified version. Whenever n is odd, n=2m+1 the columns will be permutations will always be even numbers
  
  Examples:
  ```{python}
  import pyLHD
  x = pyLHD.GoodLatticePoint(size = (7,6))
  x
  ```
  Apply Williams Transformation, with `baseline =0` the column level permutations will be (0,1,2,...,6)
  ```{python}
  pyLHD.WilliamsTransform(x)
  ```
  Apply modified Williams Transformation, with `baseline =0` the column level permutations will be (0,2,4,6)
  ```{python}
  pyLHD.WilliamsTransform(x, modified = True)
  ```
  """
  n = arr.shape[0]
  x = zero_base(arr)

  # Apply the weight transformation
  if not modified:
    wt = np.where(x < (n / 2), 2 * x + 1, 2 * (n - x))
  else:
    wt = np.where(x < (n/2), 2 * x + 1, 2 * (n - x) + 1)

  # Adjust the weight based on the baseline
  if baseline != 1:
    wt += (baseline - 1)
  return wt


def primes_range(start:int, stop:int)-> List[int]:
  """Generate prime numbers from a specified range

  Args:
      start (int): Start of interval. The interval includes this value
      stop (int): Stop of interval. If value is not a prime number it will return the previous prime integer

  Raises:
      ValueError: If `start` is less than `stop`

  Returns:
      A list of integers from the interval [start, stop]
  
  Example:
  ```{python}
  import pyLHD
  pyLHD.primes_range(start = 3, stop = 13)
  ```
  ```{python}
  pyLHD.primes_range(start = 3, stop = 20)
  ```
  """
  if start > stop:
    raise ValueError("'start' must be less than 'end'")
  if start < 2:
    start = 2
  primes = []
  for num in range(start, stop + 1):
    is_prime = True
    for i in range(2, int(num**0.5) + 1):
      if num % i == 0:
        is_prime = False
        break
    if is_prime:
      primes.append(num)
  return primes


def first_n_primes(n:int) -> List[int]:
  """Gernate the first `n` prime numbers

  Args:
      n (int): Total number of prime numbers to generate

  Returns:
      List[int]: A list of integers with the first `n` prime numbers, including 2 as the first prime number
  
  Example:
  ```{python}
  import pyLHD
  pyLHD.first_n_primes(10)
  ```
  """
  if n == 0:
    return []
  elif n == 1:
    return [2]

  primes = [2]
  num = 3
  while len(primes) < n:
    is_prime = True
    for p in primes:
      if p * p > num:
        break
      if num % p == 0:
        is_prime = False
        break
    if is_prime:
      primes.append(num)
    num += 2
  return primes

#################################
####   Checks/ Conditions #######
#################################

def check_seed(seed: Optional[Union[int,np.random.Generator]] = None) -> np.random.Generator:
  if seed is None or isinstance(seed, int):
    return np.random.default_rng(seed)
  elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
    global rng
    return seed
  else:
    raise ValueError(f'seed = {seed!r} cannot be used to seed a numpy.random.Generator instance')


def is_balanced_design(arr: npt.ArrayLike, s:int) -> NoReturn:
  """Verify a design is balanced

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      s (int): Required number of levels for each factor

  Raises:
      ValueError: n should be divisible by s
      ValueError: There should be exactly s unique levels for each factor
      ValueError: Each level should appear (n/s) times for each factor

  
  Notes:
      Let $(n,s^m)$ denote a design with $n$ runs and $m$ factors, each taking $s$ levels
  """
  n, m = arr.shape

  if n % s != 0:
    raise ValueError('n should be divisible by s')

  expected_count = n // s

  # Check the count for each level in each factor
  for factor in range(m):
    unique_levels = np.unique(arr[:, factor])
    if len(unique_levels) != s:
      raise ValueError('There should be exactly s unique levels for each factor')
    counts = np.bincount(arr[:, factor] - min(unique_levels), minlength=s)
    if not np.all(counts == expected_count):
      raise ValueError('Each level should appear (n/s) times for each factor')


def is_LHD(arr: npt.ArrayLike) -> NoReturn:
  """Verify Latinhypercube sampling conditions

  Args:
      arr (npt.ArrayLike): A numpy ndarray

  Raises:
      ValueError: If `arr` is not in unit hypercube
      ValueError: Sum of integers for each column dont add up to `n_rows * (n_rows + 1) / 2`
      ValueError: Each integer must appear once per column
  """
  # Validate range
  if not (0 <= arr).all() or not (arr <= 1).all(): 
      raise ValueError('arr must be in unit hypercube')

  # Check column sums
  n_rows, n_cols = arr.shape
  expected_sum = n_rows * (n_rows + 1) / 2
  
  integers = np.floor(arr * n_rows)
  sums = np.sum(integers + 1, axis=0)
  if not np.allclose(sums, expected_sum):
      raise ValueError('Integer sums invalid')

  # Check counts
  _, counts = np.unique(integers, return_counts=True)
  if not np.all(counts == n_cols):
      raise ValueError('Each integer must appear once per column')


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


def is_prime(n:int) -> bool:
  """Determine if a number is prime

  Args:
      n (int): Any integer

  Returns:
      [logical]: True if n is prime, False if n is not prime 
  """
  if n % 2 == 0 and n > 2:
    return False
  return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


def are_coprime(a:int, b:int) -> bool:
  """Check if two integers are coprime

  Args:
      a (int):  An integer
      b (int): An integer

  Returns:
      bool: Returns True if two integers are coprime
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.are_coprime(2,12)
  ```
  ```{python}
  pyLHD.are_coprime(3,11)
  ```
  """
  return math.gcd(a, b) == 1


def verify_generator(numbers: list[int], n: int, k: int) -> list[int]:
  """Verify generator used to construct good lattice points (GLP) design

  Args:
      numbers (list[int]): integers used for the generator
      n (int): number of rows in a GLP design
      k (int): number of columns in a GLP design

  Raises:
      ValueError: length of generator `numbers` is not the same as the number of columns `k`
      ValueError: All `numbers` should be less than `n` and coprime to `n`

  Returns:
      list[int]: If all conditions hold, `numbers` is returned
  """
  if is_prime(n):
    k = n-1
  if len(numbers) != k:
    raise ValueError('`numbers` length must be `k`')
  if any(element >= n or not are_coprime(element, n) for element in numbers):
    raise ValueError('All `numbers` should be less than `n` and coprime to `n`')
  return numbers