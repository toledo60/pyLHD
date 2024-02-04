import numpy as np 
import numpy.typing as npt
from typing import Literal
from pyLHD.helpers import distance_matrix, is_balanced_design
from scipy.spatial.distance import pdist

class Criteria:
  """A class representing a collection of criteria functions.
      This class allows for the selection and computation of various criteria functions based on the specified type. It supports all criteria found in pyLHD

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      type (str): A string representing the type of criteria function to be used.

  Raises:
      ValueError: If the specified criteria type is not recognized.

  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  phi_p = pyLHD.Criteria(random_lhd, 'phi_p')
  phi_p.compute()
  ```
  Compute `phi_p` criteria with additional arguments
  ```{python}
  phi_p = pyLHD.Criteria(random_lhd, 'phi_p')
  phi_p.compute(p=10, q=2)
  ```
  """

  def __init__(self, arr: npt.ArrayLike, type: str):
    self.arr = arr
    self.type = type
    self.criteria_functions = {
        "MaxAbsCor": MaxAbsCor,
        "AvgAbsCor": AvgAbsCor,
        "MaxProCriterion": MaxProCriterion,
        "UniformProCriterion": UniformProCriterion,
        "phi_p": phi_p,
        "discrepancy": discrepancy,
        "coverage": coverage,
        "MeshRatio": MeshRatio}
    
    if type not in self.criteria_functions:
      raise ValueError(f"Unknown criteria type: {type}")

  def update(self, new_arr: npt.ArrayLike, *args, **kwargs) -> float:
    self.arr = new_arr
    return self.compute(*args, **kwargs)

  def compute(self, *args, **kwargs) -> float:
    return self.criteria_functions[self.type](self.arr, *args, **kwargs)


def MaxAbsCor(arr: npt.ArrayLike) -> float:
  """ Calculate the Maximum Absolute Correlation

  Args:
      arr (npt.ArrayLike): A numpy ndarray

  Returns:
      Positive number indicating maximum absolute correlation. Rounded to 3 digits

  Notes:
      References for the implementation of the maximum absolute correlation

  - Georgiou, Stelios D. "Orthogonal Latin hypercube designs from generalized orthogonal designs." Journal of Statistical Planning and Inference 139.4 (2009): 1530-1540.  

  Examples:
    ```{python}
    import pyLHD
    random_lhd = pyLHD.LatinHypercube(size = (10,3))
    pyLHD.MaxAbsCor(random_lhd)
    ```
  """
  lower_matrix_corr = np.corrcoef(arr.T)[np.tril_indices(arr.shape[1],-1)]
  return np.max(np.abs(lower_matrix_corr))


def AvgAbsCor(arr: npt.ArrayLike) -> float:
  """ Calculate the Average Absolute Correlation

  Args:
      arr (npt.ArrayLike): A numpy ndarray

  Returns:
      A positive number indicating the average absolute correlation 
      of input matrix

  Examples:
  Calculate the average absolute correlation of `random_lhd`
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  pyLHD.AvgAbsCor(random_lhd)
  ```
  """
  lower_matrix_corr = np.corrcoef(arr.T)[np.tril_indices(arr.shape[1],-1)]
  return np.mean(np.abs(lower_matrix_corr))


def MaxProCriterion(arr: npt.ArrayLike) -> float:
  """ Calculate the Maximum Projection Criterion

  Args:
      arr (npt.ArrayLike): A numpy ndarray

  Returns:
      Positive number indicating maximum projection criterion
  
  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  pyLHD.MaxProCriterion(random_lhd)
  ```
  """
  n, p = arr.shape
  arr_reshaped = arr[:, np.newaxis, :]
  
  # Calculate differences between pairs of rows
  diff = arr_reshaped - arr_reshaped.transpose(1, 0, 2)
  squared_diff = diff**2

  denom = np.prod(squared_diff, axis=-1)
  temp = np.sum(1 / denom[np.triu_indices(n, k=1)])

  return (2 / (n * (n - 1)) * temp)**(1 / p)


class LqDistance:
  def __init__(self,arr: npt.ArrayLike, q=1):
    self.arr = arr
    self.q = q
  

  def pairwise(self) -> np.ndarray:
    """Calculate the Lq distance among all pairwise distances in the array

    Returns:
        The Lq distance among all pairs of points in the array
    
    Example:
    ```{python}
    import pyLHD
    sample = pyLHD.GoodLatticePoint(size = (5,3),seed =1)
    l1 = pyLHD.LqDistance(sample,q=1)
    l1.pairwise()
    ```
    """    
    return pdist(self.arr,'minkowski',p=self.q)
  
  def design(self) -> float:
    """Calculate the minimum Lq distance among all pairwise distances in the array

    Returns:
        The minimum Lq distance among all pairs of points in the array
    
    Example:
    ```{python}
    import pyLHD
    sample = pyLHD.GoodLatticePoint(size = (5,3),seed =1)
    l1 = pyLHD.LqDistance(sample,q=1)
    l1.pairwise()
    ```
    ```{python}
    l1.design()
    ```
    """
    return self.pairwise().min()
  
  def index(self,i:int, j:int, axis:int = 0) -> float:
    """ Calculate the Lq norm (distance) between two points (rows or columns) in an array.
        The points can be either two rows or two columns in the array, depending on the axis parameter

    Args:
        i (int): The index of the first point (row or column based on axis)
        j (int): The index of the second point (row or column based on axis)
        axis (int, optional): The axis along which to compute the distance
            axis = 0 for distances between rows, axis = 1 for distances between columns. Defaults to 0

    Raises:
        ValueError: If the axis is not 0 (for rows) or 1 (for columns)

    Returns:
        The Lq distance between the two specified points
    
    Example:
    ```{python}
    import pyLHD
    sample = pyLHD.GoodLatticePoint(size = (5,3),seed =1)
    l1 = pyLHD.LqDistance(sample,q=1)
    l1.index(i = 0, j = 1)
    ```
    ```{python}
    l1.index(i = 0, j = 1, axis = 1)
    ```
    """
    if axis == 0:
      return np.linalg.norm(self.arr[i, :] - self.arr[j, :], ord=self.q)
    elif axis == 1:
      return np.linalg.norm(self.arr[:, i] - self.arr[:, j], ord=self.q)
    else:
        raise ValueError("Axis can only be 0 (rows) or 1 (columns)")


def phi_p(arr: npt.ArrayLike, p: int = 15,q: int = 1) -> float:
  """ Calculate the phi_p Criterion
  
  Args:
      arr (npt.ArrayLike): A numpy ndarray

      p (int, optional): A positive integer, which is the parameter in the phi_p formula. The default is set to be 15. If (q) is 1, (inter_site) is the Manhattan (rectangular) distance. If (q) is 2, (inter_site) is the Euclidean distance.

  Returns:
      A positive number indicating phi_p

  Examples:
  Calculate the phi_p criterion for random_lhd with default settings
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  pyLHD.phi_p(random_lhd)  
  ```
  Calculate the phi_p criterion of random_lhd with p=50 and q=2 (Euclidean)
  ```{python}
  pyLHD.phi_p(random_lhd,p=50,q=2) 
  ```
  """
  lq = LqDistance(arr, q=q)
  distances = lq.pairwise()
  isd = np.sum(distances**(-p))
  return np.sum(isd)**(1/p) 


def discrepancy(arr: npt.ArrayLike,
                method: Literal["L2", "L2_star", "centered_L2", "modified_L2","balanced_centered_L2",
                                "mixture_L2", "symmetric_L2", "wrap_around_L2"] = "centered_L2") -> float:
  """ Discrepancy of a given sample

  Args:
      arr (npt.ArrayLike): A numpy ndarray

      method (str, optional): Type of discrepancy. Defaults to 'centered_L2'. Options include: 'L2', 'L2_star','centered_L2', 'modified_L2', 'mixture_L2', 'symmetric_L2', 'wrap_around_L2'

  Raises:
      ValueError: Whenever number of rows is less than number of columns

  Returns:
      Desired discrepancy type
      
  Examples:
  Calculate the centered_L2 discrepancy of `random_lhd`
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  pyLHD.discrepancy(random_lhd)
  ```
  Calculate the L2 star discrepancy of `random_lhd`
  ```{python}
  pyLHD.discrepancy(random_lhd,method='L2_star')
  ``` 
  """
  n_rows, n_columns = arr.shape
  
  if n_rows < n_columns:
    raise ValueError('Make sure number of rows is greater than number of columns')
  
  if method == 'L2':
    sum1 = np.sum(np.prod(arr * (1 - arr), axis=1))

    min_arr = np.minimum(arr[:, np.newaxis, :], arr[np.newaxis, :, :])
    max_arr = np.maximum(arr[:, np.newaxis, :], arr[np.newaxis, :, :])

    product_term = np.prod((1 - max_arr) * min_arr, axis=2)
    sum2 = np.sum(product_term)

    # Final value calculation
    value = np.sqrt(12 ** (-n_columns) - ((2 ** (1 - n_columns)) / n_rows) * sum1 + (1 / n_rows ** 2) * sum2) 
  
  if method == 'L2_star':
    one_minus_arr = 1 - arr
    # case when i != j
    max_arr = np.maximum(arr[:, np.newaxis, :], arr[np.newaxis, :, :])
    prod_term = np.prod(1 - max_arr, axis=2)
    sum_prod_term = np.sum(prod_term) - np.sum(np.prod(one_minus_arr, axis=1))
    dL2_non_equal = sum_prod_term / (n_rows ** 2)

    # case when i == j
    t1 = np.prod(one_minus_arr, axis=1)
    t2 = np.prod(1 - np.square(arr), axis=1)
    dL2_equal = np.sum(t1 / (n_rows ** 2) - ((2 ** (1 - n_columns)) / n_rows) * t2)

    # Combine the results
    dL2 = dL2_non_equal + dL2_equal
    value = np.sqrt(3 ** (-n_columns) + dL2)
      
  if method == 'centered_L2':
    abs_diff_05 = np.abs(arr - 0.5)

    sum1 = np.sum(np.prod(1 + 0.5 * abs_diff_05 - 0.5 * (abs_diff_05 ** 2), axis=1))
    sum2_matrix = 1 + 0.5 * abs_diff_05[:, np.newaxis, :] + 0.5 * abs_diff_05[np.newaxis, :, :] - 0.5 * np.abs(arr[:, np.newaxis, :] - arr[np.newaxis, :, :])
    sum2 = np.sum(np.prod(sum2_matrix, axis=2))
    value = np.sqrt(((13 / 12) ** n_columns) - ((2 / n_rows) * sum1) + ((1 / (n_rows ** 2)) * sum2))
  
  if method == 'modified_L2':
    sum1 = np.sum(np.prod(3 - np.square(arr), axis=1))
    max_arr = np.maximum(arr[:, np.newaxis, :], arr[np.newaxis, :, :])
    sum2 = np.sum(np.prod(2 - max_arr, axis=2))
    value = np.sqrt(((4 / 3) ** n_columns) - (((2 ** (1 - n_columns)) / n_rows) * sum1) + ((1 / n_rows ** 2) * sum2))
  
  if method == 'mixture_L2':
    abs_diff_05 = np.abs(arr - 0.5)
    sum1 = np.sum(np.prod(5/3 - 0.25 * abs_diff_05 - 0.25 * (abs_diff_05 ** 2), axis=1))
    diff_arr = np.abs(arr[:, np.newaxis, :] - arr[np.newaxis, :, :])
    sum2_matrix = 15/8 - 0.25 * abs_diff_05[:, np.newaxis, :] - 0.25 * abs_diff_05[np.newaxis, :, :] - 0.75 * diff_arr + 0.5 * (diff_arr ** 2)
    sum2 = np.sum(np.prod(sum2_matrix, axis=2))

    value = np.sqrt(((19 / 12) ** n_columns) - ((2 / n_rows) * sum1) + ((1 / n_rows ** 2) * sum2))
     
  if method == 'symmetric_L2':
    sum1 = np.sum(np.prod(1 + 2 * arr - 2 * np.square(arr), axis=1))
    abs_diff = np.abs(arr[:, np.newaxis, :] - arr[np.newaxis, :, :])
    sum2 = np.sum(np.prod(1 - abs_diff, axis=2))
    value = np.sqrt(((4 / 3) ** n_columns) - ((2 / n_rows) * sum1) + ((2 ** n_columns / n_rows ** 2) * sum2))

  if method == 'wrap_around_L2':
    pairwise_diff = np.abs(arr[:, np.newaxis, :] - arr[np.newaxis, :, :])
    sum1_matrix = 1.5 - pairwise_diff * (1 - pairwise_diff)
    sum1 = np.sum(np.prod(sum1_matrix, axis=2))
    value = np.sqrt((-((4 / 3) ** n_columns) + ((1 / n_rows ** 2) * sum1)))
  
  if method == 'balanced_centered_L2':
    try:
      s = np.unique(arr).size
      is_balanced_design(arr, s = s)
    except ValueError as e:
      print(f"Error: {e}")
      
    # second sum
    z = (2 * arr - s + 1) / (2 * s)
    second_sum = np.prod(1 + 0.5 * np.abs(z) - 0.5 * np.abs(z)**2, axis=1).sum()
    # first sum
    z_abs = np.abs(z)
    z_abs_diff = np.abs(z[:, np.newaxis, :] - z[np.newaxis, :, :])
    first_sum = np.prod(1 + 0.5 * (z_abs[:, np.newaxis, :] + z_abs[np.newaxis, :, :]) - 0.5 * z_abs_diff, axis=2).sum()

    value = np.sqrt((1 / n_rows**2) * first_sum - (2 / n_rows) * second_sum + (13 / 12)**n_columns)

  return value


def UniformProCriterion(arr: npt.ArrayLike) -> float:
  """Calculate the Uniform Projection Criterion

  Args:
      arr (npt.ArrayLike): A numpy ndarray

  Returns:
      Uniform projection criteria
  """
  try:
    s = np.unique(arr).size
    is_balanced_design(arr, s = s)
  except ValueError as e:
    print(f"Error: {e}")  
  
  n,m = arr.shape
  diffs = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
  l1_norms = np.sum(np.abs(diffs), axis=2)
  squared_l1_norms = l1_norms**2
  
  total_sum1 = np.sum(squared_l1_norms)
  row_sums = np.sum(l1_norms, axis=1)
  total_sum2 = np.sum(row_sums**2)
  
  gD = total_sum1 - (2/n)*total_sum2
  
  val1 = 4*m*(m-1)*(n**2)*(s**2)
  val2 = 4*(5*m - 2)*(s**4) + 30*(3*m - 5) *(s**2) + 15*m + 33
  val3 = 720 *(m-1)*(s**4)
  C_ms = (val2/val3) + (1+(-1)**s)/(64*(s**4))
  
  return gD/val1 + C_ms


def coverage(arr: npt.ArrayLike) -> float:
  """ Compute the coverage measure for a design

  Args:
      arr (npt.ArrayLike): A numpy ndarray
  Raises:
      ValueError: Whenever number of rows is less than number of columns

  Returns:
      Coverage measure
      
  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (5,5))
  pyLHD.coverage(random_lhd)
  ```
  """
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    raise ValueError('`arr` is not in unit hypercube')

  n_rows, n_columns = arr.shape
  if n_rows < n_columns:
    raise ValueError('Make sure number of rows is greater than number of columns')
  
  dist_mat = distance_matrix(arr)
  np.fill_diagonal(dist_mat,10e3)

  Dmin = np.amin(dist_mat,axis=0)
  gammabar = (1/n_rows)*np.sum(Dmin)
  sum_squares = np.sum((Dmin - gammabar) ** 2)

  return (1 / gammabar) * np.sqrt((1 / n_rows) * sum_squares)


def MeshRatio(arr: npt.ArrayLike) -> float:
  """ Compute the meshratio criterion for a given design

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      
  Raises:
      ValueError: Whenever number of rows is less than number of columns
  
  Returns:
      Calculated meshratio

  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (5,5))
  pyLHD.MeshRatio(random_lhd)
  ```
  """
  n_rows, n_columns = arr.shape
  if n_rows < n_columns:
    raise ValueError('Make sure number of rows is greater than number of columns')
  
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    raise ValueError('`arr` is not in unit hypercube')

  max_dist = -1.0e30
  min_dist = 1.0e30

  for i in range(n_rows-1):
    a = 1.0e30
    b = -1.0e30
    for k in range(n_rows):
      if i != k:
        Dist = 0 
        for j in range(n_columns):
          Dist += (arr[i,j] - arr[k,j])*(arr[i,j]-arr[k,j])
        if Dist > b:
          b = Dist
        if Dist < a:
          a = Dist
    if max_dist < a:
      max_dist = a
    if min_dist > a:
      min_dist = a
  ratio = np.sqrt(max_dist/min_dist)
  return ratio
