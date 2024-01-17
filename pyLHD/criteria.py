import numpy as np 
import numpy.typing as npt
from typing import Literal
from pyLHD.helpers import distance_matrix, is_balanced_design, lapply, column_combinations


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
        "InterSite": InterSite,
        "phi_p": phi_p,
        "discrepancy": discrepancy,
        "coverage": coverage,
        "maximin": maximin,
        "MeshRatio": MeshRatio,
        "LqDistance": LqDistance,
        "pairwise_InterSite": pairwise_InterSite}
    
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


def InterSite(arr: npt.ArrayLike, i: int, j: int,  q: int = 1, axis: int = 0)  -> float:
  """ Calculate the Inter-site Distance between the ith and jth index

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      i (int): A positive integer, which stands for the ith row of (arr)
      j (int): A positive integer, which stands for the jth row of (arr)
      q (int, optional): The default is set to be 1, and it could be either 1 or 2. If (q) is 1, (inter_site) is the Manhattan (rectangular) distance. If (q) is 2, (inter_site) is the Euclidean distance.
      axis (int, optional): The default is set to be 0, and it coult be either 1 or 0. If (axis) is 0 compute Inter-site distance row-wise otherwise columnwise.

  Returns:
      positive number indicating the distance (rectangular or Euclidean) between the ith and jth row of arr
  
  Examples:
  Calculate the inter-site distance of the 0th and 2nd index of `random_lhd` (row-wise)
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  pyLHD.InterSite(random_lhd,i=0,j=2)
  ```
  Calculate the inter-site distance of the 0th and 2nd index of `random_lhd` (column-wise)
  ```{python}
  pyLHD.InterSite(random_lhd,i=0,j=2, axis = 1)
  ```
  """
  if axis == 0:
    return np.linalg.norm(arr[i, :] - arr[j, :], ord=q)
  elif axis == 1:
    return np.linalg.norm(arr[:, i] - arr[:, j], ord=q)
  else:
      raise ValueError("Axis can only be 0 (rows) or 1 (columns).")


def pairwise_InterSite(arr: npt.ArrayLike, q:int = 1, axis:int = 0) -> npt.ArrayLike:
  """ Calculate the Inter-site Distance between all pairwise (rows/columns)

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      q (int, optional): The default is set to be 1, and it could be either 1 or 2. If (q) is 1, (inter_site) is the Manhattan (rectangular) distance. If (q) is 2, (inter_site) is the Euclidean distance.
      axis (int, optional): The default is set to be 0, and it coult be either 1 or 0. If (axis) is 0 compute Inter-site distance row-wise otherwise columnwise.


  Returns:
      All (row/column) pairwise Inter-site distances (rectangular or Euclidean)
  
  Examples:
  Calculate all row pairwise inter-site distances of `random_lhd` with q=1 (rectangular)
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  pyLHD.pairwise_InterSite(random_lhd)
  ```
  Calculate all column pairwise inter-site distances of `random_lhd` with q=2 (Euclidean)
  ```{python}
  pyLHD.pairwise_InterSite(random_lhd,q=2, axis = 1)
  ```
  """
  if axis == 0:
    diff_matrix = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
    axis_to_sum = 2
  elif axis == 1:
    diff_matrix = arr[np.newaxis, :, :] - arr[:, np.newaxis, :]
    axis_to_sum = 1 
  else:
      raise ValueError("Axis can only be 0 (rows) or 1 (columns).")

  lq_norms = np.sum(np.abs(diff_matrix)**q, axis=axis_to_sum)**(1/q)
  
  i_upper = np.triu_indices_from(lq_norms, k=1)
  lq_distances = lq_norms[i_upper]
  return lq_distances


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
  distances = pairwise_InterSite(arr, q=q)
  isd = np.sum(distances**(-p))
  return np.sum(isd)**(1/p) 


def LqDistance(arr,q=1) -> float:
  """Calculate the Lq-Distance of a Latin Hypercube Design

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      q (int, optional): If (q) is 1, (inter_site) is the Manhattan (rectangular) distance. If (q) is 2, (inter_site) is the Euclidean distance. Default is q=1.

  Returns:
      The $L_q$ distance of a LHD. Defined as $d = min \{ InterSite(arr(i,j)) : i  \\neq j, \, i,j = 1,2,...,n \}$
          The maximin $L_q$-distance design is defined as the one which maximizes $d$

  Examples:
  Calculate the $L_1$ distance of `random_lhd` with q=1 (rectangular)
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  pyLHD.LqDistance(random_lhd)
  ``` 

  Calculate the $L_2$ distance of `random_lhd` with q=2 (Euclidean)
  ```{python}
  pyLHD.LqDistance(random_lhd, q = 2)
  ```    
  """
  return pairwise_InterSite(arr,q=q).min()


def discrepancy(arr: npt.ArrayLike,
                method: Literal["L2", "L2_star", "centered_L2", "modified_L2","balanced_centered_L2"
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
  twoD_projections = column_combinations(arr, k=2)
  balanced_CD = lapply(twoD_projections, discrepancy, method='balanced_centered_L2')
  return np.mean([x**2 for x in balanced_CD])



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

  cov = (1 / gammabar) * np.sqrt((1 / n_rows) * sum_squares)
  return cov


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


def maximin(arr: npt.ArrayLike) -> float:
  """ Compute the maximin criterion for a given design. A higher value corresponds to a more regular scattering of design points.

  Args:
      arr (npt.ArrayLike): A numpy ndarray
  
  Returns:
      Calculated maximin criterion

  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  pyLHD.maximin(random_lhd)
  ```
  """  

  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    raise ValueError('`arr` is not in unit hypercube')
  
  dist_mat = distance_matrix(arr)
  np.fill_diagonal(dist_mat,1e30)
  min = np.amin(dist_mat,axis=0)
  return np.amin(min)