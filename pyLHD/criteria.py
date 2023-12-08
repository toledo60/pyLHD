import numpy as np 
import numpy.typing as npt
from typing import Literal

def MaxAbsCor(arr: npt.ArrayLike) -> float:
  """ Calculate the Maximum Absolute Correlation

  Args:
      arr (npt.ArrayLike): A numpy ndarray


  Returns:
      Positive number indicating maximum absolute correlation. Rounded to 3 digits
  
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


def InterSite(arr: npt.ArrayLike, i: int, j: int,  q: int = 1)  -> float:
  """ Calculate the Inter-site Distance

  Args:
      arr (npt.ArrayLike): A numpy ndarray

      i (int): A positive integer, which stands for the ith row of (arr)
      j (int): A positive integer, which stands for the jth row of (arr)
      q (int, optional): The default is set to be 1, and it could be either 1 or 2. If (q) is 1, (inter_site) is the Manhattan (rectangular) distance. If (q) is 2, (inter_site) is the Euclidean distance.

  Returns:
      positive number indicating the distance (rectangular or Euclidean) between the ith and jth row of arr
  
  Examples:
  Calculate the inter-site distance of the 2nd and the 4th row of `random_lhd`
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (10,3))
  pyLHD.InterSite(random_lhd,i=2,j=4)
  ```
  Calculate the inter-site distance of the 2nd and the 4th row of `random_lhd` with q=2 (Euclidean)
  ```{python}
  pyLHD.InterSite(random_lhd,i=2,j=4,q=2)
  ```
  """
  return np.sum(np.abs(arr[i, :] - arr[j, :])**q)**(1/q)


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
  n = arr.shape[0]
  distances = np.array([InterSite(arr, i=i, j=j, q=q) for i in range(n - 1) for j in range(i + 1, n)])
  isd = np.sum(distances**(-p))
  return np.sum(isd)**(1/p) 


def discrepancy(arr: npt.ArrayLike,
                method: Literal["L2", "L2_star", "centered_L2", "modified_L2",
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
  
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    raise ValueError('`arr` is not in unit hypercube')
  
  n_rows, n_columns = arr.shape
  
  if n_rows < n_columns:
    raise ValueError('Make sure number of rows is greater than number of columns')

  sum1 = 0
  sum2 = 0
  
  if method == 'L2':

    for i in range(n_rows):
      sum1 += np.prod(arr[i,:]*(1-arr[i,:]))
      for k in range(n_rows):
        q =  1
        for j in range(n_columns):
          q = q*(1-np.maximum(arr[i,j],arr[k,j]))*np.minimum(arr[i,j],arr[k,j])
        sum2 += q
    value = np.sqrt(12**(-n_columns) - (((2**(1-n_columns))/n_rows)*sum1) + ((1/n_rows**2)*sum2))  
  
  if method == 'L2_star':
    dL2 = 0
    for j in range(n_rows):
      for i in range(n_rows):
        if i!=j:
          t = []
          for l in range(n_columns):
            t.append(1-np.maximum(arr[i,l],arr[j,l]))
          t = (np.prod(t))/(n_rows**2)
        else:
          t1 = 1-arr[i,:]
          t1 = np.prod(t1)
          t2 = 1-np.square(arr[i,:])
          t2 = np.prod(t2)
          t = t1/(n_rows**2)-((2**(1-n_columns))/n_rows)*t2
    
        dL2 += t
  
    value = np.sqrt(3**(-n_columns)+dL2)
      
  if method == 'centered_L2':

    for i in range(n_rows):
      sum1 += np.prod((1+0.5*np.abs(arr[i,:]-0.5)-0.5*((abs(arr[i,:]-0.5))**2)))
      for k in range(n_rows):
        sum2 +=  np.prod((1+0.5*np.abs(arr[i,:]-0.5)+0.5*np.abs(arr[k,:]-0.5)-0.5*np.abs(arr[i,:]-arr[k,:])))
    value =  np.sqrt( ( (13/12)**n_columns)-((2/n_rows)*sum1) + ((1/(n_rows**2))*sum2)  )
  
  if method == 'modified_L2':
    
    for i in range(n_rows):
      p = 1
      p = np.prod((3-(arr[i,:]*arr[i,:])))
      sum1 += p
      
      for k in range(n_rows):
        q = 1
        for j in range(n_columns):
          q = q*(2-np.maximum(arr[i,j],arr[k,j]))
        sum2 += q
    value =  np.sqrt(((4/3)**n_columns) - (((2**(1-n_columns))/n_rows)*sum1) + ((1/n_rows**2)*sum2))
  
  if method == 'mixture_L2':
    for i in range(n_rows):
      sum1 += np.prod((5/3-0.25*np.abs(arr[i,:]-0.5)-0.25*((np.abs(arr[i,:]-0.5))**2)))
      for k in range(n_rows):
        sum2 += np.prod((15/8-0.25*np.abs(arr[i,:]-0.5)-0.25*np.abs(arr[k,:]-0.5)-
                         0.75*np.abs(arr[i,:]-arr[k,:])+0.5*((np.abs(arr[i,:]-arr[k,:]))**2)))
    value = np.sqrt(((19/12)**n_columns)-((2/n_rows)*sum1) + ((1/n_rows**2)*sum2))
     
  if method == 'symmetric_L2':

    for i in range(n_rows):
      sum1 += np.prod( (1+2*arr[i,:]) - (2*arr[i,:]*arr[i,:]))
      for k in range(n_rows):
        sum2 += np.prod( (1-np.abs(arr[i,:]-arr[k,:])) )
    value = np.sqrt(((4/3)**n_columns) - ((2/n_rows)*sum1) + ((2**n_columns/n_rows**2)*sum2))
  
  if method == 'wrap_around_L2':
    for i in range(n_rows):
      for k in range(n_rows):
        sum1 += np.prod((1.5-((np.abs(arr[i,:]-arr[k,:]))*(1-np.abs(arr[i,:]-arr[k,:])))))
  
    value =  np.sqrt((-((4/3)**n_columns) + ((1/n_rows**2)*sum1)))
  
  return value


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
  n_rows, n_columns = arr.shape
  
  if n_rows < n_columns:
    raise ValueError('Make sure number of rows is greater than number of columns')
  
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    raise ValueError('`arr` is not in unit hypercube')
    
  dist = lambda p1, p2: np.sqrt(((p1-p2)**2).sum())
  dist_mat = np.asarray([[dist(p1, p2) for p2 in arr] for p1 in arr])
  np.fill_diagonal(dist_mat,10e3)

  Dmin = np.amin(dist_mat,axis=0)
  gammabar = (1/n_rows)*np.sum(Dmin)
  sum = 0

  for i in range(n_rows):
    sum +=  (Dmin[i]-gammabar)*(Dmin[i]-gammabar)

  cov = (1/gammabar)*((1/n_rows)*sum)**(1/2)
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
  
  dist = lambda p1, p2: np.sqrt(((p1-p2)**2).sum())
  dist_mat = np.asarray([[dist(p1, p2) for p2 in arr] for p1 in arr])
  np.fill_diagonal(dist_mat,1e30)
  min = np.amin(dist_mat,axis=0)
  return np.amin(min)


def eval_design(arr: npt.ArrayLike, criteria: str = 'phi_p',p: int = 15,q: int = 1) -> float:
  """ Evaluate a design based on a chosen criteria, a simple wrapper for all `criteria` in `pyLHD`

  Args:
      arr (npt.ArrayLike): A numpy ndarray
      criteria (str, optional): Criteria to choose from. Defaults to 'phi_p'. 
          Options include 'phi_p','MaxProCriterion','AvgAbsCor','AvgAbsCor', 'coverage', 'MeshRatio', 'maximin'
          p (int): A positive integer, which is the parameter in the phi_p formula. The default is set to be 15
          q (int): If (q) is 1, (inter_site) is the Manhattan (rectangular) distance. If (q) is 2, (inter_site) is the Euclidean distance.

  Returns:
      Calculation of chosen criteria for any LHD

  Examples:
  By default `phi_p` with `p=15` and `q=1`
  ```{python}
  import pyLHD
  random_lhd = pyLHD.LatinHypercube(size = (5,3))
  pyLHD.eval_design(random_lhd)
  ```
  Evaluate design based on MaxProCriterion 
  ```{python}
  pyLHD.eval_design(random_lhd,criteria='MaxProCriterion')
  ``` 
  """
  criteria_functions = {
    'AvgAbsCor': AvgAbsCor,
    'coverage': coverage,
    'maximin': maximin,
    'MeshRatio': MeshRatio,
    'MaxProCriterion': MaxProCriterion,
    'MaxAbsCor': MaxAbsCor
  }
    
  if criteria in criteria_functions:
    return criteria_functions[criteria](arr)
  elif criteria == 'phi_p':
    return phi_p(arr, p=p, q=q)
  else:
    raise ValueError(f"Invalid criteria: {criteria}")