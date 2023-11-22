import numpy as np 
import pyLHD
import numpy.typing as npt
from typing import Literal

# Maximum Absolute Correlation

def MaxAbsCor(arr: npt.ArrayLike) -> float:
  """ Calculate the Maximum Absolute Correlation

  Args:
      arr (numpy.ndarray): A design matrix

  Returns:
      Positive number indicating maximum absolute correlation. Rounded to 3 digits
  
  Examples:
    ```{python}
    import pyLHD
    random_lhd = pyLHD.random_lhd(n_rows=10,n_columns=3)
    pyLHD.MaxAbsCor(random_lhd)
    ```
  """
  lower_matrix_corr = np.corrcoef(arr.T)[np.tril_indices(arr.shape[1],-1)]
  return np.max(np.abs(lower_matrix_corr))


# Calculate the Average Absolute Correlation

def AvgAbsCor(arr: npt.ArrayLike) -> float:
  """ Calculate the Average Absolute Correlation

  Args:
      arr (numpy.ndarray): A design matrix

  Returns:
      A positive number indicating the average absolute correlation 
      of input matrix

  Examples:
  Calculate the average absolute correlation of `random_lhd`
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=10,n_columns=3)
  pyLHD.AvgAbsCor(random_lhd)
  ```
  """
  lower_matrix_corr = np.corrcoef(arr.T)[np.tril_indices(arr.shape[1],-1)]
  return np.mean(np.abs(lower_matrix_corr))



# Calculate the Maximum Projection Criterion

def MaxProCriterion(arr: npt.ArrayLike) -> float:
  """ Calculate the Maximum Projection Criterion

  Args:
      arr (numpy.ndarray): A design matrix

  Returns:
      Positive number indicating maximum projection criterion
  
  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=10,n_columns=3)
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

# Calculate the Inter-site Distance

def inter_site(arr: npt.ArrayLike, i: int, j: int,  q: int = 1)  -> float:
  """ Calculate the Inter-site Distance

  Args:
      arr (numpy.ndarray): A design matrix
      i (int): A positive integer, which stands for the ith row of (arr)
      j (int): A positive integer, which stands for the jth row of (arr)
      q (int, optional): The default is set to be 1, and it could be either 1 or 2. If (q) is 1, (inter_site) is the Manhattan (rectangular) distance. If (q) is 2, (inter_site) is the Euclidean distance.

  Returns:
      positive number indicating the distance (rectangular or Euclidean) between the ith and jth row of arr
  
  Examples:
  Calculate the inter-site distance of the 2nd and the 4th row of `random_lhd`
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=10,n_columns=3)
  pyLHD.inter_site(random_lhd,i=2,j=4)
  ```
  Calculate the inter-site distance of the 2nd and the 4th row of `random_lhd` with q=2 (Euclidean)
  ```{python}
  pyLHD.inter_site(random_lhd,i=2,j=4,q=2)
  ```
  """
  return np.sum(np.abs(arr[i, :] - arr[j, :])**q)**(1/q)



# Calculate the phi_p Criterion

def phi_p(arr: npt.ArrayLike, p: int = 15,q: int = 1) -> float:
  """ Calculate the phi_p Criterion

  Args:
      arr (numpy.ndarray): A design matrix
      p (int, optional): A positive integer, which is the parameter in the phi_p formula. The default is set to be 15. If (q) is 1, (inter_site) is the Manhattan (rectangular) distance. If (q) is 2, (inter_site) is the Euclidean distance.

  Returns:
      A positive number indicating phi_p

  Examples:
  Calculate the phi_p criterion for random_lhd with default settings
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=10,n_columns=3)
  pyLHD.phi_p(random_lhd)  
  ```
  Calculate the phi_p criterion of random_lhd with p=50 and q=2 (Euclidean)
  ```{python}
  pyLHD.phi_p(random_lhd,p=50,q=2) 
  ```

  """
  n = arr.shape[0]
  distances = np.array([inter_site(arr, i=i, j=j, q=q) for i in range(n - 1) for j in range(i + 1, n)])
  isd = np.sum(distances**(-p))
  return np.sum(isd)**(1/p) 





# Caluclate the Discrepancy of a given sample

def discrepancy(arr: npt.ArrayLike, method: Literal["L2", "L2_star","centered_L2", "modified_L2", "mixture_L2", "symmetric_L2", "wrap_around_L2"] = "centered_L2") -> float:
  """ Discrepancy of a given sample

  Args:
      arr (numpy.ndarray): A design matrix
      method (str, optional): Type of discrepancy. Defaults to 'centered_L2'. Options include: 'L2', 'L2_star','centered_L2', 'modified_L2', 'mixture_L2', 'symmetric_L2', 'wrap_around_L2'

  Raises:
      ValueError: Whenever number of rows is less than number of columns

  Returns:
      Desired discrepancy type
      
  Examples:
  Calculate the centered_L2 discrepancy of `random_lhd`
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=10,n_columns=3)
  pyLHD.discrepancy(random_lhd)
  ```
  Calculate the L2 star discrepancy of `random_lhd`
  ```{python}
  pyLHD.discrepancy(random_lhd,method='L2_star')
  ``` 
  """
  
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    arr = pyLHD.scale(arr)
  
  n_rows = arr.shape[0]
  n_columns = arr.shape[1]
  
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

# Compute the coverage measure

def coverage(arr: npt.ArrayLike) -> float:
  """ Compute the coverage measure for a design

  Args:
      arr (numpy.ndarray): A design matrix. If design matrix is not within [0,1], the origianl design will be scaled to [0,1]
  Raises:
      ValueError: Whenever number of rows is less than number of columns

  Returns:
      Coverage measure
      
  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=5,n_columns=5)
  pyLHD.coverage(random_lhd)
  ```

  """
  n_rows = arr.shape[0]
  n_columns = arr.shape[1]
  
  if n_rows < n_columns:
    raise ValueError('Make sure number of rows is greater than number of columns')
  
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    x = pyLHD.scale(arr)
  else:
    x = arr
    
  dist = lambda p1, p2: np.sqrt(((p1-p2)**2).sum())
  dist_mat = np.asarray([[dist(p1, p2) for p2 in x] for p1 in x])
  np.fill_diagonal(dist_mat,10e3)

  Dmin = np.amin(dist_mat,axis=0)
  gammabar = (1/n_rows)*np.sum(Dmin)
  sum = 0

  for i in range(n_rows):
    sum +=  (Dmin[i]-gammabar)*(Dmin[i]-gammabar)

  cov = (1/gammabar)*((1/n_rows)*sum)**(1/2)
  return cov

# Compute the meshratio criterion

def mesh_ratio(arr: npt.ArrayLike) -> float:
  """ Compute the meshratio criterion for a given design

  Args:
      arr (numpy.ndarray): A design matrix. If design matrix is not within [0,1], the origianl design will be scaled to [0,1]
      
  Raises:
      ValueError: Whenever number of rows is less than number of columns
  
  Returns:
      Calculated meshratio

  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=5,n_columns=5)
  pyLHD.mesh_ratio(random_lhd)
  ```
  """
  
  n_rows = arr.shape[0]
  n_columns = arr.shape[1]

  if n_rows < n_columns:
    raise ValueError('Make sure number of rows is greater than number of columns')
  
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    x = pyLHD.scale(arr)
  else:
    x = arr

  max_dist = -1.0e30
  min_dist = 1.0e30

  for i in range(n_rows-1):
    a = 1.0e30
    b = -1.0e30
    for k in range(n_rows):
      if i != k:
        Dist = 0 
        for j in range(n_columns):
          Dist += (x[i,j] - x[k,j])*(x[i,j]-x[k,j])
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


# Calculate maximin criterion 

def maximin(arr: npt.ArrayLike) -> float:
  """ Compute the maximin criterion for a given design. A higher value corresponds to a more regular scattering of design points.

  Args:
      arr (numpy.ndarray): A design matrix. If design matrix is not within [0,1], the origianl design will be scaled to [0,1]      
  
  Returns:
      Calculated maximin criterion

  Examples:
  ```{python}
  import pyLHD
  random_lhd = pyLHD.random_lhd(n_rows=5,n_columns=5)
  pyLHD.maximin(random_lhd)
  ```
  """  

  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    x = pyLHD.scale(arr)
  else:
    x = arr
  
  dist = lambda p1, p2: np.sqrt(((p1-p2)**2).sum())
  dist_mat = np.asarray([[dist(p1, p2) for p2 in x] for p1 in x])
  np.fill_diagonal(dist_mat,1e30)
  min = np.amin(dist_mat,axis=0)
  return np.amin(min)
