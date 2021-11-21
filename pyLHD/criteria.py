import numpy as np 

# Maximum Absolute Correlation

def MaxAbsCor(arr):
  """ Calculate the Maximum Absolute Correlation

  Args:
      arr (numpy.ndarray): A design matrix

  Returns:
      [float]: Positive number indicating maximum absolute correlation. Rounded to 3 digits
  
  Examples:
    >>> example_LHD = rLHD(nrows=5,ncols=3)
    >>> MaxAbsCor(example_LHD)
  """
  p = arr.shape[1]  # number of columns
  corr = []
  for i in range(0, p-1):
    for j in range(i+1, p):
      corr.append(np.corrcoef(arr[:, i], arr[:, j])[0, 1])
  abs_corr_array = np.absolute(np.asarray(corr))
  return np.around(np.amax(abs_corr_array), 3)

# Calculate the Maximum Projection Criterion

def MaxProCriterion(arr):
  """ Calculate the Maximum Projection Criterion

  Args:
      arr (numpy.ndarray): A design matrix

  Returns:
      [float]: Positive number indicating maximum projection criterion
  
  Examples:
      >>> example_LHD = rLHD(nrows=5,ncols=3)
      >>> MaxProCriterion(example_LHD)
  """
  n = arr.shape[0]
  p = arr.shape[1]

  temp =0

  for i in range(0,n-1):
    for j in range(i+1,n):
      denom = 1
      for k in range(0,p):
        denom = denom*(arr[i,k]-arr[j,k])**2
      temp = temp + 1/denom
  return (2/(n*(n-1) ) *temp)**(1/p)

# Calculate the Inter-site Distance

def dij(arr,i, j, q = 1):
  """ Calculate the Inter-site Distance

  Args:
      arr (numpy.ndarray): A design matrix
      i (int): A positive integer, which stands for the ith row of (arr)
      j (int): A positive integer, which stands for the jth row of (arr)
      q (int, optional): The default is set to be 1, and it could be either 1 or 2. 
      If (q) is 1, (dij) is the Manhattan (rectangular) distance. If (q) is 2, (dij) is the Euclidean distance.

  Returns:
      [float]: positive number indicating the distance (rectangular or Euclidean) between the ith and jth row of arr
  
  Examples:
      # Calculate the inter-site distance of the 2nd and the 4th row of example_LHD
      >>> example_LHD = rLHD(nrows=5,ncols=3)
      >>> dij(example_LHD,i=2,j=4)

      # Calculate the inter-site distance of the 2nd and the 4th row of example_LHD with q=2 (Euclidean)
      >>> example_LHD = rLHD(nrows=5,ncols=3)
      >>> dij(example_LHD,i=2,j=4,q=2)
  """
  p = arr.shape[1]
  distance = np.empty(p)
  for l in range(0,p):
    distance[l] = np.absolute(arr[i,l]-arr[j,l])**q
  return np.sum(distance)**(1/q)


# Calculate the phi_p Criterion

def phi_p(arr,p=15,q=1):
  """ Calculate the phi_p Criterion

  Args:
      arr (numpy.ndarray): A design matrix
      p (int, optional): A positive integer, which is the parameter in the phi_p formula. The default is set to be 15
      If (q) is 1, (dij) is the Manhattan (rectangular) distance. If (q) is 2, (dij) is the Euclidean distance.

  Returns:
      [float]: a positive number indicating phi_p

  Examples:
      # Calculate the phi_p criterion for example_LHD with default settings
      >>> example_LHD = rLHD(nrows=5,ncols=3)
      >>> phi_p(example_LHD)

      # Calculate the phi_p criterion of example_LHD with p=50 and q=2 (Euclidean)
      >>> phi_p(example_LHD,p=50,q=2)    
  """
  n = arr.shape[0]
  isd = 0 
  for i in range(0,n-1):
    for j in range(i+1,n):
      isd = isd + dij(arr,i=i,j=j,q=q)**(-p)
  return np.sum(isd)**(1/p)

# Calculate the Average Absolute Correlation

def AvgAbsCor(arr):
  """ Calculate the Average Absolute Correlation

  Args:
      arr (numpy.ndarray): A design matrix

  Returns:
      [float]: A positive number indicating the average absolute correlation 
      of input matrix

  Examples:
      # Calculate the average absolute correlation of example_LHD
      >>> example_LHD = rLHD(nrows=5,ncols=3)
      >>> AvgAbsCor(example_LHD)
  """
  p = arr.shape[1]
  corr = []
  for i in range(0,p-1):
    for j in range(i+1,p):
      corr.append(np.corrcoef(arr[:,i],arr[:,j])[0,1])
  abs_corr_array = np.absolute(np.asarray(corr))
  return np.around(np.mean(abs_corr_array),3)

