import numpy as np 
import pyLHD

# Maximum Absolute Correlation

def MaxAbsCor(arr):
  """ Calculate the Maximum Absolute Correlation

  Args:
      arr (numpy.ndarray): A design matrix

  Returns:
      [float]: Positive number indicating maximum absolute correlation. Rounded to 3 digits
  
  Examples:
    >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
    >>> pyLHD.MaxAbsCor(example_LHD)
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
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
      >>> pyLHD.MaxProCriterion(example_LHD)
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
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
      >>> pyLHD.dij(example_LHD,i=2,j=4)

      # Calculate the inter-site distance of the 2nd and the 4th row of example_LHD with q=2 (Euclidean)
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
      >>> pyLHD.dij(example_LHD,i=2,j=4,q=2)
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
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
      >>> pyLHD.phi_p(example_LHD)

      # Calculate the phi_p criterion of example_LHD with p=50 and q=2 (Euclidean)
      >>> pyLHD.phi_p(example_LHD,p=50,q=2)    
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
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
      >>> pyLHD.AvgAbsCor(example_LHD)
  """
  p = arr.shape[1]
  corr = []
  for i in range(0,p-1):
    for j in range(i+1,p):
      corr.append(np.corrcoef(arr[:,i],arr[:,j])[0,1])
  abs_corr_array = np.absolute(np.asarray(corr))
  return np.around(np.mean(abs_corr_array),3)


# Caluclate the Discrepancy of a given sample

def discrepancy(arr, type='centered_L2'):
  """ Discrepancy of a given sample

  Args:
      arr (numpy.ndarray): A design matrix
      type (str, optional): Type of discrepancy. Defaults to 'centered_L2'. Options include:
      'L2', 'L2_star','centered_L2', 'modified_L2', 'mixture_L2', 'symmetric_L2', 'wrap_around_L2'

  Raises:
      ValueError: Whenever number of rows is less than number of columns

  Returns:
      float: Desired discrepancy type
      
  Examples:
      # Calculate the centered_L2 discrepancy of example_LHD
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
      >>> pyLHD.discrepancy(example_LHD)
      # Calculate the L2 star discrepancy of example_LHD
      >>> pyLHD.discrepancy(example_LHD,type='L2_star)     
  """
  
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    arr = pyLHD.scale(arr)
  
  nrows = arr.shape[0]
  ncols = arr.shape[1]
  
  if nrows < ncols:
    raise ValueError('Make sure number of rows is greater than number of columns')

  sum1 = 0
  sum2 = 0
  
  if type == 'L2':

    for i in range(nrows):
      sum1 += np.prod(arr[i,:]*(1-arr[i,:]))
      for k in range(nrows):
        q =  1
        for j in range(ncols):
          q = q*(1-np.maximum(arr[i,j],arr[k,j]))*np.minimum(arr[i,j],arr[k,j])
        sum2 += q
    value = np.sqrt(12**(-ncols) - (((2**(1-ncols))/nrows)*sum1) + ((1/nrows**2)*sum2))  
  
  if type == 'L2_star':
    dL2 = 0
    for j in range(nrows):
      for i in range(nrows):
        if i!=j:
          t = []
          for l in range(ncols):
            t.append(1-np.maximum(arr[i,l],arr[j,l]))
          t = (np.prod(t))/(nrows**2)
        else:
          t1 = 1-arr[i,:]
          t1 = np.prod(t1)
          t2 = 1-np.square(arr[i,:])
          t2 = np.prod(t2)
          t = t1/(nrows**2)-((2**(1-ncols))/nrows)*t2
    
        dL2 += t
  
    value = np.sqrt(3**(-ncols)+dL2)
      
  if type == 'centered_L2':

    for i in range(nrows):
      sum1 += np.prod((1+0.5*np.abs(arr[i,:]-0.5)-0.5*((abs(arr[i,:]-0.5))**2)))
      for k in range(nrows):
        sum2 +=  np.prod((1+0.5*np.abs(arr[i,:]-0.5)+0.5*np.abs(arr[k,:]-0.5)-0.5*np.abs(arr[i,:]-arr[k,:])))
    value =  np.sqrt( ( (13/12)**ncols)-((2/nrows)*sum1) + ((1/(nrows**2))*sum2)  )
  
  if type == 'modified_L2':
    
    for i in range(nrows):
      p = 1
      p = np.prod((3-(arr[i,:]*arr[i,:])))
      sum1 += p
      
      for k in range(nrows):
        q = 1
        for j in range(ncols):
          q = q*(2-np.maximum(arr[i,j],arr[k,j]))
        sum2 += q
    value =  np.sqrt(((4/3)**ncols) - (((2**(1-ncols))/nrows)*sum1) + ((1/nrows**2)*sum2))
  
  if type == 'mixture_L2':
    for i in range(nrows):
      sum1 += np.prod((5/3-0.25*np.abs(arr[i,:]-0.5)-0.25*((np.abs(arr[i,:]-0.5))**2)))
      for k in range(nrows):
        sum2 += np.prod((15/8-0.25*np.abs(arr[i,:]-0.5)-0.25*np.abs(arr[k,:]-0.5)-
                         0.75*np.abs(arr[i,:]-arr[k,:])+0.5*((np.abs(arr[i,:]-arr[k,:]))**2)))
    value = np.sqrt(((19/12)**ncols)-((2/nrows)*sum1) + ((1/nrows**2)*sum2))
     
  if type == 'symmetric_L2':

    for i in range(nrows):
      sum1 += np.prod( (1+2*arr[i,:]) - (2*arr[i,:]*arr[i,:]))
      for k in range(nrows):
        sum2 += np.prod( (1-np.abs(arr[i,:]-arr[k,:])) )
    value = np.sqrt(((4/3)**ncols) - ((2/nrows)*sum1) + ((2**ncols/nrows**2)*sum2))
  
  if type == 'wrap_around_L2':
    for i in range(nrows):
      for k in range(nrows):
        sum1 += np.prod((1.5-((np.abs(arr[i,:]-arr[k,:]))*(1-np.abs(arr[i,:]-arr[k,:])))))
  
    value =  np.sqrt((-((4/3)**ncols) + ((1/nrows**2)*sum1)))
  
  return value

# Compute the coverage measure

def coverage(arr):
  """ Compute the coverage measure for a design

  Args:
      arr (numpy.ndarray): A design matrix. If design matrix is not within [0,1], the 
      origianl design will be scaled to [0,1]
  Raises:
      ValueError: Whenever number of rows is less than number of columns

  Returns:
      float: Coverage measure
      
  Examples:
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=5)
      >>> pyLHD.coverage(example_LHD)
  """
  nrows = arr.shape[0]
  ncols = arr.shape[1]
  
  if nrows < ncols:
    raise ValueError('Make sure number of rows is greater than number of columns')
  
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    x = pyLHD.scale(arr)
  else:
    x = arr
    
  dist = lambda p1, p2: np.sqrt(((p1-p2)**2).sum())
  dist_mat = np.asarray([[dist(p1, p2) for p2 in x] for p1 in x])
  np.fill_diagonal(dist_mat,10e3)

  Dmin = np.amin(dist_mat,axis=0)
  gammabar = (1/nrows)*np.sum(Dmin)
  sum = 0

  for i in range(nrows):
    sum +=  (Dmin[i]-gammabar)*(Dmin[i]-gammabar)

  cov = (1/gammabar)*((1/nrows)*sum)**(1/2)
  return cov

# Compute the meshratio criterion

def mesh_ratio(arr):
  """ Compute the meshratio criterion for a given design

  Args:
      arr (numpy.ndarray): A design matrix. If design matrix is not within [0,1], the 
      origianl design will be scaled to [0,1]
      
  Raises:
      ValueError: Whenever number of rows is less than number of columns
  
  Returns:
      float: Calculated meshratio

  Examples:
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=5)
      >>> pyLHD.mesh_ratio(example_LHD)
  """
  
  nrows = arr.shape[0]
  ncols = arr.shape[1]

  if nrows < ncols:
    raise ValueError('Make sure number of rows is greater than number of columns')
  
  if (np.amin(arr) < 0 or np.amax(arr) > 1):
    x = pyLHD.scale(arr)
  else:
    x = arr

  max_dist = -1.0e30
  min_dist = 1.0e30

  for i in range(nrows-1):
    a = 1.0e30
    b = -1.0e30
    for k in range(nrows):
      if i != k:
        Dist = 0 
        for j in range(ncols):
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

def maximin(arr):
  """ Compute the maximin criterion for a given design. A higher value corresponds
  to a more regular scattering of design points.

  Args:
      arr (numpy.ndarray): A design matrix. If design matrix is not within [0,1], the 
      origianl design will be scaled to [0,1]
        
  Returns:
      float: Calculated maximin criterion

  Examples:
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=5)
      >>> pyLHD.maximin(example_LHD)
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