import numpy as np
import pyLHD

# Exchange two random elements in a matrix

def exchange(arr, idx, type='col'):
  """ Exchange two random elements in a matrix

  Args:
      arr (numpy.ndarray): A design matrix
      idx (int): A positive integer, which stands for the (idx) column or row of (arr) 
      type (str, optional): If type is 'col', two random elements will be exchanged within column (idx).
      If type is 'row', two random elements will be exchanged within row (idx). Defaults to 'col'.

  Returns:
      numpy.ndarray: A new design matrix after the exchange
  
  Examples:
      # Choose the first column of example_LHD and exchange two randomly selected elements
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
      >>> pyLHD.exchange(example_LHD,idx=1,type='col')

      # Choose the first row of example_LHD and exchange two randomly selected elements.
      >>> pyLHD.exchange(example_LHD,idx=1,type='row')
  """
  nrows = arr.shape[0]
  ncols = arr.shape[1]
  rng = np.random.default_rng()

  if type == 'col':
    location = rng.choice(nrows, 2, replace=False)

    arr[location[0], idx], arr[location[1],idx] = arr[location[1], idx], arr[location[0], idx]
  else:
    location = rng.choice(ncols, 2, replace=False)
    arr[idx, location[0]], arr[idx, location[1]] = arr[idx, location[1]], arr[idx, location[0]]

  return arr

# Williams Transformation

def williams_transform(arr,baseline=1):
  """ Williams Transformation

  Args:
      arr (numpy.ndarray): A design matrix
      baseline (int, optional): A integer, which defines the minimum value for each column of the matrix. Defaults to 1.

  Returns:
      numpy.ndarray: After applying Williams transformation, a matrix whose sizes are the same as input matrix
  
  Examples:
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
      >>> pyLHD.william_transformation(example_LHD)

      #Change the baseline
      >>> pyLHD.william_transformation(example_LHD,baseline=5)
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

def OA2LHD(orthogonal_array):
  """ Transfer an Orthogonal Array (OA) into an LHD

  Args:
      orthogonal_array (numpy.ndarray): An orthogonal array matrix

  Returns:
      numpy.ndarray: LHD whose sizes are the same as input OA. The assumption is that the elements of OAs must be positive
  
  Examples:
      # Create an OA(9,2,3,2) 
      >>> example_OA = numpy.array([[1,1],[1,2],[1,3],[2,1],
                          [2,2],[2,3],[3,1],[3,2],[3,3] ])
  
      # Transfer the "OA" above into a LHD according to Tang (1993)
      >>> pyLHD.OA2LHD(example_OA)        
  """
  n = orthogonal_array.shape[0]
  m = orthogonal_array.shape[1]
  s = np.unique(orthogonal_array[:,0]).size

  lhd = orthogonal_array
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

def eval_design(arr,criteria = 'phi_p',p=15,q=1):
  """ Evaluate a design based on a chosen criteria

  Args:
      arr (numpy.ndarray): A design matrix
      criteria (str, optional): Criteria to choose from. Defaults to 'phi_p'. 
      Options include 'phi_p','MaxProCriterion','AvgAbsCor','AvgAbsCor'
      p (int): A positive integer, which is the parameter in the phi_p formula. The default is set to be 15
      q (int): If (q) is 1, (dij) is the Manhattan (rectangular) distance. If (q) is 2, (dij) is the Euclidean distance.

  Returns:
      float: Calculation of chosen criteria for any LHD

  Examples:
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3)
      >>> pyLHD.eval_design(example_LHD) # phi_p with default settings
      # evaluate design based on MaxProCriterion 
      >>> pyLHD.eval_design(example_LHD,criteria='MaxProCriterion')    
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

def adjust_range(arr,min,max,digits=None):
  """ Adjust the range of a design to [min,max]

  Args:
      arr (numpy.ndarray): A design matrix
      min (float): desired lower bound of design 
      max (float): desired upper bound of design 
      digits (int): number of digits to which the design is rounded
  Returns:
      float: Design with new range [min,max]

  Examples:
      >>> example_LHD = pyLHD.rLHD(nrows=5,ncols=3,unit_cube=True)
      >>> pyLHD.adjust_range(example_LHD,-1, 1)
      >>> pyLHD.adjust_range(example_LHD, 5, 12, digits = 3)
  """  
  if (min == max):
    raise ValueError('min and max should be different values')
  if (min > max):
    raise ValueError('make sure min < max')
  
  arr_min = np.amin(arr)
  arr_max = np.amax(arr)
  
  range_diff = arr_max - arr_min
  result = (arr - arr_min)/range_diff * (max-min) + min
  
  if digits is None:
    result = result
  else:
    result = np.around(result,digits)
  return result


def scale(arr,uniformize=False):
  """ Scales design to be within [0,1]
  
  Args:
      arr (numpy.ndarray): A design matrix
      uniformize (bool): If True, Rosenblatt transformation is applied 
      (uniformize by applying the empirical cumulative distribution). 
      If False (default), scaling is done by using the minimum and maximum value
  Returns:
      float: The scaled design

  Examples:
      >>> example_LHD = rLHD(nrows=5,ncols=3,unit_cube=True)
      >>> scale(example_LHD)
      >>> scale(example_LHD,uniformize=True)
  """  
  min = np.amin(arr,axis=0)
  range = np.ptp(arr, axis=0)
  
  if uniformize:
    design = np.apply_along_axis(pyLHD.ecdf, 0, arr)
  else:
    design = (arr - min)/range
  return design