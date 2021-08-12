import numpy as np
import math

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
      example_LHD = rLHD(nrows=5,ncols=3)
      exchange(example_LHD,idx=1,type='col')

      # Choose the first row of example_LHD and exchange two randomly selected elements.
      exchange(example_LHD,idx=1,type='row')
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
      example_LHD = rLHD(nrows=5,ncols=3)
      william_transformation(example_LHD)

      #Change the baseline
      william_transformation(example_LHD,baseline=5)
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
  example_OA = numpy.array([[1,1],[1,2],[1,3],[2,1],
                      [2,2],[2,3],[3,1],[3,2],[3,3] ])
  
  # Transfer the "OA" above into a LHD according to Tang (1993)

  OA2LHD(example_OA)        
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
