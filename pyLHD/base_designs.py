import numpy as np

# Generate a random Latin Hypercube Design (LHD)

def rLHD(nrows,ncols):
  """ Generate a random Latin Hypercube Design (LHD)

  Args:
      nrows (int): A positive integer specifying the number of rows
      ncols (int): A postive integer specifying the number of columns

  Returns:
      numpy.ndarray: return a random (nrows by ncols) LHD
  
  Examples:
      rLHD(nrows=5,ncols = 4)
  """
  rng = np.random.default_rng()
  rs = np.arange(start=1, stop=nrows+1)
  space = []
  for i in range(ncols):
    space.append(rng.choice(rs, nrows, replace=False))
  return np.asarray(space).transpose()


# Good Lattice Point Design

def GLPdesign(nrows,ncols,h = None):
  """ Good Lattice Point (GLP) Design 

  Args:
      nrows (int): A positive integer specifying the number of rows
      ncols (int): A postive integer specifying the number of columns
      h (list, optional): A list whose length is same as (ncols), with its elements that are smaller than and coprime to (nrows). 
      Defaults to None. If None, a random sample of (ncols) elements between 1 and (nrows-1).

  Returns:
      numpy.ndarray: A (nrows by ncols) GLP design.
  
  Examples:
      GLPdesign(nrows=5,ncols=3)
      GLPdesign(nrows=8,ncols=4,h=[1,3,5,7])
  """
  rng = np.random.default_rng()
  if h is None:
    seq = np.arange(start=1,stop=nrows)
    h_sample = rng.choice(seq,ncols,replace=False)
  else:
    h_sample = rng.choice(h,ncols,replace=False)
  
  mat = np.zeros((nrows,ncols))

  for i in range(nrows):
    for j in range(ncols):
      mat[i,j] = ((i+1)*h_sample[j])% nrows
  return mat.astype(int)


