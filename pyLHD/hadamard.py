import numpy as np
from pyLHD.helpers import is_prime
import math
import galois
import numpy.typing as npt

def is_Hadamard(arr: npt.ArrayLike, rtol=1e-05, atol=1e-08) -> bool:
  """ Determine if a matrix is a Hadamard matrix.

  Args:
      arr (npt.ArrayLike): A numpy array.

  Raises:
      ValueError: If provided array is not a square matrix.
      ValueError: If number of rows is not a power of 2 or not divisible by 4.
      ValueError: If values are not +1 or -1.
      ValueError: If H*H.T != n*I, where I is the identity matrix of order n.

  Returns:
      True if given array follows Hadamard properties, otherwise False.
  
  Examples:
  ```{python}
  import pyLHD
  H1 = pyLHD.sylvester(n=8)
  H1
  ```
  ```{python}
  pyLHD.is_Hadamard(H1)
  ```
  ```{python}
  H2 = pyLHD.paley(p=7,k=1)
  H2
  ```
  ```{python}
  pyLHD.is_Hadamard(H2)
  ```
  """
  nrows, ncols = arr.shape

  if nrows != ncols:
    raise ValueError('Must be a square matrix.')

  # Hadamard matrices exist for orders 1, 2, and multiples of 4.
  if nrows == 2 or nrows % 4 == 0 or nrows == 1:
    pass
  else:
    raise ValueError('Number of rows must be 1, 2, or a multiple of 4.')

  if not np.all(np.isin(arr, [-1, 1])):
    raise ValueError('Elements can only be +1 or -1.')

  diag = nrows * np.eye(nrows)
  crossprod = np.dot(arr, arr.T)

  if not np.allclose(diag, crossprod, rtol=rtol, atol=atol):
    raise ValueError('Not a Hadamard matrix. H*H.T != n*I.')
  return True


def sylvester(n:int) -> np.ndarray:
  """Hadamard matrix based on Sylvester's construction

  Args:
      n (int): The order of the matrix. n must be a power of 2.

  Raises:
      ValueError: If `n` is not a positive integer and not a power of 2.

  Returns:
      The Hadamard matrix of order n.
  Examples:
  ```{python}
  import pyLHD
  pyLHD.sylvester(n=4)
  ```
  ```{python}
  pyLHD.sylvester(n=8)
  ```
  """
  # Construction obtained from scipy.linalg.hadamard
  if n < 1:
    lg2 = 0
  else:
    lg2 = int(math.log(n, 2))
  if 2 ** lg2 != n:
    raise ValueError("n must be an positive integer, and n must be "
                      "a power of 2")
  H = np.array([[1]], dtype=int)

  for _ in range(0, lg2):
    H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
  return H

####################################
#### Payley's Construction  ########
####################################


def jacobsthal_matrix(p:int, k:int) -> np.ndarray:
  """Generate a Jacobsthal matrix 

  Args:
      p (int): A prime integer
      k (int): An integer power

  Raises:
      ValueError: If `p` is not a prime number
      ValueError: If `p^k + 1` is not divisible by 4

  Returns:
      Jacobsthal matrix of order (p^k)
  """
  if not is_prime(p):
    raise ValueError("`p` must be a prime number")
  if (1 + p**k) % 4 !=0:
    raise ValueError("`p^k + 1` must be divisible by 4")
  q = p**k
  Q = np.zeros((q, q), dtype=int)  # Initialize the Jacobsthal matrix
  
  for a in range(q):
    for b in range(q):
      diff = (a - b) % q
      Q[a, b] = galois.jacobi_symbol(diff, q)
  return Q


def paley(p:int,k:int, method:int = 1) -> np.ndarray:
  """Paley Construction 

  Args:
      p (int): A prime integer
      k (int): An integer power
      method (int, optional): Paley construction I or Paley construction II. Defaults to 1.
          See https://en.wikipedia.org/wiki/Paley_construction for more details on construction

  Raises:
      ValueError: If `p` is not a prime number
      ValueError: If `p^k + 1` is not divisible by 4
      ValueError: If `method` is not 0 or 1

  Returns:
      Hadamard matrix based on Paley Constructions
  
  Examples:
  ```{python}
  import pyLHD
  pyLHD.paley(p=7,k=1)
  ```
  ```{python}
  pyLHD.paley(p=7,k=1, method = 2)
  ```      
  """
  if not is_prime(p):
    raise ValueError("`p` must be a prime number")
  if (1 + p**k) % 4 !=0:
    raise ValueError("`p^k + 1` must be divisible by 4")
  Q = jacobsthal_matrix(p=p,k=k) #(p^k x p^k)
  n = p**k
  
  ones = np.ones(n)
  ones_t = ones.reshape(1,-1)
  negative_ones = (-1)*ones.reshape(-1,1)

  S = np.zeros((n+1,n+1))
  S[0,0] = 0
  S[0,1:] = ones_t
  S[1:,0] = negative_ones[:,0]
  S[1:,1:] = Q

  I = np.identity(n=(n+1))

  if method == 1:
    return S+I
  elif method == 2:
    A = np.kron(S,np.array([[1,1],[1,-1]]))
    B = np.kron(I,np.array([[1,-1],[-1,-1]]))
    return A + B
  else:
    raise ValueError("`method` can be either 1 or 2 only.")