import math
import numpy as np
from collections import OrderedDict

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

# compute ecdf

def ecdf(x):
  ys = np.arange(1, len(x)+1)/float(len(x))
  d = dict(zip(np.argsort(x), ys.tolist()))
  l = OrderedDict(sorted(d.items(), key=lambda t: t[0])).values()
  return list(l)