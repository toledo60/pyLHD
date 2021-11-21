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