import math
import sys

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

# show progress bar for iterations

class progress_bar:
  
  def __init__(self,i,total):
    """Basic progress bar for iterations

    Args:
        i (int): current iteration
        total (int): total number of iterations
    Examples:
        >>> it = 100
        >>> for i in range(it)
        >>>   progress_bar(i,it).show()
        >>> progress_bar.end()
    """
    self.total = total
    self.i = i
  
  def show(self):
    percent = 100.0*(self.i+1)/(self.total)
    sys.stdout.write('\r')
    sys.stdout.write("Completed: [{:{}}] {:>3}%"
                    .format('='*int(percent/(100.0/50)),
                            50, int(percent)))
    sys.stdout.flush()
    
  def end():
    sys.stdout.write("\n")