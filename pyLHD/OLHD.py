import numpy as np
import pyLHD

# --- Orthogonal Latin Hypercube Designs --- #


# --- Butler, N.A. (2001) Construction --- #

def OLHD_Butler01(nrows, ncols):
  """ Orthogonal Latin Hypercube Design (OLHD). Based on the construction method of Butler (2001)

  Args:
      nrows (int): A positive integer specifying the number of rows
      ncols (int): A postive integer specifying the number of columns

  Raises:
      ValueError: If ncols is not less than or equal to nrows
      ValueError: If nrows is not greater than or equal to 3
      ValueError: If nrows is not an odd prime number

  Returns:
      numpy.ndarray: A (nrows by ncols) orthogonal LHD
  
  Examples:
      Create an orthogonal LHD with nrows =11 and ncols =5
      >>> OLHD_Butler01(nrows=11,ncols=5)
      
      Create an orthogonal LHD with nrows =11 and ncols =5
      >>> OLHD_Butler01(nrows=7,ncols=6)
  """
  if ncols >= nrows:
    raise ValueError("ncols must be less than or equal to nrows")
  if nrows < 3:
    raise ValueError("nrows must be greater than or equal to 3")
  if (not pyLHD.is_prime(nrows) or nrows % 2 != 1):
    raise ValueError("nrows must be an odd prime number")

  n0 = int((nrows-1)/2)
  rng = np.random.default_rng()

  if ncols <= n0:
    seq = np.arange(start=1, stop=n0+1)
    g = rng.choice(seq, ncols, replace=False)

    W = np.zeros((nrows, ncols))

    for i in range(nrows):
      for j in range(ncols):
        if (nrows % 4 == 1):
          W[i, j] = ((i+1)*g[j] + (nrows-1)/4) % nrows
        if(nrows % 4 == 3):
          W[i, j] = ((i+1) * g[j] + (3*nrows - 1)/4) % nrows

    X = pyLHD.williams_transform(W)

  else:
    g0 = np.arange(start=1, stop=n0+1)
    W0 = np.zeros((nrows, n0))

    for i in range(nrows):
      for j in range(n0):

        if (nrows % 4 == 1):
          W0[i, j] = ((i+1)*g0[j] + (nrows-1)/4) % nrows
        if (nrows % 4 == 3):
          W0[i, j] = ((i+1)*g0[j] + (3*nrows-1)/4) % nrows

    X0 = pyLHD.williams_transform(W0)

    r = ncols - n0
    seq = np.arange(start=1, stop=n0+1)
    g1 = rng.choice(seq, r, replace=False)

    W1 = np.zeros((nrows, r))

    for i in range(nrows):
      for j in range(r):
        W1[i, j] = ((i+1)*g1[j]) % nrows

    X1 = pyLHD.williams_transform(W1)

    X = np.column_stack((X0, X1))

  return X

# --- Sun et al. (2010) Construction --- #

def OLHD_Sun10(C, r, type='odd'):
  """Orthogonal Latin Hypercube Design (OLHD). Based on the construction method of Sun et al. (2010)

  Args:
      C (int): A positve integer.
      r (int): A positve integer.
      type (str, optional): Run size of design, this can be either odd or even. Defaults to 'odd'.
      If (type) is 'odd' the run size of the OLHD will be (r*2^(C+1)+1). If (type) is 'even' the run size of
      the OLHD will be (r*2^(C+1))

  Returns:
      numpy.ndarray: An orthogonal LHD with the following run size: (r*2^(C+1)+1) if type ='odd', or (r*2^(C+1)) if type ='even'.
      The resulting columns will be (2^(C))
  
  Examples:
      Create an orthogonal LHD with C=3, r=3, type = 'odd'
      So n = (3*2^(3+1) )+1 = 49 (rows) and k=2^(3)=8 (columns)
      >>> OLHD_Sun10(C=3,r=3,type='odd')

      Create an orthogonal LHD with C=3, r=3, type = 'even'
      So n = 3*2^(3+1) = 48 (rows) and k=2^(3)=8 (columns)
      >>> OLHD_Sun10(C=3,r=3,type='even')  
  """

  Sc = np.array([[1, 1], [1, -1]])
  Tc = np.array([[1, 2], [2, -1]])

  if C >= 2:
    counter = 2
    while counter <= C:
      Sc_star = Sc.copy()
      Tc_star = Tc.copy()

      index = int((Sc_star.shape[0])/2)

      for i in range(index):
        Sc_star[i, :] = -1*Sc_star[i, :]
        Tc_star[i, :] = -1*Tc_star[i, :]

      a = np.vstack((Sc, Sc))
      b = np.vstack((-1*Sc_star, Sc_star))
      c = np.vstack((Tc, Tc + Sc*2**(counter-1)))
      d = np.vstack((-1*(Tc_star + Sc_star*2**(counter-1)), Tc_star))

      Sc = np.hstack((a, b))
      Tc = np.hstack((c, d))
      counter = counter+1

  if type == 'odd':
    A = [Tc.copy() + Sc.copy()*(i)*2**(C) for i in range(r)]
    A_vstack = np.vstack(A)
    CP = np.zeros((1,2**C))
    X = np.concatenate((A_vstack, CP, (-1)*A_vstack), axis=0)

  if type == 'even':
    Hc = Tc.copy() - Sc.copy()*0.5
    B = [Hc + Sc.copy()*(i)*2**(C) for i in range(r)]
    B_vstack = np.vstack(B)
    X = np.vstack((B_vstack,-B_vstack))

  return X


# --- Cioppa and Lucas (2007) Constuction --- #

def OLHD_Cioppa07(m):
  """Orthogonal Latin Hyercube Design. Based on the construction method of Cioppa and Lucas (2007)

  Args:
      m (int): A positive integer, and it must be greater than or equal to 2 

  Raises:
      ValueError: If m is not greater than or equal to 2

  Returns:
      numpy.ndarray: An orthogonal LHD with the following run size: (n=2^m + 1) and 
      factor size: (k= m+ (m-1 choose 2))
  
  Examples:
      # Create an orthogonal LHD with m=4. So n=2^m+1=17 runs and k=4+3=7 factors
      >>> OLHD_Cioppa07(m=4)

      # Create an orthogonal LHD with m=5. So n=2^m+1=33 runs and k=5+7=11 factors
      >>> OLHD_Cioppa07(m=5)
  """
  if m < 2:
     raise ValueError('m must be greater than or equal to 2')
   
  q = 2**(m-1) 
  # construction of M starts  
  e = np.arange(1, q+1).reshape(-1,1)
  
  I = np.eye(2)
  R = np.array([[0,1],[1,0]])
  
  AL = np.zeros((m-1,q,q)) #there are m-1 of AL's
  
  if m==2:
    AL[m-2] = R.copy()
    M = np.hstack( (e, np.matmul(AL[m-2],e) ))
  
  if m > 2:
    
    for i in range(m-2):
      a = 1
      b = 1
      
      for j in range(m-1-(i+1)):
        a = np.kron(a,I)
      
      for k in range(i+1):
        b = np.kron(b,R)
      
      AL[i] = np.kron(a,b)
    
    c = 1
    
    for l in range(m-1):
      c = np.kron(c,R)
    
    AL[m-2] = c.copy()
    M = e.copy()
    
    for i in range(m-1):
      M = np.hstack( (M,np.matmul(AL[i],e)  ) )
    
    for i in range(m-1):
      for j in range(i+1,m-1):
        M= np.hstack((M,AL[i] @ AL[j] @ e))
    
  # construction of M ends  
  
  # Construction of S starts
  
  j = np.ones(q).reshape(-1,1)
  
  ak = np.zeros((m-1,q,1))
  B = np.ones((m-1,2,1))
  
  if m==2:
    B[m-2,0,:]=-1
    ak[m-2] = B[0]
    S = np.hstack((j,ak[m-2]))
  
  if m > 2:
    for i in range(m-1):
      temp = B.copy()
      temp[m-(i+2),0,:] = -1
      d=1
      
      for k in range(m-1):
        d = np.kron(d,temp[k])
      
      ak[i] = d.copy()
    
    S = j.copy()
    
    for i in range(m-1):
      S = np.hstack((S,ak[i]))
    
    for i in range(m-2):
      for j in range(i+1,m-1):
        S = np.hstack((S,ak[i]*ak[j]) )
          
  # construction of S ends
  
  # construction of T starts
  
  if m==2:
    T0 = np.zeros((q,2))
    
    for i in range(q):
      for k in range(2):
        T0[i,k] = M[i,k] * S[i,k]
    
    CP = np.zeros((1,2))
  
  if m>2:
    T0 = np.zeros((q,m+pyLHD.comb(m-1,2)))

    for i in range(q):
      for k in range(m+pyLHD.comb(m-1,2)):
        T0[i,k] = M[i,k]*S[i,k]
    # Construction of T ends

    CP = np.zeros((1,m+pyLHD.comb(m-1,2)))

  X = np.vstack((T0,CP,-T0))
    
  return X


# --- Ye (1998) Constuction --- #

def OLHD_Ye98(m):
  """Orthogonal Latin Hyercube Design. Based on the construction method of Ye (1998)

  Args:
      m (int): A positive integer, and it must be greater than or equal to 2 

  Raises:
      ValueError: If m is not greater than or equal to 2

  Returns:
      numpy.ndarray: An orthogonal LHD with the following run size: (n=2^m + 1) and 
      factor size: (k=2m-2)

  Examples:
      # Create an orthogonal LHD with m=4. So n=2^m+1=9 runs and k=2*m-2=4 factors
      >>> OLHD_Ye98(m=3)

      # Create an orthogonal LHD with m=5. So n=2^m+1=17 runs and k=2*m-2=6 factors
      >>> OLHD_Ye98(m=4)      
  """
  if m < 2:
    raise ValueError('m must be greater than or equal to 2')
  
  rng = np.random.default_rng()
  q = 2**(m-1)
  # construction of M starts
  e = rng.choice(np.arange(1,q+1),q,replace=False).reshape(-1,1)
  
  I = np.eye(2)
  R = np.array([[0,1],[1,0]])
  
  AL = np.zeros((m-1,q,q)) #there are m-1 of AL's  
  
  if m==2:
    AL[m-2] = R.copy()
    M = np.hstack( (e, np.matmul(AL[m-2],e) ))  

  if m > 2:
    
    for i in range(m-2):
      a = 1
      b = 1
      
      for _ in range(m-1-(i+1)):
        a = np.kron(a,I)
      
      for _ in range(i+1):
        b = np.kron(b,R)
      
      AL[i] = np.kron(a,b)
    
    c = 1
    
    for _ in range(m-1):
      c = np.kron(c,R)
    
    AL[m-2] = c.copy()
    M = e.copy()
    
    for i in range(m-1):
      M = np.hstack( (M,np.matmul(AL[i],e)  ) )
    
    for i in range(m-2):
      M= np.hstack((M,AL[i] @ AL[m-2] @ e)) 
    
  # construction of M ends  

  # Construction of S starts
  
  j = np.ones(q).reshape(-1,1)
  
  ak = np.zeros((m-1,q,1))
  B = np.ones((m-1,2,1))

  if m==2:
    B[:,0,m-2]=-1
    ak[m-2] = B[0]
    S = np.hstack((j,ak[m-2]))

  if m > 2:
    for i in range(m-1):
      temp = B.copy()
      temp[m-(i+2),0,:] = -1
      d=1
      
      for k in range(m-1):
        d = np.kron(d,temp[k])
      
      ak[i] = d.copy()
    
    S = j.copy()
    
    for i in range(m-1):
      S = np.hstack((S,ak[i]))
    
    for i in range(1,m-1):
      S = np.hstack((S,ak[0]*ak[i]) )
          
  # construction of S ends
  
  # construction of T starts
  
  T0 = np.zeros((q,2*m-2))
  
  for i in range(q):
    for k in range(2*m-2):
      T0[i,k] = M[i,k]*S[i,k]
  
  # constuction of T ends
  
  CP = np.zeros((1,2*m-2))
  X = np.vstack((T0,CP,-T0))

  return X
  

# --- Lin et al. (2009) Constuction --- #

def OLHD_Lin09(OLHD,OA):
  """Orthogonal Latin Hypercube Design. Based on the construction method of Lin et al. (2009)

  Args:
      OLHD ([type]): An orthogonal Latin hypercube design with run size (n) and factor size (p), 
      and it will be coupled with the input orthogonal array
      OA ([type]): An orthogonal array, with (n^2) rows, (2f) columns, (n) symbols, 
      strength two and index unity is available, which can be denoted as OA(n^2,2f,n,2)

  Returns:
      numpy.ndarray: orthogonal Latin hypercube design with the following run size: (n^2) 
      and the following factor size: (2fp)
  
  Examples:
  # Create a 5 by 2 OLHD
  >>> OLHD_example = OLHD_Cioppa07(m=2)
  # Create an OA(25,6,5,2)
  >>> OA_example = np.array([ [2,2,2,2,2,1],[2,1,5,4,3,5],
                              [3,2,1,5,4,5],[1,5,4,3,2,5],
                              [4,1,3,5,2,3],[1,2,3,4,5,2],
                              [1,3,5,2,4,3],[1,1,1,1,1,1],
                              [4,3,2,1,5,5],[5,5,5,5,5,1],
                              [4,4,4,4,4,1],[3,1,4,2,5,4],
                              [3,3,3,3,3,1],[3,5,2,4,1,3],
                              [3,4,5,1,2,2],[5,4,3,2,1,5],
                              [2,3,4,5,1,2],[2,5,3,1,4,4],
                              [1,4,2,5,3,4],[4,2,5,3,1,4],
                              [2,4,1,3,5,3],[5,3,1,4,2,4],
                              [5,2,4,1,3,3],[5,1,2,3,4,2],
                              [4,5,1,2,3,2]   ])
  # Construct a 25 by 12 OLHD
  >>> OLHD_Lin09(OLHD = OLHD_example,OA = OA_example)
  """
  n1 = OLHD.shape[0]
  k = OLHD.shape[1]
  
  n2 = np.unique(OA[:,0]).size
  f = int(OA.shape[1]*0.5)
  
  l = []
  for i in range(k):
    l.append(OA.copy())
  
  A = np.stack(l)
  M = np.zeros((k,n2**2,2*f))
  
  V = np.array([[1,-n2],[n2,1]])
  
  for i in range(k):
    for j in range(n2):
      for m in range(2*f):
        location = np.where(A[i,:,m]==(j+1))
        A[i,location,m] = OLHD[j,i]
  
  M_list = []
  for i in range(k):
    for j in range(f):
      M[i,:,2*(j+1)-2:2*(j+1)] = A[i,:,2*(j+1)-2:2*(j+1)] @ V 
    M_list.append(M[i])    
    
  return np.hstack(M_list)