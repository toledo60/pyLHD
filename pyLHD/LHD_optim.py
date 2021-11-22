import numpy as np
from pyLHD.base_designs import rLHD
from pyLHD.utils import eval_design,exchange
from datetime import datetime

# Lioness Algorithm for Latin hypercube designs

def LA_LHD(n,k,prun=None,m=10,N=10,criteria='phi_p',
           p=15,q=1,maxtime=5):
  if prun is None:
    prun = 1/(k-1)
  
  maxtime = maxtime * 60
  
  counter = 1 # initialize counter
  X = np.zeros((m,n,k))  # solution matrix
  
  result = []
  for i in range(m):
    X[i] = rLHD(n,k,unit_cube=False)
    result.append(eval_design(X[i],p=p,q=q,criteria=criteria))
  result = np.asarray(result).reshape(-1,1)

  while counter <= N: # step 3
    time0 = datetime.now()
    onetoM = np.arange(start=1,stop=m+1).reshape(-1,1)
    temp = np.hstack((result,onetoM))
    temp = temp[np.argsort(temp[:,0])]

    # step 4: determine the top 3 agents
    centre = X[int(temp[0,1])-1]
    LW = X[int(temp[1,1])-1]
    RW = X[int(temp[2,1])-1]

    # step 4 ends
    m=6*k+3
    Xnew = np.zeros((m,n,k)) # new position matrix
    Xnew[0] = centre.copy()
    Xnew[1] = LW.copy()
    Xnew[2] = RW.copy()
    
    # step 5 starts\centre troop
    index =3
    for j in range(k):
      Xnew[index] = centre
      Xnew[index,:,j] = LW[:,j]
      index = index+1
      
      Xnew[index] = centre
      Xnew[index,:,j] = RW[:,j]
      index = index+1

    #LW troop
    
    for j in range(k):
      Xnew[index] = LW
      Xnew[index,:,j] = centre[:,j]
      index = index+1
      
      Xnew[index] = LW
      Xnew[index,:,j] = RW[:,j]
      index = index +1
    
    # RW troop
    
    for j in range(k):
      Xnew[index] = RW
      Xnew[index,:,j] = centre[:,j]
      index = index+1
      
      Xnew[index] = RW
      Xnew[index,:,j] = LW[:,j]
      index = index+1
      
    # step 5 ends here
    X = Xnew

    for i in range(1,m): # step 6
      for j in range(k): # step 7
        z =  np.random.uniform(1,0,1) # step 8
        if z <= prun:
          X[i] = exchange(X[i],idx=j) # step 9
    
    # update criteria for all agents
    
    result = []
    for i in range(m):
      result.append(eval_design(X[i],p=p,q=q,criteria=criteria))
    result = np.asarray(result).reshape(-1,1) 
    time1 = datetime.now()
    timeDiff = time1-time0
    timeALL = timeDiff.total_seconds()
    
    if timeALL <= maxtime:
      counter = counter+1
    else:
      counter = N+1
    onetoM = np.arange(start=1,stop=m+1).reshape(-1,1)
    temp = np.hstack((result,onetoM))

    temp = temp[np.argsort(temp[:,0])]
    centre = X[int(temp[0,1])-1]
    
  return centre.astype('int')



# Simulated Annealing for Latin hypercube designs

def SA_LHD(n,k,N=10,T0=10,rate=0.1,Tmin=1,Imax=5,criteria='phi_p',
           p=15,q=1,maxtime=5):
  
  maxtime = maxtime * 60 # convert minutes to seconds
  
  counter = 1 # step 1: counter index
  
  X = rLHD(nrows=n,ncols=k,unit_cube=False) # step 2
  
  Xbest = X.copy()
  TP = T0
  Flag =1
  
  rng = np.random.default_rng()
  
  while counter <= N:
    time0 = datetime.now()
    while Flag == 1 & TP>Tmin:
      Flag = 0
      I = 1
      
      while I <= Imax:
        rs = np.arange(start=1,stop=k+1)
        rcol = rng.choice(rs, 1, replace=False) #step 3:Randomly choose a column
        
        Xnew = exchange(arr=X,idx=rcol) #step 4:Exchange two random elements from column 'rcol'
        
        #step 5 begins here
        a = eval_design(Xnew,criteria=criteria,p=p,q=q)
        b = eval_design(X,criteria=criteria,p=p,q=q)
        
        if a < b:
          X = Xnew
          Flag = 1
        else:
          prob = np.exp((b-a)/TP)
          draw = rng.choice(np.arange(0,2),1,p = [1-prob,prob],replace=False)
          
          if draw == 1:
            X=Xnew
            Flag =1 
        #step 5 ends here
        
        c = eval_design(Xbest,criteria=criteria,p=p,q=q)
        
        if a <c:
          Xbest = Xnew
          I=1
        else:
          I = I +1
      TP = TP*(1-rate)
    
    time1 = datetime.now()
    timeDiff = time1-time0
    
    timeALL = timeDiff.total_seconds()
    
    if timeALL <= maxtime:
      counter = counter+1
    else:
      counter = N+1
    TP=T0
    Flag =1

  return Xbest