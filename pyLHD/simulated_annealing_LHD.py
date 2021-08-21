import numpy as np
from pyLHD.base_designs import rLHD
from pyLHD.utils import eval_design,exchange
from pyLHD.criteria import *
from datetime import datetime


def SA(n,k,N=10,T0=10,rate=0.1,Tmin=1,Imax=5,criteria='phi_p',
       p=15,q=1,maxtime=5):
  
  maxtime = maxtime * 60 # convert minutes to seconds
  
  counter = 1 # step 1: counter index
  
  X = rLHD(nrows=n,ncols=k) # step 2
  
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