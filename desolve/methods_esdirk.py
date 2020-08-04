import numpy as np

def Default_ESDIRK_Methods():
    AllMethods_ESDIRK=[]

    # BE
    ESDIRK={}
    
    A=np.zeros((1,1))
    A[0,0]=1.
    
    b=np.zeros((1))
    
    b[:]=A[0,:]
       
    c=np.sum(A,1)
    ESDIRK['type']='ESDIRK'
    ESDIRK['A']=A
    ESDIRK['b']=b
    ESDIRK['c']=c
    ESDIRK['s']=np.size(b)
    ESDIRK['p']=1
    ESDIRK['name']='BE'
    
    AllMethods_ESDIRK.append(ESDIRK)

    # CN
    ESDIRK={}
    
    A=np.zeros((2,2))
    A[1,0]=.5
    A[1,1]=.5
    
    b=np.zeros((2))
    b[:]=A[1,:]
    
    c=np.sum(A,1)
    ESDIRK['type']='ESDIRK'
    ESDIRK['A']=A
    ESDIRK['b']=b
    ESDIRK['c']=c
    ESDIRK['s']=np.size(b)
    ESDIRK['p']=1
    ESDIRK['name']='ESDIRK-CN'

    AllMethods_ESDIRK.append(ESDIRK)

    # ARK2e
    ESDIRK={}
    
    A=np.zeros((3,3))
    A[1,0]=1.-1/np.sqrt(2.)
    A[1,1]=1.-1/np.sqrt(2.)
    A[2,0]=1./(2*np.sqrt(2.))
    A[2,1]=1./(2*np.sqrt(2.))
    A[2,2]=1.-1/np.sqrt(2.)
    
    b=np.zeros((3))
    b[:]=A[2,:]
    
    c=np.sum(A,1)
    ESDIRK['type']='ESDIRK'
    ESDIRK['A']=A
    ESDIRK['b']=b
    ESDIRK['c']=c
    ESDIRK['s']=np.size(b)
    ESDIRK['p']=1
    ESDIRK['name']='ESDIRK-ARK2e'

    AllMethods_ESDIRK.append(ESDIRK)
    

    return AllMethods_ESDIRK
