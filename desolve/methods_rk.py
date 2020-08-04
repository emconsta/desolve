import numpy as np

def Default_RK_Methods():
    AllMethods_RK=[]

    # RK4
    RK={}
    A=np.zeros((4,4))
    A[1,0]=0.5
    A[2,1]=0.5
    A[3,2]=1.
    b=np.zeros((4))
    b[0]=1/6.
    b[1]=1/3.
    b[2]=1/3.
    b[3]=1/6.
    c=np.sum(A,1)
    RK['type']='RK'
    RK['A']=A
    RK['b']=b
    RK['c']=c
    RK['s']=np.size(b)
    RK['p']=4
    RK['name']='RK4'
    AllMethods_RK.append(RK)
    
    
    
    # RK3
    RK={}
    A=np.zeros((4,4))
    A[1,0]=0.5
    A[2,1]=0.75
    A[3,0]=2./9.
    A[3,1]=1./3.
    A[3,2]=4./9.
    b=np.zeros((4))
    b[0]=2/9.
    b[1]=1/3.
    b[2]=4./9.
    b[3]=0.
    c=np.sum(A,1)
    RK['type']='RK'
    RK['A']=A
    RK['b']=b
    RK['c']=c
    RK['s']=np.size(b)
    RK['p']=3
    RK['name']='RK3BS'
    AllMethods_RK.append(RK)

    # RK2a
    RK={}
    A=np.zeros((2,2))
    A[1,0]=1.
    b=np.zeros((2))
    b[0]=1/2.
    b[1]=1/2.
    c=np.sum(A,1)
    RK['type']='RK'
    RK['A']=A
    RK['b']=b
    RK['c']=c
    RK['s']=np.size(b)
    RK['p']=2
    RK['name']='RK2a'
    AllMethods_RK.append(RK)
    
    
    # FE
    RK={}
    A=np.zeros((1,1))
    A[0,0]=0.
    b=np.zeros((1))
    b[0]=1.
    c=np.sum(A,1)
    RK['type']='RK'
    RK['A']=A
    RK['b']=b
    RK['c']=c
    RK['s']=np.size(b)
    RK['p']=1
    RK['name']='RK-FE'
    AllMethods_RK.append(RK)
    
    return AllMethods_RK

