import numpy as np

def Default_IDE_Methods():
    AllMethods_IDE=[]
    # RK2a-Trapezoidal for IDE
    IDE={}
    A=np.zeros((2,2))
    A[1,0]=1.
    b=np.zeros((2))
    b[0]=1/2.
    b[1]=1/2.
    c=np.sum(A,1)
    IDE['type']='RK-IDE'
    IDE['A']=A
    IDE['b']=b
    IDE['c']=c
    IDE['s']=np.size(b)
    IDE['p']=2
    IDE['name']='RK2a-Trap'
    AllMethods_IDE.append(IDE)

    return AllMethods_IDE

