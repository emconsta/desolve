import numpy as np
from methods_rk import Default_RK_Methods


def Default_MRK_Methods():
    AllMethods_MRK=[]

    # MPRK2
    MRK={}

    AB=np.zeros((2,2))
    AB[1,0]=1.
    bB=np.zeros((2))
    bB[0]=1/2.
    bB[1]=1/2.
    cB=np.sum(AB,1)
    sB=np.size(bB)
    
    
    AF=np.zeros((sB*2,sB*2))
    for i in range(sB):
        for j in range(sB):
            AF[i,j]=0.5*AB[i,j]
            AF[i+sB,j+sB]=0.5*AB[i,j]
    for i in range(sB):
        for j in range(sB):
            AF[sB+i,j]=bB[j]/2.

    bF=np.zeros((sB*2))
    for i in range(sB):
        bF[i]=0.5*bB[i]
        bF[i+sB]=0.5*bB[i]
    cF=np.sum(AF,1)

    AS=np.zeros((sB*2,sB*2))
    for i in range(sB):
        for j in range(sB):
            AS[i,j]=AB[i,j]
            AS[i+sB,j+sB]=AB[i,j]
    bS=np.zeros((sB*2))
    for i in range(sB):
        bS[i]=0.5*bB[i]
        bS[i+sB]=0.5*bB[i]
        
    cS=np.sum(AS,1)

    
    MRK['type']='MRK'
    MRK['AB']=AB
    MRK['bB']=bB
    MRK['cB']=cB
    MRK['sB']=sB
    MRK['pB']=2

    MRK['AF']=AF
    MRK['bF']=bF
    MRK['cF']=cF
    MRK['sF']=np.size(bF)
    MRK['pF']=2

    MRK['AS']=AS
    MRK['bS']=bS
    MRK['cS']=cS
    MRK['sS']=np.size(bS)
    MRK['pS']=2

    
    MRK['name']='MPRK2'
    AllMethods_MRK.append(MRK)



    # MPRK2
    MRK={}

    AB=np.zeros((3,3))
    AB[1,0]=1.0
    AB[2,0]=0.25
    AB[2,1]=0.25
    bB=np.zeros((3))
    bB[0]=1./6.
    bB[1]=1./6
    bB[2]=2./3.
    cB=np.sum(AB,1)
    sB=np.size(bB)
    
    
    AF=np.zeros((sB*2,sB*2))
    for i in range(sB):
        for j in range(sB):
            AF[i,j]=0.5*AB[i,j]
            AF[i+sB,j+sB]=0.5*AB[i,j]
    for i in range(sB):
        for j in range(sB):
            AF[sB+i,j]=bB[j]/2.
 

    bF=np.zeros((sB*2))
    for i in range(sB):
        bF[i]=0.5*bB[i]
        bF[i+sB]=0.5*bB[i]
    cF=np.sum(AF,1)

    AS=np.zeros((sB*2,sB*2))
    for i in range(sB):
        for j in range(sB):
            AS[i,j]=AB[i,j]
            AS[i+sB,j+sB]=AB[i,j]
    bS=np.zeros((sB*2))
    for i in range(sB):
        bS[i]=0.5*bB[i]
        bS[i+sB]=0.5*bB[i]
        
    cS=np.sum(AS,1)

    
    MRK['type']='MRK'
    MRK['AB']=AB
    MRK['bB']=bB
    MRK['cB']=cB
    MRK['sB']=sB
    MRK['pB']=3

    MRK['AF']=AF
    MRK['bF']=bF
    MRK['cF']=cF
    MRK['sF']=np.size(bF)
    MRK['pF']=3

    MRK['AS']=AS
    MRK['bS']=bS
    MRK['cS']=cS
    MRK['sS']=np.size(bS)
    MRK['pS']=3

    
    MRK['name']='MPRK2-RK3SSPHig'
    AllMethods_MRK.append(MRK)

    
    return AllMethods_MRK

