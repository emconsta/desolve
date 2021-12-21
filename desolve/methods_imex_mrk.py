import numpy as np

def Default_IMEX_MRK_Methods():
    AllMethods_IMEX_MRK=[]


    # PRK2 - IMEX
    IMEX_MRK={}

    AB=np.zeros((2,2))
    AB[1,0]=1.
    bB=np.zeros((2))
    bB[0]=1/2.
    bB[1]=1/2.
    cB=np.sum(AB,1)


    AF=np.zeros((4,4))
    AF[1,0]=0.5
    AF[2,0]=0.25
    AF[2,1]=0.25
    AF[3,0]=0.25
    AF[3,1]=0.25
    AF[3,2]=0.5
    bF=np.zeros((4))
    bF[0]=1/4.
    bF[1]=1/4.
    bF[2]=1/4.
    bF[3]=1/4.
    cF=np.sum(AF,1)
  

    AS=np.zeros((4,4))
    AS[1,0]=0.5
    AS[2,0]=0.25
    AS[2,1]=0.25
    AS[3,0]=0.25
    AS[3,1]=0.25
    AS[3,2]=0.5
    bS=np.zeros((4))
    bS=bF[:]
    cS=np.sum(AS,1)

    AT=np.zeros((4,4))
    AT[3,0]=1.
    AT[3,1]=1.
    AT[3,2]=1.
    AT[3,3]=1.
    bT=np.zeros((4))
    bT=bF[:]
    cT=np.sum(AT,1)

    
    IMEX_MRK['type']='IMEX-MRK'
    IMEX_MRK['AB']=AB
    IMEX_MRK['bB']=bB
    IMEX_MRK['cB']=cB
    IMEX_MRK['sB']=np.size(bB)
    IMEX_MRK['pB']=2

    IMEX_MRK['AF']=AF
    IMEX_MRK['bF']=bF
    IMEX_MRK['cF']=cF
    IMEX_MRK['sF']=np.size(bF)
    IMEX_MRK['pF']=2

    IMEX_MRK['AS']=AS
    IMEX_MRK['bS']=bS
    IMEX_MRK['cS']=cS
    IMEX_MRK['sS']=np.size(bS)
    IMEX_MRK['pS']=2

    IMEX_MRK['AT']=AT
    IMEX_MRK['bT']=bT
    IMEX_MRK['cT']=cT
    IMEX_MRK['sT']=np.size(bT)
    IMEX_MRK['pT']=1
    
    
    IMEX_MRK['name']='PRK2-IMEX'
    AllMethods_IMEX_MRK.append(IMEX_MRK)


    # PRK2 - IMEX2
    IMEX_MRK={}

    AB=np.zeros((2,2))
    AB[1,0]=1.
    bB=np.zeros((2))
    bB[0]=1/2.
    bB[1]=1/2.
    cB=np.sum(AB,1)


    AF=np.zeros((4,4))
    AF[1,0]=0.5
    AF[2,0]=0.25
    AF[2,1]=0.25
    AF[3,0]=0.25
    AF[3,1]=0.25
    AF[3,2]=0.5
    bF=np.zeros((4))
    bF[0]=1/4.
    bF[1]=1/4.
    bF[2]=1/4.
    bF[3]=1/4.
    cF=np.sum(AF,1)
  

    AS=np.zeros((4,4))
    AS[1,0]=0.5
    AS[2,0]=0.25
    AS[2,1]=0.25
    AS[3,0]=0.25
    AS[3,1]=0.25
    AS[3,2]=0.5
    bS=np.zeros((4))
    bS=bF[:]
    cS=np.sum(AS,1)

    AT=np.zeros((4,4))
    AT[3,0]=0.5
    AT[3,1]=0.5
    AT[3,2]=0.5
    AT[3,3]=0.5
    bT=np.zeros((4))
    bT=bF[:]
    cT=np.sum(AT,1)

    
    IMEX_MRK['type']='IMEX-MRK'
    IMEX_MRK['AB']=AB
    IMEX_MRK['bB']=bB
    IMEX_MRK['cB']=cB
    IMEX_MRK['sB']=np.size(bB)
    IMEX_MRK['pB']=2

    IMEX_MRK['AF']=AF
    IMEX_MRK['bF']=bF
    IMEX_MRK['cF']=cF
    IMEX_MRK['sF']=np.size(bF)
    IMEX_MRK['pF']=2

    IMEX_MRK['AS']=AS
    IMEX_MRK['bS']=bS
    IMEX_MRK['cS']=cS
    IMEX_MRK['sS']=np.size(bS)
    IMEX_MRK['pS']=2

    IMEX_MRK['AT']=AT
    IMEX_MRK['bT']=bT
    IMEX_MRK['cT']=cT
    IMEX_MRK['sT']=np.size(bT)
    IMEX_MRK['pT']=2
    
    
    IMEX_MRK['name']='PRK2-IMEX2'
    AllMethods_IMEX_MRK.append(IMEX_MRK)



    
    # MPRK2 - IMEX
    IMEX_MRK={}

    AB=np.zeros((2,2))
    AB[1,0]=1.
    bB=np.zeros((2))
    bB[0]=1/2.
    bB[1]=1/2.
    cB=np.sum(AB,1)


    AF=np.zeros((4,4))
    AF[1,0]=0.5
    AF[2,0]=0.25
    AF[2,1]=0.25
    AF[3,0]=0.25
    AF[3,1]=0.25
    AF[3,2]=0.5
    bF=np.zeros((4))
    bF[0]=1/4.
    bF[1]=1/4.
    bF[2]=1/4.
    bF[3]=1/4.
    cF=np.sum(AF,1)
  

    AS=np.zeros((4,4))
    AS[1,0]=1.
    AS[3,2]=1.
    bS=np.zeros((4))
    bS=bF[:]
    cS=np.sum(AS,1)

    AT=np.zeros((4,4))
    AT[3,0]=1.
    AT[3,1]=1.
    AT[3,2]=1.
    AT[3,3]=1.
    bT=np.zeros((4))
    bT=bF[:]
    cT=np.sum(AT,1)

    
    IMEX_MRK['type']='IMEX-MRK'
    IMEX_MRK['AB']=AB
    IMEX_MRK['bB']=bB
    IMEX_MRK['cB']=cB
    IMEX_MRK['sB']=np.size(bB)
    IMEX_MRK['pB']=2

    IMEX_MRK['AF']=AF
    IMEX_MRK['bF']=bF
    IMEX_MRK['cF']=cF
    IMEX_MRK['sF']=np.size(bF)
    IMEX_MRK['pF']=2

    IMEX_MRK['AS']=AS
    IMEX_MRK['bS']=bS
    IMEX_MRK['cS']=cS
    IMEX_MRK['sS']=np.size(bS)
    IMEX_MRK['pS']=2

    IMEX_MRK['AT']=AT
    IMEX_MRK['bT']=bT
    IMEX_MRK['cT']=cT
    IMEX_MRK['sT']=np.size(bT)
    IMEX_MRK['pT']=1
    
    
    IMEX_MRK['name']='MPRK2-IMEX'
    AllMethods_IMEX_MRK.append(IMEX_MRK)


    # MPRK2 - IMEX2
    IMEX_MRK={}

    AB=np.zeros((2,2))
    AB[1,0]=1.
    bB=np.zeros((2))
    bB[0]=1/2.
    bB[1]=1/2.
    cB=np.sum(AB,1)


    AF=np.zeros((4,4))
    AF[1,0]=0.5
    AF[2,0]=0.25
    AF[2,1]=0.25
    AF[3,0]=0.25
    AF[3,1]=0.25
    AF[3,2]=0.5
    bF=np.zeros((4))
    bF[0]=1/4.
    bF[1]=1/4.
    bF[2]=1/4.
    bF[3]=1/4.
    cF=np.sum(AF,1)
  

    AS=np.zeros((4,4))
    AS[1,0]=1.
    AS[3,2]=1.
    bS=np.zeros((4))
    bS=bF[:]
    cS=np.sum(AS,1)

    AT=np.zeros((4,4))
    AT[3,0]=0.5
    AT[3,1]=0.5
    AT[3,2]=0.5
    AT[3,3]=0.5
    bT=np.zeros((4))
    bT=bF[:]
    cT=np.sum(AT,1)

    
    IMEX_MRK['type']='IMEX-MRK'
    IMEX_MRK['AB']=AB
    IMEX_MRK['bB']=bB
    IMEX_MRK['cB']=cB
    IMEX_MRK['sB']=np.size(bB)
    IMEX_MRK['pB']=2

    IMEX_MRK['AF']=AF
    IMEX_MRK['bF']=bF
    IMEX_MRK['cF']=cF
    IMEX_MRK['sF']=np.size(bF)
    IMEX_MRK['pF']=2

    IMEX_MRK['AS']=AS
    IMEX_MRK['bS']=bS
    IMEX_MRK['cS']=cS
    IMEX_MRK['sS']=np.size(bS)
    IMEX_MRK['pS']=2

    IMEX_MRK['AT']=AT
    IMEX_MRK['bT']=bT
    IMEX_MRK['cT']=cT
    IMEX_MRK['sT']=np.size(bT)
    IMEX_MRK['pT']=2
    
    
    IMEX_MRK['name']='MPRK2-IMEX2'
    AllMethods_IMEX_MRK.append(IMEX_MRK)


    

    # MPRK2 - IMEX
    IMEX_MRK={}

    AB=np.zeros((2,2))
    AB[1,0]=1.
    bB=np.zeros((2))
    bB[0]=1/2.
    bB[1]=1/2.
    cB=np.sum(AB,1)


    AF=np.zeros((4,4))
    AF[1,0]=0.5
    AF[2,0]=0.25
    AF[2,1]=0.25
    AF[3,0]=0.25
    AF[3,1]=0.25
    AF[3,2]=0.5
    bF=np.zeros((4))
    bF[0]=1/4.
    bF[1]=1/4.
    bF[2]=1/4.
    bF[3]=1/4.
    cF=np.sum(AF,1)
  

    AS=np.zeros((4,4))
    AS[1,0]=1.
    AS[3,2]=1.
    bS=np.zeros((4))
    bS=bF[:]
    cS=np.sum(AS,1)

    AT=np.zeros((4,4))
    AT[1,0]=0.
    AT[2,0]=.5
    AT[2,1]=0.
    AT[3,0]=1.
    AT[3,1]=0.5
    AT[3,2]=1.
    AT[3,3]=1.
    bT=np.zeros((4))
    bT=bF[:]
    cT=np.sum(AT,1)

    
    IMEX_MRK['type']='IMEX-MRK'
    IMEX_MRK['AB']=AB
    IMEX_MRK['bB']=bB
    IMEX_MRK['cB']=cB
    IMEX_MRK['sB']=np.size(bB)
    IMEX_MRK['pB']=2

    IMEX_MRK['AF']=AF
    IMEX_MRK['bF']=bF
    IMEX_MRK['cF']=cF
    IMEX_MRK['sF']=np.size(bF)
    IMEX_MRK['pF']=2

    IMEX_MRK['AS']=AS
    IMEX_MRK['bS']=bS
    IMEX_MRK['cS']=cS
    IMEX_MRK['sS']=np.size(bS)
    IMEX_MRK['pS']=2

    IMEX_MRK['AT']=AT
    IMEX_MRK['bT']=bT
    IMEX_MRK['cT']=cT
    IMEX_MRK['sT']=np.size(bT)
    IMEX_MRK['pT']=1
    
    
    IMEX_MRK['name']='MPRK2-IMEXb'
    AllMethods_IMEX_MRK.append(IMEX_MRK)



    # MPRK2 - IMEX
    IMEX_MRK={}

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


    AT=np.zeros((2*sB,2*sB))
    
    AT[sB*2-1,0]=1./4.
    AT[sB*2-1,1]=1./4.
    AT[sB*2-1,2]=1.
    AT[sB*2-1,3]=1./4.
    AT[sB*2-1,4]=1./4
    AT[sB*2-1,5]=1.
    bT=np.zeros((2*sB))
    bT=bF[:]
    cT=np.sum(AT,1)

    
    IMEX_MRK['type']='IMEX-MRK'
    IMEX_MRK['AB']=AB
    IMEX_MRK['bB']=bB
    IMEX_MRK['cB']=cB
    IMEX_MRK['sB']=np.size(bB)
    IMEX_MRK['pB']=2

    IMEX_MRK['AF']=AF
    IMEX_MRK['bF']=bF
    IMEX_MRK['cF']=cF
    IMEX_MRK['sF']=np.size(bF)
    IMEX_MRK['pF']=2

    IMEX_MRK['AS']=AS
    IMEX_MRK['bS']=bS
    IMEX_MRK['cS']=cS
    IMEX_MRK['sS']=np.size(bS)
    IMEX_MRK['pS']=2

    IMEX_MRK['AT']=AT
    IMEX_MRK['bT']=bT
    IMEX_MRK['cT']=cT
    IMEX_MRK['sT']=np.size(bT)
    IMEX_MRK['pT']=1
    
    
    IMEX_MRK['name']='MPRK2-RK3SSPHig-IMEX'
    AllMethods_IMEX_MRK.append(IMEX_MRK)



    # MPRK2 m=4 - IMEX
    IMEX_MRK={}
    sB=2
    m=4

    AB=np.zeros((sB,sB))
    AB[1,0]=1.0
    
    bB=np.zeros((sB))
    bB[0]=1./2.
    bB[1]=1./2
    
    cB=np.sum(AB,1)
    
    AF=np.zeros((sB*m,sB*m))
    for k in range(m):
        for i in range(sB):
            for j in range(sB):
                AF[i+k*sB,j+k*sB]=(1./m)*AB[i,j]
    for k in range(1,m):
        for ell in range(k):
            for i in range(sB):
                for j in range(sB):
                    AF[k*sB+i,j+sB*ell]=bB[j]/m
 

    bF=np.zeros((sB*m))
    for k in range(m):
        for i in range(sB):
            bF[i+sB*k]=(1./m)*bB[i]
    cF=np.sum(AF,1)

    AS=np.zeros((sB*m,sB*m))
    bS=np.zeros((sB*m)) 
    for k in range(m):
        for i in range(sB):
            for j in range(sB):
                AS[i+sB*k,j+sB*k]=AB[i,j]

        for i in range(sB):
            bS[i+sB*k]=bB[i]/m
        
    cS=np.sum(AS,1)


    AT=np.zeros((m*sB,m*sB))
    AT[sB*m-1,:]=1.
   
    bT=np.zeros((m*sB))
    bT=bF[:]
    cT=np.sum(AT,1)

    
    IMEX_MRK['type']='IMEX-MRK'
    IMEX_MRK['AB']=AB
    IMEX_MRK['bB']=bB
    IMEX_MRK['cB']=cB
    IMEX_MRK['sB']=np.size(bB)
    IMEX_MRK['pB']=2

    IMEX_MRK['AF']=AF
    IMEX_MRK['bF']=bF
    IMEX_MRK['cF']=cF
    IMEX_MRK['sF']=np.size(bF)
    IMEX_MRK['pF']=2

    IMEX_MRK['AS']=AS
    IMEX_MRK['bS']=bS
    IMEX_MRK['cS']=cS
    IMEX_MRK['sS']=np.size(bS)
    IMEX_MRK['pS']=2

    IMEX_MRK['AT']=AT
    IMEX_MRK['bT']=bT
    IMEX_MRK['cT']=cT
    IMEX_MRK['sT']=np.size(bT)
    IMEX_MRK['pT']=1
    
    
    IMEX_MRK['name']='MPRK2-m4-IMEX'
    AllMethods_IMEX_MRK.append(IMEX_MRK)

    
    return AllMethods_IMEX_MRK

