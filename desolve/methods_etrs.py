import numpy as np

def Default_ETRS_Methods():
    AllMethods_ETRS=[]

    # ETRS
    ETRS={}
    
    ETRS['type']='ETRS'
    ETRS['p']=2
    ETRS['n_iter']=None
    ETRS['approx_exp']=False 
    ETRS['name']='ETRS'
    
    AllMethods_ETRS.append(ETRS)

    
    
    # ETRS
    ETRS={}
    
    ETRS['type']='ETRS'
    ETRS['p']=2
    ETRS['n_iter']=2
    ETRS['approx_exp']=True
    ETRS['name']='ETRS-approx'
    
    AllMethods_ETRS.append(ETRS)

        
    # ETRS
    ETRS={}
    
    ETRS['type']='ETRS'
    ETRS['p']=1
    ETRS['n_iter']=2
    ETRS['approx_exp']=True
    ETRS['name']='ETRS-approx-FE'
    
    AllMethods_ETRS.append(ETRS)
    
    return AllMethods_ETRS
