import numpy as np

def ProblemsPDE(name,problem_ctx=None):
    rhs_e=None
    rhs_i=None
    u_ini=None
  
    if(name=='advection_reaction'):
        rhs_e, rhs_i, u_ini, problem_setup = advection_reaction(problem_ctx)
    elif(name=='GrayScott'):
        rhs_e, rhs_i, u_ini, problem_setup = GrayScott(problem_ctx)
    elif(name=='Diffusion'):
        rhs_e, rhs_i, u_ini, problem_setup = Diffusion(problem_ctx)
    elif(name=='DiffusionComplex'):
        rhs_e, rhs_i, u_ini, problem_setup = DiffusionComplex(problem_ctx)
    else:
        raise NameError('Problem {:} has not been found.'.format(name))
    return rhs_e, rhs_i, u_ini, problem_setup
    

def GrayScott(problem_ctx=None):
    def rhs_e(t,u_in,ctx=None):
        vec=ctx['vectorize']
        unvec=ctx['unvectorize']
        
        n=ctx['n']
        mx=ctx['mx']
        my=ctx['my']
        Du=ctx['Du']
        Dv=ctx['Dv']
        Fuv=ctx['Fuv']
        kappa=ctx['kappa']
        dx=ctx['dx']
        dy=ctx['dy']
        
        F=np.zeros((n,mx,my),ctx['data-type'])
        
        y=unvec(u_in,ctx)
        
        dxx=dx*dx
        dyy=dy*dy
        
        id_uv=0
        F[id_uv,0,0]=Du*((y[id_uv,-1,0]+y[id_uv,1,0]-2*y[id_uv,0,0])/dxx + 
                         (y[id_uv,0,-1]+y[id_uv,0,1]-2*y[id_uv,0,0])/dyy)
        id_uv=1
        F[id_uv,0,0]=Dv*((y[id_uv,-1,0]+y[id_uv,1,0]-2*y[id_uv,0,0])/dxx +
                         (y[id_uv,0,-1]+y[id_uv,0,1]-2*y[id_uv,0,0])/dyy)
        
        id_uv=0
        F[id_uv,mx-1,0]=Du*((y[id_uv,mx-2,0]+y[id_uv,0,0]-2*y[id_uv,mx-1,0])/dxx +
                            (y[id_uv,mx-1,-1]+y[id_uv,mx-1,0]-2*y[id_uv,mx-1,0])/dyy)
        id_uv=1
        F[id_uv,mx-1,0]=Dv*((y[id_uv,mx-2,0]+y[id_uv,0,0]-2*y[id_uv,mx-1,0])/dxx +
                            (y[id_uv,mx-1,-1]+y[id_uv,mx-1,0]-2*y[id_uv,mx-1,0])/dyy)
        
        id_uv=0
        F[id_uv,0,my-1]=Du*((y[id_uv,0,my-2]+y[id_uv,0,0]-2*y[id_uv,0,my-1])/dyy +
                            (y[id_uv,-1,my-1]+y[id_uv,0,my-1]-2*y[id_uv,0,my-1])/dxx)
        id_uv=1
        F[id_uv,0,my-1]=Dv*((y[id_uv,0,my-2]+y[id_uv,0,0]-2*y[id_uv,0,my-1])/dyy +
                            (y[id_uv,-1,my-1]+y[id_uv,0,my-1]-2*y[id_uv,0,my-1])/dxx)
        
        id_uv=0
        F[id_uv,mx-1,my-1]=Du*((y[id_uv,mx-2,my-1]+y[id_uv,0,my-1]-2*y[id_uv,mx-1,my-1])/dxx +
                               (y[id_uv,mx-1,0]+y[id_uv,mx-1,my-2]-2*y[id_uv,mx-1,my-1])/dyy)
        id_uv=1
        F[id_uv,mx-1,my-1]=Dv*((y[id_uv,mx-2,my-1]+y[id_uv,0,my-1]-2*y[id_uv,mx-1,my-1])/dxx +
                               (y[id_uv,mx-1,0]+y[id_uv,mx-1,my-2]-2*y[id_uv,mx-1,my-1])/dyy)
        
        for id_mx in range(2,mx-2):
            for id_my in range(2,my-2):
                id_uv=0
                F[id_uv,id_mx,id_my]=Du*((y[id_uv,id_mx-1,id_my]+y[id_uv,id_mx+1,id_my]-2*y[id_uv,id_mx,id_my])/dxx + (y[id_uv,id_mx,id_my-1]+y[id_uv,id_mx,id_my+1]-2*y[id_uv,id_mx,id_my])/dyy)

                
                id_uv=1
                F[id_uv,id_mx,id_my]=Dv*((y[id_uv,id_mx-1,id_my]+y[id_uv,id_mx+1,id_my]-2*y[id_uv,id_mx,id_my])/dxx + (y[id_uv,id_mx,id_my-1]+y[id_uv,id_mx,id_my+1]-2*y[id_uv,id_mx,id_my])/dyy)
        
        
        for id_mx in range(mx):
            for id_my in range(my):
                id_uv=0
                F[id_uv,id_mx,id_my]+=-y[0,id_mx,id_my]*y[1,id_mx,id_my]*y[1,id_mx,id_my]+Fuv*(1-y[0,id_mx,id_my])
                
                id_uv=1
                F[id_uv,id_mx,id_my]+=y[0,id_mx,id_my]*y[1,id_mx,id_my]*y[1,id_mx,id_my]-(Fuv+kappa)*y[1,id_mx,id_my]
        
        u_out=vec(F,ctx)
        
        j_out=None
        
        return u_out,j_out
    
    def vectorize(u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['mx']*pde_problem_ctx['my']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['mx'],pde_problem_ctx['my']))
        return u_pde
    
    n=problem_ctx['n']
    mx=problem_ctx['mx']
    my=problem_ctx['my']
    kappa=problem_ctx['kappa']
    x_max=problem_ctx['x_max']
    x_min=problem_ctx['x_min']
    y_max=problem_ctx['y_max']
    y_min=problem_ctx['y_min']
    dx=float(x_max-x_min)/mx
    dy=float(y_max-y_min)/my
    x_coord=np.zeros((mx,))
    y_coord=np.zeros((my,))
    
    for i in range(mx):
        x_coord[i]=((i+1.)*dx)+x_min
    for i in range(my):
        y_coord[i]=((i+1.)*dy)+y_min               

    
    
    if(problem_ctx is None):
        ctx={'mx':64,'my':64,'n':2,'x_min':0.,'x_max':2.5,'y_min':0.,'y_max':2.5,
             'Du':2.0e-05,
             'Dv':1.0e-05,
             'kappa':0.054,
             'Fuv':0.034,
             'dx':dx,
             'dy':dy,
             'x_coord':x_coord,
             'y_coord':y_coord,
             'vectorize':vectorize,
             'unvectorize':unvectorize,
             'boundary':boundary,
             'j_out_ex':None,
             'j_out_im':None}
    else:
        ctx={}
        ctx['mx']=problem_ctx['mx']
        ctx['my']=problem_ctx['my']
        ctx['n']=problem_ctx['n']
        ctx['x_min']=problem_ctx['x_min']
        ctx['x_max']=problem_ctx['x_max']
        ctx['y_min']=problem_ctx['y_min']
        ctx['y_max']=problem_ctx['y_max']
        ctx['Du']=problem_ctx['Du']
        ctx['Dv']=problem_ctx['Dv']
        ctx['Fuv']=problem_ctx['Fuv']
        ctx['kappa']=problem_ctx['kappa']
        ctx['dx']=dx
        ctx['dy']=dy
        ctx['x_coord']=x_coord
        ctx['y_coord']=y_coord
        ctx['vectorize']=vectorize
        ctx['unvectorize']=unvectorize

        
    problem_setup={}
    
    problem_setup['context']=ctx
    problem_setup['context']['data-type']=np.float64
    problem_setup['DT']=1.0e-03
    problem_setup['DT_REFERENCE']=1.0e-04
    problem_setup['T_DURATION']={'start':0.,'end':500}
    problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

    u_ini_pde=np.zeros((n,mx,my),problem_setup['context']['data-type'])
    u_ini_pde[0,:,:]=1.
    u_ini_pde[0,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.5+np.reshape(np.random.uniform(low=-0.5*1e-02,high=0.5*1e-02,size=20*20),(20,20))
    u_ini_pde[1,:,:]=0.
    u_ini_pde[1,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.25+np.reshape(np.random.uniform(low=-0.25*1e-02,high=0.25*1e-02,size=20*20),(20,20))
       
    u_ini=vectorize(u_ini_pde,problem_ctx)
    problem_setup['context']['u_ini']=u_ini
    
    return rhs_e, None, u_ini, problem_setup
    
def advection_reaction(problem_ctx=None):
    def boundary(t):
        yb=np.zeros((2,),ctx['data-type'])
        yb[0]=1.-np.sin(2.*t)**4
        yb[1]=0.5*(2. - np.sin(5* 12.* t)**4 - np.cos(4.* 12.* t)**4)#0.5*(1. - np.sin(15.* 12.* t)**4 + np.cos(14.* 12.* t)**4)#(1.-np.sin(12.*12.*t)**4)*0.5#1.-np.sin(12.*t)**4
        return yb
    
    def rhs_e(t,u_in,ctx=None):
        vec=ctx['vectorize']
        unvec=ctx['unvectorize']
        
        n=ctx['n']
        m=ctx['m']
        kx=ctx['k']
        dx=ctx['dx']
        
        
        alpha=np.zeros((2,),ctx['data-type'])
        alpha[:]=ctx['alpha'][0:2,0]
        
        #print('Explicit alpha {:}, {:}'.format(alpha[0],alpha[1]))
        
        F=np.zeros((n,m),ctx['data-type'])
       
        
        
        yb=ctx['boundary'](t)
        
        y=unvec(u_in,ctx)
        
        for ni in range(n):
            F[ni,0]=alpha[ni]*((1./3.)*yb[ni]+(0.5)*y[ni,0]+(-1.)*y[ni,1]+(1./6.)*y[ni,2])/dx            
            F[ni,1]=alpha[ni]*((-1./12.)*yb[ni]+(2./3.)*y[ni,0]+(-2./3.)*y[ni,2]+(1./12.)*y[ni,3])/dx
            
            for mi in range(2,m-2):
                F[ni,mi]=alpha[ni]*((-1./12.)*y[ni,mi-2]+(+2./3.)*y[ni,mi-1]+(-2./3.)*y[ni,mi+1]+(1./12.)*y[ni,mi+2])/dx
                
            F[ni,m-2]=alpha[ni]*((-1./6.)*y[ni,m-4]+(1.)*y[ni,m-3]+(-0.5)*y[ni,m-2]+(-1./3.)*y[ni,m-1])/dx            
            F[ni,m-1]=-alpha[ni]*(y[ni,m-1]-y[ni,m-2])/dx
          
        if(ctx['j_out_ex'] is None):
            Fp=np.zeros((n,m,m),ctx['data-type'])
            
            for ni in range(n):
                Fp[ni,0,0]=alpha[ni]*(0.5)/dx
                Fp[ni,0,1]=alpha[ni]*(-1.)/dx
                Fp[ni,0,2]=alpha[ni]*(1./6.)/dx

                Fp[ni,1,0]=alpha[ni]*(2./3.)/dx
                Fp[ni,1,2]=alpha[ni]*(-2./3.)/dx
                Fp[ni,1,3]=alpha[ni]*(1./12.)/dx

                for mi in range(2,m-2):
                    Fp[ni,mi,mi-2]=alpha[ni]*(-1./12.)/dx
                    Fp[ni,mi,mi-1]=alpha[ni]*(+2./3.)/dx
                    Fp[ni,mi,mi+1]=alpha[ni]*(-2./3.)/dx
                    Fp[ni,mi,mi+2]=alpha[ni]*(1./12.)/dx

                Fp[ni,m-2,m-4]=alpha[ni]*(-1./6.)/dx
                Fp[ni,m-2,m-3]=alpha[ni]*1./dx
                Fp[ni,m-2,m-2]=alpha[ni]*(-0.5)/dx
                Fp[ni,m-2,m-1]=alpha[ni]*(-1./3.)/dx

                Fp[ni,m-1,m-1]=-alpha[ni]*1./dx
                Fp[ni,m-1,m-2]=-alpha[ni]*(-1.)/dx
                
                
            j_out=np.vstack((np.hstack((np.squeeze(Fp[0,:,:]),np.zeros((m,m),ctx['data-type']))),np.hstack((np.zeros((m,m),ctx['data-type']),np.squeeze(Fp[1,:,:])))))
            ctx['j_out_ex']=j_out
        else:
            j_out=ctx['j_out_ex']
        
        u_out=vec(F,ctx)
        
        return u_out,j_out

    def rhs_i(t,u_in,ctx=None):
        vec=ctx['vectorize']
        unvec=ctx['unvectorize']
        
        n=ctx['n']
        s=ctx['s']
        m=ctx['m']
        kx=ctx['k']
        dx=ctx['dx']
        
        
        alpha=np.zeros((2,),ctx['data-type'])
        alpha[:]=ctx['alpha'][0:2,1]
        #print('Implicit alpha {:}, {:}'.format(alpha[0],alpha[1]))
        
        F=np.zeros((n,m),ctx['data-type'])
         
        yb=ctx['boundary'](t)
       
        
        y=unvec(u_in,ctx)
        
        for ni in range(n):
            F[ni,0]=alpha[ni]*((1./3.)*yb[ni]+(0.5)*y[ni,0]+(-1.)*y[ni,1]+(1./6.)*y[ni,2])/dx            
            F[ni,1]=alpha[ni]*((-1./12.)*yb[ni]+(2./3.)*y[ni,0]+(-2./3.)*y[ni,2]+(1./12.)*y[ni,3])/dx
            
            for mi in range(2,m-2):
                F[ni,mi]=alpha[ni]*((-1./12.)*y[ni,mi-2]+(+2./3.)*y[ni,mi-1]+(-2./3.)*y[ni,mi+1]+(1./12.)*y[ni,mi+2])/dx
                
            F[ni,m-2]=alpha[ni]*((-1./6.)*y[ni,m-4]+(1.)*y[ni,m-3]+(-0.5)*y[ni,m-2]+(-1./3.)*y[ni,m-1])/dx            
            F[ni,m-1]=-alpha[ni]*(y[ni,m-1]-y[ni,m-2])/dx
            F[ni,:]=F[ni,:]+s[ni]

            
            
        for mi in range(m):
            F[0,mi]=F[0,mi]-kx[0]*y[0,mi]+kx[1]*y[1,mi]
            F[1,mi]=F[1,mi]+kx[0]*y[0,mi]-kx[1]*y[1,mi]
            
        if(ctx['j_out_im'] is None):
            Fp=np.zeros((n,m,m),ctx['data-type'])
            Fpp=np.zeros((n,m,m),ctx['data-type'])
            for ni in range(n):
                Fp[ni,0,0]=alpha[ni]*(0.5)/dx
                Fp[ni,0,1]=alpha[ni]*(-1.)/dx
                Fp[ni,0,2]=alpha[ni]*(1./6.)/dx

                Fp[ni,1,0]=alpha[ni]*(2./3.)/dx
                Fp[ni,1,2]=alpha[ni]*(-2./3.)/dx
                Fp[ni,1,3]=alpha[ni]*(1./12.)/dx

                for mi in range(2,m-2):
                    Fp[ni,mi,mi-2]=alpha[ni]*(-1./12.)/dx
                    Fp[ni,mi,mi-1]=alpha[ni]*(+2./3.)/dx
                    Fp[ni,mi,mi+1]=alpha[ni]*(-2./3.)/dx
                    Fp[ni,mi,mi+2]=alpha[ni]*(1./12.)/dx

                Fp[ni,m-2,m-4]=alpha[ni]*(-1./6.)/dx
                Fp[ni,m-2,m-3]=alpha[ni]*1./dx
                Fp[ni,m-2,m-2]=alpha[ni]*(-0.5)/dx
                Fp[ni,m-2,m-1]=alpha[ni]*(-1./3.)/dx

                Fp[ni,m-1,m-1]=-alpha[ni]*1./dx
                Fp[ni,m-1,m-2]=-alpha[ni]*(-1.)/dx
                
                
            for mi in range(m):
                Fp[0,mi,mi]=Fp[0,mi,mi]-kx[0]
                
            for mi in range(m):
                Fpp[0,mi,mi]=Fpp[1,mi,mi]+kx[1]
                
            for mi in range(m):
                Fp[1,mi,mi]=Fp[1,mi,mi]+kx[0]
                
            for mi in range(m):
                Fpp[1,mi,mi]=Fpp[1,mi,mi]-kx[1]
                
            j_out=np.vstack((np.hstack((np.squeeze(Fp[0,:,:]),np.squeeze(Fpp[0,:,:]))),np.hstack((np.squeeze(Fpp[1,:,:]),np.squeeze(Fp[1,:,:])))))
            ctx['j_out_im']=j_out
        else:
            j_out=ctx['j_out_im']
        
        u_out=vec(F,ctx)
        

        return u_out,j_out
    
    def vectorize(u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['m']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['m']))
        return u_pde
    
    m=problem_ctx['m']
    s=problem_ctx['s']
    k=problem_ctx['k']
    x_max=problem_ctx['x_max']
    x_min=problem_ctx['x_min']
    dx=float(x_max-x_min)/m
    x_coord=np.zeros((m,))
    
    for i in range(m):
        x_coord[i]=((i+1.)*dx)+x_min
                

    
    
    if(problem_ctx is None):
        ctx={'m':400,'n':2,'x_min':0.,'x_max':1.,
             's':np.asarray([0.,1.]),
             'alpha':np.asarray([[1,0],[0,1]]),
             'k':np.asarray([1.0e+03,2.0e+3]),
             'dx':dx,
             'x_coord':x_coord,
             'vectorize':vectorize,
             'unvectorize':unvectorize,
             'boundary':boundary,
             'j_out_ex':None,
             'j_out_im':None}
    else:
        ctx={}
        ctx['m']=problem_ctx['m']
        ctx['n']=problem_ctx['n']
        ctx['x_min']=problem_ctx['x_min']
        ctx['x_max']=problem_ctx['x_max']
        ctx['s']=problem_ctx['s']
        ctx['k']=problem_ctx['k']
        ctx['dx']=dx
        ctx['x_coord']=x_coord
        ctx['alpha']=problem_ctx['alpha']
        ctx['vectorize']=vectorize
        ctx['unvectorize']=unvectorize
        ctx['boundary']=boundary
        ctx['j_out_im']=None
        ctx['j_out_ex']=None
  
        

        
    problem_setup={}
    
    problem_setup['context']=ctx
    problem_setup['context']['data-type']=np.float64
    problem_setup['DT']=1.0e-03
    problem_setup['DT_REFERENCE']=1.0e-04
    problem_setup['T_DURATION']={'start':0.,'end':0.1}
    problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

    u_ini_pde=np.zeros((2,m),problem_setup['context']['data-type'])
    u_ini_pde[0,0:m]=1+x_coord#1+s[1]*x_coord
    u_ini_pde[1,0:m]=0.5*u_ini_pde[0,0:m] #(k[0]/k[1])*u_ini_pde[0,0:m]+(s[1]/k[1])    
       
    u_ini=vectorize(u_ini_pde,problem_ctx)
    problem_setup['context']['u_ini']=u_ini
    
    return rhs_e, rhs_i, u_ini, problem_setup



def Diffusion(problem_ctx=None):
 
    def diffusion_tensor(t,ctx=None):
        y_k=np.zeros((ctx['m'],),ctx['data-type'])
        
        x_max=problem_ctx['x_max']
        x_min=problem_ctx['x_min']
        x_coord=ctx['x_coord']
        y_k=(1.-np.sin(12.*t)**4)*np.sin(2*np.pi*(x_coord+t)/(x_max-x_min))**2
        return y_k
    
    
    def vectorize(u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['m']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['m']))
        return u_pde
 
    
    def rhs_e(t,u_in,ctx=None):
        
        m=ctx['m']
        dx=ctx['dx']
        kappa=ctx['kappa']
        diffusion_tensor=ctx['diffusion_tensor']
        u_out=np.zeros((m,),ctx['data-type'])
        A=np.zeros((m,m),ctx['data-type'])
        
        K=kappa*diffusion_tensor(t,ctx)
        
        
        
        A[0,0]=-2.
        A[0,1]=1.
        A[0,m-1]=1.
        for i in range(1,m-1):
            A[i,i-1]=1.
            A[i,i]=-2.
            A[i,i+1]=1.
        A[m-1,0]=1.
        A[m-1,m-1]=-2.
        A[m-1,m-2]=1.
        
        for i in range(m):
            A[i,:]=K[i]*A[i,:]
        
        A=A/(dx**2)
    
        u_out=np.matmul(A,u_in)
        
        
        j_out=A
        return u_out,j_out
 
    
    rhs_i=None
   
    m=problem_ctx['m']
    x_max=problem_ctx['x_max']
    x_min=problem_ctx['x_min']
    
    x_coord=np.zeros((m,))
    dx=float(x_max-x_min)/m
    for i in range(m):
        x_coord[i]=((i+1.)*dx)+x_min
    
     
    problem_setup={}    
    
    
 
    
    if(problem_ctx is None):
        ctx={'m':6,'n':1,'dx':dx,'x_min':0.,'x_max':1.,'kappa':0.3,
             'x_coord':x_coord,
             'diffusion_tensor':diffusion_tensor,
             'vectorize':vectorize,
             'unvectorize':unvectorize}
    else:
        ctx={}
        ctx['m']=problem_ctx['m']
        ctx['n']=problem_ctx['n']
        ctx['dx']=dx
        ctx['x_min']=problem_ctx['x_min']
        ctx['x_max']=problem_ctx['x_max']
        ctx['x_coord']=x_coord
        ctx['kappa']=problem_ctx['kappa']
        ctx['diffusion_tensor']=diffusion_tensor
        ctx['vectorize']=vectorize
        ctx['unvectorize']=unvectorize
 
        
    
    problem_setup['context']=ctx
    problem_setup['context']['data-type']=np.float64
    
    problem_setup['DT']=1.0e-02
    problem_setup['DT_REFERENCE']=1.0e-04
    problem_setup['T_DURATION']={'start':0.,'end':1.}
    problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-02}
    
    u_ini=np.zeros((m),problem_setup['context']['data-type'])
    
    for i in range(m):
        u_ini[i]=np.exp(-((i-0.5*m)**2)*dx)

    
    problem_setup['context']['u_ini']=u_ini
    
    return rhs_e, rhs_i, u_ini, problem_setup



def DiffusionComplex(problem_ctx=None):
 
    def diffusion_tensor(t,ctx=None):
        y_k=np.zeros((ctx['m'],),ctx['data-type'])
        
        x_max=problem_ctx['x_max']
        x_min=problem_ctx['x_min']
        x_coord=ctx['x_coord']
        y_k=(1.-np.sin(12.*t)**4)*np.sin(2*np.pi*(x_coord+t)/(x_max-x_min))**2
        return y_k
    
    
    def vectorize(u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['m']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['m']))
        return u_pde
 
    
    def rhs_e(t,u_in,ctx=None):
        
        m=ctx['m']
        dx=ctx['dx']
        kappa=ctx['kappa']
        diffusion_tensor=ctx['diffusion_tensor']
        u_out=np.zeros((m,),ctx['data-type'])
        A=np.zeros((m,m),ctx['data-type'])
        
        K=kappa*diffusion_tensor(t,ctx)*(4.*u_in+0.5)
       
        
        
        A[0,0]=-2.
        A[0,1]=1.
        A[0,m-1]=1.
        for i in range(1,m-1):
            A[i,i-1]=1.
            A[i,i]=-2.
            A[i,i+1]=1.
        A[m-1,0]=1.
        A[m-1,m-1]=-2.
        A[m-1,m-2]=1.
        
        for i in range(m):
            A[i,:]=K[i]*A[i,:]
        
        A=A/(dx**2)
        ii=np.complex(0,1)
        
        u_out=np.matmul(A,u_in)/ii
        
        
        j_out=A
        return u_out,j_out
 
    
    rhs_i=None
   
    m=problem_ctx['m']
    x_max=problem_ctx['x_max']
    x_min=problem_ctx['x_min']
    
    x_coord=np.zeros((m,))
    dx=float(x_max-x_min)/m
    for i in range(m):
        x_coord[i]=((i+1.)*dx)+x_min
    

    
    problem_setup={}    
    
    
 
    
    if(problem_ctx is None):
        ctx={'m':6,'n':1,'dx':dx,'x_min':0.,'x_max':1.,'kappa':0.3,
             'x_coord':x_coord,
             'diffusion_tensor':diffusion_tensor,
             'vectorize':vectorize,
             'unvectorize':unvectorize}
    else:
        ctx={}
        ctx['m']=problem_ctx['m']
        ctx['n']=problem_ctx['n']
        ctx['dx']=dx
        ctx['x_min']=problem_ctx['x_min']
        ctx['x_max']=problem_ctx['x_max']
        ctx['x_coord']=x_coord
        ctx['kappa']=problem_ctx['kappa']
        ctx['diffusion_tensor']=diffusion_tensor
        ctx['vectorize']=vectorize
        ctx['unvectorize']=unvectorize
 
        
    
    problem_setup['context']=ctx
    problem_setup['context']['data-type']=np.complex
    
    problem_setup['DT']=1.0e-02
    problem_setup['DT_REFERENCE']=1.0e-04
    problem_setup['T_DURATION']={'start':0.,'end':1.}
    problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-02}
    
    
    u_ini=np.zeros((m),problem_setup['context']['data-type'])
    
    for i in range(m):
        u_ini[i]=np.exp(-((i-0.5*m)**2)*dx)
    
    problem_setup['context']['u_ini']=u_ini
    
    return rhs_e, rhs_i, u_ini, problem_setup

