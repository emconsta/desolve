import numpy as np
from jax import jacfwd
import jax.numpy as jnp

def ProblemsPDE(name,problem_ctx=None):
    rhs_e=None
    rhs_i=None
    u_ini=None
  
    if(name=='Advection Reaction 1D'):
        problem = AdvectionReaction1D(problem_ctx)
    elif(name=='Advection Reaction 1D Split'):
        problem = AdvectionReaction1DSplit(problem_ctx)
    elif(name=='GrayScott'):
        problem = GrayScott(problem_ctx)
    elif(name=='Navier-Stokes2D'):
        problem = NavierStokes2D(problem_ctx)
    elif(name=='Diffusion'):
        problem = Diffusion(problem_ctx)
    elif(name=='DiffusionComplex'):
        problem = DiffusionComplex(problem_ctx)
    elif(name=='Advection1D'):
        problem = Advection1D(problem_ctx)
    elif(name=='AdvectionDiffusion1D'):
        problem = AdvectionDiffusion1D(problem_ctx)
    else:
        raise NameError('Problem {:} has not been found.'.format(name))


    rhs_e=problem.rhs_e
    rhs_i=problem.rhs_i
    u_ini=problem.initial_solution()
    problem_setup = problem.get_problem_setup()
    
    return rhs_e, rhs_i, u_ini, problem_setup, problem

def LinearAdvectionSemiDiscretization1D_dict(inputs, params):
    return LinearAdvectionSemiDiscretization1Dp(params['ghostPoints_l'], params['ghostPoints_r'], params['bc_l'], params['bc_r'], params['speed'], inputs, params['n'], params['mx'], params['time'], params['dx'], params['ctx'], params['ftype'], params['jac'])


def LinearAdvectionSemiDiscretization1Dp(ghostPoints_l=None, ghostPoints_r=None, bc_l=None, bc_r=None, speed=None, solution=None, n=-1, mx=-1, time=None, dx=None, ctx=None, ftype=None, jac=False):
    assert mx>0
    assert n>0

    if (jac is False):
        Jac = None
    
    if(ftype=='1stOrderUpwindFV'):  
        F=jnp.zeros((n,mx+1),ctx['data-type'])
        Flux=jnp.zeros((n,mx),ctx['data-type'])
        
        assert ghostPoints_l==1,'Need to provide exactly one ghost point'
        
        
        y=jnp.asarray(solution)
        y=jnp.reshape(y,(n,mx+2)) 

        c=jnp.asarray(speed)

        for id_mx in range(0,mx):
            Flux=Flux.at[:,id_mx].set(-(1./dx)*(c[:,id_mx+1]*y[:,id_mx+1]-c[:,id_mx]*y[:,id_mx]))
        
        Flux=jnp.reshape(Flux,(n*mx,)) 

        if(jac):
            Jac1=np.zeros((mx,mx),ctx['data-type'])
            for id_mx in range(mx-1):
                Jac1[id_mx,id_mx]=(1./dx)*(c[0,id_mx])
                Jac1[id_mx,id_mx+1]=-(1./dx)*c[0,id_mx+1]
            J0=np.zeros((mx,mx))
            
            K1=np.hstack((Jac1,J0))
            K2=np.hstack((J0,Jac1))
            Jac=np.vstack((K1,K2))
            
    elif(ctx['Flux_name']=='3rdOrderUpwindFD'):
        Flux=jnp.zeros((n,mx),ctx['data-type'])
        y=solution
        c=speed

        #assert ghostPoints_l==2,'Need to provide exactly two ghost points'

        if(ghostPoints_l==2):        
            for id_mx in range(mx):    
                Flux[:,id_mx]=(c[:,id_mx+2]/dx)*((-1./6.)*y[:,id_mx]+(1.)*y[:,id_mx+1]+(-1./2.)*y[:,id_mx+2]+(-1./3.)*y[:,id_mx+3])
        elif(ghostPoints_l==1):
            Flux[:,0]=(c[:,0+2]/dx)*((1./3.)*y[:,0]+(1./2)*y[:,1]+(-1.)*y[:,2]+(1./6.)*y[:,3])
            Flux[:,1]=(c[:,1+2]/dx)*((-1./12.)*y[:,0]+(2./3.)*y[:,1]+(-2./3.)*y[:,2]+(1./12.)*y[:,3])
            for id_mx in range(2,mx):    
                Flux[:,id_mx]=(c[:,id_mx+2]/dx)*((-1./6.)*y[:,id_mx]+(1.)*y[:,id_mx+1]+(-1./2.)*y[:,id_mx+2]+(-1./3.)*y[:,id_mx+3])
    
    elif(ftype[0:13]=='FVStagVanLeer'):
        F=jnp.zeros((n,mx+1),ctx['data-type'])
        Flux=jnp.zeros((n,mx),ctx['data-type'])
        
        assert ghostPoints_l==2,'Need to provide exactly two ghost point'
        assert ghostPoints_r==2,'Need to provide exactly two ghost point'
        if(ftype=='FVStagVanLeer-k=1'):  # second order central
            kappa = 1.
        if(ftype=='FVStagVanLeer-k=-1'): # second order upwind
            kappa = -1.
        if(ftype=='FVStagVanLeer-k=1/3'): # third orderr
            kappa = 1./3.
        if(ftype=='FVStagVanLeer-k=0'):  # Fromm scheme (second order)
            kappa = 0.
        
        y=solution
        c=speed
        for id_n in range(n):
            for id_mx in range(mx+1):
                if(c[id_n,id_mx]>=0):
                    omega_j=y[id_n,id_mx+2]
                    omega_jm1=y[id_n,id_mx+1]
                    omega_jm2=y[id_n,id_mx]
                    F[id_n,id_mx]=c[id_n,id_mx+2]*(omega_jm1+(omega_jm1-omega_jm2)*(1.-kappa)/4.+(omega_j-omega_jm1)*(1.+kappa)/4.)
                else:
                    omega_jm1=-y[id_n,id_mx+2]
                    omega_j=-y[id_n,id_mx+1]
                    omega_jp1=-y[id_n,id_mx]
                    F[id_n,id_mx]=c[id_n,id_mx+2]*(omega_j+(omega_j-omega_jp1)*(1.-kappa)/4.+(omega_jm1-omega_j)*(1.+kappa)/4.)
            
        for id_mx in range(0,mx):
            Flux[:,id_mx]=-(1./dx)*(F[:,id_mx+1]-F[:,id_mx])
    else:
        raise NameError('{:} discretization type not implemented'.format(ctx['Flux_name']))


    if(Jac is None):
        return Flux
    else:
        return Flux, Jac


def LinearAdvectionSemiDiscretization1D(ghostPoints_l=None, ghostPoints_r=None, bc_l=None, bc_r=None, speed=None, solution=None, n=-1, mx=-1, time=None, dx=None, ctx=None, ftype=None, jac=False):
    assert mx>0
    assert n>0

    if (jac is False):
        Jac = None
    
    if(ftype=='1stOrderUpwindFV'):  
        F=np.zeros((n,mx+1),ctx['data-type'])
        Flux=np.zeros((n,mx),ctx['data-type'])
        
        assert ghostPoints_l==1,'Need to provide exactly one ghost point'

        y=solution
        c=speed

        for id_mx in range(0,mx):
            Flux[:,id_mx]=-(1./dx)*(c[:,id_mx+1]*y[:,id_mx+1]-c[:,id_mx]*y[:,id_mx])

        if(jac):
            Jac1=np.zeros((mx,mx),ctx['data-type'])
            for id_mx in range(mx-1):
                Jac1[id_mx,id_mx]=(1./dx)*(c[0,id_mx])
                Jac1[id_mx,id_mx+1]=-(1./dx)*c[0,id_mx+1]
            J0=np.zeros((mx,mx))
            
            K1=np.hstack((Jac1,J0))
            K2=np.hstack((J0,Jac1))
            Jac=np.vstack((K1,K2))
            
    elif(ctx['Flux_name']=='3rdOrderUpwindFD'):
        Flux=np.zeros((n,mx),ctx['data-type'])
        y=solution
        c=speed

        #assert ghostPoints_l==2,'Need to provide exactly two ghost points'

        if(ghostPoints_l==2):        
            for id_mx in range(mx):    
                Flux[:,id_mx]=(c[:,id_mx+2]/dx)*((-1./6.)*y[:,id_mx]+(1.)*y[:,id_mx+1]+(-1./2.)*y[:,id_mx+2]+(-1./3.)*y[:,id_mx+3])
        elif(ghostPoints_l==1):
            Flux[:,0]=(c[:,0+2]/dx)*((1./3.)*y[:,0]+(1./2)*y[:,1]+(-1.)*y[:,2]+(1./6.)*y[:,3])
            Flux[:,1]=(c[:,1+2]/dx)*((-1./12.)*y[:,0]+(2./3.)*y[:,1]+(-2./3.)*y[:,2]+(1./12.)*y[:,3])
            for id_mx in range(2,mx):    
                Flux[:,id_mx]=(c[:,id_mx+2]/dx)*((-1./6.)*y[:,id_mx]+(1.)*y[:,id_mx+1]+(-1./2.)*y[:,id_mx+2]+(-1./3.)*y[:,id_mx+3])
    
    elif(ftype[0:13]=='FVStagVanLeer'):
        F=np.zeros((n,mx+1),ctx['data-type'])
        Flux=np.zeros((n,mx),ctx['data-type'])
        
        assert ghostPoints_l==2,'Need to provide exactly two ghost point'
        assert ghostPoints_r==2,'Need to provide exactly two ghost point'
        if(ftype=='FVStagVanLeer-k=1'):  # second order central
            kappa = 1.
        if(ftype=='FVStagVanLeer-k=-1'): # second order upwind
            kappa = -1.
        if(ftype=='FVStagVanLeer-k=1/3'): # third orderr
            kappa = 1./3.
        if(ftype=='FVStagVanLeer-k=0'):  # Fromm scheme (second order)
            kappa = 0.
        
        y=solution
        c=speed
        for id_n in range(n):
            for id_mx in range(mx+1):
                if(c[id_n,id_mx]>=0):
                    omega_j=y[id_n,id_mx+2]
                    omega_jm1=y[id_n,id_mx+1]
                    omega_jm2=y[id_n,id_mx]
                    F[id_n,id_mx]=c[id_n,id_mx+2]*(omega_jm1+(omega_jm1-omega_jm2)*(1.-kappa)/4.+(omega_j-omega_jm1)*(1.+kappa)/4.)
                else:
                    omega_jm1=-y[id_n,id_mx+2]
                    omega_j=-y[id_n,id_mx+1]
                    omega_jp1=-y[id_n,id_mx]
                    F[id_n,id_mx]=c[id_n,id_mx+2]*(omega_j+(omega_j-omega_jp1)*(1.-kappa)/4.+(omega_jm1-omega_j)*(1.+kappa)/4.)
            
        for id_mx in range(0,mx):
            Flux[:,id_mx]=-(1./dx)*(F[:,id_mx+1]-F[:,id_mx])
    else:
        raise NameError('{:} discretization type not implemented'.format(ctx['Flux_name']))


    if(Jac is None):
        return Flux
    else:
        return Flux, Jac

class Advection1D:
    def __init__(self,problem_ctx=None):

        self.rhs_i=None
        
        
        n=problem_ctx['n']
        mx=problem_ctx['mx']
        Flux_type=problem_ctx['Flux']
        Flux_name=problem_ctx['Flux_name']
        if('Flux_c' in problem_ctx.keys()):
            Flux_c=problem_ctx['Flux_c']
        elif('Flux_cv' in problem_ctx.keys()):
            Flux_cv=problem_ctx['Flux_cv']
        else:
            raise NotImplemented
        BC_type=problem_ctx['BC']
        x_max=problem_ctx['x_max']
        x_min=problem_ctx['x_min']
        
        dx=float(x_max-x_min)/mx
        
        x_coord=np.zeros((mx,))

        for i in range(mx):
            x_coord[i]=((i+1.)*dx)+x_min

        if(problem_ctx is None):
           raise Error
        else:
            ctx={}
            ctx['mx']=problem_ctx['mx']
            ctx['n']=problem_ctx['n']
            ctx['x_min']=problem_ctx['x_min']
            ctx['x_max']=problem_ctx['x_max']
            ctx['Flux']=problem_ctx['Flux']
            if('Flux_c' in problem_ctx.keys()):
                ctx['Flux_c']=problem_ctx['Flux_c']
            elif('Flux_cv' in problem_ctx.keys()):
                ctx['Flux_cv']=problem_ctx['Flux_cv']
            else:
                raise NotImplemented
            ctx['Flux_name']=problem_ctx['Flux_name']
            ctx['dx']=dx
            ctx['x_coord']=x_coord
            ctx['vectorize']=self.vectorize
            ctx['unvectorize']=self.unvectorize
            ctx['SplitSolution']=self.split_solution
            ctx['MergeSolution']=self.merge_solution

        


        problem_setup={}
        problem_setup['name']='Advection1D'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-01
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':5}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        u_ini_pde=np.zeros((n,mx),problem_setup['context']['data-type'])


        #u_ini_pde[0,:]=1.
        #u_ini_pde[0,mx//3:2*mx//3]=0.5
        u_ini_pde[0,:]=1.0+0.5*np.cos(2*np.pi*x_coord)
        #u_ini_pde[0,:]=x_coord[:]
        self._dx=ctx['dx']
        
        self._n=ctx['n']
        self._nF=int(ctx['mx']/2)
        self._nS=ctx['mx']-int(ctx['mx']/2)
        self._mx=ctx['mx']
        ctx['nF']=self._nF
        ctx['nS']=self._nS
        
        self.u_ini=self.vectorize(u_ini_pde,problem_ctx)
        problem_setup['context']['u_ini']=self.u_ini
        self.problem_setup=problem_setup

    def rhs_e_fast(self,t,uS_in,uF_in,ctx=None):
        
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        if('Flux_c' in ctx.keys()):
            c=ctx['Flux_c']
            cv=np.zeros((n,mx))
            cv[:]=c
        elif('Flux_cv' in ctx.keys()):
            cv=ctx['Flux_cv']
        else:
            raise NotImplemented
        cv_v=self.vectorize(cv,ctx)
        cv_vF,cv_vS=self.split_solution(cv_v)
        cS=self.unvectorize_partition(cv_vS,'S',ctx)
        cF=self.unvectorize_partition(cv_vF,'F',ctx)
        
        nF=ctx['nF']#self._nF
        nS=ctx['nS']#self._nS

        uS_in=self.unvectorize_partition(uS_in,'S',ctx)
        uF_in=self.unvectorize_partition(uF_in,'F',ctx)

        if(ctx['Flux_name']=='1stOrderUpwindFV'):
            y=np.hstack((np.reshape(uS_in[:,-1],(n,1)),uF_in,np.reshape(uS_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nF, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name']=='1stOrderUpwindFVStag'):       
            y=np.hstack((np.reshape(uS_in[:,-1],(n,1)),uF_in,np.reshape(uS_in[:,0],(n,1))))
            cm=np.hstack((np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1))))
            cp=np.hstack((cF,np.reshape(cS[:,0],(n,1)),np.reshape(cS[:,1],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nF, dx=dx, ctx=ctx, ftype='1stOrderUpwindFV')
    
        elif(ctx['Flux_name']=='3rdOrderUpwindFD'):            
            y=np.hstack((np.reshape(uS_in[:,-2],(n,1)),np.reshape(uS_in[:,-1],(n,1)),uF_in,np.reshape(uS_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cS[:,-2],(n,1)),np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nF, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name'][0:13]=='FVStagVanLeer'):
            y=np.hstack((np.reshape(uS_in[:,-2],(n,1)),np.reshape(uS_in[:,-1],(n,1)),uF_in,np.reshape(uS_in[:,0],(n,1)),np.reshape(uS_in[:,1],(n,1))))
            
            cm=np.hstack((np.reshape(cS[:,-3],(n,1)),np.reshape(cS[:,-2],(n,1)),np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1)),np.reshape(cS[:,1],(n,1))))
            cp=np.hstack((np.reshape(cS[:,-2],(n,1)),np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1)),np.reshape(cS[:,1],(n,1)),np.reshape(cS[:,2],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=2, speed=c, solution=y, n=n, mx=nF, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])

        else:
            raise NameError('Flux name {:} not implemented'.format(ctx['Flux_name']))
        
        u_out=self.vectorize_partition(Flux,'F',ctx)
        j_out=None
        return u_out,j_out

        
    def rhs_e_slow(self,t,uS_in,uF_in,ctx=None):
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        if('Flux_c' in ctx.keys()):
            c=ctx['Flux_c']
            cv=np.zeros((n,mx))
            cv[:]=c
        elif('Flux_cv' in ctx.keys()):
            cv=ctx['Flux_cv']
        else:
            raise NotImplemented
        cv_v=self.vectorize(cv,ctx)
        cv_vF,cv_vS=self.split_solution(cv_v)
        cS=self.unvectorize_partition(cv_vS,'S',ctx)
        cF=self.unvectorize_partition(cv_vF,'F',ctx)

        nF=ctx['nF']#self._nF
        nS=ctx['nS']#self._nS


        uS_in=self.unvectorize_partition(uS_in,'S',ctx)
        uF_in=self.unvectorize_partition(uF_in,'F',ctx)
        
        if(ctx['Flux_name']=='1stOrderUpwindFV'):
            y=np.hstack((np.reshape(uF_in[:,-1],(n,1)),uS_in,np.reshape(uF_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nS, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name']=='1stOrderUpwindFVStag'):
            y=np.hstack((np.reshape(uF_in[:,-1],(n,1)),uS_in,np.reshape(uF_in[:,0],(n,1))))
            cm=np.hstack((np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1))))
            cp=np.hstack((cS,np.reshape(cF[:,0],(n,1)),np.reshape(cF[:,1],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nS, dx=dx, ctx=ctx, ftype='1stOrderUpwindFV')
        elif(ctx['Flux_name']=='3rdOrderUpwindFD'):
            y=np.hstack((np.reshape(uF_in[:,-2],(n,1)),np.reshape(uF_in[:,-1],(n,1)),uS_in,np.reshape(uF_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cF[:,-2],(n,1)),np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nS, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name'][0:13]=='FVStagVanLeer'):
            y=np.hstack((np.reshape(uF_in[:,-2],(n,1)),np.reshape(uF_in[:,-1],(n,1)),uS_in,np.reshape(uF_in[:,0],(n,1)),np.reshape(uF_in[:,1],(n,1))))
            
            cm=np.hstack((np.reshape(cF[:,-3],(n,1)),np.reshape(cF[:,-2],(n,1)),np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1)),np.reshape(cF[:,1],(n,1))))
            cp=np.hstack((np.reshape(cF[:,-2],(n,1)),np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1)),np.reshape(cF[:,1],(n,1)),np.reshape(cF[:,-2],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=2, speed=c, solution=y, n=n, mx=nS, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        else:
            raise NameError('Flux name {:} not implemented'.format(ctx['Flux_name']))

        
        u_out=self.vectorize_partition(Flux,'S',ctx)
        j_out=None    
        return u_out,j_out

    
    def rhs_e(self,t,u_in,ctx=None):
        vec=ctx['vectorize']
        unvec=ctx['unvectorize']
        
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        if('Flux_c' in ctx.keys()):
            c=ctx['Flux_c']
            cv=np.zeros((n,mx))
            cv[:]=c
        elif('Flux_cv' in ctx.keys()):
            cv=ctx['Flux_cv']
        else:
            raise NotImplemented

        y_in=unvec(u_in,ctx)
 #       if(ctx['Flux_name']=='1stOrderUpwindFV'):
 ##           F=np.zeros((n,mx+1),ctx['data-type'])
 #           Flux=np.zeros((n,mx),ctx['data-type'])
 #           y=y_in
 #           F[:,0]=cv[:,-1]*y[:,-1]
 #           for id_mx in range(1,mx):
 #               F[:,id_mx]=cv[:,id_mx-1]*y[:,id_mx-1]
 #           F[:,mx]=cv[:,mx-1]*y[:,mx-1]
 #           for id_mx in range(0,mx):
 #               Flux[:,id_mx]=-(1./dx)*(F[:,id_mx+1]-F[:,id_mx])        
 #           
        if(ctx['Flux_name']=='1stOrderUpwindFV'):
            y=np.hstack((np.reshape(y_in[:,-1],(n,1)),y_in,np.reshape(y_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name']=='1stOrderUpwindFVStag'):
            y=np.hstack((np.reshape(y_in[:,-1],(n,1)),y_in,np.reshape(y_in[:,0],(n,1))))
            cm=np.hstack((np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1))))
            cp=np.hstack((cv,np.reshape(cv[:,0],(n,1)),np.reshape(cv[:,1],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=mx, dx=dx, ctx=ctx, ftype='1stOrderUpwindFV')
        elif(ctx['Flux_name'][0:13]=='FVStagVanLeer'):
            y=np.hstack((np.reshape(y_in[:,-2],(n,1)),np.reshape(y_in[:,-1],(n,1)),y_in,np.reshape(y_in[:,0],(n,1)),np.reshape(y_in[:,1],(n,1))))
            
            cm=np.hstack((np.reshape(cv[:,-3],(n,1)),np.reshape(cv[:,-2],(n,1)),np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1)),np.reshape(cv[:,1],(n,1))))
            cp=np.hstack((np.reshape(cv[:,-2],(n,1)),np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1)),np.reshape(cv[:,1],(n,1)),np.reshape(cv[:,-2],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=2, speed=c, solution=y, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name']=='3rdOrderUpwindFD'):
            y=np.hstack((np.reshape(y_in[:,-2],(n,1)),np.reshape(y_in[:,-1],(n,1)),y_in,np.reshape(y_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cv[:,-2],(n,1)),np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1))))     
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=1, speed=c, solution=y, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        else:
            raise NameError('Flux name {:} not implemented'.format(ctx['Flux_name']))

        u_out=vec(Flux,ctx)
        j_out=None
        return u_out,j_out
    
    def TotalMass(self,u_pde):
        mass=0.
        for i in range(self._mx):
            for j in range(self._n):
                mass+=self._dx*u_pde[j,i]
        
        return mass
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde.copy(),(pde_problem_ctx['mx']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode.copy(),(pde_problem_ctx['n'],pde_problem_ctx['mx']))
        return u_pde

    def vectorize_partition(self,u_pde,partition_id,pde_problem_ctx):
        if(partition_id=='F'):
            u_ode=np.reshape(u_pde.copy(),(pde_problem_ctx['nF']*pde_problem_ctx['n']))
        if(partition_id=='S'):
            u_ode=np.reshape(u_pde.copy(),(pde_problem_ctx['nS']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize_partition(self,u_ode,partition_id,pde_problem_ctx):
        if(partition_id=='F'):
            u_pde=np.reshape(u_ode.copy(),(pde_problem_ctx['n'],pde_problem_ctx['nF']))
        if(partition_id=='S'):
            u_pde=np.reshape(u_ode.copy(),(pde_problem_ctx['n'],pde_problem_ctx['nS']))
        return u_pde

    def split_solution(self,u_ode):
        nF=self._nF
        nS=self._nS
        mx=self._mx
        n=self._n
        
        uF_ode=np.reshape(u_ode[0:nF].copy(),(nF,))
        uS_ode=np.reshape(u_ode[nF:mx].copy(),(nS,))
        return uF_ode,uS_ode

    def merge_solution(self,uF_ode,uS_ode):
        nF=self._nF
        nS=self._nS
        mx=self._mx
        n=self._n
        u_ode=np.zeros((n*mx))
        u_ode[0:nF]=uF_ode.copy()
        u_ode[nF:mx]=uS_ode.copy()
        return u_ode
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)


class AdvectionDiffusion1D:
    def __init__(self,problem_ctx=None):

        self.rhs_i=None
        
        
        n=problem_ctx['n']
        mx=problem_ctx['mx']
        Flux_name=problem_ctx['Flux_name']
        Flux_type=problem_ctx['Flux']

        if('Flux_c' in problem_ctx.keys()):
            Flux_c=problem_ctx['Flux_c']
        elif('Flux_cv' in problem_ctx.keys()):
            Flux_cv=problem_ctx['Flux_cv']
        else:
            raise NotImplemented
        
        BC_type=problem_ctx['BC']
        x_max=problem_ctx['x_max']
        x_min=problem_ctx['x_min']
        kappa=problem_ctx['kappa']
        dx=float(x_max-x_min)/mx
        
        x_coord=np.zeros((mx,))

        for i in range(mx):
            x_coord[i]=((i+1.)*dx)+x_min

        if(problem_ctx is None):
           raise Error
        else:
            ctx={}
            ctx['mx']=problem_ctx['mx']
            ctx['n']=problem_ctx['n']
            ctx['x_min']=problem_ctx['x_min']
            ctx['x_max']=problem_ctx['x_max']
            ctx['Flux']=problem_ctx['Flux']
            if('Flux_c' in problem_ctx.keys()):
                ctx['Flux_c']=problem_ctx['Flux_c']
            elif('Flux_cv' in problem_ctx.keys()):
                ctx['Flux_cv']=problem_ctx['Flux_cv']
            else:
                raise NotImplemented
            ctx['Flux_name']=problem_ctx['Flux_name']
            ctx['dx']=dx
            ctx['x_coord']=x_coord
            ctx['vectorize']=self.vectorize
            ctx['unvectorize']=self.unvectorize
            ctx['SplitSolution']=self.split_solution
            ctx['MergeSolution']=self.merge_solution
            ctx['kappa']=problem_ctx['kappa']
            ctx['diffusion_tensor']=self.diffusion_tensor
            ctx['x_max']=problem_ctx['x_max']
            ctx['x_min']=problem_ctx['x_min']

        problem_setup={}
        problem_setup['name']='AdvectionDiffusion1D'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-01
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':5}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        u_ini_pde=np.zeros((n,mx),problem_setup['context']['data-type'])


        #u_ini_pde[0,:]=1.
        #u_ini_pde[0,mx//3:2*mx//3]=0.5

        u_ini_pde[0,:]=1.0+0.5*np.cos(2*np.pi*x_coord)

        #u_ini_pde[0,:]=1.0
        #u_ini_pde[0,int(mx*0.1):int(mx*0.3)]=1.+0.2*np.exp(-(x_coord[int(mx*0.1):int(mx*0.3)]-x_coord[int(mx*0.2)])**2/0.001)

        
        
        self._dx=ctx['dx']
        self._n=ctx['n']
        self._nF=int(ctx['mx']/2)
        self._nS=ctx['mx']-self._nF
        self._mx=ctx['mx']


        
        ctx['nF']=self._nF
        ctx['nS']=self._nS
        
        self.u_ini=self.vectorize(u_ini_pde,problem_ctx)
        problem_setup['context']['u_ini']=self.u_ini
        self.problem_setup=problem_setup

    def diffusion_tensor(self,t,ctx=None):
        y_k=np.zeros((ctx['mx'],),ctx['data-type'])
        
        x_max=ctx['x_max']
        x_min=ctx['x_min']
        x_coord=ctx['x_coord']
        y_k[:]=1.#(1.-np.sin(12.*t)**4)*np.sin(2*np.pi*(x_coord+t)/(x_max-x_min))**2
        return y_k
    
    def rhs_mr_implicit(self,t,u_in,ctx=None):
        
        mx=ctx['mx']
        dx=ctx['dx']
        kappa=ctx['kappa']
        diffusion_tensor=ctx['diffusion_tensor']
        u_out=np.zeros((mx,),ctx['data-type'])
        A=np.zeros((mx,mx),ctx['data-type'])
        
        K=kappa*diffusion_tensor(t,ctx)
        
        #print(K)
        
        A[0,0]=-2.
        A[0,1]=1.
        A[0,mx-1]=1.
        for i in range(1,mx-1):
            A[i,i-1]=1.
            A[i,i]=-2.
            A[i,i+1]=1.
        A[mx-1,0]=1.
        A[mx-1,mx-1]=-2.
        A[mx-1,mx-2]=1.
        
        for i in range(mx):
            A[i,:]=K[i]*A[i,:]
        
        A=A/(dx**2)
    
        u_out=np.matmul(A,u_in)
               
        j_out=A
        return u_out,j_out
    
    def rhs_e_fast(self,t,uS_in,uF_in,ctx=None):            
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        if('Flux_c' in ctx.keys()):
            c=ctx['Flux_c']
            cv=np.zeros((n,mx))
            cv[:]=c
        elif('Flux_cv' in ctx.keys()):
            cv=ctx['Flux_cv']
        else:
            raise NotImplemented
        cv_v=self.vectorize(cv,ctx)
        cv_vF,cv_vS=self.split_solution(cv_v)
        cS=self.unvectorize_partition(cv_vS,'S',ctx)
        cF=self.unvectorize_partition(cv_vF,'F',ctx)
        
        nF=ctx['nF']#self._nF
        nS=ctx['nS']#self._nS

        uS_in=self.unvectorize_partition(uS_in,'S',ctx)
        uF_in=self.unvectorize_partition(uF_in,'F',ctx)

        if(ctx['Flux_name']=='1stOrderUpwindFV'):
            y=np.hstack((np.reshape(uS_in[:,-1],(n,1)),uF_in,np.reshape(uS_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nF, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name']=='1stOrderUpwindFVStag'):       
            y=np.hstack((np.reshape(uS_in[:,-1],(n,1)),uF_in,np.reshape(uS_in[:,0],(n,1))))
            cm=np.hstack((np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1))))
            cp=np.hstack((cF,np.reshape(cS[:,0],(n,1)),np.reshape(cS[:,1],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nF, dx=dx, ctx=ctx, ftype='1stOrderUpwindFV')
    
        elif(ctx['Flux_name']=='3rdOrderUpwindFD'):            
            y=np.hstack((np.reshape(uS_in[:,-2],(n,1)),np.reshape(uS_in[:,-1],(n,1)),uF_in,np.reshape(uS_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cS[:,-2],(n,1)),np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nF, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name'][0:13]=='FVStagVanLeer'):
            y=np.hstack((np.reshape(uS_in[:,-2],(n,1)),np.reshape(uS_in[:,-1],(n,1)),uF_in,np.reshape(uS_in[:,0],(n,1)),np.reshape(uS_in[:,1],(n,1))))
            
            cm=np.hstack((np.reshape(cS[:,-3],(n,1)),np.reshape(cS[:,-2],(n,1)),np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1)),np.reshape(cS[:,1],(n,1))))
            cp=np.hstack((np.reshape(cS[:,-2],(n,1)),np.reshape(cS[:,-1],(n,1)),cF,np.reshape(cS[:,0],(n,1)),np.reshape(cS[:,1],(n,1)),np.reshape(cS[:,2],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=2, speed=c, solution=y, n=n, mx=nF, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])

        else:
            raise NameError('Flux name {:} not implemented'.format(ctx['Flux_name']))
        
        u_out=self.vectorize_partition(Flux,'F',ctx)
        j_out=None
        return u_out,j_out

        
    def rhs_e_slow(self,t,uS_in,uF_in,ctx=None):
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        if('Flux_c' in ctx.keys()):
            c=ctx['Flux_c']
            cv=np.zeros((n,mx))
            cv[:]=c
        elif('Flux_cv' in ctx.keys()):
            cv=ctx['Flux_cv']
        else:
            raise NotImplemented
        cv_v=self.vectorize(cv,ctx)
        cv_vF,cv_vS=self.split_solution(cv_v)
        cS=self.unvectorize_partition(cv_vS,'S',ctx)
        cF=self.unvectorize_partition(cv_vF,'F',ctx)

        nF=ctx['nF']#self._nF
        nS=ctx['nS']#self._nS


        uS_in=self.unvectorize_partition(uS_in,'S',ctx)
        uF_in=self.unvectorize_partition(uF_in,'F',ctx)
        
        if(ctx['Flux_name']=='1stOrderUpwindFV'):
            y=np.hstack((np.reshape(uF_in[:,-1],(n,1)),uS_in,np.reshape(uF_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nS, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name']=='1stOrderUpwindFVStag'):
            y=np.hstack((np.reshape(uF_in[:,-1],(n,1)),uS_in,np.reshape(uF_in[:,0],(n,1))))
            cm=np.hstack((np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1))))
            cp=np.hstack((cS,np.reshape(cF[:,0],(n,1)),np.reshape(cF[:,1],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nS, dx=dx, ctx=ctx, ftype='1stOrderUpwindFV')
        elif(ctx['Flux_name']=='3rdOrderUpwindFD'):
            y=np.hstack((np.reshape(uF_in[:,-2],(n,1)),np.reshape(uF_in[:,-1],(n,1)),uS_in,np.reshape(uF_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cF[:,-2],(n,1)),np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=1, speed=c, solution=y, n=n, mx=nS, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name'][0:13]=='FVStagVanLeer'):
            y=np.hstack((np.reshape(uF_in[:,-2],(n,1)),np.reshape(uF_in[:,-1],(n,1)),uS_in,np.reshape(uF_in[:,0],(n,1)),np.reshape(uF_in[:,1],(n,1))))
            
            cm=np.hstack((np.reshape(cF[:,-3],(n,1)),np.reshape(cF[:,-2],(n,1)),np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1)),np.reshape(cF[:,1],(n,1))))
            cp=np.hstack((np.reshape(cF[:,-2],(n,1)),np.reshape(cF[:,-1],(n,1)),cS,np.reshape(cF[:,0],(n,1)),np.reshape(cF[:,1],(n,1)),np.reshape(cF[:,-2],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=2, speed=c, solution=y, n=n, mx=nS, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        else:
            raise NameError('Flux name {:} not implemented'.format(ctx['Flux_name']))

        
        u_out=self.vectorize_partition(Flux,'S',ctx)
        j_out=None    
        return u_out,j_out
            
    def rhs_e(self,t,u_in,ctx=None):

        vec=ctx['vectorize']
        unvec=ctx['unvectorize']
        
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        if('Flux_c' in ctx.keys()):
            c=ctx['Flux_c']
            cv=np.zeros((n,mx))
            cv[:]=c
        elif('Flux_cv' in ctx.keys()):
            cv=ctx['Flux_cv']
        else:
            raise NotImplemented

        y_in=unvec(u_in,ctx)
 #       if(ctx['Flux_name']=='1stOrderUpwindFV'):
 ##           F=np.zeros((n,mx+1),ctx['data-type'])
 #           Flux=np.zeros((n,mx),ctx['data-type'])
 #           y=y_in
 #           F[:,0]=cv[:,-1]*y[:,-1]
 #           for id_mx in range(1,mx):
 #               F[:,id_mx]=cv[:,id_mx-1]*y[:,id_mx-1]
 #           F[:,mx]=cv[:,mx-1]*y[:,mx-1]
 #           for id_mx in range(0,mx):
 #               Flux[:,id_mx]=-(1./dx)*(F[:,id_mx+1]-F[:,id_mx])        
 #           
        if(ctx['Flux_name']=='1stOrderUpwindFV'):
            y=np.hstack((np.reshape(y_in[:,-1],(n,1)),y_in,np.reshape(y_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1))))
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name']=='1stOrderUpwindFVStag'):
            y=np.hstack((np.reshape(y_in[:,-1],(n,1)),y_in,np.reshape(y_in[:,0],(n,1))))
            cm=np.hstack((np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1))))
            cp=np.hstack((cv,np.reshape(cv[:,0],(n,1)),np.reshape(cv[:,1],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=n, mx=mx, dx=dx, ctx=ctx, ftype='1stOrderUpwindFV')
        elif(ctx['Flux_name'][0:13]=='FVStagVanLeer'):
            y=np.hstack((np.reshape(y_in[:,-2],(n,1)),np.reshape(y_in[:,-1],(n,1)),y_in,np.reshape(y_in[:,0],(n,1)),np.reshape(y_in[:,1],(n,1))))
            
            cm=np.hstack((np.reshape(cv[:,-3],(n,1)),np.reshape(cv[:,-2],(n,1)),np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1)),np.reshape(cv[:,1],(n,1))))
            cp=np.hstack((np.reshape(cv[:,-2],(n,1)),np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1)),np.reshape(cv[:,1],(n,1)),np.reshape(cv[:,-2],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=2, speed=c, solution=y, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        elif(ctx['Flux_name']=='3rdOrderUpwindFD'):
            y=np.hstack((np.reshape(y_in[:,-2],(n,1)),np.reshape(y_in[:,-1],(n,1)),y_in,np.reshape(y_in[:,0],(n,1))))
            c=np.hstack((np.reshape(cv[:,-2],(n,1)),np.reshape(cv[:,-1],(n,1)),cv,np.reshape(cv[:,0],(n,1))))     
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=1, speed=c, solution=y, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
        else:
            raise NameError('Flux name {:} not implemented'.format(ctx['Flux_name']))

        u_out=vec(Flux,ctx)
        j_out=None
        return u_out,j_out

    def TotalMass(self,u_pde):
        mass=0.
        for i in range(self._mx):
            for j in range(self._n):
                mass+=self._dx*u_pde[j,i]
        
        return mass
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde.copy(),(pde_problem_ctx['mx']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode.copy(),(pde_problem_ctx['n'],pde_problem_ctx['mx']))
        return u_pde

    def vectorize_partition(self,u_pde,partition_id,pde_problem_ctx):
        if(partition_id=='F'):
            u_ode=np.reshape(u_pde.copy(),(pde_problem_ctx['nF']*pde_problem_ctx['n']))
        if(partition_id=='S'):
            u_ode=np.reshape(u_pde.copy(),(pde_problem_ctx['nS']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize_partition(self,u_ode,partition_id,pde_problem_ctx):
        if(partition_id=='F'):
            u_pde=np.reshape(u_ode.copy(),(pde_problem_ctx['n'],pde_problem_ctx['nF']))
        if(partition_id=='S'):
            u_pde=np.reshape(u_ode.copy(),(pde_problem_ctx['n'],pde_problem_ctx['nS']))
        return u_pde

    def split_solution(self,u_ode):
        nF=self._nF
        nS=self._nS
        mx=self._mx
        n=self._n
        
        uF_ode=np.reshape(u_ode[0:nF].copy(),(nF,))
        uS_ode=np.reshape(u_ode[nF:mx].copy(),(nS,))
        return uF_ode,uS_ode

    def merge_solution(self,uF_ode,uS_ode):
        nF=self._nF
        nS=self._nS
        mx=self._mx
        n=self._n
        u_ode=np.zeros((n*mx))
        u_ode[0:nF]=uF_ode.copy()
        u_ode[nF:mx]=uS_ode.copy()
        return u_ode    
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)



class AdvectionReaction1D:
    def __init__(self,problem_ctx=None):
        self.exact_solution=None
        s=problem_ctx['s']
        alpha=problem_ctx['alpha']
        n=problem_ctx['n']
        mx=problem_ctx['mx']
        Flux_type=problem_ctx['Flux']
        Flux_name=problem_ctx['Flux_name']

        if('Flux_c' in problem_ctx.keys()):
            Flux_c=problem_ctx['Flux_c']
        elif('Flux_cv' in problem_ctx.keys()):
            Flux_cv=problem_ctx['Flux_cv']
        else:
            raise NotImplemented
        
        BC_type=problem_ctx['BC']
        x_max=problem_ctx['x_max']
        x_min=problem_ctx['x_min']
        kappa=problem_ctx['kappa']
        dx=float(x_max-x_min)/mx
        
        x_coord=np.zeros((mx,))

        for i in range(mx):
            x_coord[i]=((i+1.)*dx)+x_min

        if(problem_ctx is None):
           raise Error
        else:
            ctx={}
            if('alpha' in problem_ctx.keys()):
                ctx['alpha']=problem_ctx['alpha']
            if('alpha_vec' in problem_ctx.keys()):
                ctx['alpha_vec']=problem_ctx['alpha_vec']
            
            if('alpha_i' in problem_ctx.keys()):
                ctx['alpha_i']=problem_ctx['alpha_i']
            if('alpha_i_vec' in problem_ctx.keys()):
                ctx['alpha_i_vec']=problem_ctx['alpha_vec_i_vec']

            ctx['s']=problem_ctx['s']
            ctx['mx']=problem_ctx['mx']
            ctx['n']=problem_ctx['n']
            ctx['x_min']=problem_ctx['x_min']
            ctx['x_max']=problem_ctx['x_max']
            ctx['Flux']=problem_ctx['Flux']
            if('Flux_c' in problem_ctx.keys()):
                ctx['Flux_c']=problem_ctx['Flux_c']
            elif('Flux_cv' in problem_ctx.keys()):
                ctx['Flux_cv']=problem_ctx['Flux_cv']
            else:
                raise NotImplemented
            ctx['Flux_name']=problem_ctx['Flux_name']
            ctx['BC']=problem_ctx['BC']
            ctx['IC']=problem_ctx['IC']
            ctx['dx']=dx
            ctx['x_coord']=x_coord
            ctx['vectorize']=self.vectorize
            ctx['unvectorize']=self.unvectorize
            ctx['kappa']=problem_ctx['kappa']
            ctx['x_max']=problem_ctx['x_max']
            ctx['x_min']=problem_ctx['x_min']

        problem_setup={}
        problem_setup['name']='Advection Reaction 1D'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-01
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':5}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        u_ini_pde=np.zeros((n,mx),problem_setup['context']['data-type'])


        if(ctx['IC']=='Hundsdorfer'):
            u_ini_pde[0,:]=1.0+s[1]*x_coord
            u_ini_pde[1,:]=(kappa[0]/kappa[1])* u_ini_pde[0,:]+s[1]/kappa[1]
        elif(ctx['IC']=='Test1'):
            u_ini_pde[0,:]=1.0+np.sin(2*x_coord*np.pi*2)
            u_ini_pde[1,:]=np.sin(2*x_coord*np.pi*4)

        self._dx=ctx['dx']
        self._n=ctx['n']
        self._nF=int(ctx['mx']/2)
        self._nS=ctx['mx']-int(ctx['mx']/2)
        self._mx=ctx['mx']
        ctx['nF']=self._nF
        ctx['nS']=self._nS
        
        self.u_ini=self.vectorize(u_ini_pde,problem_ctx)
        problem_setup['context']['u_ini']=self.u_ini
        self.problem_setup=problem_setup
    
    def rhs_e(self,t,u_in,ctx=None):
        vec=ctx['vectorize']
        unvec=ctx['unvectorize']

        s=ctx['s']
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        alpha=ctx['alpha']

        if('Flux_c' in ctx.keys()):
            c=ctx['Flux_c']
            cv=np.zeros((n,mx))
            cv[:]=c
        elif('Flux_cv' in ctx.keys()):
            cv=ctx['Flux_cv']
        else:
            raise NotImplemented

        cv=np.zeros((n,mx))
        cv[0,:]=alpha[0]
        cv[1,:]=alpha[1]

        
        F=np.zeros((n,mx+1),ctx['data-type'])
        Flux=np.zeros((n,mx),ctx['data-type'])
        
        y=unvec(u_in,ctx)


        
        
        #F[:,0]=cv[:,-1]*yb[:]
        #for id_mx in range(1,mx):
        #    F[:,id_mx]=cv[:,id_mx-1]*y[:,id_mx-1]
        #F[:,mx]=cv[:,mx-1]*y[:,mx-1]
    
        #for id_mx in range(0,mx):
        #    Flux[:,id_mx]=-(1./dx)*(F[:,id_mx+1]-F[:,id_mx])+np.reshape(s[:],(2,))
        


        if('alpha' in ctx.keys()):
            c=ctx['alpha']
            cv=np.zeros((n,mx))
            cv[0,:]=c[0]
            cv[1,:]=c[1]
        elif('alpha_vec' in ctx.keys()):
            cv=ctx['alpha_vec']
        else:
            raise NotImplemented

        if(ctx['BC']=='Hundsdorfer'):
            yb=np.asarray([1-np.sin(12*t)**4,0]).reshape(2,)
            yb_left=yb
            yb_right=0
        elif(ctx['BC']=='periodic'):
            yb_left=np.zeros((n,2))
            yb_right=np.zeros((n,2))

            yb_left[:,0]=y[:,-2]
            yb_left[:,1]=y[:,-1]
            yb_right[:,0]=y[:,0]
            yb_right[:,1]=y[:,1]

            c_left=np.zeros((n,2))
            c_right=np.zeros((n,2))

            c_left[:,0]=cv[:,-2]
            c_left[:,1]=cv[:,-1]
            c_right[:,0]=cv[:,0]
            c_right[:,1]=cv[:,1]

        if(ctx['Flux_name']=='1stOrderUpwindFV'):
            u=np.hstack((np.reshape(yb_left[:,1],(n,1)),y,np.reshape(yb_right[:,1],(n,1))))
            w=np.hstack((np.reshape(c_left[:,1],(n,1)),cv,np.reshape(c_right[:,1],(n,1))))
            
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=w, solution=u, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])

        elif(ctx['Flux_name']=='1stOrderUpwindFVStag'):       
            u=np.hstack((np.reshape(yb_left[:,1],(n,1)),y,np.reshape(yb_right[:,1],(n,1))))

            cm=np.hstack((np.reshape(c_left[:,:],(n,2)),cv))
            cp=np.hstack((np.reshape(c_left[:,1],(n,1)),cv,np.reshape(c_right[:,1],(n,1))))
            c=0.5*(cm+cp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=u, n=n, mx=mx, dx=dx, ctx=ctx, ftype='1stOrderUpwindFV')
    
        elif(ctx['Flux_name']=='3rdOrderUpwindFD'):            
            u=np.hstack((np.reshape(yb_left[:,:],(n,2)),y,np.reshape(yb_right[:,:],(n,2))))
            w=np.hstack((np.reshape(c_left[:,:],(n,2)),cv,np.reshape(c_right[:,:],(n,2))))

            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=1, speed=w, solution=u, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])

        elif(ctx['Flux_name'][0:13]=='FVStagVanLeer'):
            u=np.hstack((np.reshape(yb_left[:,:],(n,2)),y,np.reshape(yb_right[:,:],(n,2))))
            wm=np.hstack((np.reshape(cv[:,-3],(n,1)),np.reshape(c_left[:,:],(n,2)),cv,np.reshape(c_right[:,:],(n,2))))
            wp=np.hstack((np.reshape(c_left[:,:],(n,2)),cv,np.reshape(c_right[:,:],(n,2)),np.reshape(cv[:,2],(n,1))))

            c=0.5*(wm+wp)
            Flux=LinearAdvectionSemiDiscretization1D(ghostPoints_l=2, ghostPoints_r=2, speed=c, solution=u, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])

        elif(ctx['Flux_name']=='Hundsdorfer'):
            Flux[0,0]=alpha[0]*((1./3.)*yb_left[0]+(0.5)*y[0,0]+(-1.)*y[0,1]+(1./6.)*y[0,2])/dx
            if(alpha[1]!=0):
                Flux[1,0]=alpha[1]*((1./3.)*yb_left[1]+(0.5)*y[1,0]+(-1.)*y[1,1]+(1./6.)*y[1,2])/dx
            
            Flux[0,1]=alpha[0]*((-1./12.)*yb_left[0]+(2./3.)*y[0,0]+(-2./3.)*y[0,2]+(1./12.)*y[0,3])/dx
            if(alpha[1]!=0):
                Flux[1,1]=alpha[1]*((-1./12.)*yb_left[1]+(2./3.)*y[1,0]+(-2./3.)*y[1,2]+(1./12.)*y[1,3])/dx
            
            for i in range(2,mx-2): 
                Flux[0,i]=alpha[0]*((-1./12.)*y[0,i-2]+(2./3.)*y[0,i-1]+(-2./3.)*y[0,i+1]+(1./12.)*y[0,i+2])/dx

            if(alpha[1]!=0):
                for i in range(2,mx-2):
                    Flux[1,i]=alpha[1]*((-1./12.)*y[1,i-2]+(2./3.)*y[1,i-1]+(-2./3.)*y[1,i+1]+(1./12.)*y[1,i+2])/dx
            
            
            Flux[0,mx-2]=alpha[0]*((-1./6.)*y[0,mx-4]+(1.)*y[0,mx-3]+(-0.5)*y[0,mx-2]+(-1./3.)*y[0,mx-1])/dx
            if(alpha[1]!=0):
                Flux[1,mx-2]=alpha[1]*((-1./6.)*y[1,mx-4]+(1.)*y[1,mx-3]+(-0.5)*y[1,mx-2]+(-1./3.)*y[2,mx-1])/dx
            
            
            Flux[0,mx-1]=-alpha[0]*(y[0,mx-1]-y[0,mx-2])/dx
            
            if(alpha[1]!=0):
                Flux[1,mx-1]=-alpha[1]*(y[1,mx-1]-y[1,mx-2])/dx


            for i in range(mx):
                Flux[:,i]=Flux[:,i]+np.reshape(s[:],(2,))
        else:
            raise NameError('Flux name {:} not implemented'.format(ctx['Flux_name']))
 
        u_out=vec(Flux,ctx)
        
        j_out=None
        
        return u_out,j_out
    
    def rhs_i(self,t,u_in,ctx=None):
        vec=ctx['vectorize']
        unvec=ctx['unvectorize']

        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        kappa=ctx['kappa']
        
        F=np.zeros((n,mx),ctx['data-type'])
        
        K=np.asarray([[-kappa[0],kappa[1]],[kappa[0], -kappa[1]]])
              
        y=unvec(u_in,ctx)

        for id_mx in range(mx):
            F[:,id_mx]=np.matmul(K,y[:,id_mx])
        
        # Advection
        
        if('alpha_i' in ctx.keys()):
            c=ctx['alpha_i']
            cv=np.zeros((n,mx))
            cv[0,:]=c[0]
            cv[1,:]=c[1]
            ImplicitAdvection=True
        elif('alpha_i_vec' in ctx.keys()):
            cv=ctx['alpha_i_vec']
            ImplicitAdvection=True
        else:
            ImplicitAdvection=False

        if(ImplicitAdvection==True):
            if(ctx['BC']=='Hundsdorfer'):
                yb=np.asarray([1-np.sin(12*t)**4,0]).reshape(2,)
                yb_left=yb
                yb_right=0
            elif(ctx['BC']=='periodic'):
                yb_left=np.zeros((n,2))
                yb_right=np.zeros((n,2))

                yb_left[:,0]=y[:,-2]
                yb_left[:,1]=y[:,-1]
                yb_right[:,0]=y[:,0]
                yb_right[:,1]=y[:,1]

                c_left=np.zeros((n,2))
                c_right=np.zeros((n,2))

                c_left[:,0]=cv[:,-2]
                c_left[:,1]=cv[:,-1]
                c_right[:,0]=cv[:,0]
                c_right[:,1]=cv[:,1]

            if(ctx['Flux_name']=='1stOrderUpwindFV'):
                u=np.hstack((np.reshape(yb_left[:,1],(n,1)),y,np.reshape(yb_right[:,1],(n,1))))
                w=np.hstack((np.reshape(c_left[:,1],(n,1)),cv,np.reshape(c_right[:,1],(n,1))))
                
                Flux,_ = LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, 
                                speed=w, solution=u, n=n, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'], jac=False)
                if('noJacobian' in ctx.keys()):
                    JacAd=np.zeros((mx*n,mx*n))
                else:
                    JacAd = jacfwd(LinearAdvectionSemiDiscretization1D_dict)(np.reshape(u,(n*(mx+2),)),{'ghostPoints_l':1, 'ghostPoints_r':1, 
                                        'bc_l':None, 'bc_r':None, 'speed':w, 'n':n, 'mx':mx,
                                        'time':None, 'dx':dx, 'ctx':ctx, 'ftype':ctx['Flux_name'], 'jac':False})
                    JacAd=np.asarray(JacAd)
                    JacAd=JacAd[:,1:n*mx+1] 
            elif(ctx['Flux_name']=='1stOrderUpwindFVStag'):       
                raise NameError('Flux name {:} not implemented on the implicit side'.format(ctx['Flux_name']))
            elif(ctx['Flux_name']=='3rdOrderUpwindFD'):            
                raise NameError('Flux name {:} not implemented on the implicit side'.format(ctx['Flux_name']))
            elif(ctx['Flux_name'][0:13]=='FVStagVanLeer'):
                raise NameError('Flux name {:} not implemented on the implicit side'.format(ctx['Flux_name']))
            elif(ctx['Flux_name']=='Hundsdorfer'):
                raise NameError('Flux name {:} not implemented on the implicit side'.format(ctx['Flux_name']))
            else:
                raise NameError('Flux name {:} not implemented on th eimplicit side'.format(ctx['Flux_name']))

        

        if(ImplicitAdvection==False):
            u_out=vec(F,ctx)
        else:
            u_out=vec(F+Flux,ctx)
        
        
        K11=kappa[0]*np.eye(mx)
        K12=kappa[1]*np.eye(mx)
        K1=np.hstack(((-1)*K11,K12))
        K2=np.hstack((K11,(-1)*K12))
        j_out=np.vstack((K1,K2))

        

        if(ImplicitAdvection==True):   
            j_out+=JacAd

        return u_out,j_out

    def TotalMass(self,u_pde):
        mass=0.
        for i in range(self._mx):
            for j in range(self._n):
                mass+=self._dx*u_pde[j,i]
        
        return mass
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde.copy(),(pde_problem_ctx['mx']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode.copy(),(pde_problem_ctx['n'],pde_problem_ctx['mx']))
        return u_pde
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)


class AdvectionReaction1DSplit:
    def __init__(self,problem_ctx=None):

        s=problem_ctx['s']
        alpha=problem_ctx['alpha']
        alpha_i=problem_ctx['alpha_i']
        n=problem_ctx['n']
        mx=problem_ctx['mx']
        Flux_type=problem_ctx['Flux']
        Flux_name=problem_ctx['Flux_name']

        if('Flux_c' in problem_ctx.keys()):
            Flux_c=problem_ctx['Flux_c']
        elif('Flux_cv' in problem_ctx.keys()):
            Flux_cv=problem_ctx['Flux_cv']
        else:
            raise NotImplemented
        
        BC_type=problem_ctx['BC']
        x_max=problem_ctx['x_max']
        x_min=problem_ctx['x_min']
        kappa=problem_ctx['kappa']
        dx=float(x_max-x_min)/mx
        
        x_coord=np.zeros((mx,))

        for i in range(mx):
            x_coord[i]=((i+1.)*dx)+x_min

        if(problem_ctx is None):
           raise Error
        else:
            ctx={}
            ctx['alpha']=problem_ctx['alpha']
            ctx['alpha_i']=problem_ctx['alpha_i']
            ctx['s']=problem_ctx['s']
            ctx['mx']=problem_ctx['mx']
            ctx['n']=problem_ctx['n']
            ctx['x_min']=problem_ctx['x_min']
            ctx['x_max']=problem_ctx['x_max']
            ctx['Flux']=problem_ctx['Flux']
            if('Flux_c' in problem_ctx.keys()):
                ctx['Flux_c']=problem_ctx['Flux_c']
            elif('Flux_cv' in problem_ctx.keys()):
                ctx['Flux_cv']=problem_ctx['Flux_cv']
            else:
                raise NotImplemented
            ctx['Flux_name']=problem_ctx['Flux_name']
            ctx['dx']=dx
            ctx['x_coord']=x_coord
            ctx['vectorize']=self.vectorize
            ctx['unvectorize']=self.unvectorize
            ctx['kappa']=problem_ctx['kappa']
            ctx['x_max']=problem_ctx['x_max']
            ctx['x_min']=problem_ctx['x_min']

        problem_setup={}
        problem_setup['name']='Advection Reaction 1D Split'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-01
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':5}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        u_ini_pde=np.zeros((n,mx),problem_setup['context']['data-type'])



        #u_ini_pde[0,:]=1.0+s[1]*x_coord
        #u_ini_pde[1,:]=(kappa[0]/kappa[1])* u_ini_pde[0,:]+s[1]/kappa[1]


        u_ini_pde[0,:]=1.0+x_coord
        u_ini_pde[1,:]= u_ini_pde[0,:]*0.5

        self._dx=ctx['dx']
        self._n=ctx['n']
        self._nF=int(ctx['mx']/2)
        self._nS=ctx['mx']-int(ctx['mx']/2)
        self._mx=ctx['mx']
        ctx['nF']=self._nF
        ctx['nS']=self._nS
        
        self.u_ini=self.vectorize(u_ini_pde,problem_ctx)
        problem_setup['context']['u_ini']=self.u_ini
        self.problem_setup=problem_setup
    
    def rhs_e(self,t,u_in,ctx=None):
        vec=ctx['vectorize']
        unvec=ctx['unvectorize']

        s=ctx['s']
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        alpha=ctx['alpha']
        alpha_i=ctx['alpha_i']

        if('Flux_c' in ctx.keys()):
            c=ctx['Flux_c']
            cv=np.zeros((n,mx))
            cv[:]=c
        elif('Flux_cv' in ctx.keys()):
            cv=ctx['Flux_cv']
        else:
            raise NotImplemented

        cv=np.zeros((n,mx))
        cv[0,:]=alpha[0]
        cv[1,:]=alpha[1]

        Flux=np.zeros((n,mx),ctx['data-type'])
        y_in=unvec(u_in,ctx)


        yb=np.asarray([1-np.sin(12*t)**4,0]).reshape(2,)
        
        if(ctx['Flux_name']=='1stOrderUpwindFV'):
            for i in range(n):
                y=np.hstack((np.reshape(y_in[i,-1],(1,1)),np.reshape(y_in[i,:],(1,mx)),np.reshape(y_in[i,0],(1,1))))
                c=np.hstack((np.reshape(cv[i,-1],(1,1)),np.reshape(cv[i,:],(1,mx)),np.reshape(cv[i,0],(1,1))))
                Flux[i,:]=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=1, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'])
                


            
        u_out=vec(Flux,ctx)
        j_out=None
        
        return u_out,j_out

    def rhs_i(self,t,u_in,ctx=None):
        vec=ctx['vectorize']
        unvec=ctx['unvectorize']
        
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        kappa=ctx['kappa']
        alpha_i=ctx['alpha_i']
        
        F=np.zeros((n,mx),ctx['data-type'])


        
        K=np.asarray([[-kappa[0],kappa[1]],[kappa[0], -kappa[1]]])
        
       
        y_in=unvec(u_in,ctx)

        for id_mx in range(mx):
            F[:,id_mx]=np.matmul(K,y_in[:,id_mx])
            
        u_out=vec(F,ctx)




        if('Flux_c' in ctx.keys()):
            c=ctx['Flux_c']
            cv=np.zeros((n,mx))
            cv[:]=c
        elif('Flux_cv' in ctx.keys()):
            cv=ctx['Flux_cv']
        else:
            raise NotImplemented

        cv=np.zeros((n,mx))
        cv[0,:]=alpha_i[0]
        cv[1,:]=alpha_i[1]

        Flux=np.zeros((n,mx),ctx['data-type'])

        Jacs=[]
        
        if(ctx['Flux_name']=='1stOrderUpwindFV'):
            for i in range(n):
                y=np.hstack((np.reshape(y_in[i,-1],(1,1)),np.reshape(y_in[i,:],(1,mx)),np.reshape(y_in[i,0],(1,1))))
                c=np.hstack((np.reshape(cv[i,-1],(1,1)),np.reshape(cv[i,:],(1,mx)),np.reshape(cv[i,0],(1,1))))
                Flux[i,:], _=LinearAdvectionSemiDiscretization1D(ghostPoints_l=1, ghostPoints_r=1, speed=c, solution=y, n=1, mx=mx, dx=dx, ctx=ctx, ftype=ctx['Flux_name'], jac=False)
                #Jacs.append(Jac)
                
        u_out+=vec(Flux,ctx)

        
        K11=kappa[0]*np.eye(mx)
        K12=kappa[1]*np.eye(mx)
        K1=np.hstack(((-1)*K11,K12))
        K2=np.hstack((K11,(-1)*K12))
        j_out=np.vstack((K1,K2))

        ZeroBlock=np.zeros((mx,mx))

        B1=np.hstack((Jacs[0],ZeroBlock))
        B2=np.hstack((ZeroBlock,Jacs[1]))
        j_out+=np.vstack((B1,B2))

        return u_out,j_out

    def TotalMass(self,u_pde):
        mass=0.
        for i in range(self._mx):
            for j in range(self._n):
                mass+=self._dx*u_pde[j,i]
        
        return mass
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde.copy(),(pde_problem_ctx['mx']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode.copy(),(pde_problem_ctx['n'],pde_problem_ctx['mx']))
        return u_pde
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)


    
class GrayScott:

    def __init__(self,problem_ctx=None):

        self.rhs_i=None

        
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
                 'vectorize':self.vectorize,
                 'unvectorize':self.unvectorize,
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
            ctx['vectorize']=self.vectorize
            ctx['unvectorize']=self.unvectorize

        


        problem_setup={}
        problem_setup['name']='Gray-Scott'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-00
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':5}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        u_ini_pde=np.zeros((n,mx,my),problem_setup['context']['data-type'])
        u_ini_pde[0,:,:]=1.
        u_ini_pde[0,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.5+np.reshape(np.random.uniform(low=-0.5*1e-02,high=0.5*1e-02,size=20*20),(20,20))
        u_ini_pde[1,:,:]=0.
        u_ini_pde[1,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.25+np.reshape(np.random.uniform(low=-0.25*1e-02,high=0.25*1e-02,size=20*20),(20,20))

        self.u_ini=self.vectorize(u_ini_pde,problem_ctx)
        problem_setup['context']['u_ini']=self.u_ini
        self.problem_setup=problem_setup

        
    def rhs_e(self,t,u_in,ctx=None):
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
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['mx']*pde_problem_ctx['my']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['mx'],pde_problem_ctx['my']))
        return u_pde
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)

class NavierStokes2D:

    def __init__(self,problem_ctx=None):

        self.rhs_i=None

        
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
                 'vectorize':self.vectorize,
                 'unvectorize':self.unvectorize,
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
            ctx['vectorize']=self.vectorize
            ctx['unvectorize']=self.unvectorize

        


        problem_setup={}
        problem_setup['name']='Navier-Stokes 2D'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-00
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':5}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        u_ini_pde=np.zeros((n,mx,my),problem_setup['context']['data-type'])
        u_ini_pde[0,:,:]=1.
        u_ini_pde[0,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.5+np.reshape(np.random.uniform(low=-0.5*1e-02,high=0.5*1e-02,size=20*20),(20,20))
        u_ini_pde[1,:,:]=0.
        u_ini_pde[1,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.25+np.reshape(np.random.uniform(low=-0.25*1e-02,high=0.25*1e-02,size=20*20),(20,20))

        self.u_ini=self.vectorize(u_ini_pde,problem_ctx)
        problem_setup['context']['u_ini']=self.u_ini
        self.problem_setup=problem_setup

        
    def rhs_e(self,t,u_in,ctx=None):
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
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['mx']*pde_problem_ctx['my']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['mx'],pde_problem_ctx['my']))
        return u_pde
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)


class Diffusion:

    def __init__(self,problem_ctx=None):

        self.rhs_i=None

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
            


        problem_setup={}
        problem_setup['name']='AdvectionDiffusion1D'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-01
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':5}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        u_ini_pde=np.zeros((n,mx),problem_setup['context']['data-type'])


        #u_ini_pde[0,:]=1.
        #u_ini_pde[0,mx//3:2*mx//3]=0.5
        u_ini_pde[0,:]=1.0+0.5*np.sin(2*np.pi*x_coord)


        self._nF=int(ctx['mx']/2)
        self._nS=ctx['mx']-int(ctx['mx']/2)
        self._mx=ctx['mx']
        ctx['nF']=self._nF
        ctx['nS']=self._nS
        
        self.u_ini=self.vectorize(u_ini_pde,problem_ctx)
        problem_setup['context']['u_ini']=self.u_ini
        self.problem_setup=problem_setup

    def diffusion_tensor(self,t,ctx=None):
        y_k=np.zeros((ctx['mx'],),ctx['data-type'])
        
        x_max=problem_ctx['x_max']
        x_min=problem_ctx['x_min']
        x_coord=ctx['x_coord']
        y_k=(1.-np.sin(12.*t)**4)*np.sin(2*np.pi*(x_coord+t)/(x_max-x_min))**2
        return y_k
    
    def rhs_mr_implicit(self,t,u_in,ctx=None):
        
        mx=ctx['mx']
        dx=ctx['dx']
        kappa=ctx['kappa']
        diffusion_tensor=ctx['diffusion_tensor']
        u_out=np.zeros((m,),ctx['data-type'])
        A=np.zeros((m,m),ctx['data-type'])
        
        K=kappa*diffusion_tensor(t,ctx)
        
        
        
        A[0,0]=-2.
        A[0,1]=1.
        A[0,mx-1]=1.
        for i in range(1,mx-1):
            A[i,i-1]=1.
            A[i,i]=-2.
            A[i,i+1]=1.
        A[mx-1,0]=1.
        A[mx-1,mx-1]=-2.
        A[mx-1,mx-2]=1.
        
        for i in range(mx):
            A[i,:]=K[i]*A[i,:]
        
        A=A/(dx**2)
    
        u_out=np.matmul(A,u_in)
               
        j_out=A
        return u_out,j_out
    
    def rhs_e_fast(self,t,uS_in,uF_in,ctx=None):
        
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        c=ctx['Flux_c']
        nF=ctx['nF']#self._nF
        nS=ctx['nS']#self._nS

        uS_in=self.unvectorize_partition(uS_in,'S',ctx)
        uF_in=self.unvectorize_partition(uF_in,'F',ctx)

        
        F=np.zeros((n,nF+1),ctx['data-type'])
        Flux=np.zeros((n,nF),ctx['data-type'])
        
        y=uF_in

        F[:,0]=c*uS_in[:,-1]
        for id_mx in range(1,nF):
            F[:,id_mx]=c*y[:,id_mx-1]
        F[:,nF]=c*y[:,nF-1]
        
        for id_mx in range(0,nF):
            Flux[:,id_mx]=-(1./dx)*(F[:,id_mx+1]-F[:,id_mx])        
        
        u_out=Flux
        
        j_out=None

        u_out=self.vectorize_partition(u_out,'F',ctx)
        
        return u_out,j_out

        
    def rhs_e_slow(self,t,uS_in,uF_in,ctx=None):
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        c=ctx['Flux_c']
        nF=ctx['nF']#self._nF
        nS=ctx['nS']#self._nS


        uS_in=self.unvectorize_partition(uS_in,'S',ctx)
        uF_in=self.unvectorize_partition(uF_in,'F',ctx)
        
        F=np.zeros((n,nS+1),ctx['data-type'])
        Flux=np.zeros((n,nS),ctx['data-type'])
        
        y=uS_in

        F[:,0]=c*uF_in[:,nF-1]
        for id_mx in range(1,nS):
            F[:,id_mx]=c*y[:,id_mx-1]
        F[:,nS]=c*y[:,nS-1]
        
        for id_mx in range(0,nS):
            Flux[:,id_mx]=-(1./dx)*(F[:,id_mx+1]-F[:,id_mx])        
        
        u_out=Flux
        
        u_out=self.vectorize_partition(u_out,'S',ctx)
    
        j_out=None
        
        return u_out,j_out

    
    def rhs_e(self,t,u_in,ctx=None):
        vec=ctx['vectorize']
        unvec=ctx['unvectorize']
        
        n=ctx['n']
        mx=ctx['mx']
        dx=ctx['dx']
        c=ctx['Flux_c']
        
        F=np.zeros((n,mx+1),ctx['data-type'])
        Flux=np.zeros((n,mx),ctx['data-type'])
        
        y=unvec(u_in,ctx)

        F[:,0]=c*y[:,-1]
        for id_mx in range(1,mx):
            F[:,id_mx]=c*y[:,id_mx-1]
        F[:,mx]=c*y[:,mx-1]
        
        for id_mx in range(0,mx):
            Flux[:,id_mx]=-(1./dx)*(F[:,id_mx+1]-F[:,id_mx])        
        
        u_out=vec(Flux,ctx)
        
        j_out=None
        
        return u_out,j_out
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['mx']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['mx']))
        return u_pde

    def vectorize_partition(self,u_pde,partition_id,pde_problem_ctx):
        if(partition_id=='F'):
            u_ode=np.reshape(u_pde,(pde_problem_ctx['nF']*pde_problem_ctx['n']))
        if(partition_id=='S'):
            u_ode=np.reshape(u_pde,(pde_problem_ctx['nS']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize_partition(self,u_ode,partition_id,pde_problem_ctx):
        if(partition_id=='F'):
            u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['nF']))
        if(partition_id=='S'):
            u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['nS']))
        return u_pde

    def split_solution(self,u_ode):
        nF=self._nF
        nS=self._nS
        mx=self._mx
        
        uF_ode=np.reshape(u_ode[0:int(mx/2)],(nF,))
        uS_ode=np.reshape(u_ode[int(mx/2):mx],(nS,))
        return uF_ode,uS_ode

    def merge_solution(self,uF_ode,uS_ode):
        u_ode=np.hstack((uF_ode,uS_ode))
        return u_ode
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)

    
class GrayScott:

    def __init__(self,problem_ctx=None):

        self.rhs_i=None

        
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
                 'vectorize':self.vectorize,
                 'unvectorize':self.unvectorize,
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
            ctx['vectorize']=self.vectorize
            ctx['unvectorize']=self.unvectorize

        


        problem_setup={}
        problem_setup['name']='Gray-Scott'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-00
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':5}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        u_ini_pde=np.zeros((n,mx,my),problem_setup['context']['data-type'])
        u_ini_pde[0,:,:]=1.
        u_ini_pde[0,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.5+np.reshape(np.random.uniform(low=-0.5*1e-02,high=0.5*1e-02,size=20*20),(20,20))
        u_ini_pde[1,:,:]=0.
        u_ini_pde[1,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.25+np.reshape(np.random.uniform(low=-0.25*1e-02,high=0.25*1e-02,size=20*20),(20,20))

        self.u_ini=self.vectorize(u_ini_pde,problem_ctx)
        problem_setup['context']['u_ini']=self.u_ini
        self.problem_setup=problem_setup

        
    def rhs_e(self,t,u_in,ctx=None):
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
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['mx']*pde_problem_ctx['my']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['mx'],pde_problem_ctx['my']))
        return u_pde
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)

class NavierStokes2D:

    def __init__(self,problem_ctx=None):

        self.rhs_i=None

        
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
                 'vectorize':self.vectorize,
                 'unvectorize':self.unvectorize,
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
            ctx['vectorize']=self.vectorize
            ctx['unvectorize']=self.unvectorize

        


        problem_setup={}
        problem_setup['name']='Navier-Stokes 2D'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-00
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':5}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        u_ini_pde=np.zeros((n,mx,my),problem_setup['context']['data-type'])
        u_ini_pde[0,:,:]=1.
        u_ini_pde[0,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.5+np.reshape(np.random.uniform(low=-0.5*1e-02,high=0.5*1e-02,size=20*20),(20,20))
        u_ini_pde[1,:,:]=0.
        u_ini_pde[1,mx//2-10:mx//2+10,my//2-10:my//2+10]=0.25+np.reshape(np.random.uniform(low=-0.25*1e-02,high=0.25*1e-02,size=20*20),(20,20))

        self.u_ini=self.vectorize(u_ini_pde,problem_ctx)
        problem_setup['context']['u_ini']=self.u_ini
        self.problem_setup=problem_setup

        
    def rhs_e(self,t,u_in,ctx=None):
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
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['mx']*pde_problem_ctx['my']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['mx'],pde_problem_ctx['my']))
        return u_pde
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)


class Diffusion:

    def __init__(self,problem_ctx=None):

        self.rhs_i=None

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

        self.u_ini=u_ini
        self.problem_setup=problem_setup

        
    def diffusion_tensor(self,t,ctx=None):
        y_k=np.zeros((ctx['m'],),ctx['data-type'])
        
        x_max=problem_ctx['x_max']
        x_min=problem_ctx['x_min']
        x_coord=ctx['x_coord']
        y_k=(1.-np.sin(12.*t)**4)*np.sin(2*np.pi*(x_coord+t)/(x_max-x_min))**2
        return y_k
    
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['m']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['m']))
        return u_pde
 
    
    def rhs_e(self,t,u_in,ctx=None):
        
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
 

class DiffusionComplex:

    def __init__(self,problem_ctx=None):

        self.rhs_i=None

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
        self.u_ini=u_ini
        self.problem_setup=problem_setup

        
    def diffusion_tensor(self,t,ctx=None):
        y_k=np.zeros((ctx['m'],),ctx['data-type'])
        
        x_max=problem_ctx['x_max']
        x_min=problem_ctx['x_min']
        x_coord=ctx['x_coord']
        y_k=(1.-np.sin(12.*t)**4)*np.sin(2*np.pi*(x_coord+t)/(x_max-x_min))**2
        return y_k
    
    
    def vectorize(self,u_pde,pde_problem_ctx):
        u_ode=np.reshape(u_pde,(pde_problem_ctx['m']*pde_problem_ctx['n']))
        return u_ode
    
    def unvectorize(self,u_ode,pde_problem_ctx):
        u_pde=np.reshape(u_ode,(pde_problem_ctx['n'],pde_problem_ctx['m']))
        return u_pde
 
    
    def rhs_e(self,t,u_in,ctx=None):
        
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
 
    
