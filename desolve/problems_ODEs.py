import numpy as np
import re 

def ProblemsODE(name,problem_ctx=None):
    rhs_e=None
    rhs_i=None
    u_ini=None
    
    if re.match(r'\ACos( )*(_|-)*Sine( )*(_|-)*1D\Z',name,re.IGNORECASE):
        problem = Cos_Sine_OneD(problem_ctx)      
    elif re.match(r'\AHarmonic( )*(_|-)*Oscillator\Z',name,re.IGNORECASE):
        problem = HarmonicOscillator(problem_ctx)
    elif re.match(r'\APrince( )*(_|-)*1978( )*(_|-)*A\Z',name,re.IGNORECASE):
        problem = Prince1978A(problem_ctx)
    elif re.match(r'\AProthero( )*(_|-)*Robinson\Z',name,re.IGNORECASE):
        problem = ProtheroRobinson(problem_ctx)
    elif re.match(r'\AHull1972( )*(_|-)*B4\Z',name,re.IGNORECASE):
        problem = Hull1972_B4(problem_ctx)
    elif re.match(r'\Avdp\Z',name,re.IGNORECASE):
        problem = vdP(problem_ctx)
    elif re.match(r'\AKulikov-III\Z',name,re.IGNORECASE):
        problem = Kulikov_III(problem_ctx)    
    elif re.match(r'\AKulikov-IV\Z',name,re.IGNORECASE):
        problem = Kulikov_IV(problem_ctx)
    elif re.match(r'\ABarnes\Z',name,re.IGNORECASE):
        problem = Barnes(problem_ctx)
    elif re.match(r'\ALorenz\Z',name,re.IGNORECASE):
        problem = Lorenz(problem_ctx)
    elif re.match(r'\ALorenz96\Z',name,re.IGNORECASE):
        problem = Lorenz96(problem_ctx)
    elif re.match(r'\ABEKiller\Z',name,re.IGNORECASE):
        problem = BEKiller(problem_ctx)   
    else:
        raise NameError('Problem {:} has not been found.'.format(name))
    
    
    rhs_e=problem.rhs_e
    rhs_i=problem.rhs_i
    u_ini=problem.initial_solution()
    problem_setup = problem.get_problem_setup()
    
    
    return rhs_e, rhs_i, u_ini, problem_setup, problem


class Cos_Sine_OneD():
    def __init__(self,problem_ctx=None):
        self.u_ini=np.zeros((1))+2.
        
        self.exact_solution=None
        
        problem_setup={}
        ctx={}
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-02
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':10.}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}
        self.problem_setup=problem_setup
        pass
    
    def rhs_e(self,t,u_in,ctx=None):
        u_out=np.zeros((1,))
        u_out=-(4./3.)*np.pi*np.sin((2.*np.pi*t)/3.)
        j_out=np.zeros((1,1))
        return u_out,j_out

    def rhs_i(self,t,u_in,ctx=None):
        u_out=np.zeros((1,))
        u_out=5*np.pi*np.cos(5*np.pi*t)
        j_out=np.zeros((1,1))
        return u_out,j_out
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)


class HarmonicOscillator():
    """Separable Hamiltonian test problem with both full and split RHS forms.

    The Hamiltonian is

        H(q, p) = 0.5 * p^2 + 0.5 * omega^2 * q^2

    so the canonical equations are

        q' = p,
        p' = -omega^2 q.

    The class deliberately exposes two compatible interfaces:

    - ``rhs_e(t, u, ctx)`` returns the full two-component vector field, which
      lets standard explicit methods such as ``RK4`` solve the same problem.
    - ``get_symplectic_rhs()`` returns the drift/kick split expected by the new
      symplectic table-driven methods.
    """

    def __init__(self, problem_ctx=None):
        self.rhs_i = None

        if(problem_ctx is None):
            ctx = {'omega': 1.0}
        else:
            ctx = {'omega': problem_ctx['omega']}

        self.u_ini = np.zeros((2,), dtype=np.float64)
        self.u_ini[0] = 1.0
        self.u_ini[1] = 0.0

        problem_setup = {}
        problem_setup['name'] = 'HarmonicOscillator'
        problem_setup['context'] = ctx
        problem_setup['context']['data-type'] = np.float64
        problem_setup['context']['SplitSolution'] = self.split_solution
        problem_setup['context']['MergeSolution'] = self.merge_solution
        problem_setup['DT'] = 5.0e-02
        problem_setup['DT_REFERENCE'] = 1.0e-04
        problem_setup['T_DURATION'] = {'start': 0., 'end': 10.}
        problem_setup['DT_INTERVAL'] = {'start': 1.0e-02, 'end': 2.0e-01}

        self.problem_setup = problem_setup

    def split_solution(self, u_in):
        return u_in[0:1], u_in[1:2]

    def merge_solution(self, q_in, p_in):
        return np.concatenate((q_in, p_in))

    def rhs_e(self, t, u_in, ctx=None):
        omega = ctx['omega']
        q_in, p_in = self.split_solution(u_in)

        u_out = np.zeros((2,), dtype=ctx['data-type'])
        u_out[0] = p_in[0]
        u_out[1] = -(omega**2) * q_in[0]

        j_out = np.zeros((2, 2), dtype=ctx['data-type'])
        j_out[0, 1] = 1.0
        j_out[1, 0] = -(omega**2)
        return u_out, j_out

    def rhs_symplectic_drift(self, t, q_in, p_in, ctx=None):
        dqdt = np.zeros(q_in.shape, dtype=ctx['data-type'])
        dqdt[:] = p_in
        return dqdt, None

    def rhs_symplectic_kick(self, t, q_in, p_in, ctx=None):
        omega = ctx['omega']
        dpdt = np.zeros(p_in.shape, dtype=ctx['data-type'])
        dpdt[:] = -(omega**2) * q_in
        return dpdt, None

    def get_symplectic_rhs(self):
        return {
            'symplectic_drift': self.rhs_symplectic_drift,
            'symplectic_kick': self.rhs_symplectic_kick,
        }

    def initial_solution(self):
        return self.u_ini

    def get_problem_setup(self):
        return self.problem_setup

    def exact_solution(self, t, ctx=None):
        omega = ctx['omega']
        q0 = self.u_ini[0]
        p0 = self.u_ini[1]

        if(isinstance(t, np.ndarray)):
            u = np.zeros((2, len(t)), dtype=ctx['data-type'])
            u[0, :] = q0 * np.cos(omega * t) + (p0 / omega) * np.sin(omega * t)
            u[1, :] = p0 * np.cos(omega * t) - omega * q0 * np.sin(omega * t)
            return u
        else:
            u = np.zeros((2,), dtype=ctx['data-type'])
            u[0] = q0 * np.cos(omega * t) + (p0 / omega) * np.sin(omega * t)
            u[1] = p0 * np.cos(omega * t) - omega * q0 * np.sin(omega * t)
            return u

class Prince1978A:
    def __init__(self,problem_ctx=None):
        self.rhs_i=None
        #self.exact_solution=None
        self.u_ini=np.zeros((1))+1.
        problem_setup={}    
        ctx={}
        problem_setup['name']='Prince1978A'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-02
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':10.}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}
        self.problem_setup=problem_setup
        
    def rhs_e(self,t,u_in,ctx=None):
        u_out=np.zeros((1,))
        u_out=-u_in[0]
        
        
        j_out=np.zeros((1,1))
        j_out[0,0]=-1.
        return u_out,j_out
     
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)
    
    def exact_solution(self,t,ctx=None):
        if(isinstance(t,np.ndarray)):
            u=np.zeros((1,len(t)))
            for i in range(len(t)):
                u[0,i]=np.exp(-t[i])
            return u
        else:
            return(np.exp(-t))
        
class BEKiller:
    def __init__(self,problem_ctx=None):
        
        self.rhs_i=None
        #self.exact_solution=None
        
        if(problem_ctx is None):
            ctx={'sigma': 10.,'rho':28.,'beta':8./3.}
        else:
            ctx={'sigma': problem_ctx['sigma'],'rho': problem_ctx['rho'],'beta': problem_ctx['beta']}

        self.u_ini=np.zeros((3))
        self.u_ini[:]=np.asarray([1.,0.,2.])
            
        problem_setup={}
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['name']='BEKiller'
        problem_setup['DT']=0.1
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':50.}
        problem_setup['DT_INTERVAL']={'start':1e-02,'end':1e-01}

        self.problem_setup=problem_setup
        
    def rhs_e(self,t,u_in,ctx=None):
        A=np.zeros((3,3),ctx['data-type'])
        A[0,0]=42.2
        A[0,1]=50.1
        A[0,2]=-42.1
        A[1,0]=-66.1
        A[1,1]=-58
        A[1,2]=58.1
        A[2,0]=26.1
        A[2,1]=42.1
        A[2,2]=-34
        u_out=np.matmul(A,u_in)
        j_out=A
        
        
        return u_out,j_out
 
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)
    
    def exact_solution(self,t,ctx=None):
        if(isinstance(t,np.ndarray)):
            u=np.zeros((3,len(t)))
            for i in range(len(t)):
                u[0,i]=np.exp(0.1*t[i])*np.sin(8*t[i])+np.exp(-50*t[i])
                u[1,i]=np.exp(0.1*t[i])*np.cos(8*t[i])-np.exp(-50*t[i])
                u[2,i]=np.exp(0.1*t[i])*(np.cos(8*t[i])+np.sin(8*t[i]))+np.exp(-50*t[i])
            return u
        else:
            u=np.zeros((3,))
            u[0]=np.exp(0.1*t)*np.sin(8*t)+np.exp(-50*t)
            u[1]=np.exp(0.1*t)*np.cos(8*t)-np.exp(-50*t)
            u[2]=np.exp(0.1*t)*(np.cos(8*t)+np.sin(8*t))+np.exp(-50*t)
            return u
        
class ProtheroRobinson:
    def __init__(self,problem_ctx=None):
    
        self.rhs_i=None

        self.u_ini=np.zeros((1))+np.sin(0.)
        problem_setup={}    
        if(problem_ctx is None):
            ctx={'Epsilon': -2.0e+6}
        else:
            ctx={'Epsilon': problem_ctx['Epsilon']}
        problem_setup['name']='Prothero-Robinson'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-02
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':100.}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

    
        self.problem_setup=problem_setup
        
    def rhs_e(self,t,u_in,ctx=None):
        Epsilon=ctx['Epsilon']
        u_out=np.zeros((1,))
        u_out=Epsilon*(u_in[0]-np.sin(t))+np.cos(t)
        
        
        j_out=np.zeros((1,1))
        j_out[0,0]=Epsilon
        return u_out,j_out

    def exact_solution(self,t,ctx=None):
        if(isinstance(t,np.ndarray)):
            u=np.zeros((1,len(t)))
            for i in range(len(t)):
                u[0,i]=np.sin(t[i])
            return u
        else:
           return(np.sin(t))
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)

class Hull1972_B4:
    def __init__(self,problem_ctx=None):
        self.exact_solution=None

        ctx={}

        problem_setup={}    
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-02
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':1.}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-02}

        self.u_ini=np.zeros((3),dtype=problem_setup['context']['data-type'])
        self.u_ini[0]=3.

        
        self.problem_setup=problem_setup
        
        
    def rhs_e(self,t,u_in,ctx=None):
        sy1y2=np.sqrt(u_in[0]**2+u_in[1]**2)
        u_out=np.zeros((3,),dtype=ctx['data-type'])
        u_out[0]=-u_in[1]-(u_in[0]*u_in[2])/sy1y2
        u_out[1]=u_in[0]-(u_in[1]*u_in[2])/sy1y2
        j_out=np.zeros((3,3),dtype=ctx['data-type'])
        
        j_out[0,0]=-((u_in[1]**2 * u_in[2])/(u_in[0]**2 + u_in[1]**2)**(3./2.))
        j_out[0,1]=-1. + (u_in[0]* u_in[1]* u_in[2])/(u_in[0]**2 + u_in[1]**2)**(3./2.)
        j_out[0,2]=-(u_in[0]/np.sqrt(u_in[0]**2 + u_in[1]**2))
        
        j_out[1,0]=1. - (u_in[1]**2 * u_in[2])/(u_in[0]**2 + u_in[1]**2)**(3./2.)
        j_out[1,1]=(u_in[0]* u_in[1]* u_in[2])/(u_in[0]**2 + u_in[1]**2)**(3./2.)
        j_out[1,2]=-( u_in[0]/np.sqrt(u_in[0]**2 + u_in[1]**2))
        
        return u_out,j_out

    def rhs_i(self,t,u_in,ctx=None):
        sy1y2=np.sqrt(u_in[0]**2+u_in[1]**2)
        u_out=np.zeros((3,),dtype=ctx['data-type'])
        u_out[2]=u_in[0]
        j_out=np.zeros((3,3),dtype=ctx['data-type'])
        j_out[2,0]= u_in[1]**2/(u_in[0]**2 + u_in[1]**2)**(3./2.)
        j_out[2,1]= -(( u_in[0] *u_in[1])/(u_in[0]**2 + u_in[1]**2)**(3./2.))
        return u_out,j_out

    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)
    
class vdP:
    def __init__(self,problem_ctx=None):
           
        self.u_ini=np.zeros((2))
        self.u_ini[0]=2.


        self.exact_solution=None
        if(problem_ctx is None):
            ctx={'Epsilon': 1.0e-2}
        else:
            ctx={'Epsilon': problem_ctx['Epsilon']}

        problem_setup={}
        problem_setup['name']='van der Pol'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-03
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':10.}
        problem_setup['DT_INTERVAL']={'start':1e-03,'end':1e-01}

        self.problem_setup=problem_setup
        
    def rhs_e(self,t,u_in,ctx=None):
        u_out=np.zeros((2,))
        u_out[0]=u_in[1]
        
        j_out=np.zeros((2,2))
        j_out[0,1]=1.
        
        return u_out,j_out

    def rhs_i(self,t,u_in,ctx=None):
        u_out=np.zeros((2,))
        Epsilon=ctx['Epsilon']
        u_out[1]=((1.-u_in[0]**2)*u_in[1]-u_in[0])/Epsilon
        
        j_out=np.zeros((2,2))
        j_out[1,0]=(-2*u_in[0]*u_in[1]-1.)/Epsilon
        j_out[1,1]=(1.-u_in[0]**2)/Epsilon
        return u_out,j_out
 
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)

class Lorenz96:
    def __init__(self,problem_ctx=None):
        
        self.rhs_i=None
        self.exact_solution=None
        
        if(problem_ctx is None):
            ctx={'m': 40,'F':8}
        else:
            ctx={'m': problem_ctx['m'],'F': problem_ctx['F']}

        m=ctx['m']

        self.u_ini=np.zeros((m))
        self.u_ini[:]=0.
            
        problem_setup={}
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['name']='Lorenz 96'
        problem_setup['DT']=0.05
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':10.}
        problem_setup['DT_INTERVAL']={'start':1e-02,'end':1e-01}

        self.problem_setup=problem_setup
        
    def rhs_e(self,t,u_in,ctx=None):
        m=ctx['m']
        F=ctx['F']
        
        u_out=np.zeros((m,))
        u_out[0] = (u_in[1]-u_in[m-2])*u_in[m-1] - u_in[0]
        u_out[1] = (u_in[2]-u_in[m-1])*u_in[0]   - u_in[1]
        for i in range(2,m-2):
            u_out[i]=(u_in[i+1]-u_in[i-2])*u_in[i-1] - u_in[i]
        u_out[m-2]=(u_in[m-1]-u_in[m-4])*u_in[m-3] - u_in[m-2]
        u_out[m-1]=(u_in[0]  -u_in[m-3])*u_in[m-2] - u_in[m-1]
        u_out=u_out+F
        
        j_out=None#np.zeros((m,m))
        
        return u_out,j_out
 
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)    
    
class Lorenz:
    def __init__(self,problem_ctx=None):
        
        self.rhs_i=None
        self.exact_solution=None
        
        if(problem_ctx is None):
            ctx={'sigma': 10.,'rho':28.,'beta':8./3.}
        else:
            ctx={'sigma': problem_ctx['sigma'],'rho': problem_ctx['rho'],'beta': problem_ctx['beta']}

        self.u_ini=np.zeros((3))
        self.u_ini[:]=1.
            
        problem_setup={}
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['name']='Lorenz'
        problem_setup['DT']=0.1
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':50.}
        problem_setup['DT_INTERVAL']={'start':1e-02,'end':1e-01}

        self.problem_setup=problem_setup
        
    def rhs_e(self,t,u_in,ctx=None):
        sigma=ctx['sigma']
        rho=ctx['rho']
        beta=ctx['beta']
        
        u_out=np.zeros((3,))
        u_out[0] = sigma*(u_in[1]-u_in[0]) # sigma*(y-x);
        u_out[1] = rho * u_in[0] - u_in[1] - u_in[0]*u_in[2] # rho*x - y - x*z;
        u_out[2] = u_in[0]*u_in[1] - beta * u_in[2] # x*y - beta*z;
        
        j_out=np.zeros((3,3))
        j_out[0,0]=-sigma
        j_out[0,1]=sigma

        j_out[1,0]=rho-u_in[2]
        j_out[1,1]=-1
        j_out[1,2]=-u_in[0]

        j_out[2,0]=u_in[1]
        j_out[2,1]=u_in[0]
        j_out[2,2]=-beta
        
        return u_out,j_out
 
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)    
class Kulikov_III:
    def __init__(self,problem_ctx=None):
    
        self.rhs_i=None

        self.u_ini=np.zeros((1))+1.0
        problem_setup={}    
        if(problem_ctx is None):
            ctx={'Lambda': 1.0e+6}
        else:
            ctx={'Lambda': problem_ctx['Lambda']}
        problem_setup['name']='Kulikov-III'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-02
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':10}
        problem_setup['DT_INTERVAL']={'start':0.001,'end':0.01}

    
        self.problem_setup=problem_setup
        
    def rhs_e(self,t,u_in,ctx=None):
        Lambda=ctx['Lambda']
        u_out=np.zeros((1,))
        u_out[0]=Lambda*(np.sin(4.*t)-u_in[0])+4.*np.cos(4.*t)
        j_out=np.zeros((1,1))
        j_out[0,0]=-Lambda
        return u_out,j_out

    def exact_solution(self,t,ctx=None):
        Lambda=ctx['Lambda']
        if(isinstance(t,np.ndarray)):
            u=np.zeros((1,len(t)))
            for i in range(len(t)):
                u[0,i]=np.exp(-Lambda*t[i])+np.sin(4.*t[i])
            return u
        else:
            return(np.exp(-Lambda*t)+np.sin(4.*t))
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)    
        

class Kulikov_IV:       
    def __init__(self,problem_ctx=None):
    
        self.rhs_i=None

        self.u_ini=np.zeros((4))+1.0
        self.u_ini[2]=0.0
        
        problem_setup={}    
        if(problem_ctx is None):
            ctx={'Lambda': 1.0e+6}
        else:
            ctx={'Lambda': problem_ctx['Lambda']}
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-01
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':1.5}
        problem_setup['DT_INTERVAL']={'start':0.001,'end':0.01}

    
        self.problem_setup=problem_setup
        
    def rhs_e(self,t,u_in,ctx=None):
        Lambda=ctx['Lambda']
        u_out=np.zeros((4,))
        
        x1=u_in[0]
        x2=u_in[1]
        x3=u_in[2]
        x4=u_in[3]
        
        u_out[0]=Lambda*((x4**4)/x2 - x1**2 - x3**2) - x3
        u_out[1]=Lambda*((x4**4) - x2) - 2*x2
        u_out[2]=x1
        u_out[3]=-(x2**(1/4.))/2
        
        j_out=np.zeros((4,4))
        j_out[0,0]=-2.*Lambda*x1;
        j_out[0,1]=-Lambda*(x4**4)/(x2**2);
        j_out[0,2]=-1.-2.*Lambda*x3;
        j_out[0,3]=4.*Lambda*(x4**3)/x2;
        j_out[1,1]=-Lambda-2.;
        j_out[1,3]=4.*Lambda*x4**3;
        j_out[2,0]=1.;
        j_out[3,1]=-1./(8.*(x2**(3./4.)));
        
        return u_out,j_out

    def exact_solution(self,t,ctx=None):
        if(isinstance(t,np.ndarray)):
            u=np.zeros((4,len(t)))
            for i in range(len(t)):
                u[0,i]=np.cos(t[i])
                u[1,i]=np.exp(-2*t[i])
                u[2,i]=np.sin(t[i])
                u[3,i]=np.exp(-t[i]/2.)
            return u
        else:
            u=np.zeros((4,))
            u[0]=np.cos(t)
            u[1]=np.exp(-2*t)
            u[2]=np.sin(t)
            u[3]=np.exp(-t/2.)  
            
            return(u)
    
    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)    
        

class Barnes:
    def __init__(self,problem_ctx=None):
        
        self.rhs_i=None
        self.u_ini=np.zeros((2))
        self.u_ini[0]=1.
        self.u_ini[1]=.3

        self.exact_solution=None
        if(problem_ctx is None):
            ctx={'p1': 1.,'p2':1.,'p3':1.}
        else:
            ctx={'p1': problem_ctx['p1'],'p2': problem_ctx['p2'],'p3': problem_ctx['p3']}

        problem_setup={}
        problem_setup['name']='Barnes'
        problem_setup['context']=ctx
        problem_setup['context']['data-type']=np.float64
        problem_setup['DT']=1.0e-02
        problem_setup['DT_REFERENCE']=1.0e-04
        problem_setup['T_DURATION']={'start':0.,'end':20.}
        problem_setup['DT_INTERVAL']={'start':1e-02,'end':1e-01}

        self.problem_setup=problem_setup
        
    def rhs_e(self,t,u_in,ctx=None):
        
        p1=ctx['p1']
        p2=ctx['p2']
        p3=ctx['p3']

        u_out=np.zeros((2,))
        u_out[0]=p1*u_in[0]-p2*u_in[0]*u_in[1]
        u_out[1]=p2*u_in[0]*u_in[1]-p3*u_in[1]
        
        j_out=np.zeros((2,2))
        
        j_out[0,0]=p1-p2*u_in[1]
        j_out[0,1]=-p2*u_in[0]
        j_out[1,0]=p2*u_in[1]
        j_out[1,1]=p2*u_in[0]-p3
        
        return u_out,j_out

    def initial_solution(self):
        return (self.u_ini)
    
    def get_problem_setup(self):
        return (self.problem_setup)
