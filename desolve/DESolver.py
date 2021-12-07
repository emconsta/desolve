import numpy as np
from scipy.optimize import fsolve
import scipy

from methods_rk import Default_RK_Methods
from methods_etrs import Default_ETRS_Methods
from methods_ark import Default_ARK_Methods
from methods_esdirk import Default_ESDIRK_Methods
from methods_glee import Default_GLEE_Methods
from methods_glee_eimex import Default_GLEE_EIMEX_Methods
from methods_ide import Default_IDE_Methods
from methods_mrk import Default_MRK_Methods
from methods_imex_mrk import Default_IMEX_MRK_Methods

class DESolver:
    """
    Time-integration schemes. This includes:
        - FE: Forward uler.
        - RK4: Runge-Kutta
        - RK2a: Runge-Kutta
    """

    def __init__(self):
        self._keep_history = True
        self._dt = 1
        self._max_n_steps = 100000
        self._t_start = 0.
        self._t_end = 1

        self._rhs = None
        self._rhs_fast = None
        self._rhs_slow = None
        self._rhs_mr_implicit = None
        self._rhs_e = None
        self._rhs_i = None
        self._rhs_gint = None

        self._r = None
        self._n = None

        self._Ystage = None
        self._function_context = None

        self._info = 0
        self._method_set = []
        self._methods = {}

        self.reset()

    def reset(self,soft=False):
        self._solved = False
        self._step = 0
        self._In = None
        self._t = None
        self._global_error = None
        self._has_global_error = False
        self._current_method = None
        self._setup = False
        self._solution = None
        self._solution_aux = None
        self._t_trajectory = None
        self._u_trajectory = None
        self._epsilon_trajectory = None
        self._epsilon_local_trajectory = None
        self._u_high_trajectory = None
        pass
    
    def GenerateOrderTestGrid(self, NRuns, t_min, t_max, dt_min, dt_max):

        T_tot = float(t_max-t_min)

        NSteps_min = np.int(T_tot/dt_max)
        NSteps_max = np.int(T_tot/dt_min)

        NSteps_tab = np.asarray(np.logspace(
            np.log10(NSteps_min), np.log10(NSteps_max), NRuns)).astype('int')

        dt_tab = T_tot/NSteps_tab

        return dt_tab, NSteps_tab

    def _RegisterDefaultMethods(self):

        AllMethods_RK = Default_RK_Methods()
        for i in range(len(AllMethods_RK)):
            A_method = AllMethods_RK[i]
            if(self._info >= 1):
                print('registering {:} of type {:}'.format(
                    A_method['name'], A_method['type']))
            self._methods[A_method['name']] = A_method
            self._method_set.append(A_method['name'])

        AllMethods_ETRS = Default_ETRS_Methods()
        for i in range(len(AllMethods_ETRS)):
            A_method = AllMethods_ETRS[i]
            if(self._info >= 1):
                print('registering {:} of type {:}'.format(
                    A_method['name'], A_method['type']))
            self._methods[A_method['name']] = A_method
            self._method_set.append(A_method['name'])

        AllMethods_ARK = Default_ARK_Methods()
        for i in range(len(AllMethods_ARK)):
            A_method = AllMethods_ARK[i]
            if(self._info >= 1):
                print('registering {:} of type {:}'.format(
                    A_method['name'], A_method['type']))
            self._methods[A_method['name']] = A_method
            self._method_set.append(A_method['name'])

        AllMethods_ESDIRK = Default_ESDIRK_Methods()
        for i in range(len(AllMethods_ESDIRK)):
            A_method = AllMethods_ESDIRK[i]
            if(self._info >= 1):
                print('registering {:} of type {:}'.format(
                    A_method['name'], A_method['type']))
            self._methods[A_method['name']] = A_method
            self._method_set.append(A_method['name'])

        AllMethods_GLEE = Default_GLEE_Methods()
        for i in range(len(AllMethods_GLEE)):
            A_method = AllMethods_GLEE[i]
            if(self._info >= 1):
                print('registering {:} of type {:}'.format(
                    A_method['name'], A_method['type']))
            self._methods[A_method['name']] = A_method
            self._method_set.append(A_method['name'])

        AllMethods_IDE = Default_IDE_Methods()
        for i in range(len(AllMethods_IDE)):
            A_method = AllMethods_IDE[i]
            if(self._info >= 1):
                print('registering {:} of type {:}'.format(
                    A_method['name'], A_method['type']))
            self._methods[A_method['name']] = A_method
            self._method_set.append(A_method['name'])

        AllMethods_GLEE_EIMEX = Default_GLEE_EIMEX_Methods()
        for i in range(len(AllMethods_GLEE_EIMEX)):
            A_method = AllMethods_GLEE_EIMEX[i]
            if(self._info >= 1):
                print('registering {:} of type {:}'.format(
                    A_method['name'], A_method['type']))
            self._methods[A_method['name']] = A_method
            self._method_set.append(A_method['name'])


        AllMethods_MRK = Default_MRK_Methods()
        for i in range(len(AllMethods_MRK)):
            A_method = AllMethods_MRK[i]
            if(self._info >= 1):
                print('registering {:} of type {:}'.format(
                    A_method['name'], A_method['type']))
            self._methods[A_method['name']] = A_method
            self._method_set.append(A_method['name'])

            

        AllMethods_IMEX_MRK = Default_IMEX_MRK_Methods()
        for i in range(len(AllMethods_IMEX_MRK)):
            A_method = AllMethods_IMEX_MRK[i]
            if(self._info >= 1):
                print('registering {:} of type {:}'.format(
                    A_method['name'], A_method['type']))
            self._methods[A_method['name']] = A_method
            self._method_set.append(A_method['name'])
            
    def _compound_imex_rhs(self, tt, uu, ctx=None):

        u_ex, J_ex = self._rhs_e(tt, uu, self._function_context)
        u_im, J_im = self._rhs_i(tt, uu, self._function_context)
        u_out = u_ex+u_im
        if(J_ex is None or J_im is None):
            return u_out, None
        else:
            J_out = J_ex+J_im

        return u_out, J_out

    def setup(self,keep_history=True):
        self._RegisterDefaultMethods()
        self._step = 0
        self._keep_history=keep_history
        pass

    def GLEE_Starting_Procedure(self):
        METHOD = self._methods[str(self._current_method)]
        self._r = METHOD['r']
        self._solution_aux = np.zeros(
            (self._n, self._r-1), dtype=self._function_context['data-type'])
        for i in range(self._r-1):
            self._solution_aux[:, i] = self._solution.copy()
        pass

    def solve(self, u_ini=None):
        if(u_ini != None):
            self.set_initial_solution(u_ini)
        if(self._setup):
            self.setup()

        METHOD = self._methods[str(self._current_method)]

        if(METHOD['type'] == 'GLEE' or METHOD['type'] == 'GLEE-EIMEX'):
            self.GLEE_Starting_Procedure()

        while True:
            self.step()

            if(self._step == self._max_n_steps):
                if(self._info >= 1):
                    print('Exit on exceeding the number of steps (step {:g} = max number of steps = {:g})'.format(
                        self._step, self._max_n_steps))
                break
            if(self._t >= self._t_end):
                if(self._info >= 1):
                    print('Exit on exceeding the duration (current time {:g} >= tend = {:g})'.format(
                        self._t, self._t_end))
                break

        self._solved = True
        return 0

    def step(self):
        """
        Advance solution one step
        """

        METHOD = self._methods[str(self._current_method)]

        n = self._n

        u_in = self._solution

        dt = self._dt
        t = self._t

        if(self._info >= 1):
            print(' ---------- step {:03d}: t={:05g} {:}:{:}'.format(
                self._step, t, METHOD['type'], self._current_method))

        if(METHOD['type'] == 'GLEE'):  # or METHOD['type']=='GLEE-EIMEX'):

            if(self._solution_aux is None):
                raise NameError('Auxiliary solution is not initialized.')
            else:
                q_in = np.zeros((self._n, self._r),
                                dtype=self._function_context['data-type'])
                q_in[:, 0] = self._solution[0:n]
                for i in range(self._r-1):
                    q_in[:, i+1] = self._solution_aux[:, i]

            #print('Solutions: {:} {:}'.format(q_in[0,0],q_in[0,1]))

            A = METHOD['A']
            U = METHOD['U']
            B = METHOD['B']
            V = METHOD['V']
            gamma = METHOD['gamma']
            s = METHOD['s']
            r = METHOD['r']
            p = METHOD['p']
            q = METHOD['q']
            c = METHOD['c']

            if(self._info >= 2):
                print(' ---------- step {:d}: t={:g}, u_in = u[{:d}], u_out=[{:d}],'.format(
                    self._step, t, self._step, self._step+1))
            Ystage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            Kstage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])

            if(self._info >= 2):
                print('Step {:d}: Y1=u_in'.format(self._step, t))

            for istage in range(s):
                Ystage[0:n, istage] = 0.
                for jstage in range(istage):  # should be istage
                    Ystage[0:n, istage] = Ystage[0:n, istage] + \
                        dt*A[istage, jstage]*Kstage[0:n, jstage]
                for jstage in range(r):
                    Ystage[0:n, istage] = Ystage[0:n, istage] + \
                        U[istage, jstage]*q_in[0:n, jstage]

                t_stage = t+c[istage]*dt

                if(A[istage, istage] != 0.):
                    def IFunction(X, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        f, _ = self._rhs(t_stage_in, X, self._function_context)
                        F = X-alpha*f-Res_in
                        return F

                    def IFunction_Jac(X, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        f, Jac = self._rhs(
                            t_stage_in, X, self._function_context)
                        J = np.eye(Jac.shape[0])-alpha*Jac
                        return J

                    params = (dt*A[istage, istage],
                              t_stage, Ystage[0:n, istage])
                    Ystage[0:n, istage] = fsolve(
                        func=IFunction, fprime=IFunction_Jac, x0=q_in[0:n, 0], args=params)

                Kstage[0:n, istage], _ = self._rhs(
                    t_stage, Ystage[0:n, istage], self._function_context)

            q_out = np.zeros(
                q_in.shape, dtype=self._function_context['data-type'])
            for rstage in range(r):
                for jstage in range(s):
                    q_out[0:n, rstage] = q_out[0:n, rstage] + \
                        dt*B[rstage, jstage]*Kstage[0:n, jstage]
                for jstage in range(r):
                    q_out[0:n, rstage] = q_out[0:n, rstage] + \
                        V[rstage, jstage]*q_in[0:n, jstage]

            u_out = np.zeros(self._solution.shape,
                             dtype=self._function_context['data-type'])
            u_out[0:n] = q_out[0:n, 0]
            for i in range(self._r-1):
                self._solution_aux[0:n, i] = q_out[0:n, i+1]

            self._u_high = (1./(1.-gamma)) * \
                q_out[0:n, 1]-(gamma/(1-gamma))*q_out[0:n, 0]
            self._epsilon = (1./(1.-gamma))*(q_out[0:n, 1]-q_out[0:n, 0])
            self._epsilon_local = (
                1./(1.-gamma))*(q_out[0:n, 1]-q_out[0:n, 0])-(1./(1.-gamma))*(q_in[0:n, 1]-q_in[0:n, 0])
            self._global_error = self._epsilon

        elif(METHOD['type'] == 'GLEE-EIMEX'):

            if(self._solution_aux is None):
                raise NameError('Auxiliary solution is not initialized.')
            else:
                q_in = np.zeros((self._n, self._r),
                                dtype=self._function_context['data-type'])
                q_in[:, 0] = self._solution[0:n]
                for i in range(self._r-1):
                    q_in[:, i+1] = self._solution_aux[:, i]

            if(self._rhs_e is None or self._rhs_i is None):
                raise NameError(
                    'rhs_e, rhs_i need to be defined in order to use GLEE-EIMEX integrators.')
            #print('Solutions: {:} {:}'.format(q_in[0,0],q_in[0,1]))

            A = METHOD['A']
            At = METHOD['AT']
            bt = METHOD['bt']
            U = METHOD['U']
            B = METHOD['B']
            V = METHOD['V']
            gamma = METHOD['gamma']
            s = METHOD['s']
            r = METHOD['r']
            p = METHOD['p']
            q = METHOD['q']
            c = METHOD['c']

            if(self._info >= 2):
                print(' ---------- step {:d}: t={:g}, u_in = u[{:d}], u_out=[{:d}],'.format(
                    self._step, t, self._step, self._step+1))

            Ystage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            KstageE = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            KstageI = np.zeros(
                (n, s), dtype=self._function_context['data-type'])

            if(self._info >= 2):
                print('Step {:d}: Y1=u_in'.format(self._step, t))

            for istage in range(s):
                Ystage[0:n, istage] = 0.
                for jstage in range(s):
                    Ystage[0:n, istage] = Ystage[0:n, istage]+dt*A[istage, jstage] * \
                        KstageE[0:n, jstage]+dt * \
                        At[istage, jstage]*KstageI[0:n, jstage]
                for jstage in range(r):
                    Ystage[0:n, istage] = Ystage[0:n, istage] + \
                        U[istage, jstage]*q_in[0:n, jstage]

                t_stage = t+c[istage]*dt

                if(At[istage, istage] != 0.):
                    def IMEX_Function(X, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        f, _ = self._rhs_i(
                            t_stage_in, X, self._function_context)
                        F = X-alpha*f-Res_in
                        return F

                    def IMEX_Function_Jac(X, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        f, Jac = self._rhs_i(
                            t_stage_in, X, self._function_context)
                        J = np.eye(Jac.shape[0])-alpha*Jac
                        return J

                    params = (dt*At[istage, istage],
                              t_stage, Ystage[0:n, istage])
                    # x,_=IMEX_Function(X,*params)
                    Ystage[0:n, istage] = fsolve(
                        func=IMEX_Function, fprime=IMEX_Function_Jac, x0=q_in[0:n, 0], args=params)

                KstageE[0:n, istage], _ = self._rhs_e(
                    t_stage, Ystage[0:n, istage], self._function_context)
                KstageI[0:n, istage], _ = self._rhs_i(
                    t_stage, Ystage[0:n, istage], self._function_context)

            q_out = np.zeros(
                q_in.shape, dtype=self._function_context['data-type'])
            for rstage in range(r):
                for jstage in range(s):
                    q_out[0:n, rstage] = q_out[0:n, rstage] + \
                        dt*B[rstage, jstage]*KstageE[0:n, jstage]
                for jstage in range(r):
                    q_out[0:n, rstage] = q_out[0:n, rstage] + \
                        V[rstage, jstage]*q_in[0:n, jstage]

            self._u_high = (1./(1.-gamma)) * \
                q_out[0:n, 1]-(gamma/(1-gamma))*q_out[0:n, 0]
            self._epsilon = (1./(1.-gamma))*(q_out[0:n, 1]-q_out[0:n, 0])
            self._epsilon_local = (
                1./(1.-gamma))*(q_out[0:n, 1]-q_out[0:n, 0])-(1/(1-gamma))*(q_in[0:n, 1]-q_in[0:n, 0])

            self._global_error = self._epsilon

            for rstage in range(r):
                for jstage in range(s):
                    q_out[0:n, rstage] = q_out[0:n, rstage] + \
                        dt*bt[jstage]*KstageI[0:n, jstage]

            u_out = np.zeros(self._solution.shape,
                             dtype=self._function_context['data-type'])
            u_out[0:n] = q_out[0:n, 0]
            for i in range(self._r-1):
                self._solution_aux[0:n, i] = q_out[0:n, i+1]

        elif(METHOD['type'] == 'RK-IDE'):
            if(self._rhs == None or self._rhs_gint == None):
                raise NameError(
                    'rhs_e, rhs_gint need to be defined in order to use RK-IDE integrators.')
            if(self._info >= 2):
                print(' ---------- step {:d}: t={:g}, u_in = u[{:d}], u_out=[{:d}],'.format(
                    self._step, t, self._step, self._step+1))

            A = METHOD['A']
            b = METHOD['b']
            c = METHOD['c']
            s = METHOD['s']

            Ystage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            Kstage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])

            if(self._info >= 2):
                print('Step {:d}: Y1=u_in'.format(self._step, t))
            Ystage[0:n, 0] = u_in

            t_stage = t

            if(self._info >= 2):
                print('Step {:d}: Y1=u_in, K1=f(Y1)'.format(self._step, t))
            Kstage[0:n, 0], _ = self._rhs(
                t_stage, Ystage[0:n, 0], self._function_context)

            if(self._step >= 1):
                if(self._info >= 2):
                    print('Step {:d}: In(1)={:g}, In(2)={:g}'.format(
                        self._step, self._In[0], self._In[1]))
                Kstage[0:n, 0] = Kstage[0:n, 0]+0.5*dt * \
                    (self._In+self._rhs_gint(t, t, u_in, self._function_context))
                if(self._info >= 2):
                    print(
                        'Step {:d}: K1=K1+0.5*dt*(In+g(t={:g},t={:g},u_in))'.format(self._step, t, t))

            t_stage = t+dt

            if(self._info >= 2):
                print('Step {:d}: Y2=u_in+dt*K1'.format(self._step))

            Ystage[0:n, 1] = u_in+dt*Kstage[0:n, 0]

            if(self._info >= 2):
                print('Step {:d}: In=g(t+dt={:g},t0={:g},u0), u[{:d}](1)= {:g} u[{:d}](2)={:g}'.format(
                    self._step, t+dt, self._t_start, 0, self._u_trajectory[0, 0], 0, self._u_trajectory[1, 0]))
            self._In = self._rhs_gint(
                t+dt, self._t_start, self._u_trajectory[0:n, 0], self._function_context)

            for i in range(1, self._step):
                if(self._info >= 2):
                    print('Step {:d}: In=In+2*g(t+dt={:g},t={:g},u[{:d}]), u[{:d}](1)= {:g} u[{:d}](2)={:g}'.format(
                        self._step, t+dt, self._t_trajectory[i], i, i, self._u_trajectory[0, i], i, self._u_trajectory[1, i]))
                self._In = self._In+2. * \
                    self._rhs_gint(
                        t+dt, self._t_trajectory[i], self._u_trajectory[0:n, i], self._function_context)

            if(self._step >= 1):
                if(self._info >= 2):
                    print(
                        'Step {:d}: In=In+2*g(t+dt={:g},t={:g},u_in)'.format(self._step, t+dt, t))
                self._In = self._In+2. * \
                    self._rhs_gint(t+dt, t, u_in, self._function_context)

            Kstage[0:n, 1], _ = self._rhs(
                t_stage, Ystage[0:n, 1], self._function_context)

            if(self._info >= 2):
                print(
                    'Step {:d}: K2=f(t,Y2)+0.5*dt*(In+g(t={:g},t={:g},Y2))'.format(self._step, t_stage, t_stage))
            Kstage[0:n, 1] = Kstage[0:n, 1]+0.5*dt*(self._In+self._rhs_gint(
                t_stage, t_stage, Ystage[0:n, 1], self._function_context))
            if(self._info >= 2):
                print('Step {:d}: In(1)= {:g}, In(2)={:g}'.format(
                    self._step, self._In[0], self._In[1]))
            u_out = u_in.copy()

            # for i in range(s):
            u_out = u_out+dt*0.5*Kstage[0:n, 0]+dt*0.5*Kstage[0:n, 1]

        elif(METHOD['type'] == 'MRK'):
            AS = METHOD['AB']
            bS = METHOD['bB']
            cS = METHOD['cB']
            sS = METHOD['sB']

            AF = METHOD['AF']
            bF = METHOD['bF']
            cF = METHOD['cF']
            sF = METHOD['sF']

            AS = METHOD['AS']
            bS = METHOD['bS']
            cS = METHOD['cS']
            sS = METHOD['sS']

            uF_in,uS_in=self._function_context['SplitSolution'](u_in)

            nF=self._function_context['nF']
            nS=self._function_context['nS']
            
            YstageF = np.zeros(
                (nF, sF), dtype=self._function_context['data-type'])
            KstageF = np.zeros(
                (nF, sF), dtype=self._function_context['data-type'])

            YstageS = np.zeros(
                (nS, sS), dtype=self._function_context['data-type'])
            KstageS = np.zeros(
                (nS, sS), dtype=self._function_context['data-type'])

            for istage in range(sF):
                YstageF[0:nF, istage] = uF_in
                YstageS[0:nS, istage] = uS_in
                
                for jstage in range(sF):
                    YstageF[0:nF, istage] = YstageF[0:nF, istage] + \
                        dt*AF[istage, jstage]*KstageF[0:nF, jstage]
                    YstageS[0:nS, istage] = YstageS[0:nS, istage] + \
                        dt*AS[istage, jstage]*KstageS[0:nS, jstage]

                t_stageF = t+cF[istage]*dt
                t_stageS = t+cS[istage]*dt

                KstageF[0:nF, istage], _ = self._rhs_fast(
                    t_stageF, YstageS[0:nS, istage], YstageF[0:nF, istage], self._function_context)
                KstageS[0:nS, istage], _ = self._rhs_slow(
                    t_stageS, YstageS[0:nS, istage], YstageF[0:nF, istage], self._function_context)

            uF_out = uF_in.copy()
            uS_out = uS_in.copy()

            for i in range(sF):
                uF_out = uF_out+dt*bF[i]*KstageF[0:nF, i]
            for i in range(sS):
                uS_out = uS_out+dt*bS[i]*KstageS[0:nS, i]
            
            u_out=self._function_context['MergeSolution'](uF_out,uS_out)



        elif(METHOD['type'] == 'IMEX-MRK'):
            AS = METHOD['AB']
            bS = METHOD['bB']
            cS = METHOD['cB']
            sS = METHOD['sB']

            AF = METHOD['AF']
            bF = METHOD['bF']
            cF = METHOD['cF']
            sF = METHOD['sF']

            AS = METHOD['AS']
            bS = METHOD['bS']
            cS = METHOD['cS']
            sS = METHOD['sS']

            AT = METHOD['AT']
            bT = METHOD['bT']
            cT = METHOD['cT']
            sT = METHOD['sT']
            
            uF_in,uS_in=self._function_context['SplitSolution'](u_in)

            nF=self._function_context['nF']
            nS=self._function_context['nS']
            
            YstageF = np.zeros(
                (nF, sF), dtype=self._function_context['data-type'])
            KstageF = np.zeros(
                (nF, sF), dtype=self._function_context['data-type'])

            YstageS = np.zeros(
                (nS, sS), dtype=self._function_context['data-type'])
            KstageS = np.zeros(
                (nS, sS), dtype=self._function_context['data-type'])

            #This is for the implicit part
            YstageG = np.zeros(
                (n, sT), dtype=self._function_context['data-type'])
            KstageG = np.zeros(
                (n, sT), dtype=self._function_context['data-type'])

            assert sF==sS, "Slow and fast methods should have the same number of stages"

            assert nF+nS==n, "Spatial partitioning is not consistent"
            
            for istage in range(sF):
                YstageF[0:nF, istage] = uF_in.copy()
                YstageS[0:nS, istage] = uS_in.copy()
                
                for jstage in range(sF):
                    YstageF[0:nF, istage] = YstageF[0:nF, istage] + \
                        dt*AF[istage, jstage]*KstageF[0:nF, jstage]
                    YstageS[0:nS, istage] = YstageS[0:nS, istage] + \
                        dt*AS[istage, jstage]*KstageS[0:nS, jstage]

                if(self._info >= 2): print("Before YStage: {:}  {:}".format(YstageF[0:2, istage],YstageS[0:2, istage]))
                    
                #The implicit part (explicit portion)
                YstageG[0:n, istage] = self._function_context['MergeSolution'](YstageF[0:nF, istage],YstageS[0:nS, istage])

                if(self._info >= 2): print("Sanity check (before stage): ||YG-u_in|| = {:}".format(np.linalg.norm(np.squeeze(YstageG[0:n, istage])-u_in)))
                if(self._info >= 2): print("Stage {:}:".format(istage))
                for jstage in range(istage):
                    if(self._info >= 2): print("YG_{:} += AT[{:},{:}] * dt * KG_{:} (AT = {:})".format(istage,istage,jstage,jstage,AT[istage, jstage]))
                    YstageG[0:n, istage] = YstageG[0:n, istage] + \
                        dt*AT[istage, jstage]*KstageG[0:n, jstage]
            
                
                if(AT[istage, istage] == 0):
                    #This stage is explicit
                    pass
                else:
                    if(self._info >= 2): print("Stage {:} is implicit: AT[{:},:]={:}".format(istage,istage,AT[istage, :]))
                    def IMEX_Function(X, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        f, _ = self._rhs_mr_implicit(
                            t_stage_in, X, self._function_context)
                        F = X-alpha*f-Res_in
                        return F

                    def IMEX_Function_Jac(X, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        f, Jac = self._rhs_mr_implicit(
                            t_stage_in, X, self._function_context)
                        J = np.eye(Jac.shape[0])-alpha*Jac
                        return J

                    
                    t_stageI = t+cT[istage]*dt
                    Res=YstageG[0:n, istage].copy()
                    params = (dt*AT[istage, istage], t_stageI, Res)
                    
                    if(self._info >= 2):
                        x=IMEX_Function(u_in,*params)
                        print("X  in: {:}".format(np.linalg.norm(x)))
                        
                    sol_fsolve = fsolve(func=IMEX_Function, fprime=IMEX_Function_Jac,x0=u_in, args=params, xtol=1.0e-12,full_output=True)
                    
                    sol_f=sol_fsolve[0]
                    if(self._info >= 2):
                        info_dict=sol_fsolve[1]
                        print("FSOLVE> Number of function evaluations: {:}".format(info_dict["nfev"]))
                        print("FSOLVE> Number of Jacobian evaluations: {:}".format(info_dict["nfev"]))
                        print("FSOLVE> Norm of function at output: {:}".format(np.linalg.norm(info_dict["fvec"])))
                    YstageG[0:n, istage] = sol_f.copy()
                    
                    if(self._info >= 2):
                        x=IMEX_Function(YstageG[0:n, istage],*params)
                        print("X out: {:}".format(np.linalg.norm(x)))


                
                t_stageI = t+cT[istage]*dt

                if(self._info >= 2): print("Eval: KG_{:} = g({:},YG{:})".format(istage,t_stageI,istage))
                KstageG[0:n, istage], _ = self._rhs_mr_implicit(
                    t_stageI, YstageG[0:n, istage], self._function_context)
                
                YstageF[0:nF, istage], YstageS[0:nS, istage] = self._function_context['SplitSolution'](YstageG[0:n, istage])
                if(self._info >= 2): print("After  YStage: {:}  {:}".format(YstageF[0:2, istage],YstageS[0:2, istage]))
                
                t_stageF = t+cF[istage]*dt
                t_stageS = t+cS[istage]*dt

                KstageF[0:nF, istage], _ = self._rhs_fast(
                    t_stageF, YstageS[0:nS, istage], YstageF[0:nF, istage], self._function_context)
                KstageS[0:nS, istage], _ = self._rhs_slow(
                    t_stageS, YstageS[0:nS, istage], YstageF[0:nF, istage], self._function_context)

                if(self._info >= 2): print('Norms: K={:} KF={:} KS={:}'.format(np.linalg.norm(np.squeeze(KstageG[0:n, istage])),np.linalg.norm(np.squeeze(KstageF[0:nF, istage])),np.linalg.norm(np.squeeze(KstageS[0:nF, istage]))))
            uF_out = uF_in.copy()
            uS_out = uS_in.copy()

            for i in range(sF):
                uF_out = uF_out+dt*bF[i]*KstageF[0:nF, i]
            for i in range(sS):
                uS_out = uS_out+dt*bS[i]*KstageS[0:nS, i]
            
            u_out=self._function_context['MergeSolution'](uF_out,uS_out)
            if(self._info >= 3):
                import matplotlib.pyplot as plt
                plt.plot(u_out)
                plt.title('Before adding the implicit part')
                plt.show()


            dd=np.zeros(n)
            if(self._info >= 3):
                plt.figure()
                for i in range(sT):
                    dd+=np.squeeze(KstageG[0:n, i])
                    plt.plot(np.squeeze(KstageG[0:n, i]),label='{:}'.format(i))
                plt.legend()
                plt.title('K implicit')
                plt.show()

            
                plt.figure()
                plt.plot(dt*dd)
            
                plt.title('Sum of K implicit * dt')
                plt.show()
            
            if(self._info >= 2): print("*Sanity check (before implicit stage): ||u_out-u_in|| = {:}".format(np.linalg.norm(u_out-u_in)))
            
            for i in range(sT):
                if(self._info >= 2): print("u_out += bT[{:}] * dt * KG_{:} (bT = {:})".format(i,i,bT[i]))
                u_out = u_out+dt*bT[i]*KstageG[0:n, i]

            if(self._info >= 3):
                plt.figure()
                plt.plot(u_out)
                plt.title('Done with the step')
                plt.show()
        elif(METHOD['type'] == 'RK'):
            A = METHOD['A']
            b = METHOD['b']
            c = METHOD['c']
            s = METHOD['s']
            Ystage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            Kstage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            for istage in range(s):
                Ystage[0:n, istage] = u_in
                for jstage in range(s):
                    Ystage[0:n, istage] = Ystage[0:n, istage] + \
                        dt*A[istage, jstage]*Kstage[0:n, jstage]

                t_stage = t+c[istage]*dt
                # print 't=',t
                Kstage[0:n, istage], _ = self._rhs(
                    t_stage, Ystage[0:n, istage], self._function_context)
                # print Kstage[0:n,istage]
            u_out = u_in.copy()

            for i in range(s):
                u_out = u_out+dt*b[i]*Kstage[0:n, i]
                # print u_out

        elif(METHOD['type'] == 'ETRS'):
            approx_exp = METHOD['approx_exp']
            n_iter = METHOD['n_iter']
            p = METHOD['p']
            ii = np.complex(0, 1)
            if(approx_exp is False):
                _, A1 = self._rhs(t, u_in, self._function_context)

                eA = scipy.linalg.expm(-ii*A1*dt)
                u_A = np.matmul(eA, u_in)

                _, A2 = self._rhs(t+dt, u_A, self._function_context)

                eA1 = scipy.linalg.expm(-ii*A1*dt*0.5)
                eA2 = scipy.linalg.expm(-ii*A2*dt*0.5)

                u_out = np.matmul(eA1, u_in)
                u_out = np.matmul(eA2, u_out)
            else:
                if(p == 2):
                    _, Ht = self._rhs(t, u_in, self._function_context)

                    A = -ii*Ht*dt*0.5

                    u_half = u_in
                    u_acc = u_in
                    for i in range(n_iter):
                        u_acc = np.matmul(A, u_acc)/(i+1.)
                        u_half = u_half+u_acc

                    A = -ii*Ht*dt
                    u_full_approx = u_in
                    u_acc = u_in
                    for i in range(n_iter):
                        u_acc = np.matmul(A, u_acc)/(i+1.)
                        u_full_approx = u_full_approx+u_acc

                    _, Htdt = self._rhs(t+dt, u_full_approx,
                                        self._function_context)
                    # _,Htdt=self._rhs(t,u_in,self._function_context)

                    A = -ii*Htdt*dt*0.5
                    u_out = u_half
                    u_acc = u_half
                    for i in range(n_iter):
                        u_acc = np.matmul(A, u_acc)/(i+1.)
                        u_out = u_out+u_acc
                elif(p == 1):
                    _, Ht = self._rhs(t, u_in, self._function_context)

                    A = -ii*Ht*dt

                    u_half = u_in
                    u_acc = u_in
                    for i in range(n_iter):
                        u_acc = np.matmul(A, u_acc)/(i+1.)
                        u_half = u_half+u_acc

                    u_out = u_half
                else:
                    error('Not implemented')

        elif(METHOD['type'] == 'ESDIRK'):

            At = METHOD['A']
            bt = METHOD['b']
            ct = METHOD['c']
            s = METHOD['s']
            Ystage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            Kstage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])

            for istage in range(s):
                t_stage = t+ct[istage]*dt
                Res = u_in
                for jstage in range(istage+1):
                    Res = Res+dt*At[istage, jstage]*Kstage[0:n, jstage]
                if(At[istage, istage] == 0):
                    Ystage[0:n, istage] = Res
                else:
                    def complexify(X_in):
                        size_of_x = X_in.size//2
                        X = np.zeros((size_of_x), np.complexfloating)
                        for i in range(size_of_x):
                            X[i] = X_in[i]+(1j)*X_in[i+size_of_x]
                        return X

                    def decomplexify(X_in):
                        size_of_x = X_in.size
                        X = np.zeros((2*size_of_x), np.float64)
                        for i in range(size_of_x):
                            X[i] = np.real(X_in[i])
                            X[i+size_of_x] = np.imag(X_in[i])
                        return X

                    def IM_Function(X_in, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        data_type = data[3]
                        if(data_type == np.dtype('complex128')):
                            X = complexify(X_in)
                        else:
                            X = X_in
                        f, _ = self._rhs(t_stage_in, X, self._function_context)
                        F = X-alpha*f-Res_in
                        # return np.ndarray.tolist(F)
                        if(data_type == np.dtype('complex128')):
                            return decomplexify(F)
                        else:
                            return F

                    def IM_Function_Jac(X, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        data_type = data[3]
                        f, Jac = self._rhs(
                            t_stage_in, X, self._function_context)
                        J = np.eye(Jac.shape[0])-alpha*Jac
                        return J

                    params = (dt*At[istage, istage], t_stage, Res, u_in.dtype)
                    # x,_=IMEX_Function(X,*params)
                    if(u_in.dtype == np.dtype('complex128')):
                        outp = fsolve(func=IM_Function, x0=decomplexify(
                            u_in), args=params, xtol=1.0e-12)
                        Ystage[0:n, istage] = complexify(outp)
                    else:
                        Ystage[0:n, istage] = fsolve(
                            func=IM_Function, fprime=IM_Function_Jac, x0=u_in, args=params, xtol=1.0e-12)
                    #from mpmath import findroot
                    #Ystage[0:n,istage] = findroot(IM_Function, np.ndarray.tolist(u_in))

                Kstage[0:n, istage], _ = self._rhs(
                    t_stage, Ystage[0:n, istage], self._function_context)

            u_out = u_in.copy()
            for i in range(s):
                u_out = u_out+dt*bt[i]*Kstage[0:n, i]

        elif(METHOD['type'] == 'ARK'):
            if(self._rhs_e == None or self._rhs_i == None):
                raise NameError(
                    'rhs_e, rhs_i need to be defined in order to use ARKq integrators.')

            A = METHOD['A']
            b = METHOD['b']
            c = METHOD['c']
            At = METHOD['At']
            bt = METHOD['bt']
            ct = METHOD['ct']
            s = METHOD['s']

            if(self._info >= 2): print(At,bt)
            Ystage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            KstageE = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            KstageI = np.zeros(
                (n, s), dtype=self._function_context['data-type'])

            for istage in range(s):
                t_stageE = t+c[istage]*dt
                t_stageI = t+ct[istage]*dt
                Res = u_in
                for jstage in range(istage+1):
                    Res = Res+dt*A[istage, jstage]*KstageE[0:n, jstage] + \
                        dt*At[istage, jstage]*KstageI[0:n, jstage]
                if(At[istage, istage] == 0):
                    Ystage[0:n, istage] = Res
                else:
                    def IMEX_Function(X, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        f, _ = self._rhs_i(
                            t_stage_in, X, self._function_context)
                        F = X-alpha*f-Res_in
                        return F

                    def IMEX_Function_Jac(X, *data):
                        alpha = data[0]
                        t_stage_in = data[1]
                        Res_in = data[2]
                        f, Jac = self._rhs_i(
                            t_stage_in, X, self._function_context)
                        J = np.eye(Jac.shape[0])-alpha*Jac
                        return J

                    params = (dt*At[istage, istage], t_stageI, Res)
                    x=IMEX_Function(u_in,*params)
                    if(self._info >= 2): print("X  in: {:}".format(np.linalg.norm(x)))
                    Ystage[0:n, istage] = fsolve(
                        func=IMEX_Function, fprime=IMEX_Function_Jac, x0=u_in, args=params)
                    x=IMEX_Function(Ystage[0:n, istage],*params)
                    if(self._info >= 2): print("X out: {:}".format(np.linalg.norm(x)))
                    
                KstageE[0:n, istage], _ = self._rhs_e(
                    t_stageE, Ystage[0:n, istage], self._function_context)
                KstageI[0:n, istage], _ = self._rhs_i(
                    t_stageI, Ystage[0:n, istage], self._function_context)

            u_out = u_in.copy()
            
            for i in range(s):
                if(self._info >= 2): print('Norm of KE_{:}={:}'.format(i,np.linalg.norm(np.squeeze(KstageE[0:n, i]))))
                if(self._info >= 2): print('** KI_{:}={:}'.format(i,KstageI[0:3, i]))
                u_out = u_out+dt*b[i]*KstageE[0:n, i]+dt*bt[i]*KstageI[0:n, i]
        else:
            raise NotImplemented

        self._solution = u_out.copy()

        self._t = t+dt
        self._step = self._step+1

        if(self._keep_history == True):
            self._t_trajectory = np.hstack(
                (self._t_trajectory, np.asarray(self._t)))
            self._u_trajectory = np.hstack(
                (self._u_trajectory, np.reshape(self._solution, (n, 1))))
            if(self._has_global_error == True):
                self._epsilon_trajectory = np.hstack(
                    (self._epsilon_trajectory, np.reshape(self._epsilon, (n, 1))))
                self._epsilon_local_trajectory = np.hstack(
                    (self._epsilon_local_trajectory, np.reshape(self._epsilon_local, (n, 1))))
                self._u_high_trajectory = np.hstack(
                    (self._u_high_trajectory, np.reshape(self._u_high, (n, 1))))

    def set_rhs(self, rhs):
        if(type(rhs) == dict):
            if('imex_explicit' in rhs.keys() and 'imex_implicit' in rhs.keys()):
                self._rhs_e = rhs['imex_explicit']
                self._rhs_i = rhs['imex_implicit']
                self._rhs = self._compound_imex_rhs

            if('explicit' in rhs.keys() and 'integrand' in rhs.keys()):
                self._rhs = rhs['explicit']
                self._rhs_gint = rhs['integrand']

            if('mr_explicit_fast' in rhs.keys() and 'mr_explicit_slow' in rhs.keys()):
                self._rhs_fast = rhs['mr_explicit_fast']
                self._rhs_slow = rhs['mr_explicit_slow']
            if('mr_implicit' in rhs.keys()):
                self._rhs_mr_implicit = rhs['mr_implicit']
        
        else:
            self._rhs = rhs

    def set_duration(self, t_start=0., t_end=1., dt=1.):
        self._t_start = t_start
        self._t_end = t_end
        self._dt = dt

        self._t = self._t_start

        self._max_n_steps = 1+int((self._t_end-self._t_start)/self._dt)

        if(self._keep_history == True):
            # self._t_trajectory=np.zeros((self._n,1),dtype=np.float64)
            self._t_trajectory = np.zeros((1,), dtype=np.float64)
            self._t_trajectory[0] = self._t_start

    def set_method(self, method):
        if method in self._method_set:
            self._current_method = method
            METHOD = self._methods[str(self._current_method)]
            if(METHOD['type'] == 'GLEE' or METHOD['type'] == 'GLEE-EIMEX'):
                self._has_global_error = True
            else:
                self._has_global_error = False
        else:
            raise NotImplemented

    def get_method(self):
        if(self._current_method is None):
            return None
        else:
            return self._methods[str(self._current_method)]
        
    def get_method_name(self):
        return self._current_method

    def view_registered_methods(self):
        print(self._method_set)
        return self._method_set

    def view_complete_status(self):

        print('------------------------------------------')
        print('Current Method')
        METHOD = self._methods[str(self._current_method)]
        
        if(METHOD['type'] is None):
            print(' No method is currently selected')
        elif(METHOD['type'] == 'GLEE'):
            raise NotImplemented
        elif(METHOD['type'] == 'ARK'):
            A = METHOD['A']
            b = METHOD['b']
            c = METHOD['c']
            At = METHOD['At']
            bt = METHOD['bt']
            ct = METHOD['ct']
            s = METHOD['s'] 
            print('Selected method type: {:}'.format(METHOD['type']))
            print('Method coefficients:')
            
            print('A =') 
            print(A)
            print('b =') 
            print(b)
            print('c =') 
            print(c)

            print('At =') 
            print(At)
            print('bt =') 
            print(bt)
            print('ct =') 
            print(ct)

        elif(METHOD['type'] == 'MRK'):
            AB = METHOD['AB']
            bB = METHOD['bB']
            cB = METHOD['cB']
            sB = METHOD['sB']

            AF = METHOD['AF']
            bF = METHOD['bF']
            cF = METHOD['cF']
            sF = METHOD['sF']

            AS = METHOD['AS']
            bS = METHOD['bS']
            cS = METHOD['cS']
            sS = METHOD['sS']

            print('Selected method type: {:}'.format(METHOD['type']))
            print('Method coefficients:')
            
            print('A_f =') 
            print(AF)
            print('b_f =') 
            print(bF)
            print('c_f =') 
            print(cF)

            print('A_s =') 
            print(AS)
            print('b_s =') 
            print(bS)
            print('c_s =') 
            print(cS)

            print('A base =') 
            print(AB)
            print('b base =') 
            print(bB)
            print('c base  =') 
            print(cB)
        
        elif(METHOD['type'] == 'IMEX-MRK'):
            AB = METHOD['AB']
            bB = METHOD['bB']
            cB = METHOD['cB']
            sB = METHOD['sB']

            AF = METHOD['AF']
            bF = METHOD['bF']
            cF = METHOD['cF']
            sF = METHOD['sF']

            AS = METHOD['AS']
            bS = METHOD['bS']
            cS = METHOD['cS']
            sS = METHOD['sS']

            AT = METHOD['AT']
            bT = METHOD['bT']
            cT = METHOD['cT']
            sT = METHOD['sT']

            print('Selected method type: {:}'.format(METHOD['type']))
            print('Method coefficients:')
            
            print('A_f =') 
            print(AF)
            print('b_f =') 
            print(bF)
            print('c_f =') 
            print(cF)

            print('A_s =') 
            print(AS)
            print('b_s =') 
            print(bS)
            print('c_s =') 
            print(cS)

            print('At =') 
            print(AT)
            print('bt =') 
            print(bT)
            print('ct =') 
            print(cT)

        elif(METHOD['type'] == 'RK'):
            A = METHOD['A']
            b = METHOD['b']
            c = METHOD['c']
           
            s = METHOD['s'] 
            print('Selected method type: {:}'.format(METHOD['type']))
            print('Method coefficients:')
            
            print('A =') 
            print(A)
            print('b =') 
            print(b)
            print('c =') 
            print(c)

        else:
            raise NotImplemented
        print('------------------------------------------')
        return self._method_set

    def set_initial_solution(self, U0):
        self._n = U0.size
        self._solution = U0.copy()
        if(self._keep_history == True):
            self._u_trajectory = np.zeros(
                (self._n, 1), dtype=self._function_context['data-type'])
            self._u_trajectory[0:self._n, 0] = U0

            self._epsilon_trajectory = np.zeros(
                (self._n, 1), dtype=self._function_context['data-type'])
            self._epsilon_local_trajectory = np.zeros(
                (self._n, 1), dtype=self._function_context['data-type'])
            self._u_high_trajectory = np.zeros(
                (self._n, 1), dtype=self._function_context['data-type'])
            self._u_high_trajectory[0:self._n, 0] = U0

    def get_solution(self):
        return self._solution.copy()

    def get_has_global_error(self):
        return self._has_global_error

    def get_global_error(self):
        return self._global_error

    def set_max_steps(self, steps):
        self._max_n_steps = steps
        return 0

    def set_info(self, info):
        self._info = info
        return 0

    def set_function_context(self, function_ctx):
        self._function_context = function_ctx
        return 0

    def get_trajectory(self, time_points=None):
        if not self._solved:
            print("Warning: problem not solved")
        if(self._keep_history == True):
            if(time_points is None):
                return self._t_trajectory.copy(), self._u_trajectory.copy()
            else:
                finterp = scipy.interpolate.interp1d(
                    self._t_trajectory, self._u_trajectory, axis=1)
                u_trajectory = finterp(time_points)
                return time_points, u_trajectory
        else:
            return None, None

    def get_trajectory_GLEE(self, time_points=None):
        if not self._solved:
            print("Warning: problem not solved")
        tt = None
        uu = None
        ee = None
        el = None
        uh = None
        if(self._keep_history == True):
            if(time_points is None):
                tt = self._t_trajectory.copy()
                uu = self._u_trajectory.copy()
            else:
                finterp = scipy.interpolate.interp1d(
                    self._t_trajectory, self._u_trajectory, axis=1)
                u_trajectory = finterp(time_points)
                tt = time_points
                uu = u_trajectory
            if(self._has_global_error == True):
                if(time_points is None):
                    tt = self._t_trajectory.copy()
                    ee = self._epsilon_trajectory.copy()
                    el = self._epsilon_local_trajectory.copy()
                    uh = self._u_high_trajectory.copy()
                else:
                    finterp = scipy.interpolate.interp1d(
                        self._t_trajectory, self._epsilon_trajectory, axis=1)
                    ee = finterp(time_points)
                    finterp = scipy.interpolate.interp1d(
                        self._t_trajectory, self._epsilon_local_trajectory, axis=1)
                    el = finterp(time_points)
                    finterp = scipy.interpolate.interp1d(
                        self._t_trajectory,  self._u_high_trajectory, axis=1)
                    uh = finterp(time_points)
        return tt, uu, ee, el, uh

    def grid_convergence(self, rhs_e, rhs_i, u_ini, problem_setup, problem, method, duration, grid=None, method_ref=None, ref_dt=None, RefSol=None):

        
        dt_tab=np.logspace(grid['start_exponent'], grid['end_exponent'], grid['points'])
        #print(dt_tab)
        T_end=problem_setup['T_DURATION']['end']
        NSteps=np.zeros(dt_tab.shape,dtype=int)
        for i in range(dt_tab.size):
            NSteps[i]=round(T_end/dt_tab[i])
        
        dt_tab=T_end/NSteps
        #print(dt_tab)
        #print(NSteps)

        
        self.set_info(0)
        self.setup()
        self.set_function_context(problem_setup['context'])
        self.set_initial_solution(u_ini)

        self.set_method(method)


        if(rhs_i is not None):
            self.set_rhs({'imex_explicit':rhs_e,'imex_implicit':rhs_i})
        else:
            self.set_rhs(rhs_e)

        


        
        if(problem.exact_solution is None and RefSol is None):
            print("Reference solution is not provided. Need to compute a reference solution first.")
            problem_setup['DT']=ref_dt
            
            problem_setup['T_DURATION']['start']=duration['t_start']
            problem_setup['T_DURATION']['end']=duration['t_end']*11./10.
        
            self.set_duration(t_start=problem_setup['T_DURATION']['start'],
                              t_end=problem_setup['T_DURATION']['end'],
                              dt=problem_setup['DT'])
            Ref_Steps=round(duration['t_end']/ref_dt)
            self.set_max_steps(Ref_Steps)

            
            print('Solving {:} by using {:} and using a time step of {:} ({:} steps)'.format(problem_setup['name'],self.get_method_name(),ref_dt,Ref_Steps))
            
            self.solve()
            
            t_ref,u_ref,glee_ref,_,_=self.get_trajectory_GLEE()
            uref_end=self.get_solution()
        elif(problem.exact_solution is None and RefSol is not None):
            print("Reference solution is provided. Reusing it.")
            u_ref=RefSol['u_ref']
            method_ref=RefSol['method_ref']
            ref_dt=RefSol['ref_dt']
            t_ref=RefSol['t_ref']
            glee_ref=RefSol['glee_ref']
            uref_end=RefSol['uref_end']
            print("Reference solution at final time {:} computed by {:} with time step {:}.".format(t_ref[-1],method_ref,ref_dt))
        else:
            raise NotImplementedError
        

            
        sol_t = []
        sol_u = []
        sol_uf= []
        sol_tf= []
        err_tab_tf=[]
        sol_glee = []
        sol_gleef = []
        sol_el=[]
        sol_uh=[]
        sol_uhf=[]

        for i in range(dt_tab.size):
            self.reset()
            self.set_info(0)
            self.setup()
            self.set_initial_solution(u_ini)
            self.set_method(method)


            problem_setup['T_DURATION']['start']=duration['t_start']
            problem_setup['T_DURATION']['end']=duration['t_end']*11./10.
            problem_setup['DT']=dt_tab[i]
           
            self.set_duration(t_start=problem_setup['T_DURATION']['start'],
                                t_end=problem_setup['T_DURATION']['end'],
                                dt=problem_setup['DT'])
            print('Solving {:} by using {:} and using a time step of {:} ({:} steps)'.format(problem_setup['name'],self.get_method_name(),dt_tab[i],NSteps[i]))

            self.set_max_steps(NSteps[i])
    
            self.solve()

            t,u,glee,el,uh=self.get_trajectory_GLEE()
            u_end=self.get_solution()
            sol_t.append(t.copy())
            sol_u.append(u.copy())
            sol_tf.append(t[-1])
            sol_uf.append(u_end)
            sol_glee.append(glee)
            if(glee is not None):
                sol_gleef.append(glee[:,-1])
                sol_el.append(el)
                sol_uh.append(uh)
                sol_uhf.append(uh[:,-1])

                
            err_tab_tf.append(np.linalg.norm(np.squeeze(np.asarray(u_end)[:]-u_ref[:,-1]),ord=2))

        RS={'t_ref':t_ref,'u_ref':u_ref,'uref_end':uref_end,'glee_ref':glee_ref,'method_ref':method_ref, 'ref_dt':ref_dt}
        return dt_tab, sol_t, sol_u, sol_uf, sol_tf, err_tab_tf, sol_glee, sol_gleef,RS,sol_el,sol_uh,sol_uhf
