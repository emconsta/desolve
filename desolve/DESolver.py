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
            Ystage = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            KstageE = np.zeros(
                (n, s), dtype=self._function_context['data-type'])
            KstageI = np.zeros(
                (n, s), dtype=self._function_context['data-type'])

            for istage in range(s):
                t_stage = t+c[istage]*dt
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

                    params = (dt*At[istage, istage], t_stage, Res)
                    # x,_=IMEX_Function(X,*params)
                    Ystage[0:n, istage] = fsolve(
                        func=IMEX_Function, fprime=IMEX_Function_Jac, x0=u_in, args=params)

                KstageE[0:n, istage], _ = self._rhs_e(
                    t_stage, Ystage[0:n, istage], self._function_context)
                KstageI[0:n, istage], _ = self._rhs_i(
                    t_stage, Ystage[0:n, istage], self._function_context)

            u_out = u_in.copy()
            for i in range(s):
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
