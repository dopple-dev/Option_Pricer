from NMFF.Fourier.Fourier import FFT, Fourier_inversion
from NMFF.Chf_PDF.Charateristic_functions import BS_chf

import numpy as np
import scipy.stats as st
from scipy.integrate import quad
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse.linalg import splu
from time import time

class Black_Scholes:
    
    def __init__(self, Option_info, Process_info):

        self.S0 = Option_info.S0 # underlying 
        self.K = Option_info.K # strike 
        self.T = Option_info.T # tenor 
        self.payoff = Option_info.payoff # payoff 
        self.exercise_style = Option_info.exercise_style
        self.price = 0

        self.r = Process_info.r # interest rate 
        self.sigma = Process_info.sigma # diffusion coefficient 
        self.rd = Process_info.rd
        self.rf = Process_info.rf

        self.S_arr = None

    def get_payoff(self, S): # obtain payoff function 
        if (self.payoff == 'call' or self.payoff == 'c' or self.payoff == 'Call' or self.payoff == 'C'):
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == 'put' or self.payoff == 'p' or self.payoff == 'Put' or self.payoff == 'P':
            Payoff = np.maximum(self.K - S, 0)
        return Payoff
    
    def BS_pricer_self(self, Time=False):

        t_init = time()

        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.S0 / self.K) + (self.r - self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))

        if (self.payoff == 'call' or self.payoff == 'c' or self.payoff == 'Call' or self.payoff == 'C'):
            self.price = self.S0 * st.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * st.norm.cdf(d2)
        elif self.payoff == 'put' or self.payoff == 'p' or self.payoff == 'Put' or self.payoff == 'P':
            self.price = self.K * np.exp(-self.r * self.T) * st.norm.cdf(-d2) - self.S0 * st.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
        
        if Time is True:
                elapsed = time() - t_init
                return self.price, elapsed
        else:
            return self.price
        
    def BS_F_inv(self, Time=False):
        t_init = time()

        BS_partial_chf = BS_chf.chf_GBM_partial(self.sigma, self.r, self.T) 
        self.price = Fourier_inversion(self.S0, self.K, self.r, self.T, BS_partial_chf, self.payoff)

        if Time is True:
                elapsed = time() - t_init
                return self.price, elapsed
        else:
            return self.price 

    def BS_FFT(self, K, method='carr_madan', N=2**12, B=200, Time=True): 
        t_init = time()
        
        BS_partial_chf = BS_chf.chf_GBM_partial(self.sigma, self.r, self.T) 
        self.price = FFT(self.S0, self.K, K, self.r, self.T, BS_partial_chf, self.payoff, 
                         method = method, N = 2**12, B = 200, interp="cubic")
        
        if Time is True:
                elapsed = time() - t_init
                return self.price, elapsed
        else:
            return self.price

    def PDE_pricer_self(self, Nspace, Ntime, Time=False, solver="splu"):
            """
            steps = tuple with number of space steps and time steps
            payoff = "call" or "put"
            exercise = "European" or "American"
            Time = Boolean. Execution time.
            Solver = spsolve or splu or Thomas or SOR
            """
            t_init = time()

            S_max = 6 * float(self.K)
            S_min = float(self.K) / 6
            x_max = np.log(S_max)
            x_min = np.log(S_min)
            x0 = np.log(self.S0)  # current log-price

            x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)
            t, dt = np.linspace(0, self.T, Ntime, retstep=True)

            self.S_arr = np.exp(x)  # vector of S
            Payoff = self.get_payoff(self.S_arr)

            V = np.zeros((Nspace, Ntime))
            if (self.payoff == 'call' or self.payoff == 'c' or self.payoff == 'Call' or 
                self.payoff == 'C' or self.payoff == 1):
                V[:, -1] = Payoff
                V[-1, :] = np.exp(x_max) - self.K * np.exp(-self.r * t[::-1])
                V[0, :] = 0
            elif (self.payoff == 'put' or self.payoff == 'p' or self.payoff == 'Put' or 
                self.payoff == 'P' or self.payoff == -1):
                V[:, -1] = Payoff
                V[-1, :] = 0
                V[0, :] = Payoff[0] * np.exp(-self.r * t[::-1])  # Instead of Payoff[0] I could use K
                # For s to 0, the limiting value is e^(-rT)(K-s)
            else:
                    raise ValueError("invalid type. Set 'call' or 'put'")

            sig2 = self.sigma**2
            dxx = dx**2
            a = (dt / 2) * ((self.r - 0.5 * sig2) / dx - sig2 / dxx)
            b = 1 + dt * (sig2 / dxx + self.r)
            c = -(dt / 2) * ((self.r - 0.5 * sig2) / dx + sig2 / dxx)

            D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()

            offset = np.zeros(Nspace - 2)

            if solver == "spsolve":
                if self.exercise_style == "European":
                    for i in range(Ntime - 2, -1, -1):
                        offset[0] = a * V[0, i]
                        offset[-1] = c * V[-1, i]
                        V[1:-1, i] = spsolve(D, (V[1:-1, i + 1] - offset))
                else:
                    raise AttributeError('The exercise style ', self.exercise_style, ' needs to be implemented')
            elif solver == "splu":
                DD = splu(D)
                if self.exercise_style == "European":
                    for i in range(Ntime - 2, -1, -1):
                        offset[0] = a * V[0, i]
                        offset[-1] = c * V[-1, i]
                        V[1:-1, i] = DD.solve(V[1:-1, i + 1] - offset)
                else:
                    raise AttributeError('The exercise style ', self.exercise_style, ' needs to be implemented')
            else:
                raise ValueError("Solver is splu or spsolve")

            self.price = np.interp(x0, x, V[:, 0])
            self.price_vec = V[:, 0]
            self.mesh = V

            if Time is True:
                elapsed = time() - t_init
                return self.price, elapsed
            else:
                return self.price

    @staticmethod 
    def binom_tree(payoff, S0, T, r, sigma, strikes):

        t_init = time()

        N = 30000  # number of periods or number of time steps

        dT = float(T) / N  # Delta t
        u = np.exp(sigma * np.sqrt(dT))  # up factor
        d = 1.0 / u  # down factor

        V = np.zeros(N + 1)  # initialize the price vector

        # price S_T at time T
        S_T = np.array([(S0 * u**j * d ** (N - j)) for j in range(N + 1)])

        a = np.exp(r * dT)  # risk free compounded return
        p = (a - d) / (u - d)  # risk neutral up probability
        q = 1.0 - p  # risk neutral down probability

        BS_tree_price = np.zeros(len(strikes))

        for k, K in enumerate(strikes): 
            if payoff == "call":
                V[:] = np.maximum(S_T - K, 0.0)
            else:
                V[:] = np.maximum(K - S_T, 0.0)

            for i in range(N - 1, -1, -1):
                # the price vector is overwritten at each step
                V[:-1] = np.exp(-r * dT) * (p * V[1:] + q * V[:-1])

            BS_tree_price[k] = V[0]

        elapsed = time() - t_init

        return BS_tree_price, elapsed

    @staticmethod 
    def PC_Parity(given, find, option_value, S0, K, r=0, T=0):

        if ((find == 'call' or find == 'c' or find == 'Call' or find == 'C') and 
            (given == 'put' or given == 'p' or given == 'Put' or given == 'P')):
            return option_value + S0 - K * np.exp(-r * T)
        
        elif ((given == 'call' or given == 'c' or given == 'Call' or given == 'C') and 
            (find == 'put' or find == 'p' or find == 'Put' or find == 'P')):
            return option_value - S0 + K * np.exp(-r * T)

        else:
            raise ValueError('invalid option payoff type.')
        
    @staticmethod
    def BS_pricer(payoff, S0, K, T, r, sigma):
        
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

        if payoff == 'call' or payoff == 'c' or payoff == 'Call' or payoff == 'C':
            return S0 * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
        elif payoff == 'put' or payoff == 'p' or payoff == 'Put' or payoff == 'P':
            return K * np.exp(-r * T) * st.norm.cdf(-d2) - S0 * st.norm.cdf(-d1)
        else:
            raise ValueError('invalid option payoff type.')
        
    
class BS_main:

    def closed_formula(BS_p, strikes):
        BS_p_closed_formula = np.zeros_like(strikes, dtype=float)
        BS_p_closed_f_time = 0

        for i, K in enumerate(strikes):
            BS_p.K = K
            BS_pt = BS_p.BS_pricer_self(Time=True)
            BS_p_closed_formula[i] = BS_pt[0]
            BS_p_closed_f_time = BS_p_closed_f_time + BS_pt[1]

        print('BS closed form prices:')
        print(BS_p_closed_formula)
        print()
        print('Pricing time:', BS_p_closed_f_time)

    def f_transform(BS_p, strikes):
        BS_p_F_inversion = np.zeros_like(strikes, dtype=float)
        BS_p_F_inv_time = 0
        for i, K in enumerate(strikes):
            BS_p.K = K
            BS_pt = BS_p.BS_F_inv(Time=True)
            BS_p_F_inversion[i] = BS_pt[0]
            BS_p_F_inv_time = BS_p_F_inv_time + BS_pt[1]

        print('BS Fourier inversion prices:')
        print(BS_p_F_inversion)
        print()
        print('Pricing time:', BS_p_F_inv_time )

    def FFT_lew(BS_p, strike):
        strikes = np.array(strike)
        BS_pt = BS_p.BS_FFT(K=strikes, method='lewis', Time=True)
        BS_p_lewis = BS_pt[0]
        BS_p_FFT_Lew_time = BS_pt[1]

        print('BS FFT Lewis prices:')
        print(BS_p_lewis)
        print()
        print('Pricing time:', BS_p_FFT_Lew_time )

    def FFT_carr(BS_p, strikes):
        BS_pt = BS_p.BS_FFT(K=strikes, method='carr_madan', Time=True)
        BS_p_carr = BS_pt[0]
        BS_p_FFT_Carr_time = BS_pt[1]

        print('BS FFT Carr-Madan prices:')
        print(BS_p_carr)
        print()
        print('Pricing time:', BS_p_FFT_Carr_time )

    def bin_tree(payoff, S0, T, r, sigma, strikes):
        BS_tree_price, elapsed = Black_Scholes.binom_tree(payoff, S0, T, r, sigma, strikes)
        print("BS Tree Prices: ")
        print(BS_tree_price)
        print()
        print('Pricing time:', elapsed )

    def PDE(BS_p, strikes):
        BS_PDE_price_splu = np.zeros(len(strikes))
        pricing_time_splu = 0

        BS_PDE_price_spsolve = np.zeros(len(strikes))
        pricing_time_spsolve = 0

        for k, K in enumerate(strikes): 
            BS_p.K = K
            PDE_price = BS_p.PDE_pricer_self(Nspace=5000, Ntime=4000, Time=True, solver="splu")
            BS_PDE_price_splu[k] = PDE_price[0]
            pricing_time_splu = pricing_time_splu + PDE_price[1]

        for k, K in enumerate(strikes): 
            BS_p.K = K
            PDE_price = BS_p.PDE_pricer_self(Nspace=5000, Ntime=4000, Time=True, solver="spsolve")
            BS_PDE_price_spsolve[k] = PDE_price[0]
            pricing_time_spsolve = pricing_time_spsolve + PDE_price[1]

        print('BS PDE-splu prices:')
        print(BS_PDE_price_splu)
        print()
        print('Pricing time:', pricing_time_splu)
        print()
        print('BS PDE-spsolve prices:')
        print(BS_PDE_price_spsolve)
        print()
        print('Pricing time:', pricing_time_spsolve)

