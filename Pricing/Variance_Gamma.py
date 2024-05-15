from NMFF.Fourier.Fourier import FFT, Fourier_inversion
from NMFF.Chf_PDF.Charateristic_functions import VG_chf
from NMFF.Pricing.Black_Scholes import Black_Scholes

import numpy as np 
from math import factorial
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.special as scsp
from scipy.integrate import quad
import warnings
from time import time

class VG_pricer: 

    def __init__(self, Option_info, Process_info):

        self.mu = Process_info.mu  # interest rate
        self.sigma = Process_info.sigma  # VG parameter
        self.theta = Process_info.theta  # VG parameter
        self.kappa = Process_info.kappa  # VG parameter
        self.price = 0 

        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years
        self.exercise_style = Option_info.exercise_style
        self.payoff = Option_info.payoff

    def get_payoff(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def VG_F_inv(self, Time=True): 
        t_init = time()
        w = -np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma**2) / self.kappa
        correction = self.mu - w
        VG_partial_chf = VG_chf.chf_VG_partial(self.T, correction, self.theta, self.sigma, self.kappa)
        self.price = Fourier_inversion(self.S0, self.K, self.mu, self.T, VG_partial_chf, self.payoff)
        if Time is True:
                elapsed = time() - t_init
                return self.price, elapsed
        else:
            return self.price 
        
    def VG_FFT(self, K, method= 'carr_madan', N = 2**12, B = 200, Time=True): 
        t_init = time()
        w = -np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma**2) / self.kappa
        correction = self.mu - w
        VG_partial_chf = VG_chf.chf_VG_partial(self.T, correction, self.theta, self.sigma, self.kappa)
        self.price = FFT(self.S0, self.K, K, self.mu, self.T, 
                         VG_partial_chf, self.payoff, method = method, N = 2**12, B = 200, interp="cubic")
        if Time is True:
                elapsed = time() - t_init
                return self.price, elapsed
        else:
            return self.price 

    
    def plot_VG_chf(self, u1=np.linspace(0,100,1000)):

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        print('FIGURE 1: T, mu, theta, sigma, kappa = ', 
              self.T, self.mu, self.theta, self.sigma, self.kappa)
        plt.figure(1)
        ax = plt.axes(projection='3d')
        cf = VG_chf.chf_VG(u1, self.T, self.mu, self.theta, self.sigma, self.kappa)
        x = np.real(cf)
        y = np.imag(cf)
        ax.plot3D(u1, x, y, 'blue')
        ax.view_init(30, -120)

        T = 15
        mu = 8.0 
        theta = 0.1  
        sigma = 1.0
        kappa = 1.0
        u2=np.linspace(0,1.2,1000)

        print('FIGURE 2: T, mu, theta, sigma, kappa = ', 
              T, mu, theta, sigma, kappa)
        plt.figure(2)  
        ax = plt.axes(projection='3d')
        chf = VG_chf.chf_VG(u2, T, mu, theta, sigma, kappa)
        x = np.real(chf)
        y = np.imag(chf)
        ax.plot3D(u2, x, y, 'blue')
        ax.view_init(30, -120)

class VG_main:
     
    def f_transform(VG_p, strikes):
        VG_p_F_inversion = np.zeros_like(strikes, dtype=float)
        VG_p_F_inv_time = 0

        for i, K in enumerate(strikes):
            VG_p.K = K
            VG_pt = VG_p.VG_F_inv(Time=True)
            VG_p_F_inversion[i] = VG_pt[0]
            VG_p_F_inv_time = VG_p_F_inv_time + VG_pt[1]
        
        print('VG Fourier inversion prices:')
        print(VG_p_F_inversion)
        print()
        print('Pricing time:', VG_p_F_inv_time )

    def FFT_lew(VG_p, strike):
        strikes = np.array(strike)
        VG_pt = VG_p.VG_FFT(K=strikes, method='lewis',Time=True)
        VG_p_lewis = VG_pt[0]
        VG_p_FFT_Lew_time = VG_pt[1]

        print('VG FFT Lewis prices:')
        print(VG_p_lewis)
        print()
        print('Pricing time:', VG_p_FFT_Lew_time )

    def FFT_carr(VG_p, strikes):
        VG_pt = VG_p.VG_FFT(K=strikes, method='carr_madan', Time=True)
        VG_p_carr = VG_pt[0]
        VG_p_FFT_Carr_time = VG_pt[1]

        print('VG FFT Carr-Madan prices:')
        print(VG_p_carr)
        print()
        print('Pricing time:', VG_p_FFT_Carr_time )