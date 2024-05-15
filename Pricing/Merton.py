from NMFF.Fourier.Fourier import FFT, Fourier_inversion
from NMFF.Chf_PDF.Charateristic_functions import MerJD_chf
from NMFF.Pricing.Black_Scholes import Black_Scholes

import numpy as np 
from math import factorial
import matplotlib.pyplot as plt
import warnings
from time import time

class Merton_pricer: 

    def __init__(self, Option_info, Process_info):

        self.mu = Process_info.mu  # interest rate
        self.sigma = Process_info.sigma  # diffusion coefficient
        self.lambd = Process_info.lambd  # jump activity
        self.muJ = Process_info.muJ  # jump mean
        self.sigmaJ = Process_info.sigmaJ  # jump std
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

    def MerJD_pricer_self(self, Time=True): # closed formula
        t_init = time()
        m = self.lambd * (np.exp(self.muJ + (self.sigmaJ**2) / 2) - 1)  # coefficient m
        lam2 = self.lambd * np.exp(self.muJ + (self.sigmaJ**2) / 2)

        self.price = 0
        for i in range(18):
            bs = Black_Scholes.BS_pricer(self.payoff,self.S0,self.K,self.T,
                                        self.mu - m + i * (self.muJ + 0.5 * self.sigmaJ**2) / self.T,
                                        np.sqrt(self.sigma**2 + (i * self.sigmaJ**2) / self.T))
            self.price += (np.exp(-lam2 * self.T) * (lam2 * self.T) ** i / factorial(i)) * bs
        if Time is True:
                elapsed = time() - t_init
                return self.price, elapsed
        else:
            return self.price

    def MerJD_F_inv(self, Time=True):
        t_init = time()
        m = self.lambd * (np.exp(self.muJ + (self.sigmaJ**2) / 2) - 1)
        correction = self.mu - 0.5 * self.sigma**2 - m
        MerJD_partial_chf = MerJD_chf.chf_MerJD_partial(self.T, correction, self.sigma, self.lambd, self.muJ,  self.sigmaJ)
        self.price = Fourier_inversion(self.S0, self.K, self.mu, self.T, MerJD_partial_chf, self.payoff)
        if Time is True:
                elapsed = time() - t_init
                return self.price, elapsed
        else:
            return self.price
    
    def MerJD_FFT(self, K, method= 'carr_madan', N = 2**12, B = 200, Time=True): 
        t_init = time()
        m = self.lambd * (np.exp(self.muJ + (self.sigmaJ**2) / 2) - 1)
        correction = self.mu - 0.5 * self.sigma**2 - m
        MerJD_partial_chf = MerJD_chf.chf_MerJD_partial(self.T, correction, self.sigma, self.lambd, self.muJ,  self.sigmaJ)
        self.price =  FFT(self.S0, self.K, K, self.mu, self.T, 
                          MerJD_partial_chf, self.payoff, method = method, N = 2**12, B = 200, interp="cubic")
        if Time is True:
                elapsed = time() - t_init
                return self.price, elapsed
        else:
            return self.price

    def plot_MerJD_chf(self, u1=np.linspace(0,100,1000)):

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        print('FIGURE 1:  T, mu, sigma, lambd, muJ,  sigmaJ = ', 
              self.T, self.mu, self.sigma, self.lambd, self.muJ,  self.sigmaJ)
        plt.figure(1)
        ax = plt.axes(projection='3d')
        cf = MerJD_chf.chf_MerJD(u1, self.T, self.mu, self.sigma, self.lambd, self.muJ,  self.sigmaJ)
        x = np.real(cf)
        y = np.imag(cf)
        ax.plot3D(u1, x, y, 'blue')
        ax.view_init(30, -120)

        T = 15 
        mu = 8.0 
        sigma = 1.0
        lambd = 0.8  
        muJ = 3  
        sigmaJ = 0.1  
        u2=np.linspace(0,0.5,2000)
        print('FIGURE 2:  T, mu, sigma, lambd, muJ,  sigmaJ = ', 
               T, mu, sigma, lambd, muJ,  sigmaJ)
        plt.figure(2)  
        ax = plt.axes(projection='3d')
        chf = MerJD_chf.chf_MerJD(u2,  T, mu, sigma, lambd, muJ,  sigmaJ)
        x = np.real(chf)
        y = np.imag(chf)
        ax.plot3D(u2, x, y, 'blue')
        ax.view_init(30, -120)

class ME_main:
 
    def closed_formula(ME_p, strike):
        strikes = np.array(strike)
        ME_p_closed_formula = np.zeros_like(strikes, dtype=float)
        ME_p_closed_f_time = 0

        for i, K in enumerate(strikes):
            ME_p.K = K
            ME_pt = ME_p.MerJD_pricer_self(Time=True)
            ME_p_closed_formula[i] = ME_pt[0]
            ME_p_closed_f_time = ME_p_closed_f_time + ME_pt[1]

        print('ME closed form prices:')
        print(ME_p_closed_formula)
        print()
        print('Pricing time:', ME_p_closed_f_time)

    def f_transform(ME_p, strikes):
        ME_p_F_inversion = np.zeros_like(strikes, dtype=float)
        ME_p_F_inv_time = 0
        for i, K in enumerate(strikes):
            ME_p.K = K
            ME_pt = ME_p.MerJD_F_inv(Time=True)
            ME_p_F_inversion[i] = ME_pt[0]
            ME_p_F_inv_time = ME_p_F_inv_time + ME_pt[1]

        print('ME Fourier inversion prices:')
        print(ME_p_F_inversion)
        print()
        print('Pricing time:', ME_p_F_inv_time )

    def FFT_lew(ME_p, strike):
        strikes = np.array(strike)
        ME_pt = ME_p.MerJD_FFT(K=strikes, method='lewis', Time=True)
        ME_p_lewis = ME_pt[0]
        ME_p_FFT_Lew_time = ME_pt[1]

        print('ME FFT Lewis prices:')
        print(ME_p_lewis)
        print()
        print('Pricing time:', ME_p_FFT_Lew_time )

    def FFT_carr(ME_p, strikes):
        ME_pt = ME_p.MerJD_FFT(K=strikes, method='carr_madan', Time=True)
        ME_p_carr = ME_pt[0]
        ME_p_FFT_Carr_time = ME_pt[1]

        print('ME FFT Carr-Madan prices:')
        print(ME_p_carr)
        print()
        print('Pricing time:', ME_p_FFT_Carr_time )