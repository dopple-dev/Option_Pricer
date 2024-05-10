from NMFF.Fourier.Fourier import FFT, Fourier_inversion
from NMFF.Chf_PDF.Charateristic_functions import MerJD_chf
from NMFF.Pricing.Black_Scholes import Black_Scholes

import numpy as np 
from math import factorial
import matplotlib.pyplot as plt
import warnings

class Merton_pricer: 

    def __init__(self, Option_info, Process_info):

        self.mu = Process_info.mu  # interest rate
        self.sigma = Process_info.sigma  # diffusion coefficient
        self.lambd = Process_info.lambd  # jump activity
        self.muJ = Process_info.muJ  # jump mean
        self.sigmaJ = Process_info.sigmaJ  # jump std

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

    def MerJD_pricer_self(self): # closed formula

        m = self.lambd * (np.exp(self.muJ + (self.sigmaJ**2) / 2) - 1)  # coefficient m
        lam2 = self.lambd * np.exp(self.muJ + (self.sigmaJ**2) / 2)

        price = 0
        for i in range(18):
            bs = Black_Scholes.BS_pricer(self.payoff,self.S0,self.K,self.T,
                                        self.mu - m + i * (self.muJ + 0.5 * self.sigmaJ**2) / self.T,
                                        np.sqrt(self.sigma**2 + (i * self.sigmaJ**2) / self.T))
            price += (np.exp(-lam2 * self.T) * (lam2 * self.T) ** i / factorial(i)) * bs
        return price

    def MerJD_F_inv(self):
        m = self.lambd * (np.exp(self.muJ + (self.sigmaJ**2) / 2) - 1)
        correction = self.mu - 0.5 * self.sigma**2 - m
        MerJD_partial_chf = MerJD_chf.chf_MerJD_partial(self.T, correction, self.sigma, self.lambd, self.muJ,  self.sigmaJ)
        return Fourier_inversion(self.S0, self.K, self.mu, self.T, MerJD_partial_chf, self.payoff)
    
    def MerJD_FFT(self, K, method= 'carr_madan', N = 2**12, B = 200 ): 
        m = self.lambd * (np.exp(self.muJ + (self.sigmaJ**2) / 2) - 1)
        correction = self.mu - 0.5 * self.sigma**2 - m
        MerJD_partial_chf = MerJD_chf.chf_MerJD_partial(self.T, correction, self.sigma, self.lambd, self.muJ,  self.sigmaJ)
        return FFT(self.S0, self.K, K, self.mu, self.T, 
                   MerJD_partial_chf, self.payoff, method = method, N = 2**12, B = 200, interp="cubic")

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