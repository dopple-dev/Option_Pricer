from NMFF.Fourier.Fourier import FFT, Fourier_inversion
from NMFF.Chf_PDF.Charateristic_functions import Hes_chf

import numpy as np
import scipy.stats as st
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings

class Heston_pricer: 

    def __init__(self, Option_info, Process_info):
        
        self.mu = Process_info.mu
        self.rd = Process_info.rd
        self.rf = Process_info.rf
        self.fx = Process_info.fx
        self.S0 = Option_info.S0 # underlying 
        self.v0 = Option_info.v0  # spot variance
        self.K = Option_info.K # strike 
        self.T = Option_info.T # tenor 
        self.payoff = Option_info.payoff # payoff 
        self.exercise_style = Option_info.exercise_style

        self.sigma = Process_info.sigma # Heston pars
        self.theta = Process_info.theta  
        self.kappa = Process_info.kappa  
        self.rho = Process_info.rho 

    def get_payoff(self, S): # obtain payoff function 
        if (self.payoff == 'call' or self.payoff == 'c' or self.payoff == 'Call' or self.payoff == 'C'):
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == 'put' or self.payoff == 'p' or self.payoff == 'Put' or self.payoff == 'P':
            Payoff = np.maximum(self.K - S, 0)
        return Payoff
        
    def Hes_F_inv(self):
        Hes_partial_chf = Hes_chf.chf_Heston_Schoutens_partial(self.T, self.v0, self.mu, self.kappa, self.theta, self.sigma, self.rho, self.fx, self.rd, self.rf)
        return Fourier_inversion(self.S0, self.K, self.mu, self.T, Hes_partial_chf, self.payoff, self.fx, self.rd, self.rf)

    def Hes_FFT(self, K, method= 'lewis', N = 2**12, B = 200 ): 
        Hes_partial_chf = Hes_chf.chf_Heston_Schoutens_partial(self.T, self.v0, self.mu, self.kappa, self.theta, self.sigma, self.rho, self.fx, self.rd, self.rf)
        return FFT(self.S0, self.K, K, self.mu, self.T, Hes_partial_chf, self.payoff, method=method, fx=self.fx, rd=self.rd, rf=self.rf)
    
    def plot_HES_chf(self, u1=np.linspace(0,100,1000)):

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        print('FIGURE 1: T, v0, mu, kappa, theta, sigma, rho = ', 
              self.T, self.v0, self.mu, self.kappa, self.theta, self.sigma, self.rho)
        plt.figure(1)
        ax = plt.axes(projection='3d')
        cf = Hes_chf.chf_Heston_Schoutens(u1, self.T, self.v0, self.mu, self.kappa, self.theta, self.sigma, self.rho)
        x = np.real(cf)
        y = np.imag(cf)
        ax.plot3D(u1, x, y, 'blue')
        ax.view_init(30, -120)

        mu = 0.05  
        rho = -0.8  
        kappa = 3  
        theta = 0.1  
        mu = 8.0
        sigma = 1.0
        v0 = 0.08 
        T = 15
        u2=np.linspace(0,3.1,1000)

        print('FIGURE 2: T, v0, mu, kappa, theta, sigma, rho = ', 
              T, v0, mu, kappa, theta, sigma, rho)
        plt.figure(2)  
        ax = plt.axes(projection='3d')
        chf = Hes_chf.chf_Heston_Schoutens(u2, T, v0, mu, kappa, theta, sigma, rho)
        x = np.real(chf)
        y = np.imag(chf)
        ax.plot3D(u2, x, y, 'blue')
        ax.view_init(30, -120)