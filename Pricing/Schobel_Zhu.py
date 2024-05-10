from NMFF.Fourier.Fourier import FFT, Fourier_inversion
from NMFF.Chf_PDF.Charateristic_functions import Hes_chf
from NMFF.Chf_PDF.Charateristic_functions import SZ_chf



import numpy as np
import scipy.stats as st
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings

class Schobel_Zhu: 

    def __init__(self, Option_info, Process_info):
        
        self.mu = Process_info.mu
        self.S0 = Option_info.S0 # underlying 
        self.sigma0 = Option_info.sigma0  # spot variance
        self.K = Option_info.K # strike 
        self.T = Option_info.T # tenor 
        self.payoff = Option_info.payoff # payoff 
        self.exercise_style = Option_info.exercise_style

        self.sigma = Process_info.sigma # pars
        self.theta = Process_info.theta  
        self.kappa = Process_info.kappa  
        self.rho = Process_info.rho 

    def get_payoff(self, S): # obtain payoff function 
        if (self.payoff == 'call' or self.payoff == 'c' or self.payoff == 'Call' or self.payoff == 'C'):
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == 'put' or self.payoff == 'p' or self.payoff == 'Put' or self.payoff == 'P':
            Payoff = np.maximum(self.K - S, 0)
        return Payoff
        
    def SZ_F_inv(self):
        SZ_partial_chf = SZ_chf.chf_Schobel_Zhu_partial(self.T, self.sigma0, self.mu, self.kappa, self.theta, self.sigma, self.rho)
        return Fourier_inversion(self.S0, self.K, self.mu, self.T, SZ_partial_chf, self.payoff, sz=1)

    def SZ_FFT(self, K, method= 'carr_madan', N = 2**12, B = 200 ): 
        SZ_partial_chf = SZ_chf.chf_Schobel_Zhu_partial(self.T, self.sigma0, self.mu, self.kappa, self.theta, self.sigma, self.rho)
        return FFT(self.S0, self.K, K, self.mu, self.T, 
                   SZ_partial_chf, self.payoff, method = method, N = 2**12, B = 200, interp="cubic")
    
    def plot_SZ_chf(self, u1=np.linspace(0,100,1000)):

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        print('FIGURE 1: T, sigma0, mu, kappa, theta, sigma, rho = ', 
              self.T, self.sigma0, self.mu, self.kappa, self.theta, self.sigma, self.rho)
        plt.figure(1)
        ax = plt.axes(projection='3d')
        cf = SZ_chf.chf_Schobel_Zhu(u1, self.T, self.sigma0, self.mu, self.kappa, self.theta, self.sigma, self.rho)
        x = np.real(cf)
        y = np.imag(cf)
        ax.plot3D(u1, x, y, 'blue')
        ax.view_init(30, -120)

        T = 15
        mu = 8.0 
        theta = 0.1  
        sigma = 1.0
        kappa = 1.0
        sigma0 = 0.04
        rho = -0.3
        u2=np.linspace(0,2,1000)

        print('FIGURE 2: T, sigma0, mu, kappa, theta, sigma, rho = ', 
              T, sigma0, mu, kappa, theta, sigma, rho)
        plt.figure(2)  
        ax = plt.axes(projection='3d')
        chf = SZ_chf.chf_Schobel_Zhu(u2, T, sigma0, mu, kappa, theta, sigma, rho)
        x = np.real(chf)
        y = np.imag(chf)
        ax.plot3D(u2, x, y, 'blue')
        ax.view_init(30, -120)

