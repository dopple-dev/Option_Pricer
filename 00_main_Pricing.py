from NMFF.Pricing.Black_Scholes import Black_Scholes as BS 
from NMFF.Pricing.Parameters import Par 
from NMFF.Pricing.Processes import BS_diffusion
from NMFF.Pricing.Black_Scholes import  BS_main as bsm

from NMFF.Pricing.Schobel_Zhu import Schobel_Zhu as SZ
from NMFF.Pricing.Processes import Schobel_Zhu_process
from NMFF.Pricing.Schobel_Zhu import SZ_main as szm

from NMFF.Pricing.Heston import Heston_pricer as Hes
from NMFF.Pricing.Processes import Heston_process
from NMFF.Pricing.Heston import HE_main as hem

from NMFF.Pricing.Merton import Merton_pricer as Mer
from NMFF.Pricing.Processes import Merton_process
from NMFF.Pricing.Merton import ME_main as mem

from NMFF.Pricing.Variance_Gamma import VG_pricer as VG
from NMFF.Pricing.Processes import Variance_Gamma_process
from NMFF.Pricing.Variance_Gamma import VG_main as vgm

from NMFF.Pricing.SABR import SABR 

import numpy as np


def main():

    change_pricing_model = False
    while change_pricing_model == False:
        models = ['BS', 'ME', 'SZ', 'VG', 'HE', 'SB']
        print('Pricing models available (code: description):\n\n'
            '\'BS\': Black-Scholes,\n'
            '\'ME\': Merton,\n' 
            '\'SZ\': Schobel_Zhu,\n' 
            '\'VG\': Variance-Gamma,\n'
            '\'HE\': Heston,\n' 
            '\'SB\': SABR.\n')
        model = str(input("Enter model code: "))
        
        check = False
        while check == False:
            if model in models:
                check = True
                break
            else:
                model = str(input("Model code not recognized. Please enter valid code: "))

        print('Please provide information on the option you want to price.. ')
        S0 = float(input("Enter the underlying spot price (float): ")) 
        K_inp = input("Enter the option strike (float or list of floats separated by \',\'): ")
        strikes = list(map(float, K_inp.split(',')))
        T = float(input("Enter the option maturity in years (float): ")) 
        r = float(input("Enter the risk free rate (float): ")) 
        if model != 'SB':
            sigma = float(input("Enter the implied volatility (float): "))
        payoff = str(input("Enter option type (call or put): "))
 
        if model == 'BS':

            opt_param = Par(S0=S0, T=T, payoff=payoff, exercise_style='European')
            proc_param = BS_diffusion(r=r, sigma=sigma)
            BS_p = BS(opt_param, proc_param)

            change_pricing_technique = False
            while change_pricing_technique == False:
                ptcs = ['01','02','03','04','05','06']
                print('Schobel-Zhu model selected. Select the code of the desired pricing techniques among the following:\n\n'
                    '\'01\': Closed formula,\n'
                    '\'02\': Fourier inversion,\n' 
                    '\'03\': Fast Fourier Transformation - Lewis,\n' 
                    '\'04\': Fast Fourier Transformation - Carr-Madan,\n'
                    '\'05\': Binomial Tree,\n' 
                    '\'06\': PDE numerical solution.\n')

                pricing_tec = str(input("Enter pricing technique code: "))
                print()
                check = False
                while check == False:
                    if pricing_tec in ptcs:
                        check = True
                        break
                    else:
                        pricing_tec = str(input("Pricing code not recognized. Please enter valid code: "))

                if pricing_tec == '01':
                    bsm.closed_formula(BS_p, strikes)
                elif pricing_tec == '02':
                    bsm.f_transform(BS_p, strikes)
                elif pricing_tec == '03':
                    bsm.FFT_lew(BS_p, strikes)
                elif pricing_tec == '04':
                    bsm.FFT_carr(BS_p, strikes)
                elif pricing_tec == '05':
                    bsm.bin_tree(payoff, S0, T, r, sigma, strikes)
                else:
                    bsm.PDE(BS_p, strikes)
            
                price_tec_change = str(input("Would you like to use another pricing technique (Y/n): "))
                if price_tec_change == 'n':
                    change_pricing_technique = True
                elif price_tec_change == 'Y':
                    change_pricing_technique = False

        elif model == 'SZ':
            change_pricing_technique = False
            while change_pricing_technique == False:
                
                ptcs = ['01','02','03']
                print('Schobel-Zhu model selected. Select the code of the desired pricing techniques among the following:\n\n'
                    '\'01\': Fourier inversion,\n'
                    '\'02\': Fast Fourier Transformation - Lewis,\n' 
                    '\'03\': Fast Fourier Transformation - Carr-Madan.\n')

                pricing_tec = str(input("Enter pricing technique code: "))
                print()
                check = False
                while check == False:
                    if pricing_tec in ptcs:
                        check = True
                        break
                    else:
                        pricing_tec = str(input("Pricing code not recognized. Please enter valid code: "))
                    
                print()
                print('The model requires some other inputs:')

                rho = float(input("Enter correlation parameter rho (float): ")) 
                kappa = float(input("Enter mean reversion parameter kappa (float): ")) 
                gamma = float(input("Enter volatility of volatility gamma (float): ")) 
                theta = float(input("Enter long term mean parameter theta (float): ")) 

                opt_pars = Par(S0=S0, T=T, sigma0=sigma, payoff=payoff, exercise_style='European')
                SZ_pars = Schobel_Zhu_process(r, gamma, theta, kappa, rho)
                SZ_p = SZ(opt_pars, SZ_pars)

                if pricing_tec == '01':
                    szm.f_transform(SZ_p, strikes)
                elif pricing_tec == '02':
                    szm.FFT_lew(SZ_p, strikes)
                else:
                    szm.FFT_carr(SZ_p, strikes)
            
                price_tec_change = str(input("Would you like to use another pricing technique (Y/n): "))
                if price_tec_change == 'n':
                    change_pricing_technique = True
                elif price_tec_change == 'Y':
                    change_pricing_technique = False
                    
        elif model == 'HE':
            change_pricing_technique = False
            while change_pricing_technique == False:
                
                ptcs = ['01','02','03']
                print('Heston model selected. Select the code of the desired pricing techniques among the following:\n\n'
                    '\'01\': Fourier inversion,\n'
                    '\'02\': Fast Fourier Transformation - Lewis,\n' 
                    '\'03\': Fast Fourier Transformation - Carr-Madan.\n')

                pricing_tec = str(input("Enter pricing technique code: "))
                print()
                check = False
                while check == False:
                    if pricing_tec in ptcs:
                        check = True
                        break
                    else:
                        pricing_tec = str(input("Pricing code not recognized. Please enter valid code: "))
                    
                print()
                print('The model requires some other inputs:')

                rho = float(input("Enter correlation parameter rho (float): ")) 
                kappa = float(input("Enter mean reversion parameter kappa (float): ")) 
                nu = float(input("Enter volatility of volatility nu (float): ")) 
                theta = float(input("Enter long term mean parameter theta (float): ")) 

                opt_pars = Par(S0=S0, T=T, v0=sigma, payoff=payoff, exercise_style='European')
                Heston_pars = Heston_process(mu=r, rho=rho, sigma=nu, theta=theta, kappa=kappa)
                Hes_p = Hes(opt_pars, Heston_pars)

                if pricing_tec == '01':
                    hem.f_transform(Hes_p, strikes)
                elif pricing_tec == '02':
                    hem.FFT_lew(Hes_p, strikes)
                else:
                    hem.FFT_carr(Hes_p, strikes)
            
                price_tec_change = str(input("Would you like to use another pricing technique (Y/n): "))
                if price_tec_change == 'n':
                    change_pricing_technique = True
                elif price_tec_change == 'Y':
                    change_pricing_technique = False

        elif model == 'VG':
            change_pricing_technique = False
            while change_pricing_technique == False:
                
                ptcs = ['01','02','03']
                print('Heston model selected. Select the code of the desired pricing techniques among the following:\n\n'
                    '\'01\': Fourier inversion,\n'
                    '\'02\': Fast Fourier Transformation - Lewis,\n' 
                    '\'03\': Fast Fourier Transformation - Carr-Madan.\n')

                pricing_tec = str(input("Enter pricing technique code: "))
                print()
                check = False
                while check == False:
                    if pricing_tec in ptcs:
                        check = True
                        break
                    else:
                        pricing_tec = str(input("Pricing code not recognized. Please enter valid code: "))
                    
                print()
                print('The model requires some other inputs:')

                kappa = float(input("Enter mean reversion parameter kappa (float): ")) 
                theta = float(input("Enter long term mean parameter theta (float): ")) 

                opt_pars = Par(S0=S0, T=T,  payoff=payoff, exercise_style='European')
                VG_pars = Variance_Gamma_process(mu=r, theta=theta, sigma=sigma, kappa=kappa)
                VG_p = VG(opt_pars, VG_pars)

                if pricing_tec == '01':
                    vgm.f_transform(VG_p, strikes)
                elif pricing_tec == '02':
                    vgm.FFT_lew(VG_p, strikes)
                else:
                    vgm.FFT_carr(VG_p, strikes)
            
                price_tec_change = str(input("Would you like to use another pricing technique (Y/n): "))
                if price_tec_change == 'n':
                    change_pricing_technique = True
                elif price_tec_change == 'Y':
                    change_pricing_technique = False            
        
        elif model == 'SB':
            change_pricing_technique = False
            while change_pricing_technique == False:
                
                ptcs = ['01']
                print('Heston model selected. Select the code of the desired pricing techniques among the following:\n\n'
                    '\'01\': Closed formula,\n')

                pricing_tec = str(input("Enter pricing technique code: "))
                print()
                check = False
                while check == False:
                    if pricing_tec in ptcs:
                        check = True
                        break
                    else:
                        pricing_tec = str(input("Pricing code not recognized. Please enter valid code: "))
                    
                print()
                print('The model requires some other inputs:')

                beta = float(input("Enter SABR beta parameter (float): ")) 
                rho = float(input("Enter SABR rho parameter (float): ")) 
                nu = float(input("Enter SABR nu paramter (float): ")) 
                alpha = float(input("Enter SABR alpha paramter (float): ")) 

                f = S0*np.exp(r*T)
                if len(strikes) == 1:
                    sigma_SABR  = SABR.Hagan_vol_float(strikes,T,f,alpha,beta,rho,nu )
                else:
                    sigma_SABR  = SABR.SABR_vol_arr(strikes,T,f,alpha,beta,rho,nu )
                opt_param = Par(S0=S0, T=T, payoff=payoff, exercise_style='European')
                BS_proc_param_SABR = BS_diffusion(r=r, sigma=sigma_SABR[0])
                BS_SABR_p = BS(opt_param, BS_proc_param_SABR) 

                BS_SABR_p.BS_pricer_self()
            
                price_tec_change = str(input("Would you like to use another pricing technique (Y/n): "))
                if price_tec_change == 'n':
                    change_pricing_technique = True
                elif price_tec_change == 'Y':
                    change_pricing_technique = False
       
        elif model == 'ME':
            
            change_pricing_technique = False
            while change_pricing_technique == False:
                
                ptcs = ['01','02','03','04']
                print('Heston model selected. Select the code of the desired pricing techniques among the following:\n\n'
                    '\'01\': Closed formula,\n'
                    '\'02\': Fourier inversion,\n'
                    '\'03\': Fast Fourier Transformation - Lewis,\n' 
                    '\'04\': Fast Fourier Transformation - Carr-Madan.\n')

                pricing_tec = str(input("Enter pricing technique code: "))
                print()
                check = False
                while check == False:
                    if pricing_tec in ptcs:
                        check = True
                        break
                    else:
                        pricing_tec = str(input("Pricing code not recognized. Please enter valid code: "))
                    
                print()
                print('The model requires some other inputs:')

                lambd = float(input("Enter lambda parameter (float): ")) 
                muJ = float(input("Enter jump mean parameter (float): ")) 
                sigmaJ = float(input("Enter jump volatility paramter (float): ")) 

                opt_pars = Par(S0=S0, T=T, payoff=payoff, exercise_style='European')
                MerJD_pars = Merton_process(mu=r, sigma=sigma, lambd=lambd, muJ=muJ, sigmaJ=sigmaJ)
                ME_p = Mer(opt_pars, MerJD_pars)

                if pricing_tec == '01':
                    mem.closed_formula(ME_p, strikes)
                elif pricing_tec == '02':
                    mem.f_transform(ME_p, strikes)
                elif pricing_tec == '03':
                    mem.FFT_lew(ME_p, strikes)
                else:
                    mem.FFT_carr(ME_p, strikes) 
            
                price_tec_change = str(input("Would you like to use another pricing technique (Y/n): "))
                if price_tec_change == 'n':
                    change_pricing_technique = True
                elif price_tec_change == 'Y':
                    change_pricing_technique = False
        
        price_mod_change = str(input("Would you like to use another pricing model (Y/n): "))
        if price_mod_change == 'n':
            change_pricing_model = True
            print('Program ended')
            break
        elif price_mod_change == 'Y':
            change_pricing_model = False
        else:
            break

main() 