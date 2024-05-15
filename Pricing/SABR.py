import numpy as np
import scipy.optimize as opt

class SABR:

    def SABR_vol_arr(K,T,f,alpha,beta,rho,gamma):

        z        = gamma/alpha*np.power(f*K,(1.0-beta)/2.0)*np.log(f/K);
        x_z      = np.log((np.sqrt(1.0-2.0*rho*z+z*z)+z-rho)/(1.0-rho))
        A        = alpha/(np.power(f*K,((1.0-beta)/2.0))*(1.0+np.power(1.0-beta,2.0)/24.0*
                                np.power(np.log(f/K),2.0)+np.power((1.0-beta),4.0)/1920.0*
                                np.power(np.log(f/K),4.0)))
        B1       = 1.0 + (np.power((1.0-beta),2.0)/24.0*alpha*alpha/(np.power((f*K),
                    1-beta))+1/4*(rho*beta*gamma*alpha)/(np.power((f*K),
                                ((1.0-beta)/2.0)))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T
        impVol   = A*(z/x_z) * B1

        B2 = 1.0 + (np.power(1.0-beta,2.0)/24.0*alpha*alpha/
                    (np.power(f,2.0-2.0*beta))+1.0/4.0*(rho*beta*gamma*
                    alpha)/np.power(f,(1.0-beta))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T;

        # Special treatment for ATM strike price

        impVol[np.where(K==f)] = alpha / np.power(f,(1-beta)) * B2;

        return impVol
    
    def Hagan_vol_float(K,T,f,alpha,beta,rho,gamma):
        
        z        = gamma/alpha*np.power(f*K,(1.0-beta)/2.0)*np.log(f/K)
        x_z      = np.log((np.sqrt(1.0-2.0*rho*z+z*z)+z-rho)/(1.0-rho))
        A        = alpha/(np.power(f*K,((1.0-beta)/2.0))*(1.0+np.power(1.0-beta,2.0)/24.0*
                                np.power(np.log(f/K),2.0)+np.power((1.0-beta),4.0)/1920.0*
                                np.power(np.log(f/K),4.0)))
        B1       = 1.0 + (np.power((1.0-beta),2.0)/24.0*alpha*alpha/(np.power((f*K),
                    1-beta))+1/4*(rho*beta*gamma*alpha)/(np.power((f*K),
                                ((1.0-beta)/2.0)))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T
        impVol   = A*(z/x_z) * B1
        B2 = 1.0 + (np.power(1.0-beta,2.0)/24.0*alpha*alpha/
                    (np.power(f,2.0-2.0*beta))+1.0/4.0*(rho*beta*gamma*
                    alpha)/np.power(f,(1.0-beta))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T

        # Special treatment of ATM strike value
        if K == f:

            impVol = alpha / np.power(f,(1-beta)) * B2
        return impVol
    
    def DetermineOptimalAlpha(iv_ATM, K_ATM,t,f,beta,rho,gamma):
            target = lambda alpha: SABR.Hagan_vol(K_ATM,t,f,alpha,beta,rho,gamma)-iv_ATM
            alpha_est = opt.newton(target,1.05,tol=0.0000001)
            return alpha_est