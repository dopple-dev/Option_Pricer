import numpy as np
from functools import partial
import matplotlib as plt

class BS_chf:

    def chf_normal(u, mu=1, sigma=2):
        return np.exp(1j * u * mu - 0.5 * u**2 * sigma**2)

    def chf_GBM(u, sigma, r, T ):
                return BS_chf.chf_normal(u, mu=(r - 0.5 * sigma**2) * T, sigma=sigma * np.sqrt(T))
                
    def chf_GBM_partial(sigma, r, T):
        return partial(BS_chf.chf_normal, mu=(r - 0.5 * sigma**2) * T, sigma=sigma * np.sqrt(T))

class SZ_chf:

    def chf_Schobel_Zhu(u, t, sigma0, mu, kappa, theta, sigma, rho):
        
        alpha = -0.5 * u*(1j+u)
        beta = 2*(kappa - rho*sigma*u*1j )
        gamma = 2*sigma**2
        D = np.sqrt(beta**2- 4* alpha*gamma)
        G = (beta - D)/(beta + D)
        B = kappa * theta * (beta-D)/(D*sigma**2) * ((1-np.exp(-0.5*D*t))**2 )/(1-G*np.exp(-D*t))
        A = ((beta-D)*kappa**2*theta**2)/(2*(D**3) * sigma**2) * (
            beta * (D*t-4)+(D*(D*t-2)) +
            (4*np.exp(-0.5*D*t)*((D**2-2*beta**2)/(beta+D) * np.exp(-0.5*D*t) + 2*beta)) / 
            (1-G*np.exp(-D*t))
        )
        cf = Hes_chf.chf_Heston_Schoutens(u,t,sigma0**2,mu,2*kappa,sigma**2/2*kappa,2*sigma,rho) * (np.exp(A+B*sigma0))
        return cf 
    
    def chf_Schobel_Zhu_partial(T, sigma0, r, kappa, theta, sigma, rho):
        return partial(SZ_chf.chf_Schobel_Zhu, t=T, sigma0=sigma0, mu=r, theta=theta, sigma=sigma, kappa=kappa, rho=rho)
    
class Hes_chf:

    def chf_Heston_Schoutens(u, t, v0, mu=0, kappa=0, theta=0, sigma=0, rho=0, fx=0, rd=0, rf=0):
        
        if fx == 1:
            d2 = np.sqrt((rho*sigma*u*1j - kappa)**2 - sigma**2 * (-u*1j-u**2))
            g2 = (kappa - rho*sigma*u*1j -d2) / (kappa - rho*sigma*1j + d2)
            C2 = (rd-rf)*u*1j*t + (kappa*theta)/sigma**2 * ((kappa - rho*sigma*u*1j - d2)*t - 2*np.log((1-g2*np.exp(-d2 * t))/(1 -g2)))
            D2 = (kappa - rho*sigma*u*1j - d2)/ sigma**2 * ((1-np.exp(-d2*t))/(1 - g2*np.exp(-d2*t)))
            cf = np.exp(C2+D2*v0+1j*u)
            return cf
        else: 
            xi = kappa - sigma * rho * u * 1j
            d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
            g1 = (xi + d) / (xi - d)
            g2 = 1 / g1
            cf = np.exp(
                1j * u * mu * t
                + (kappa * theta) / (sigma**2) * ((xi - d) * t - 2 * np.log((1 - g2 * np.exp(-d * t)) / (1 - g2)))
                + (v0 / sigma**2) * (xi - d) * (1 - np.exp(-d * t)) / (1 - g2 * np.exp(-d * t))
            )
            return cf

    def chf_Heston_Schoutens_partial(T, v0, r=0, kappa=0, theta=0, sigma=0, rho=0, fx=0, rd=0, rf=0):
        return partial(Hes_chf.chf_Heston_Schoutens, t=T, v0=v0, mu=r, theta=theta, sigma=sigma, kappa=kappa, rho=rho,
                       fx=fx, rd=rd, rf=rf)
    
class MerJD_chf:

     def chf_MerJD(u, t, mu, sigma, lambd, muJ, sigmaJ):
          cf = np.exp(t * (1j * u * mu - 0.5 * u**2 * sigma**2 + lambd * (np.exp(1j * u * muJ - 0.5 * u**2 * sigmaJ**2) - 1)))
          return cf 
     
     def chf_MerJD_partial(T, mu, sigma, lambd, muJ, sigmaJ):
          return partial(MerJD_chf.chf_MerJD, t=T, mu=mu, sigma=sigma, lambd=lambd, muJ=muJ, sigmaJ=sigmaJ)
     
class VG_chf:
   
   def chf_gamma(u, a=1, b=2):
    return (1 - b * u * 1j) ** (-a)
   
   def chf_VG(u, t, mu, theta, sigma, kappa):
    return np.exp(t * (1j * mu * u - np.log(1 - 1j * theta * kappa * u + 0.5 * kappa * sigma**2 * u**2) / kappa))
   
   def chf_VG_partial(T, mu, theta, sigma, kappa):
       return partial(VG_chf.chf_VG, t=T, mu=mu, theta=theta, sigma=sigma, kappa=kappa)