from NMFF.Pricing.Black_Scholes import Black_Scholes as BS 

import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad

class implied_vol:

    def __init__(self, S0, K, T, payoff, r, price, method):

        self.S0 = S0 # underlying 
        self.K = K # strike 
        self.T = T # tenor 
        self.payoff = payoff # payoff 
        self.r = r # interest rate 
        self.price = price
        self.method = method

    def iv_NM(price, S0, K, T, r=0, fx=0, rd=0, rf=0, payoff="call", method="fsolve"):
        if fx==0:
            if method == 'brent':
                obj_f = lambda sigma: price - BS.BS_pricer(payoff=payoff, S0=S0, K=K, T=T, r=r, sigma=sigma)

                x, r = opt.brentq(obj_f, a=1e-15, b=500, full_output=True)
                if r.converged == True:
                    return x
                
            elif method == "fsolve":
                obj_f = lambda sigma: price - BS.BS_pricer(payoff=payoff, S0=S0, K=K, T=T, r=r, sigma=sigma)
                X0 = [0.1, 0.5, 1, 3]  # set of initial guess points
                for x0 in X0:
                    x, _, solved, _ = opt.fsolve(obj_f, x0, full_output=True, xtol=1e-8)
                    if solved == 1:
                        x = x[0]
                        return x
            
            elif method == 'minimize': 
                n = 2  # must be even
                def obj_fun(vol):
                    return (BS.BS_pricer(payoff=payoff, S0=S0, K=K, T=T, r=r, sigma=vol) - price) ** n

                res = opt.minimize_scalar(obj_fun, bounds=(1e-15, 8), method="bounded")
                if res.success == True:
                    x = res.x
                    return x
        elif fx==1:
            # method == 'minimize'
            n = 2  # must be even
            def obj_fun(vol):
                return (BS.BS_pricer(payoff, S0, K, T, r=0, sigma=vol, fx=1, rd=rd, rf=rf) - price) ** n

            res = opt.minimize_scalar(obj_fun, bounds=(1e-15, 8), method="bounded")
            if res.success == True:
                x = res.x
                return x
            
    def IV_Gatheral(K, S0, T, r, chf):
        k = np.log(S0 / K)

        def obj_fun(sigma):
            integrand = (
                lambda u: 1 / (u**2 + 0.25) * np.real( np.exp(- u * k * 1j) * (
                        chf(u - 0.5*1j) - np.exp(1j * u * r * T + 0.5 * r * T) * np.exp(-0.5 * T * (u**2 + 0.25) * sigma**2))) )
            int_value = quad(integrand, 1e-15, 2000, limit=2000, full_output=1)[0]
            return int_value

        X0 = [0.2, 1, 2, 4, 0.0001]  # set of initial guess points
        for x0 in X0:
            x, _, solved, msg = opt.fsolve(obj_fun,[x0],full_output=True,xtol=1e-4)
            if solved == 1:
                return x[0]
            else:
                print('Error occured')