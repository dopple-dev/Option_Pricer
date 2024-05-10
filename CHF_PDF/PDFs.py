import numpy as np 
from math import factorial
import scipy.special as scsp

class PDF_s:

    def Merton_pdf(x, T, mu, sig, lam, muJ, sigJ):
        """
        Merton density function
        """
        tot = 0
        for k in range(20):
            tot += (
                (lam * T) ** k
                * np.exp(-((x - mu * T - k * muJ) ** 2) / (2 * (T * sig**2 + k * sigJ**2)))
                / (factorial(k) * np.sqrt(2 * np.pi * (sig**2 * T + k * sigJ**2)))
            )
        return np.exp(-lam * T) * tot
    
def VG_pdf(x, T, c, theta, sigma, kappa):
    return (
        2 * np.exp(theta * (x - c) / sigma**2)
        / (kappa ** (T / kappa) * np.sqrt(2 * np.pi) * sigma * scsp.gamma(T / kappa))
        * ((x - c) ** 2 / (2 * sigma**2 / kappa + theta**2)) ** (T / (2 * kappa) - 1 / 4)
        * scsp.kv(
            T / kappa - 1 / 2,
            sigma ** (-2) * np.sqrt((x - c) ** 2 * (2 * sigma**2 / kappa + theta**2)),
        )
    )