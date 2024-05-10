import numpy as np 
from scipy.integrate import quad
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d

    
def Q1(k, cf, right_lim, sz=0):
    '''
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration'''
    if sz == 1:
        integrand = lambda u: np.real((np.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.000000000001j)) 
        return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]
    else: 
        integrand = lambda u: np.real((np.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.00000000000001j))

        return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]
        

def Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """

    integrand = lambda u: np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))

    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]

def Fourier_inversion(S0=1, K=1, r=0, T=1, partial_chf=0, payoff='call', sz=0):
    """
    Price obtained by inversion of the characteristic function
    """
    k = np.log(K / S0)

    if payoff == 'call' or payoff == 'c' or payoff == 'Call' or payoff == 'C' or payoff == 1:
        call = S0 * Q1(k, partial_chf, np.inf, sz = sz) - K * np.exp(-r * T) * Q2( k, partial_chf, np.inf )
        return call
    elif payoff == 'put' or payoff == 'p' or payoff == 'Put' or payoff == 'P' or payoff == -1:
        put = K * np.exp(-r * T) * (1 - Q2(k, partial_chf, np.inf)) - S0 * (1 - Q1(k, partial_chf, np.inf, sz = sz))
        return put
    else:
        raise ValueError('invalid payoff type.')
    
def FFT(S0, K, KK, r, T, partial_chf, payoff, method = 'carr_madan', N = 2**12, B = 200, interp="cubic"):

        if method == 'lewis':
            return FFT_Lewis(KK, S0, r, T, partial_chf, N = 2**12, B = 200, interp="cubic")
        elif method == 'carr_madan':
            return FFT_Carr_Madan(KK, S0, r, T, partial_chf, N = 2**12, B = 200, interp="cubic")
        else: 
            raise ValueError("invalid method type (set 'lewis' or 'carr_madan')")

def FFT_Lewis(K, S0, r, T, cf, N = 2**12, B = 200, interp="cubic"):

    #N = FFT more efficient for N power of 2
    # B = integration limit
    dx = B / N
    x = np.arange(N) * dx  # the final value B is excluded

    weight = np.arange(N)  # Simpson weights
    weight = 3 + (-1) ** (weight + 1)
    weight[0] = 1
    weight[N - 1] = 1

    dk = 2 * np.pi / B
    b = N * dk / 2
    ks = -b + dk * np.arange(N)

    integrand = np.exp(-1j * b * np.arange(N) * dx) * cf(x - 0.5j) * 1 / (x**2 + 0.25) * weight * dx / 3
    integral_value = np.real(ifft(integrand) * N)

    if interp == "linear":
        spline_lin = interp1d(ks, integral_value, kind="linear")
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * spline_lin(np.log(S0 / K))
    elif interp == "cubic":
        spline_cub = interp1d(ks, integral_value, kind="cubic")
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * spline_cub(np.log(S0 / K))
    return prices

def FFT_Carr_Madan(K, S0, r, T, cf, N = 2**12, B = 200, interp="cubic"):

    alpha = 0.75
    # N = FFT more efficient for N power of 2
    # B = integration limit
    dx = B / N
    x = np.arange(N) * dx  
    dk = (2 * np.pi) / (N*dx)
    b = 0.5 * N * dk 
    ks = -b + dk * np.arange(N)

    weight=dx / 3 * (3+(-1)**(np.arange(1,N+1))-np.eye(1,N))

    psi = cf(x-(alpha+1)*1j)*np.exp(-r*T) / (alpha**2 + alpha - x**2 + 1j*(2*alpha+1)*x) 
    psi_hat = fft(np.exp(-1j * b * x) * psi * weight)
    integral_value = S0*np.real((np.exp(-alpha*ks) / np.pi) * psi_hat)
    
    if interp == "linear":
        spline_lin = interp1d(S0*np.exp(ks), integral_value, kind="linear")
        prices = spline_lin(K)
    elif interp == "cubic":
        spline_cub = interp1d(S0*np.exp(ks), integral_value, kind="cubic")
        prices =  spline_cub(K)

    return prices