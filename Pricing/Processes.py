import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

class BS_diffusion:

    def __init__(self, r=0.01, sigma=0.02, fx=0, rd=0, rf=0):

        self.r = r
        self.rd = rd
        self.rf = rf
        self.fx = fx

        if sigma < 0:
            raise ValueError('invalid sigma parameter: must be non-negative')
        else:
            self.sigma = sigma

    def BS_proc(self, S0, T, N):
        W = st.norm.rvs((self.r - 0.5 * self.sigma**2) * T, np.sqrt(T) * self.sigma, N)
        S_T = S0 + np.exp(W)
        
        return S_T.reshape((N, 1))
    
class OU_process:

    def __init__(self, sigma=0.2, theta=0.1, kappa=0.1):
        
        if sigma < 0 or kappa < 0 or theta<0:
            raise ValueError("sigma, theta and kappa must be non negative")
        else:
            self.sigma = sigma
            self.theta = theta
            self.kappa = kappa

    def path(self, X0=2.5, T=5, N=2000, paths=10):

        T_vec, dt = np.linspace(0, T, N, retstep=True)
        X = np.zeros((N, paths))
        X[0, :] = X0
        W = st.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

        std_dt = np.sqrt(self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * dt)))

        for t in range(0, N - 1):
            X[t + 1, :] = self.theta + np.exp(-self.kappa * dt) * (X[t, :] - self.theta) + std_dt * W[t, :]

        plt.figure(figsize=(15, 5))
        plt.plot(T_vec, X, linewidth=0.5)
        plt.plot(T_vec, self.theta * np.ones_like(T_vec), label="Long term mean - Theta")
        plt.legend(loc="upper right")
        plt.title(f"{paths} OU processes")
        plt.xlabel("T")
        plt.legend()
        plt.grid(True)
        plt.show()

        #return X
    
class Schobel_Zhu_process: 

    def __init__(self, mu=0.1, sigma=0.2, theta=-0.1, kappa=0.1, rho=0):
        
        self.mu = mu

        if np.abs(rho) > 1:
            raise ValueError("|rho| must be <=1")
        self.rho = rho

        if sigma < 0 or kappa < 0 or theta<0:
            raise ValueError("sigma, theta and kappa must be non negative")
        else:
            self.sigma = sigma
            self.theta = theta
            self.kappa = kappa
        
    def path(self, S0, sigma0, T=5, N=2000, paths=10):

        MU = np.array([0, 0])
        COV = np.matrix([[1, self.rho], [self.rho, 1]])
        W = st.multivariate_normal.rvs(mean=MU, cov=COV, size=(N - 1, paths))
        W_S = W[:, :, 0] # Stock Brownian motion --> selects all rows (:) and all columns (:) from the first "plane" (0) of the 3D array.
        W_sigma = W[:, :, 1] # Volatility Brownian motion

        T_arr, dt = np.linspace(0, T, N, retstep=True)
        sqrt_dt = np.sqrt(dt)
        X = np.zeros((N, paths))
        X[0, :] = S0
        sig = np.zeros((N, paths))
        sig[0, :] = sigma0

        std_dt = np.sqrt(self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * dt)))

        for t in range(0, N - 1):
            sig[t+1,:] = self.theta + np.exp(-self.kappa * dt) * (sig[t,:] - self.theta) + std_dt * W_sigma[t,:]
            X[t+1, :] = X[t,:]+(self.mu-0.5*sig[t,:])*dt+sig[t,:]*sqrt_dt*W_S[t,:]

        #print(sig)
        
        fig = plt.figure(figsize=(16, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(T_arr, X)
        ax1.set_title(f"Schobel-Zhu - Underlying process ({paths} paths) ")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Underlying")
        ax2.plot(T_arr, sig)
        ax2.set_title(f"Schobel-Zhu - OU processes ({paths} paths)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Variance")
        ax2.plot(T_arr, self.theta * np.ones_like(T_arr), label="Long term mean - Theta")
        ax2.legend(loc="upper right")
        plt.grid(True)
        plt.show()

class CIR_process:

    def __init__(self, sigma=0.2, theta=0.1, kappa=0.1):
        
        if sigma < 0 or kappa < 0 or theta<0:
            raise ValueError("sigma, theta and kappa must be non negative")
        else:
            self.sigma = sigma
            self.theta = theta
            self.kappa = kappa

    def path(self, X0=2.5, T=5, N=2000, paths=10):

        T_vec, dt = np.linspace(0, T, N, retstep=True)
        X = np.zeros((N, paths))
        X[0, :] = X0
        W = st.norm.rvs(loc=0, scale=np.sqrt(dt), size=(N - 1, paths))

        for t in range(0, N - 1):
            X[t + 1, :] = np.abs(X[t, :] + self.kappa * (self.theta - X[t, :]) * dt + self.sigma * np.sqrt(X[t, :]) * W[t, :])

        print("Feller condition is: ", 2 * self.kappa * self.theta > self.sigma**2)

        plt.figure(figsize=(15, 5))
        plt.plot(T_vec, X, linewidth=0.5)
        plt.plot(T_vec, self.theta * np.ones_like(T_vec), label="Long term mean - Theta")
        plt.legend(loc="upper right")
        plt.title(f"{paths} CIR processes")
        plt.xlabel("T")
        plt.legend()
        plt.grid(True)
        plt.show()


        #return X
    
class Heston_process:

    def __init__(self, mu=0.1, rho=0, sigma=0.2, theta=-0.1, kappa=0.1, fx=0, rd=0, rf=0):
        self.mu = mu
        self.rd = rd
        self.rf = rf
        self.fx = fx
        if np.abs(rho) > 1:
            raise ValueError("|rho| must be <=1")
        self.rho = rho
        if theta < 0 or sigma < 0 or kappa < 0:
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.theta = theta
            self.sigma = sigma
            self.kappa = kappa

    def path(self, S0, v0, N=10000, paths = 5, T=1):

        std = np.sqrt(self.theta * self.sigma**2 / (2 * self.kappa))  # asymptotic standard deviation for the CIR process
        assert 2 * self.kappa * self.theta > self.sigma**2  # Feller condition

        # Generate random Brownian Motion
        MU = np.array([0, 0])
        COV = np.matrix([[1, self.rho], [self.rho, 1]])
        W = st.multivariate_normal.rvs(mean=MU, cov=COV, size=(N - 1, paths)) # Draw random samples from a multivariate normal distr.
        W_S = W[:, :, 0]  # Stock Brownian motion --> selects all rows (:) and all columns (:) from the first "plane" (0) of the 3D array.
        W_v = W[:, :, 1]  # Variance Brownian motion

        # Initialize vectors
        T_arr, dt = np.linspace(0, T, N, retstep=True) # time step array
        sqrt_dt = np.sqrt(dt) # square root of dt
        X0 = np.log(S0)  # log spot price --> NB: use log var and log st to avoid negative values for proces v
        Y0 = np.log(v0)  # log spot variance

        Y = np.zeros((N, paths))
        Y[0, :] = Y0
        X = np.zeros((N, paths))
        X[0, :] = X0
        v = np.zeros(N)

        # Generate paths
        for t in range(0, N - 1):
            v = np.exp(Y[t, :])  # variance
            sqrt_v = np.sqrt(v)  # square root of variance

            Y[t+1, :] = Y[t,:]+(1/v)*(self.kappa*(self.theta-v)-0.5*self.sigma**2)*dt+self.sigma*(1/sqrt_v)*sqrt_dt*W_v[t,:]
            X[t+1, :] = X[t,:]+(self.mu-0.5*v)*dt+sqrt_v*sqrt_dt*W_S[t,:]
        
        fig = plt.figure(figsize=(16, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(T_arr, np.exp(X))
        ax1.set_title("Heston - Underlying process (5 paths) ")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Underlying")
        ax2.plot(T_arr, np.exp(Y))
        ax2.set_title("Heston - Variance process (5 paths)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Variance")
        ax2.plot(T_arr, self.theta * np.ones_like(T_arr), label="Long term mean")
        ax2.legend(loc="upper right")
        plt.grid(True)
        plt.show()

class Merton_process:

    def __init__(self, mu=0.1, sigma=0.2, lambd=0.8, muJ=0, sigmaJ=0.5):
        self.mu = mu
        self.lambd = lambd
        self.muJ = muJ
        if sigma < 0 or sigmaJ < 0:
            raise ValueError("sigma and sigmaJ must be non negative")
        else:
            self.sigma = sigma
            self.sigmaJ = sigmaJ

        # moments
        self.variance = self.sigma**2 + self.lambd * self.sigmaJ**2 + self.lambd * self.muJ**2
        self.skew = self.lambd * (3 * self.sigmaJ**2 * self.muJ + self.muJ**3) / self.variance ** (1.5)
        self.kurt = self.lambd * (3 * self.sigmaJ**3 + 6 * self.sigmaJ**2 * self.muJ**2 + self.muJ**4) / self.variance**2
    
    # def path(self, S0, T=1, N=10000):
    #     m = self.lambd * (np.exp(self.muJ + (self.sigmaJ**2) / 2) - 1)  # coefficient m
    #     W = st.norm.rvs(0, 1, N)  # The normal RV vector
    #     P = st.poisson.rvs(self.lambd * T, size=N)  # Poisson random vector (number of jumps)
    #     Jumps = np.asarray([st.norm.rvs(self.muJ, self.sigmaJ, ind).sum() for ind in P])  # Jumps vector
    #     S_T = S0 * np.exp(
    #         (self.mu - 0.5 * self.sigma**2 - m) * T + np.sqrt(T) * self.sigma * W + Jumps)  # Martingale exponential Merton
    #     return S_T.reshape((N, 1))
    
    def path(self, S0, T=1, N=10000, paths = 5):
        dt = T / N 
        t = np.linspace(0, T, N+1)

        # Generate normal and Poisson random variables
        Z = np.random.normal(size=(paths, N)) * np.sqrt(dt)  # Standard Brownian motion increments
        P = np.random.poisson(self.lambd * dt, size=(paths, N))  # Poisson process for jump occurrence

        # Simulate Merton jump diffusion process
        X = np.zeros((paths, N+1))
        X[:, 0] = S0  # Initial value

        for i in range(N):
            Jumps = np.random.normal(self.muJ, self.sigmaJ, size=(paths,)) * P[:, i]  # Jump sizes
            X[:, i+1] = X[:, i] * (1 + (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * Z[:, i] + Jumps)
        for i in range(paths):
            plt.plot(t, X[i])

        plt.title('Merton Jump Diffusion - Underlyng (5 paths)')
        plt.xlabel('Time')
        plt.ylabel('Underlying')
        plt.grid(True)
        plt.show()
        
class Variance_Gamma_process:

    def __init__(self, mu=0.1, sigma=0.2, theta=-0.1, kappa=0.1):
        self.mu = mu
        self.theta = theta
        self.kappa = kappa
        if sigma < 0:
            print()
            raise ValueError("invalid sigma value ", sigma, ". sigma must be positive")
        else:
            self.sigma = sigma

        # moments
        self.mean = self.mu + self.theta
        self.var = self.sigma**2 + self.theta**2 * self.kappa
        self.skew = (2 * self.theta**3 * self.kappa**2 + 3 * self.sigma**2 * self.theta * self.kappa) / (
            self.var ** (1.5)
        )
        self.kurt = (
            3 * self.sigma**4 * self.kappa
            + 12 * self.sigma**2 * self.theta**2 * self.kappa**2
            + 6 * self.theta**4 * self.kappa**3
        ) / (self.var**2)

    def VG_process(self, S0, T, N):
        w = -np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma**2) / self.kappa  # coefficient w
        rho = 1 / self.kappa
        G = st.gamma(rho * T).rvs(N) / rho  # The gamma RV
        Norm = st.norm.rvs(0, 1, N)  # The normal RV
        VG = self.theta * G + self.sigma * np.sqrt(G) * Norm  # VG process at final time G
        S_T = S0 * np.exp((self.r - w) * T + VG)  # Martingale exponential VG
        return S_T.reshape((N, 1))

    def path(self, T=1, N=10000, paths=5):
        t = np.linspace(0, T, N)
        dt = T / (N - 1)  # time interval
        X0 = np.zeros((paths, 1))
        G = st.gamma(dt / self.kappa, scale=self.kappa).rvs(size=(paths, N - 1))  # The gamma RV
        Norm = st.norm.rvs(loc=0, scale=1, size=(paths, N - 1))  # The normal RV
        increments = self.mu * dt + self.theta * G + self.sigma * np.sqrt(G) * Norm
        X = np.concatenate((X0, increments), axis=1).cumsum(1)
        
        for i in range(paths):
            plt.plot(t, X[i])
        plt.title('Variance-Gamma - Underlyng (5 paths)')
        plt.xlabel('Time')
        plt.ylabel('Underlying')
        plt.grid(True)
        plt.show()