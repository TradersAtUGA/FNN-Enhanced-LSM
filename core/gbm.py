import numpy as np
import matplotlib.pyplot as plt

def generate_gbm_paths(S0, ir, sigma, T, N, M):
    """
    Generates geometric bronwian motion stock paths for a single asset

    This function simulates M paths over N time steps using the GBM formula
    This is used later in the LSM

    Args:
        S0 (float): Intial stock price
        ir (float): risk-free interest rate (drift term)
        sigma (float): Volatility of the underlying
        T (float): Total time to maturity (in years)
        N (int): Number of discrete time steps
        M (int): Number of simulated paths

    Returns:
        np.ndarray: A 2d numpy array of shape (M, N+1) where each row is a path, each column is a time step, and array[i][j] == a price at the given time
    """

    dt = T / N # Time increment per step
    S_paths = np.zeros((M, N + 1)) # Init path matrix with all 0
    S_paths[:, 0] = S0 # Init all paths with the init price

    for i in range(1, N + 1):
        z = np.random.normal(0, 1, M) # Sample std normal noise foreach path

        # Apply GBM equation S_t+1 = S_t * exp((mu - 0.5sigma^2)*dt + sigma * root(dt) * Z)
        S_paths[:, i] = S_paths[:, i - 1] * np.exp(
            (ir - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )

    return S_paths



def generate_multidim_gbm_paths(S0, ir, sigma, corr_matrix, T, N, M):
    """
    Generates multi-dimensional geometric bronwian motion paths for correlated assets

    Each asset follows its own GBM process, but the brownian motion are correlated using a Cholesky decomposiotn of the correlation matrix. 
    This helps to great paths for basket options, rainbow options, and other multi-asset derivatives

    Args:
        S0 (np.ndarray): Init prices of each asset. Shape: (D,)
        ir (np.ndarry): risk-free interest rate (drift term) of each asset. Shape: (D,)
        sigma (np.ndarry): Volatility of each asset. Shape: (D,)
        corr_matrix (np.ndarry): Correlation matrix between assets. Shape: (D,D)
        T (float): Total time to maturity (in years)
        N (int): Number of discrete time steps
        M (int): Number of simulated paths

    Returns:
        np.ndarry: A 3d numpy array of shape (M, N + 1, D). Each path is a matrix
            of shape (N + 1, D), where D is the number of assets
    """
    D = len(S0) # Number of assets
    dt = T / N # Time increment

    # Cholseky decomposition for correlation
    L = np.linalg.cholesky(corr_matrix)

    # Sample standard normals: shape (M, N, D)
    Z = np.random.normal(size=(M, N, D))
    # Corrolate Z to L transposed
    Z = Z @ L.T

    # Initialize paths (M simulations, N+1 steps, D assets)
    paths = np.zeros((M, N + 1, D))
    paths[:, 0, :] = S0 # Set init prices for each asset

    for t in range(1, N + 1):
        # Drift and diffusion components for each asset
        drift = (ir - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z[:, t - 1, :]

        # Apply GBM equation to all paths for all assets
        paths[:, t, :] = paths[:, t - 1, :] * np.exp(drift + diffusion)

    return paths


D = 3  # Number of assets
M = 1000
N = 250
T = 1.0
dt = T / N

S0 = np.array([100, 90, 110])
mu = np.array([0.05, 0.03, 0.04])
sigma = np.array([0.2, 0.15, 0.25])
corr = np.array([
    [1.0, 0.8, 0.2],
    [0.8, 1.0, 0.4],
    [0.2, 0.4, 1.0]
])

paths = generate_multidim_gbm_paths(S0, mu, sigma, corr, T, N, M)

# Average path for each asset
mean_paths = paths.mean(axis=0)  # Shape: (N+1, D)

# Plot
plt.figure(figsize=(10, 6))
for d in range(D):
    plt.plot(mean_paths[:, d], label=f"Asset {d+1}")

plt.title("Average GBM Path for Each Asset")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))

for d in range(D):
    for i in range(M):
        plt.plot(paths[i, :, d], lw=0.5, alpha=0.4)

plt.title("All GBM Paths for All Assets")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.grid(True)
plt.show()