import numpy as np


def generate_gbm_paths(S0, ir, sigma, T, N, M):
    """
    Generates geometric bronwian motion stock paths for LSM
    """
    dt = T / N
    S_paths = np.zeros((M, N + 1))
    S_paths[:, 0] = S0
    for i in range(1, N + 1):
        z = np.random.normal(0, 1, M)

        # brownian motion
        S_paths[:, i] = S_paths[:, i - 1] * np.exp((ir - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return S_paths