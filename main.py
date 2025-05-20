import numpy as np
import matplotlib.pyplot as plt

from enum import Enum

from least_square_trad import lsm_traditional
from least_square_fnn import lsm_fnn
from binomial_tree import binomial_tree

class OptionType(Enum):
    PUT = "put"
    CALL = "call"

def generate_gbm_paths(S0, ir, sigma, T, N, M):
    dt = T / N
    S_paths = np.zeros((M, N + 1))
    S_paths[:, 0] = S0
    for i in range(1, N + 1):
        z = np.random.normal(0, 1, M)

        # brownian motion
        S_paths[:, i] = S_paths[:, i - 1] * np.exp((ir - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return S_paths


def main():
    num_of_paths = 100  # num of simulated paths
    num_of_steps = 365  # num of time steps
    time_to_exp = 1  # time to expiration
    init_stock_price = 100  # initial stock price
    drift = 0.0417  # drift (assumed to equal risk-free interest rate b/c black scholes assumes risk-neutral)
    risk_free_interest = 0.0417  # risk-free interest rate
    volatility = 0.2  # volatility
    strike_price = 110  # strike price
    time_step = time_to_exp / num_of_steps  # time step
    poly_degree = 3  # degree of polynomial regression
    option_type = OptionType.CALL 

    # Paths used for LSM
    S_paths = generate_gbm_paths()

    poly_price = lsm_traditional(S_paths)
    fnn_price = lsm_fnn(S_paths)

    binomial_price = binomial_tree() # Baseline for results


    # Uncomment this for plot of GBM paths
    # t_grid = np.linspace(0, T, N + 1)
    # plt.figure(figsize=(10, 6))
    # for i in range(M):
    #     plt.plot(t_grid, S_paths[i], lw=1)
    # plt.xlabel('Time (years)')
    # plt.ylabel('Stock Price')
    # plt.title('Simulated Stock Price Paths')

    # plt.show()

    

if __name__ == "__main__":
    main()