import numpy as np
import matplotlib.pyplot as plt

from enum import Enum

from least_square_trad import lsm_traditional
from least_square_fnn import lsm_global_fnn
from binomial_tree import binomial_tree

class OptionType(Enum):
    AMERICAN = 'american'
    BERMUDAN = 'bermudan'
    EUROPEAN = 'european'

class OptionSide(Enum):
    PUT = 'put'
    CALL = 'call'

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


def get_nn_sizes(d: int) -> int:
    """
    Returns hidden layer sizes based on the number of dimensions d.
    """
    if d <= 4:
        return [32, 16, 8]
    elif d <= 11:
        return [64, 32, 16]
    elif d <= 20:
        return [128, 64, 32]
    else:
        return [256, 128, 64]


def main():
    num_of_paths = 10_000  # number of simulated Monte Carlo paths
    num_of_steps = 2_000  # number of time steps in each path
    time_to_exp = 1  # time to expiration (in years)
    init_stock_price = 540  # initial stock price of the underlying asset
    drift = 0.0417  # drift (assumed to equal risk-free interest rate due to risk-neutral measure)
    risk_free_interest = 0.0417  # risk-free interest rate
    volatility = 0.2  # volatility of the underlying asset
    strike_price = 600  # strike price of the option
    time_step = time_to_exp / num_of_steps  # size of each time step
    poly_degree = 3  # degree of polynomial regression used in baseline LSM
    option_side = OptionSide.CALL  # whether it's a call or put option
    option_type = OptionType.AMERICAN  # whether it's an American or European option
    dimensions = 1  # number of underlying assets (1 = single asset)
    nn_layers = get_nn_sizes(dimensions)  # feedforward neural net layer sizes based on input dimension
    epochs = 300  # number of training epochs for the neural net



    OPTION_DETAILS = f"""
    Option Parameters:
    - Number of paths: {num_of_paths}
    - Number of steps: {num_of_steps}
    - Time to expiration: {time_to_exp}
    - Initial stock price: {init_stock_price}
    - Drift: {drift}
    - Risk-free interest rate: {risk_free_interest}
    - Volatility: {volatility}
    - Strike price: {strike_price}
    - Time step (dt): {time_step}
    - Option side: {option_side}
    - Option type: {option_type}
    - Dimensions: {dimensions}
    - Neural Network Layers: {nn_layers}
    - Epochs: {epochs}
    """


    # Paths used for LSM
    S_paths = generate_gbm_paths(init_stock_price, drift, volatility, time_to_exp, num_of_steps, num_of_paths)

    poly_price3 = lsm_traditional(S_paths, strike_price, risk_free_interest, time_step, option_side, 3)
    poly_price2 = lsm_traditional(S_paths, strike_price, risk_free_interest, time_step, option_side, 2)
    poly_price1 = lsm_traditional(S_paths, strike_price, risk_free_interest, time_step, option_side, 1)
    # fnn_price = lsm_fnn(S_paths)

    binomial_price = binomial_tree(init_stock_price, strike_price, time_to_exp, risk_free_interest, volatility, num_of_steps, option_side) # Baseline for results

    
    fnn_price = lsm_global_fnn(S_paths, strike_price, risk_free_interest, time_step, option_side, nn_layers, 300)
    
    print(OPTION_DETAILS)
    print(f"Binomial Price: {binomial_price}")
    print(f"Poly Price 3-degree: {poly_price3}")
    print(f"Poly Price 2-degree: {poly_price2}")
    print(f"Poly Price 1-degree: {poly_price1}")
    print(f"Global - FNN Price: {fnn_price}")


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