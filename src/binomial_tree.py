import numpy as np


def binomial_tree(S0, K, T, r, sigma, N, option_side) -> float:
    """
    This binomial tree is the baseline for american options that are dependent on one underlying
    """

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))      # up factor
    d = 1 / u                             # down factor
    p = (np.exp(r * dt) - d) / (u - d)    # risk-neutral prob
    discount = np.exp(-r * dt)

    # Step 1: Compute stock prices at maturity
    stock_prices = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

    # Step 2: Compute option payoffs at maturity
    if option_side.value == 'put':
        # Creates an np array of options values where each entrie is the max(k - stock_prices[j], 0)
        option_values = np.maximum(K - stock_prices, 0)
    elif option_side.value == 'call':
        option_values = np.maximum(stock_prices - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")
    
    # Step 3: Backward induction
    for i in range(N - 1, -1, -1):
        option_values = discount * (p * option_values[1:i+2] + (1 - p) * option_values[0:i+1])

        # stock_prices = stock_prices[0:i+1] / u  # Move one step back in stock prices
        stock_prices = stock_prices[0:i + 1] / d 

        # For American options: early exercise condition
        if option_side.value == 'put':
            option_values = np.maximum(option_values, K - stock_prices)
        else:
            option_values = np.maximum(option_values, stock_prices - K)

    return option_values[0]

