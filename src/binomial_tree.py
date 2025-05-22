import numpy as np

def binomial_tree(S0, K, T, r, sigma, N, option_type):
    

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))      # up factor
    d = 1 / u                             # down factor
    p = (np.exp(r * dt) - d) / (u - d)    # risk-neutral prob
    discount = np.exp(-r * dt)

    # Step 1: Compute stock prices at maturity
    stock_prices = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

    # Step 2: Compute option payoffs at maturity
    if option_type.value == 'put':
        option_values = np.maximum(K - stock_prices, 0)
    elif option_type.value == 'call':
        option_values = np.maximum(stock_prices - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    # Step 3: Backward induction
    for i in range(N - 1, -1, -1):
        option_values = discount * (p * option_values[1:i+2] + (1 - p) * option_values[0:i+1])

        stock_prices = stock_prices[0:i+1] / u  # Move one step back in stock prices

        # For American options: early exercise condition
        if option_type == 'put':
            option_values = np.maximum(option_values, K - stock_prices)
        else:
            option_values = np.maximum(option_values, stock_prices - K)

    return option_values[0]
