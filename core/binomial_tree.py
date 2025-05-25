import numpy as np
from enums import OptionSide, OptionType

def binomial_tree(S0, K, T, r, sigma, N, option_side, option_type, exercise_points=None) -> float:
    """
    This binomial tree is accepts American, European and Bermudan options
    that are dependent on one underlying
    """

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))      # up factor
    d = 1 / u                             # down factor
    p = (np.exp(r * dt) - d) / (u - d)    # risk-neutral prob
    discount = np.exp(-r * dt)

    # Step 1: Compute stock prices at maturity
    stock_prices = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

    # Step 2: Compute option payoffs at maturity
    if option_side == OptionSide.PUT:
        # Creates an np array of options values where each entrie is the max(k - stock_prices[j], 0)
        option_values = np.maximum(K - stock_prices, 0)
    elif option_side == OptionSide.CALL:
        option_values = np.maximum(stock_prices - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")
    
    # Step 3: Backward induction
    for i in range(N - 1, -1, -1):
        option_values = discount * (p * option_values[1:i+2] + (1 - p) * option_values[0:i+1])

        stock_prices = stock_prices[0:i + 1] / d 

        # Early Exercise logic for american and bermudan options
        is_exercise_time = (
            (option_type == OptionType.AMERICAN) or
            (option_type == OptionType.BERMUDAN and exercise_points is not None and i in exercise_points)
        )

        if not is_exercise_time:
            continue

        if option_side == OptionSide.PUT:
            option_values = np.maximum(option_values, K - stock_prices)
        else:
            option_values = np.maximum(option_values, stock_prices - K)

    return option_values[0]

