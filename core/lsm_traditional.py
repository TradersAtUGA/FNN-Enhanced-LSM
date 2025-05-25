import numpy as np
from enums import OptionType, OptionSide
from typing import Optional

def should_exercise_early(t: int, option_style: OptionType, excercise_pts: Optional[np.ndarray]) -> bool:
    if option_style == OptionType.AMERICAN:
        return True
    elif option_style == OptionType.BERMUDAN and excercise_pts is not None:
        return t in excercise_pts
    return False


def lsm_traditional(S_paths: np.ndarray, K: float, r: float, dt: float, poly_degree: int, 
                    option_side: OptionSide, option_type: OptionType, 
                    exercise_points: Optional[np.ndarray]) -> float:
    
    M, N_plus_1 = S_paths.shape
    N = N_plus_1 - 1

    # find payoff of each path
    if option_side == OptionSide.PUT:
        payoff = np.maximum(K - S_paths, 0)
    elif option_side == OptionSide.CALL:
        payoff = np.maximum(S_paths - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    # payoff at expiration
    cashflow = payoff[:, -1].copy()

    # time index on exercise
    exercise_time = np.full(M, N)

    # backwards induction
    for t in range(N - 1, 0, -1):
        if not should_exercise_early(t, option_type, exercise_points):
            continue

        # exclude paths that have alr been exercised
        alive = np.where(exercise_time > t)[0]
        if len(alive) == 0:
            continue

        # only include itm paths for regression
        itm_mask = payoff[alive, t] > 0
        if np.sum(itm_mask) == 0:
            continue

        itm_indices = alive[itm_mask]

        # future discounted cash flows
        Y = cashflow[itm_indices] * np.exp(-r * dt * (exercise_time[itm_indices] - t))

        # current asset prices (itm only)
        X = S_paths[itm_indices, t]

        # ran into issues when there were too few asset prices in the money
        if len(X) < 4:
            continue

        # regress future discounted cash flows onto asset price at present w/ polynomial regression
        # This is the part of the algo that has been swapped out for NN
        coeffs = np.polyfit(X, Y, poly_degree)
        continuation_value = np.polyval(coeffs, X)

        # value of option at present
        immediate_exercise = payoff[itm_indices, t]

        # exercise if immediate value is greater than continuation value
        exercise_now = immediate_exercise > continuation_value

        # update where early exercise is optimal
        exercise_indices = itm_indices[exercise_now]
        cashflow[exercise_indices] = payoff[exercise_indices, t]
        exercise_time[exercise_indices] = t

    # discount cash flows from exercise to t=0
    option_values = cashflow * np.exp(-r * dt * exercise_time)
    option_price = np.mean(option_values)

    return option_price