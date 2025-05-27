import numpy as np
import torch
import torch.nn as nn

from enums import OptionSide, OptionType
from .neural_net import LSMContinuationNN
from typing import Optional


def european_price(r: float, dt: float, N: int, payoff: np.ndarray) -> float:
    """
    Prices a European option using Monte Carlo — no regression needed.
    Only uses final payoffs at maturity.
    """
    discount_factor = np.exp(-r * dt * N)
    option_price = np.mean(payoff) * discount_factor

    return option_price


def lsm_global_fnn(S_paths: np.ndarray, K: float, r: float, dt: float, 
                   option_side: OptionSide, option_type: OptionType, 
                   exercise_points: Optional[np.ndarray], nn_layers: list, num_of_epochs: int) -> float:
    """
    This function creates only 1 global FNN trains the data on that then it makes its predictions
    """
    M, N_plus_1 = S_paths.shape
    N = N_plus_1 - 1
    d = 1 if S_paths.ndim == 2 else S_paths.shape[2] # multidimensions later

    # Step 1: Compute intrisct value
    if option_side == OptionSide.PUT:
        payoff = np.maximum(K - S_paths, 0)
    elif option_side == OptionSide.CALL:
        payoff = np.maximum(S_paths - K, 0)
    else:
        raise ValueError("Option must either be put or call")
    

    # Skip training and backward induction for European options — no early exercise allowed
    if option_type == OptionType.EUROPEAN:
        return european_price(r, dt, N, payoff)
    
    
    # Initialize cashflow at expiry
    cashflow = payoff[:, -1].copy()
    exercise_time = np.full(M, N)

    # Step 2: Collect training data (X = [S_t, t], Y = discounted cashflow)
    X_all = []
    Y_all = []

    for t in range(N - 1, 0, -1):
        alive = np.where(exercise_time > t)[0]
        if len(alive) == 0:
            continue

        itm_mask = payoff[alive, t] > 0
        if np.sum(itm_mask) == 0:
            continue
        itm_indices = alive[itm_mask]

        S_t = S_paths[itm_indices, t]
        t_norm = t / N

        # For 1D input: shape [n, 2]
        X_t = np.column_stack((S_t, np.full_like(S_t, t_norm)))
        Y_t = cashflow[itm_indices] * np.exp(-r * dt * (exercise_time[itm_indices] - t))

        X_all.append(X_t)
        Y_all.append(Y_t)

    X_all = np.vstack(X_all)
    Y_all = np.hstack(Y_all).reshape(-1, 1)

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_all, dtype=torch.float32)


    # Device to support gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to gpu if available
    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    # Step 3: Train global FNN
    model = LSMContinuationNN(X_all.shape[1], nn_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(num_of_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, Y_tensor)
        loss.backward()
        optimizer.step()


    # Step 4: Re-run backward induction using trained model
    cashflow = payoff[:, -1].copy()
    exercise_time = np.full(M, N)

    for t in range(N - 1, 0, -1):
        alive = np.where(exercise_time > t)[0]
        if len(alive) == 0:
            continue

        itm_mask = payoff[alive, t] > 0
        if np.sum(itm_mask) == 0:
            continue
        itm_indices = alive[itm_mask]

        S_t = S_paths[itm_indices, t]
        t_norm = t / N
        X_pred = torch.tensor(np.column_stack((S_t, np.full_like(S_t, t_norm))), dtype=torch.float32).to(device)

        with torch.no_grad():
            continuation_value = model(X_pred).squeeze().cpu().numpy()

        immediate_exercise = payoff[itm_indices, t]
        exercise_now = immediate_exercise > continuation_value

        exercise_indices = itm_indices[exercise_now]
        cashflow[exercise_indices] = immediate_exercise[exercise_now]
        exercise_time[exercise_indices] = t

    # Step 5: Discount to present
    option_values = cashflow * np.exp(-r * dt * exercise_time)
    return np.mean(option_values)





def lsm_local_fnn():
    """
    During the backwards induction step this function creates a local NN at each time step
    """