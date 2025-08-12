import numpy as np
import torch

from enums import OptionSide, OptionType
from .neural_net import LSMContinuationNN
from typing import Optional

def should_exercise_early(t: int, option_style: OptionType, exercise_pts: Optional[np.ndarray]) -> bool:
    if option_style == OptionType.AMERICAN:
        return True
    if option_style == OptionType.BERMUDAN and exercise_pts is not None:
        return t in exercise_pts
    return False


def european_price(r: float, dt: float, N: int, payoff: np.ndarray) -> float:
    """
    Prices a European option using Monte Carlo — no regression needed.
    Only uses final payoffs at maturity.
    """
    discount_factor = np.exp(-r * dt * N)
    option_price = np.mean(payoff) * discount_factor

    return option_price


def compute_intrisic_val(S_paths: np.ndarray, K: float, option_side: OptionSide, d: int) -> np.ndarray:
    K_scaler = K.sum()
    basket_price = S_paths.sum(axis=2)
    if option_side == OptionSide.PUT:
        payoff = np.maximum(K_scaler - basket_price, 0)
    elif option_side == OptionSide.CALL:
        payoff = np.maximum(basket_price - K_scaler, 0)
    else:
        raise ValueError("Option must either be put or call")
    
    return payoff


def collect_training_data(S_paths: np.ndarray, payoff: np.ndarray, N: int, M: int, r: float, dt: float, option_type: OptionType, exercise_points: np.ndarray):
    # Init cashflow and exercise times
    cashflow = payoff[:, -1].copy()
    exercise_time = np.full(M, N)

    X_all, Y_all = [], []

    # Induct backwards
    for t in range(N - 1, 0, -1):
        if not should_exercise_early(t, option_type, exercise_points):
            continue

        alive = np.where(exercise_time > t)[0]
        if alive.size == 0:
            continue

        # In the money mask
        itm_mask = payoff[alive, t] > 0
        if not np.any(itm_mask):
            continue
        itm_indices = alive[itm_mask]

        # Shape: (num_alive, d)
        S_t = S_paths[itm_indices, t]

        # Normalized time column
        t_col = np.full((S_t.shape[0], 1), t / N)

        # Features: all asset prices + time
        X_t = np.hstack((S_t, t_col))
        
        # Discounted continuation values
        Y_t = cashflow[itm_indices] * np.exp(-r * dt * (exercise_time[itm_indices] - t))

        X_all.append(X_t)
        Y_all.append(Y_t)

    # Stack all collected data
    X_all = np.vstack(X_all)
    Y_all = np.hstack(Y_all).reshape(-1, 1)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_all, dtype=torch.float32)

    return (X_tensor, Y_tensor, X_all.shape[1])


def calculate_option_values(
        S_paths: np.ndarray, payoff: np.ndarray, 
        N: int, M: int, r: float, dt: float, 
        model: LSMContinuationNN, device: str,
        option_type: OptionType, exercise_points: np.ndarray
    ) -> np.ndarray:

    # Init cashflow and exercise times
    cashflow = payoff[:, -1].copy()
    exercise_time = np.full(M, N)

    # Induct backwards
    for t in range(N - 1, 0, -1):
        if not should_exercise_early(t, option_type, exercise_points):
            continue

        alive = np.where(exercise_time > t)[0]
        if alive.size == 0:
            continue
        
        # In the money mask
        itm_mask = payoff[alive, t] > 0
        if not np.any(itm_mask):
            continue
        itm_indices = alive[itm_mask]

        # Shape: (num_alive, d)
        S_t = S_paths[itm_indices, t]
        
        # Normalize time column
        t_col = np.full((S_t.shape[0], 1), t / N)

        # Concate asset prices and time
        X_pred = torch.tensor(np.column_stack((S_t, t_col)), dtype=torch.float32).to(device)

        with torch.no_grad():
            continuation_value = model(X_pred).squeeze().cpu().numpy()

        immediate_exercise = payoff[itm_indices, t]
        exercise_now = immediate_exercise > continuation_value

        exercise_indices = itm_indices[exercise_now]
        cashflow[exercise_indices] = immediate_exercise[exercise_now]
        exercise_time[exercise_indices] = t
    
    # Discount to present
    return cashflow * np.exp(-r * dt * exercise_time)

def lsm_global_fnn(S_paths: np.ndarray, K: float, r: float, dt: float, 
                   option_side: OptionSide, option_type: OptionType, 
                   exercise_points: Optional[np.ndarray], dim:int, nn_layers: list, num_of_epochs: int) -> float:
    """
    This function creates only 1 global FNN trains the data on that then it makes its predictions
    """
    M, N_plus_1 = S_paths.shape[:2]
    N = N_plus_1 - 1
    

    # Step 1: Compute intrisct value
    payoff = compute_intrisic_val(S_paths, K, option_side, dim)
    

    # Skip training and backward induction for European options, no early exercise allowed
    if option_type == OptionType.EUROPEAN:
        return european_price(r, dt, N, payoff)
    

    # Step 2: Collect training data (X = [S_t, t], Y = discounted cashflow)
    X_tensor, Y_tensor, X_all_shape = collect_training_data(S_paths, payoff, N, M, r, dt, option_type, exercise_points)


    # Device to support gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to gpu if available
    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    # Step 3: Train global FNN
    model = LSMContinuationNN(X_all_shape, nn_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(num_of_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, Y_tensor)
        loss.backward()
        optimizer.step()


    # Step 4: Re-run backward induction using trained model
    option_values = calculate_option_values(S_paths, payoff, N, M, r, dt, model, device, option_type, exercise_points)

    return np.mean(option_values)



def lsm_local_fnn():
    """
    During the backwards induction step this function creates a local NN at each time step
    """