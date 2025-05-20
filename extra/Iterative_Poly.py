import numpy as np


# ------------------------------------------
# 1. Asset Price Path Simulation (GBM)
# ------------------------------------------
def generate_gbm_paths(S0, r, sigma, T, N, M):
    """
    Generate M GBM paths over N time steps.
    S0: initial asset price.
    r: risk-free rate (risk-neutral drift).
    sigma: volatility.
    T: time to maturity (years).
    N: number of time steps.
    M: number of paths.
    Returns: an array of shape (M, N+1).
    """
    dt = T / N
    paths = np.zeros((M, N + 1))
    paths[:, 0] = S0
    for i in range(1, N + 1):
        z = np.random.normal(0, 1, M)
        paths[:, i] = paths[:, i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return paths


# ------------------------------------------
# 2. Payoff Function
# ------------------------------------------
def payoff(S, K, option_type='put'):
    """
    American option payoff.
    For a put: max(K - S, 0)
    For a call: max(S - K, 0)
    """
    if option_type == 'put':
        return np.maximum(K - S, 0)
    elif option_type == 'call':
        return np.maximum(S - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")


# ------------------------------------------
# 3. Design Matrix and Regression Functions
# ------------------------------------------
def build_design_matrix(X):
    """
    Given an array X of shape (n_samples, 2) where each row is [t, S],
    build a design matrix for a quadratic polynomial:
      [1, t, S, t^2, t*S, S^2]
    """
    t = X[:, 0].reshape(-1, 1)
    S = X[:, 1].reshape(-1, 1)
    ones = np.ones_like(t)
    t2 = t ** 2
    tS = t * S
    S2 = S ** 2
    return np.hstack((ones, t, S, t2, tS, S2))


def regression_fit(X, y):
    """
    Solve the least-squares problem to fit a quadratic function.
    Returns the coefficient vector beta (length 6).
    """
    X_design = build_design_matrix(X)
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    return beta


def evaluate_polynomial(t, S, beta):
    """
    Given scalar t and S and regression coefficients beta,
    evaluate the quadratic approximator:
      f(t, S) = beta0 + beta1*t + beta2*S + beta3*t^2 + beta4*t*S + beta5*S^2.
    """
    return (beta[0] + beta[1] * t + beta[2] * S + beta[3] * t ** 2 +
            beta[4] * t * S + beta[5] * S ** 2)


# ------------------------------------------
# 4. Iterative Algorithm Functions
# ------------------------------------------
def construct_stopping_times(paths, beta, r, dt, K, option_type):
    """
    For each simulated path, use the current approximator f(t,S)
    to decide when to exercise.
    The stopping rule is: at time t, exercise if payoff(S_t) >= f(t, S_t).
    Returns an array of stopping times (one per path).
    """
    M, N_plus_1 = paths.shape
    N = N_plus_1 - 1
    stopping_times = np.full(M, N, dtype=int)  # default: exercise at maturity
    for i in range(M):
        for t in range(N):
            S_t = paths[i, t]
            if payoff(S_t, K, option_type) >= evaluate_polynomial(t, S_t, beta):
                stopping_times[i] = t
                break
    return stopping_times


def build_regression_data(paths, stopping_times, r, dt, K, option_type):
    """
    For each path and for each time step t up to the stopping time tau,
    record the feature [t, S_t] and target value equal to the discounted payoff
    from exercising at time tau.

    Returns:
      X: array of features (each row [t, S_t])
      y: array of target values.
    """
    M, N_plus_1 = paths.shape
    N = N_plus_1 - 1
    gamma = np.exp(-r * dt)
    X_list = []
    y_list = []
    for i in range(M):
        tau = stopping_times[i]
        exercise_payoff = payoff(paths[i, tau], K, option_type)
        for t in range(tau):
            discount = gamma ** (tau - t)
            X_list.append([t, paths[i, t]])
            y_list.append(discount * exercise_payoff)
        # Also add the point at the stopping time (discount factor = 1)
        X_list.append([tau, paths[i, tau]])
        y_list.append(exercise_payoff)
    return np.array(X_list), np.array(y_list)


def iterative_algorithm(S0, r, sigma, T, N, M, K, option_type, num_iter, tol):
    """
    Implements the iterative algorithm:
      - At each iteration, generate M new sample paths.
      - Compute stopping times using the current approximator f.
      - Build regression data (features and discounted targets).
      - Fit a quadratic regression to update beta.
      - Iterate until the change in beta is below tol.

    Returns:
      beta_final: the final regression coefficient vector.
    """
    dt = T / N
    # Initialize with a function that always returns a high value (forcing exercise at maturity).
    # For example, set f(t,S) = C where C is a very high constant.
    beta_current = np.array([1e6, 0, 0, 0, 0, 0])

    for it in range(num_iter):
        paths = generate_gbm_paths(S0, r, sigma, T, N, M)
        stopping_times = construct_stopping_times(paths, beta_current, r, dt, K, option_type)
        X, y = build_regression_data(paths, stopping_times, r, dt, K, option_type)
        beta_new = regression_fit(X, y)
        diff = np.linalg.norm(beta_new - beta_current)
        print(f"Iteration {it + 1}: beta = {beta_new}, change = {diff:.4f}")
        if diff < tol:
            beta_current = beta_new
            break
        beta_current = beta_new
    return beta_current


def estimate_option_price(beta, S0, r, sigma, T, N, M_est, K, option_type):
    """
    Estimate the option price using the final stopping rule defined by beta.
    Generate M_est new sample paths and, for each, exercise at the time when
    payoff >= f(t,S). Discount the payoff to t=0 and average over paths.
    """
    dt = T / N
    gamma = np.exp(-r * dt)
    paths = generate_gbm_paths(S0, r, sigma, T, N, M_est)
    M, _ = paths.shape
    prices = np.zeros(M)
    for i in range(M):
        for t in range(N):
            S_t = paths[i, t]
            if payoff(S_t, K, option_type) >= evaluate_polynomial(t, S_t, beta):
                tau = t
                break
        else:
            tau = N
        prices[i] = payoff(paths[i, tau], K, option_type) * (gamma ** tau)
    return np.mean(prices)


# ------------------------------------------
# 5. Main Execution
# ------------------------------------------
if __name__ == "__main__":
    # Parameters
    S0 = 100  # initial asset price
    r = 0.05  # risk-free rate
    sigma = 0.2  # volatility
    T = 1.0  # time to maturity in years
    N = 50  # number of exercise dates (discrete times)
    M = 10000  # number of paths per iteration
    K = 105  # strike price
    option_type = 'put'  # pricing an American put option
    num_iter = 10  # maximum number of iterations
    tol = 1e-3  # tolerance for convergence

    print("Running iterative algorithm for American option pricing...")
    beta_final = iterative_algorithm(S0, r, sigma, T, N, M, K, option_type, num_iter, tol)
    print("\nFinal regression coefficients (beta):", beta_final)

    M_est = 50000  # number of paths for price estimation
    price_estimate = estimate_option_price(beta_final, S0, r, sigma, T, N, M_est, K, option_type)
    print("Estimated American option price (lower bound):", price_estimate)