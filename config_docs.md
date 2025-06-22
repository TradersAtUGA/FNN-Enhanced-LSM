# Config File Documentation

This document outlines the structure and valid values for the `config.yaml` file. Each section below corresponds to a configuration parameter.

---

## `option_type`
- **Type**: `string`  
- **Options**:
  - `"AMERICAN"` — Option can be exercised at any time before expiration.
  - `"EUROPEAN"` — Option can be exercised only at expiration.
  - `"BERMUDAN"` — Option can be exercised at predefined points (see `exercise_frequency`).
- **Default**: `"BERMUDAN"`

---

## `option_side`
- **Type**: `string`  
- **Options**:
  - `"PUT"` — Grants the right to sell an asset at the strike price.
  - `"CALL"` — Grants the right to buy an asset at the strike price.
- **Default**: `"PUT"`

---

## `dimensions`
- **Type**: `int`  
- Specifies the number of underlying assets to price. Put 1 for vanilla options
- **Default**: `5`

---

## `risk_free_interest`
- **Type**: `float`  
- **Symbol**: `r`  
- In a risk-neutral framework, the risk-free rate also serves as the expected return (`μ`).  
- **Default**: `0.05`

---

## `time_to_exp`
- **Type**: `float`  
- **Symbol**: `T`  
- Time to expiration, measured in years.
- **Default**: `0.25`

---

## `exercise_frequency`
- **Type**: `string`  
- Only applicable if `option_type = "BERMUDAN"`.
- **Options**:
  - `"MONTHLY"`
  - `"QUARTERLY"`
  - `"SEMI_MONTHLY"`
  - `"CUSTOM"` — Requires the `custom_exercise_points` parameter.
- **Default**: `"MONTHLY"`

---

## `exercise_points`
- **Type**: `Array[int]`  
- Only applicable if `option_type = "BERMUDAN"` and `exercise_frequency = "CUSTOM"`.  
- Specifies custom exercise times **in days**.  
- **Example**: `[1, 50, 100, 300]` — each value represents a day on which early exercise is allowed.


---

## `init_stock_prices`
- **Type**: `Array[float]`  
- **Symbol**: `[S₀₁, S₀₂, ..., S₀_d]`  
- Initial stock prices for each asset.  
- Array length must equal `dimensions`.
- **Default**: `[80, 82, 78, 81, 83]`

---

## `strike_prices`
- **Type**: `Array[float]`  
- **Symbol**: `[K₁, K₂, ..., K_d]`  
- Strike price for each asset.  
- Array length must equal `dimensions`.
- **Default**: `[110, 110, 110, 110, 110]`

---

## `volatilities`
- **Type**: `Array[float]`  
- **Symbol**: `[σ₁, σ₂, ..., σ_d]`  
- Volatility of each asset.  
- Array length must equal `dimensions`.
- **Default**: `[0.1, 0.12, 0.09, 0.11, 0.1]`

---

## `num_of_paths`
- **Type**: `int`  
- **Symbol**: `M`  
- Number of Monte Carlo simulation paths.
- **Default**: `10000`

---

## `num_of_steps`
- **Type**: `int`  
- **Symbol**: `N`  
- Number of discrete time steps used in each path.
- **Default**: `250`

---

## `poly_degree`
- **Type**: `int`  
- Degree of the polynomial used in the regression step of the Longstaff-Schwartz method (LSM).
- **Default**: `3`

---

## `epochs`
- **Type**: `int`  
- Number of training epochs for the feedforward neural network (FNN) when used.
- **Default**: `300`
