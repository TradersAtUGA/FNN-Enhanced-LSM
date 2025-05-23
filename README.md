# DeepBermuda

DeepBermuda is a neural network-enhanced implementation of the Longstaff-Schwartz Monte Carlo (LSM) method for pricing Bermudan options, with a focus on high-dimensional, over-the-counter (OTC) derivatives. This project extends traditional LSM by replacing polynomial regression with a feedforward neural network to more accurately estimate continuation values in complex multi-asset settings. It is specifically designed to handle basket-style Bermudan options with many underlyings, reflecting real-world OTC structures where early exercise occurs at discrete intervals. The goal is to explore the intersection of deep learning and advanced option pricing for flexible, high-dimensional derivatives.

To see findings and results check `Conclusions.pdf`

## Installation


1. Clone the repository  
   git clone https://github.com/TradersAtUGA/FNN-Enhanced-LSM.git   
   cd your-repo-name

2. Create a virtual environment  
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies  
   pip install -r requirements.txt

4. Run the project  
   python main.py

## Configurable Parameters

You can customize the simulation by editing the following variables in `main.py`:

Note: These settings are for American Options

- `num_of_paths`: Number of simulated paths (e.g. 100)
- `num_of_steps`: Number of time steps per path (e.g. 365 for daily)
- `time_to_exp`: Time to expiration (in years)
- `init_stock_price`: Initial stock price
- `drift`: Expected return (assumed to be equal to the risk-free rate under risk-neutral measure)
- `risk_free_interest`: Risk-free interest rate
- `volatility`: Annualized volatility of the stock
- `strike_price`: Strike price of the option
- `time_step`: Computed as `time_to_exp / num_of_steps`; represents time delta between steps
- `poly_degree`: Degree of polynomial regression used in the LSM algorithm
- `option_side`: Direction of the option — use `OptionSide.CALL` for a call or `OptionSide.PUT` for a put
- `option_type`: Set the option type — use `OptionType.AMERICAN` for American options or `OptionType.EUROPEAN` for European options
- `dimensions`: Number of underlying assets (1 = single asset, >1 = basket option)
- `nn_layers`: Feedforward neural network layer sizes based on `dimensions`
- `epochs`: Number of training epochs for the neural network
