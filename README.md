# FNN-Enhanced LSM for American Option Pricing

This project explores a modern approach to pricing American-style options, which are challenging to value due to their early exercise features.

Rather than relying on traditional techniques, we integrated machine learning by replacing the standard regression model with a neural network. This shift enabled a more flexible and data-driven estimation of future payoffs, resulting in a more robust and accurate valuation framework.

To see findings and results check `Conclusions.pdf`

## Installation


1. Clone the repository  
   git clone https://github.com/your-username/your-repo-name.git  
   cd your-repo-name

2. Create a virtual environment  
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies  
   pip install -r requirements.txt

4. Run the project  
   python main.py

### Configurable Parameters

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
- `poly_degree`: Degree of polynomial regression used in the LSM algorithm
- `option_type`: Set the option type — use `OptionType.CALL` for a call option or `OptionType.PUT` for a put option