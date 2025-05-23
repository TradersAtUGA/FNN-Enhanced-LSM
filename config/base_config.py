from montecarlo.enums import OptionSide, OptionType
from montecarlo.core import get_nn_sizes


class Config:
    # Option Configs
    option_side = OptionSide.CALL
    option_type = OptionType.AMERICAN
    time_to_exp = 1
    init_stock_price = 540
    drift = 0.0417
    risk_free_interest = 0.0417
    volatility = 0.2
    strike_price = 600

    # LSM Configs
    poly_degree = 3
    num_of_paths = 10_000
    num_of_steps = 2_000
    time_step = time_to_exp / num_of_steps

    # LSM-FNN Configs
    dimensions = 1
    nn_layers = get_nn_sizes(dimensions)
    epochs = 300

    @classmethod
    def get_details(cls):
        return f"""
        Option Parameters:
        - Number of paths: {cls.num_of_paths}
        - Number of steps: {cls.num_of_steps}
        - Time to expiration: {cls.time_to_exp}
        - Initial stock price: {cls.init_stock_price}
        - Drift: {cls.drift}
        - Risk-free interest rate: {cls.risk_free_interest}
        - Volatility: {cls.volatility}
        - Strike price: {cls.strike_price}
        - Time step (dt): {cls.time_step}
        - Option side: {cls.option_side}
        - Option type: {cls.option_type}
        - Dimensions: {cls.dimensions}
        - Neural Network Layers: {cls.nn_layers}
        - Epochs: {cls.epochs}
        """