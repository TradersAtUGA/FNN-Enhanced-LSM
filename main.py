import time

import numpy as np
from config import Config
from enums import OptionType, ExerciseFrequency
from core import binomial_tree 
from core import generate_gbm_paths
from core import lsm_traditional
from core import lsm_global_fnn


def main():
    cfg = Config()
    cfg1 = Config(option_type=OptionType.BERMUDAN, exercise_frequency=ExerciseFrequency.CUSTOM, custom_exercise_points=np.array([167,333,500]))
    cfg2 = Config(option_type=OptionType.EUROPEAN)

    
    
    # Paths used for LSM
    S_paths = generate_gbm_paths(
        S0=cfg.init_stock_price, 
        ir=cfg.drift, 
        sigma=cfg.volatility, 
        T=cfg.time_to_exp, 
        N=cfg.num_of_steps, 
        M=cfg.num_of_paths
    )

    poly_price1 = lsm_traditional(S_paths, cfg.strike_price, cfg.risk_free_interest, 
                                  cfg.time_step, cfg.poly_degree, cfg.option_side, cfg.option_type, cfg.exercise_points)
    
    poly_price2 = lsm_traditional(S_paths, cfg1.strike_price, cfg1.risk_free_interest, 
                                  cfg1.time_step, cfg1.poly_degree, cfg1.option_side, cfg1.option_type, cfg1.exercise_points)

    poly_price3 = lsm_traditional(S_paths, cfg2.strike_price, cfg2.risk_free_interest, 
                                  cfg2.time_step, cfg2.poly_degree, cfg2.option_side, cfg2.option_type, cfg2.exercise_points)

    print(cfg.get_details())
    print(f"Poly Price: {poly_price1}")
    print(cfg1.get_details())
    print(f"Poly Price: {poly_price2}")
    print(cfg2.get_details())
    print(f"Poly Price: {poly_price3}")
    # print(f"Binomial Tree took {end - start:.4f} seconds")
    # print(f"Poly Price 3-degree: {poly_price3}")
    # print(f"Poly Price 2-degree: {poly_price2}")
    # print(f"Poly Price 1-degree: {poly_price1}")
    # print(f"Global - FNN Price: {fnn_price}")


if __name__ == "__main__":
    main()