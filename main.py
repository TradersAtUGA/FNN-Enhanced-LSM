import time
import torch


import numpy as np
from config import Config
from enums import OptionType, ExerciseFrequency
from core import binomial_tree 
from core import generate_gbm_paths
from core import lsm_traditional
from core import lsm_global_fnn


def main():
    cfg = Config()
    
    # Paths used for LSM
    S_paths = generate_gbm_paths(
        S0=cfg.init_stock_price, 
        ir=cfg.drift, 
        sigma=cfg.volatility, 
        T=cfg.time_to_exp, 
        N=cfg.num_of_steps, 
        M=cfg.num_of_paths
    )

    binomial_price = binomial_tree(cfg.init_stock_price, cfg.strike_price, cfg.time_to_exp, cfg.risk_free_interest, cfg.volatility, cfg.num_of_steps, cfg.option_side, cfg.option_type, cfg.exercise_points)

    poly_price1 = lsm_traditional(S_paths, cfg.strike_price, cfg.risk_free_interest, 
                                  cfg.time_step, cfg.poly_degree, cfg.option_side, cfg.option_type, cfg.exercise_points)
    
    
    start_time = time.time()
    fnn_price = lsm_global_fnn(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, cfg.option_type, cfg.exercise_points, cfg.nn_layers, cfg.epochs)
    end_time = time.time()

    print(cfg.get_details())
    print(f"Binomial Tree Price: {binomial_price:.6f}")
    print(f"Poly LSM Price: {poly_price1:.6f}")
    print(f"Global FNN-Enhanced LSM Price: {fnn_price:6f}")
    print(f"Using {torch.cuda.get_device_name(0)}, took {end_time - start_time:.4f} seconds")
    

    # print(f"Binomial Tree took {end - start:.4f} seconds")
    # print(f"Poly Price 3-degree: {poly_price3}")
    # print(f"Poly Price 2-degree: {poly_price2}")
    # print(f"Poly Price 1-degree: {poly_price1}")
    # print(f"Global - FNN Price: {fnn_price}")


if __name__ == "__main__":
    main()