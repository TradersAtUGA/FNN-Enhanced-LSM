import time
import torch
import argparse


import numpy as np
from config import Config, load_config_from_yaml
from enums import OptionType, ExerciseFrequency
from core import binomial_tree 
from core import generate_gbm_paths, generate_multidim_gbm_paths
from core import lsm_traditional
from core import lsm_global_fnn



def get_args():
    parser = argparse.ArgumentParser(description="Run your model with custom configs")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    return parser.parse_args()

def main():
    # Load Config
    args = get_args()

    cfg = load_config_from_yaml(args.config)

    if cfg.dimensions == 1:
        S_paths = generate_gbm_paths(
            S0=cfg.init_stock_prices,
            ir=cfg.risk_free_interest,
            sigma=cfg.volatilities,
            T=cfg.time_to_exp,
            N=cfg.num_of_steps,
            M=cfg.num_of_paths
        )
        # Normalize shape of S_paths -> (paths, time_step, # of assets)
        S_paths = S_paths[:, :, np.newaxis]
    else:
        S_paths = generate_multidim_gbm_paths(
            S0=cfg.init_stock_prices,
            ir=cfg.risk_free_interest,
            sigma=cfg.volatilities,
            corr_matrix=cfg.correlation_matrix,
            T=cfg.time_to_exp,
            N=cfg.num_of_steps,
            M=cfg.num_of_paths
        )

    
    fnn_price = lsm_global_fnn(
        S_paths=S_paths, 
        K=cfg.strike_prices,
        r=cfg.risk_free_interest,
        dt=cfg.time_step,
        option_side=cfg.option_side,
        option_type=cfg.option_type,
        exercise_points=cfg.exercise_points,
        dim=cfg.dimensions,
        nn_layers=cfg.nn_layers,
        num_of_epochs=cfg.epochs
    )
    

    with open(f"{args.config}.txt", "w") as fptr:
        fptr.write(str(cfg))
        fptr.write(f"{fnn_price:6f}")


    # binomial_price = binomial_tree(cfg.init_stock_prices, cfg.strike_prices, cfg.time_to_exp, cfg.risk_free_interest, cfg.volatilities, cfg.num_of_steps, cfg.option_side, cfg.option_type, cfg.exercise_points)

    # poly_price1 = lsm_traditional(S_paths, cfg.strike_price, cfg.risk_free_interest, 
    #                               cfg.time_step, cfg.poly_degree, cfg.option_side, cfg.option_type, cfg.exercise_points)
    
    
    # start_time = time.time()
    # fnn_price = lsm_global_fnn(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, cfg.option_type, cfg.exercise_points, cfg.nn_layers, cfg.epochs)
    # end_time = time.time()

    # print(cfg.get_details())
    # print(f"Binomial Tree Price: {binomial_price}")
    # print(f"Poly LSM Price: {poly_price1:.6f}")
    # print(f"Global FNN-Enhanced LSM Price: {fnn_price:6f}")
    # print(f"Using {torch.cuda.get_device_name(0)}, took {end_time - start_time:.4f} seconds")
    

    # print(f"Binomial Tree took {end - start:.4f} seconds")
    # print(f"Poly Price 3-degree: {poly_price3}")
    # print(f"Poly Price 2-degree: {poly_price2}")
    # print(f"Poly Price 1-degree: {poly_price1}")
    # print(f"Global - FNN Price: {fnn_price}")


if __name__ == "__main__":
    main()