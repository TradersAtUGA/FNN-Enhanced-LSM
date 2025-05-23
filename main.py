import time

from config import Config
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

    # poly_price3 = lsm_traditional(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, 3)
    # poly_price2 = lsm_traditional(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, 2)
    # poly_price1 = lsm_traditional(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, 1)
    start = time.time()
    binomial_price = binomial_tree(
        cfg.init_stock_price, 
        cfg.strike_price, 
        cfg.time_to_exp, 
        cfg.risk_free_interest, 
        cfg.volatility,
        cfg.num_of_steps, 
        cfg.option_side
    )
    
    # fnn_price = lsm_global_fnn(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, cfg.nn_layers, cfg.epochs)
    end = time.time()

    print(cfg.get_details())
    print(f"Binomial Price: {binomial_price}")
    print(f"Binomial Tree took {end - start:.4f} seconds")
    # print(f"Poly Price 3-degree: {poly_price3}")
    # print(f"Poly Price 2-degree: {poly_price2}")
    # print(f"Poly Price 1-degree: {poly_price1}")
    # print(f"Global - FNN Price: {fnn_price}")


if __name__ == "__main__":
    main()