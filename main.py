import time

from config import Config
from enums import OptionType, ExerciseFrequency
from core import binomial_tree 
from core import generate_gbm_paths
from core import lsm_traditional
from core import lsm_global_fnn


def main():
    cfg = Config()
    cfg1 = Config()
    cfg2 = Config()
    cfg1.option_type = OptionType.BERMUDAN
    cfg1.exercise_frequency = ExerciseFrequency.MONTHLY
    cfg1.exercise_points = cfg1.get_excercise_points()
    cfg2.option_type = OptionType.EUROPEAN

    
    
    # Paths used for LSM
    # S_paths = generate_gbm_paths(
    #     S0=cfg.init_stock_price, 
    #     ir=cfg.drift, 
    #     sigma=cfg.volatility, 
    #     T=cfg.time_to_exp, 
    #     N=cfg.num_of_steps, 
    #     M=cfg.num_of_paths
    # )

    # poly_price3 = lsm_traditional(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, 3)
    # poly_price2 = lsm_traditional(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, 2)
    # poly_price1 = lsm_traditional(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, 1)
    # start = time.time()
    binomial_price = binomial_tree(
        cfg.init_stock_price, 
        cfg.strike_price, 
        cfg.time_to_exp, 
        cfg.risk_free_interest, 
        cfg.volatility,
        cfg.num_of_steps, 
        cfg.option_side,
        cfg.option_type
    )

    binomial_price1 = binomial_tree(
        cfg1.init_stock_price, 
        cfg1.strike_price, 
        cfg1.time_to_exp, 
        cfg1.risk_free_interest, 
        cfg1.volatility,
        cfg1.num_of_steps, 
        cfg1.option_side,
        cfg1.option_type,
        cfg1.exercise_points
    )

    binomial_price2 = binomial_tree(
        cfg2.init_stock_price, 
        cfg2.strike_price, 
        cfg2.time_to_exp, 
        cfg2.risk_free_interest, 
        cfg2.volatility,
        cfg2.num_of_steps, 
        cfg2.option_side,
        cfg2.option_type,
    )
    
    # fnn_price = lsm_global_fnn(S_paths, cfg.strike_price, cfg.risk_free_interest, cfg.time_step, cfg.option_side, cfg.nn_layers, cfg.epochs)
    # end = time.time()

    print(cfg.get_details())
    print(f"Binomial Price: {binomial_price}")
    print(cfg1.get_details())
    print(f"Binomial Price: {binomial_price1}")
    print(cfg2.get_details())
    print(f"Binomial Price: {binomial_price2}")
    # print(f"Binomial Tree took {end - start:.4f} seconds")
    # print(f"Poly Price 3-degree: {poly_price3}")
    # print(f"Poly Price 2-degree: {poly_price2}")
    # print(f"Poly Price 1-degree: {poly_price1}")
    # print(f"Global - FNN Price: {fnn_price}")


if __name__ == "__main__":
    main()