from typing import Optional, List
import numpy as np
from dataclasses import dataclass, field
from enums import OptionSide, OptionType, ExerciseFrequency
from core import get_nn_sizes


@dataclass
class Config:
    """
    Class that controls all the params for options
    """
    # === Option Configs ===
    option_type: OptionType = OptionType.AMERICAN
    option_side: OptionSide = OptionSide.PUT
    dimensions: int = 1               # dₛ → number of underlying assets (1 for vanilla options)
    risk_free_interest: float = 0.10  # r  → risk-free interest rate
    time_to_exp: float = 0.25         # T  → time to expiration

    # === Bermudan Configs ===
    exercise_frequency: ExerciseFrequency = None
    custom_exercise_points: Optional[np.ndarray] = None
    exercise_points: Optional[np.ndarray] = None

    # === Multi-dim Configs ===
    init_stock_prices: np.ndarray = np.array([80])      # S₀ → initial stock price
    strike_prices: np.ndarray = np.array([110])         # K  → strike price
    volatilities: np.ndarray = np.array([0.1])           # σ  → volatility of the underlying asset
    
    # === LSM Configs ===
    num_of_paths: int = 10_000        # M  → number of Monte Carlo simulation paths
    num_of_steps: int = 250           # N  → number of discrete time steps
    poly_degree: int = 3              # Degree of polynomial used in regression (LSM)

    # === FNN Configs ===
    epochs: int = 300                 # e  → number of training epochs (if using FNN)
    time_step: float = field(init=False)
    nn_layers: List[int] = field(init=False)


    def __post_init__(self):
        self.time_step = self.time_to_exp / self.num_of_steps
        self.nn_layers = get_nn_sizes(self.dimensions)

        if self.option_type == OptionType.BERMUDAN:
            if self.exercise_frequency is None:
                raise ValueError("Bermudan options require an exercise frequency.")
        else:
            if self.exercise_frequency is not None:
                raise ValueError("Only Bermudan options should specify exercise frequency.")
            
        self.exercise_points = self.get_excercise_points()


    def get_excercise_points(self):
        if self.option_type != OptionType.BERMUDAN:
            return None
        
        if self.exercise_frequency == ExerciseFrequency.QUARTERLY:
            num_of_dates = 4
        elif self.exercise_frequency == ExerciseFrequency.MONTHLY:
            num_of_dates = 12
        elif self.exercise_frequency == ExerciseFrequency.SEMI_MONTHLY:
            num_of_dates = 24
        elif self.exercise_frequency == ExerciseFrequency.CUSTOM:
            if self.custom_exercise_points:
                return self.custom_exercise_points
            else:
                raise ValueError("Custom exercise points must be provided for custom frequency.")

        else:
            raise ValueError("Invalid or missing exercise frequency for Bermudan option.")

        return np.linspace(0, self.num_of_steps, num_of_dates, dtype=int)
    
    
    def get_details(self):
        return f"""
        Option Parameters:
        - Number of paths: {self.num_of_paths}
        - Number of steps: {self.num_of_steps}
        - Time to expiration: {self.time_to_exp}
        - Initial stock price: {self.init_stock_price}
        - Drift: {self.drift}
        - Risk-free interest rate: {self.risk_free_interest}
        - Volatility: {self.volatility}
        - Strike price: {self.strike_price}
        - Time step (dt): {self.time_step}
        - Option side: {self.option_side}
        - Option type: {self.option_type}
        - Excercise Frequency: {self.exercise_frequency}
        - Excercise Points: {self.exercise_points}
        - Dimensions: {self.dimensions}
        - Neural Network Layers: {self.nn_layers}
        - Epochs: {self.epochs}
        """