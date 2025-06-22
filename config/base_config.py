from typing import Optional, List
import numpy as np
from dataclasses import dataclass, field
from enums import OptionSide, OptionType, ExerciseFrequency, CorrelationType
from core import get_nn_sizes


EPSILON = 1e-8

@dataclass
class Config:
    """
    Represents the configuration for a basket option pricing experiment
    Loaded from a YAML file
    """

    # === Required (no defaults) ===
    option_type: OptionType
    option_side: OptionSide
    dimensions: int
    risk_free_interest: float
    time_to_exp: float
    init_stock_prices: np.ndarray
    strike_prices: np.ndarray
    volatilities: np.ndarray
    correlation_rho: float
    num_of_paths: int
    num_of_steps: int
    poly_degree: int
    epochs: int

    # === Optional (default values) ===
    exercise_frequency: Optional[ExerciseFrequency] = None
    exercise_points: Optional[List[int]] = None
    correlation_matrix: Optional[List[List[float]]] = None
    correlation_type: Optional[CorrelationType] = None

    # === Computed fields ===
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
        self.correlation_matrix = self.get_correlation_matrix()


    def get_excercise_points(self):
        if self.option_type != OptionType.BERMUDAN:
            return None
        
        if self.exercise_points is not None and self.exercise_frequency == ExerciseFrequency.CUSTOM:
            self.validate_exercise_points()
            return self.exercise_points
        
        elif self.exercise_points is not None:
            raise ValueError(f"Attempted to load exercise points: {self.exercise_points}, however exercise frequency: {self.exercise_frequency} mismatch.")
        
        elif self.exercise_frequency == ExerciseFrequency.CUSTOM:
            raise ValueError(f"Exercise frequency: {self.exercise_frequency} == CUSTOM, however exercise points: {self.exercise_points} are empty")
        
        if self.exercise_frequency == ExerciseFrequency.QUARTERLY:
            num_of_dates = 4
        elif self.exercise_frequency == ExerciseFrequency.MONTHLY:
            num_of_dates = 12
        elif self.exercise_frequency == ExerciseFrequency.SEMI_MONTHLY:
            num_of_dates = 24
        else:
            raise ValueError(f"Exercise frequency is unkown: {self.exercise_frequency}")

        return np.linspace(0, self.num_of_steps, num_of_dates, dtype=int)
    

    def get_correlation_matrix(self):
        if self.dimensions == 1:
            return None
        
        if self.correlation_type == CorrelationType.UNIFORM:
            matrix = np.full((self.dimensions, self.dimensions), self.correlation_rho)
            np.fill_diagonal(matrix, 1.0)
            return matrix
        
        if self.correlation_type == CorrelationType.IDENTITY:
            return np.eye(self.dimensions)

        if self.correlation_type == CorrelationType.CUSTOM:
            self.validate_correlation_matrix()
            return self.correlation_matrix
        
        if self.correlation_type is None:
            raise ValueError("Correlation type must be set for multi-dimension options")


    def validate_exercise_points(self):
        if not isinstance(self.exercise_points, list):
            raise TypeError("Exercise points must be a list")
        
        if len(self.exercise_points) == 0:
            raise ValueError("Exercise points must contain at least one integer")
        
        if not all(isinstance(x, int) for x in self.exercise_points):
            raise TypeError("All entries of exercise points must be integers")


    def validate_correlation_matrix(self):
        matrix = self.correlation_matrix

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Correlation matrix must be a square")
        
        if not np.allclose(matrix, matrix.T, atol=EPSILON):
            raise ValueError("Correlation matrix must be symmetric")
        
        if not np.allclose(np.diag(matrix), 1.0, atol=EPSILON):
            raise ValueError("Diagnol of correlation matrix must all be 1s")
    
        eigenvalues = np.linalg.eigvals(matrix)
        if np.any(eigenvalues < -EPSILON):
            raise ValueError("Correlation matrix must be positive semi-definite (PSD)")

    
    def __str__(self):
        return f"""
        Option Parameters:
        - Option type: {self.option_type}
        - Option side: {self.option_side}
        - Dimensions: {self.dimensions}
        - Inital prices: {self.init_stock_prices}
        - Strike prices: {self.strike_prices}
        - Volatilities: {self.volatilities}
        - Risk-free interest rate: {self.risk_free_interest}
        - Time to expiration: {self.time_to_exp}
        - Excercise Frequency: {self.exercise_frequency}
        - Excercise Points: {self.exercise_points}
        - Correlation Type: {self.correlation_type}
        - Correlation Rho: {self.correlation_rho}
        - Correlation Matrix: {self.correlation_matrix}
        
        Model Parameters:  
        - Number of paths: {self.num_of_paths}
        - Number of steps: {self.num_of_steps} 
        - Poly degree: {self.poly_degree}      
        - Neural Network Layers: {self.nn_layers}
        - Epochs: {self.epochs}
        """