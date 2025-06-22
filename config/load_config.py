import yaml
import numpy as np
from config import Config
from enums import OptionType, OptionSide, ExerciseFrequency, CorrelationType

def load_config_from_yaml(path: str):
    with open(path, "r") as f:
        # Turns yaml data into dict
        # safe_load = no sys cmds can be ran (rm -rf)
        data = yaml.safe_load(f)

    # Convert string to enums
    data["option_type"] = OptionType[data["option_type"]]
    data["option_side"] = OptionSide[data["option_side"]]
    data["correlation_type"] = CorrelationType[data["correlation_type"]]

    if data["exercise_frequency"] is not None:
        data["exercise_frequency"] = ExerciseFrequency[data["exercise_frequency"]]

    # Convert custom exercise points
    if data["exercise_points"] is not None:
        data["exercise_points"] = np.array(data["exercise_points"])

    if data["correlation_matrix"] is not None:
        data["correlation_matrix"] = np.array(data["correlation_matrix"])

    # Convert list to numpy arrays
    for arr in ["init_stock_prices", "strike_prices", "volatilities"]:
        data[arr] = np.array(data[arr])

    return Config(**data)
