# Config File Documentation

## option_type
- Type: String
- Options:
    - "AMERICAN": Exercisable any time before expiration
    - "EUROPEAN": Exercisable only at expiration
    - "BERMUDAN": Exercisable at specific points (see `exercise_frequency`)
- Default: "AMERICAN"

## option_side
- Type: String
- Options:
    - "PUT": The right to sell an asset in the future at X price
    - "CALL": The right to buy an asset in the future at X price
- Default: "PUT"

## dimensions
- Type: int
- Specify how many assets you would like to price
- Default: 1

## risk_free_interest
- Type: float
- For this pricing model we are assuming a risk-netural world so risk functions as mu
- Defualt: 0.10

## time_to_exp
- Type: float
- How long the option has until experation; time is in years
- Default: 0.25

## exercise_frequency
- Type: String
- Only used if `option_type = "BERMUDAN`
- Options:
    - "MONTHLY"
    - "QUARTERLY"
    - "SEMI_MONTHLY"
    - "CUSTOM" (requires `custom_exercise_points`)
- Defualt: None

## custom_exercise_points
- TO-DO

## 

