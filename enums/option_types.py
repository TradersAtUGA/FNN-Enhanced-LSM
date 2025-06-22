from enum import Enum


class OptionType(Enum):
    AMERICAN = 'american'
    BERMUDAN = 'bermudan'
    EUROPEAN = 'european'

# Only needed if the option type is bermudan
class ExerciseFrequency(Enum):
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    SEMI_MONTHLY = 'semi_monthly'
    CUSTOM = 'custom'  # optional override

class OptionSide(Enum):
    CALL = 'call'
    PUT = 'put'

class CorrelationType(Enum):
    UNIFORM = "uniform",
    IDENTITY = "identity",
    CUSTOM = "custom"