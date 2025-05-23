from enum import Enum


class OptionType(Enum):
    AMERICAN = 'american'
    BERMUDAN = 'bermudan'
    EUROPEAN = 'european'


class OptionSide(Enum):
    CALL = 'call'
    PUT = 'put'