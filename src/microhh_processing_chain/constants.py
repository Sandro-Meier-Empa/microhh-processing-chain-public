from dataclasses import dataclass
from enum import Enum


class PrintColor(str, Enum):
    """ANSI color codes for terminal output."""

    CYAN = "\033[96m"
    VIOLET = "\033[95m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    UNDERLINE = "\033[4m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


@dataclass(frozen=True)
class GasProperties:
    """Properties of a gas species.

    Attributes:
        name: Human-readable name of the gas
        molar_mass: Molar mass in g/mol
    """

    name: str
    molar_mass: float  # g/mol


# Gas properties dictionary with molar masses
GAS_PROPERTIES: dict[str, GasProperties] = {
    "air": GasProperties("dry air", 28.9664),
    "co": GasProperties("carbon monoxide", 28.01),
    "co2": GasProperties("carbon dioxide", 44.01),
    "h2o": GasProperties("water vapor", 18.01528),
    "no2": GasProperties("nitrogen dioxide", 46.0055),
    "no": GasProperties("nitric oxide", 30.01),
    "o3": GasProperties("ozone", 48.00),
    "so2": GasProperties("sulfur dioxide", 64.066),
    "ch4": GasProperties("methane", 16.0425),
}

# Simple molar mass lookup (for backward compatibility)
GAS_MOLAR_MASS: dict[str, float] = {
    key: props.molar_mass for key, props in GAS_PROPERTIES.items()
}

# Physical constants
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s - Earth's angular velocity


__all__ = [
    "EARTH_ROTATION_RATE",
    "GAS_MOLAR_MASS",
    "GAS_PROPERTIES",
    "GasProperties",
    "PrintColor",
]
