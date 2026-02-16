"""MicroHH Processing - A package for processing MicroHH model output."""

__all__ = [
    "constants",
    "enums",
    "utils",
    "cscs",
    "initialisation",
    "io",
    "misc",
    "ml",
    "plume",
    "postprocessing",
    "preprocessing",
    "sources",
    "spatial",
    "visualisation",
    "wind",
]

from . import (
    constants,
    enums,
    utils,
    cscs,
    initialisation,
    io,
    misc,
    ml,
    plume,
    postprocessing,
    preprocessing,
    sources,
    spatial,
    visualisation,
    wind,
)

# Convenience exports for commonly used classes/functions
from .enums import (
    Cluster,
    Processor,
    Partition,
    MailType,
    InitMethod,
    WindIntegrationMethod,
    Stability,
    Precision,
    FileType,
    RunMode,
)
from .constants import (
    PrintColor,
    GAS_PROPERTIES,
    GAS_MOLAR_MASS,
    EARTH_ROTATION_RATE,
)
from .initialisation import Case, initialise_case, read_case
from .cscs import AlpsSetup
from .preprocessing import PreProcessing
from .postprocessing import PostProcessing
from .wind import WindProcessing
