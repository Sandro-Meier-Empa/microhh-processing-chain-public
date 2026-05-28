"""Enumeration types used throughout the microhh_processing_chain package."""

from enum import Enum


class Cluster(str, Enum):
    """CSCS cluster names."""

    CLARIDEN = "clariden"
    DAINT = "daint"
    EIGER = "eiger"
    SANTIS = "santis"
    TODI = "todi"


class Processor(str, Enum):
    """Processor types available on CSCS clusters."""

    CPU = "cpu"
    GPU = "gpu"


class Partition(str, Enum):
    """SLURM partitions available on CSCS Alps."""

    DEBUG = "debug"
    NORMAL = "normal"
    XFER = "xfer"
    PREPOST = "prepost"
    LOW = "low"


class MailType(str, Enum):
    """SLURM mail notification types."""

    NONE = "NONE"
    BEGIN = "BEGIN"
    END = "END"
    FAIL = "FAIL"
    REQUEUE = "REQUEUE"
    ALL = "ALL"


class InitMethod(str, Enum):
    """Methods for initializing a MicroHH case."""

    DEFAULT = "default"
    INTERACTIVE = "interactive"
    RERUN = "rerun"


class WindIntegrationMethod(str, Enum):
    """Methods for vertical integration of wind fields."""

    MEAN = "mean"
    MEDIAN = "median"
    HEIGHT = "height"
    CONCENTRATION_WEIGHTED = "concentration_weighted"


class Stability(str, Enum):
    """Atmospheric stability regimes."""

    NEUTRAL = "neutral"
    UNSTABLE = "unstable"
    STABLE = "stable"
    NEUTRAL_UNSTABLE = "neutral_unstable"


class Precision(str, Enum):
    """Numerical precision for output files."""

    SINGLE = "single"
    DOUBLE = "double"


class FileType(str, Enum):
    """MicroHH file types that can be opened in VS Code."""

    INI = "ini"
    OUT = "out"
    SBATCH = "sbatch"
    SBATCH_OUT = "sbatch_out"


class RunMode(str, Enum):
    """MicroHH run modes."""

    RUN = "run"
    RERUN = "rerun"
    RESTART = "restart"


__all__ = [
    "Cluster",
    "FileType",
    "InitMethod",
    "MailType",
    "Partition",
    "Precision",
    "Processor",
    "RunMode",
    "Stability",
    "WindIntegrationMethod",
]
