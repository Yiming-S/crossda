"""crossda — cross-session EEG classification and domain-adaptation pipelines.

Public API
----------
    Config                 runtime configuration dataclass
    run                    orchestrate a full multi-dataset run from a Config
    process_subject        run all pipelines for a single subject
    generate_method_bank   build the feature × classifier × DA search space
    MAP, DWP, MMP, BDP     the four pipeline-family entry functions

Example
-------
    from crossda import Config, run
    run(Config(datasets=["bnci004"], method_mode="smoke", data_dir="/path/to/data"))
"""

from .config import Config
from .core.method_bank import generate_method_bank
from .core.workers import process_subject
from .pipelines.map_pipeline import MAP
from .pipelines.dwp_pipeline import DWP
from .pipelines.mmp_pipeline import MMP
from .pipelines.bdp_pipeline import BDP
from .cli import run

__version__ = "0.1.0"

__all__ = [
    "Config",
    "run",
    "process_subject",
    "generate_method_bank",
    "MAP",
    "DWP",
    "MMP",
    "BDP",
    "__version__",
]
