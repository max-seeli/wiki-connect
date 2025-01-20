from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent
ROOT_PATH = MODULE_PATH.parent.parent
DATA_PATH = ROOT_PATH / "data"
PLOTS_PATH = ROOT_PATH / "plots"