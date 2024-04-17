import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.append(REPO_DIR.as_posix())
from limap_extension.utils.path_fixer import allow_limap_imports
