import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
LIMAP_DIR = REPO_DIR / "limap"
HYPERSIM_DIR = LIMAP_DIR / "runners" / "hypersim"


def allow_limap_imports():
    """Adds the necessary paths to allow imports from the LIMAP and extension repositories"""
    # Assuming repository directory is already on the path since you're running this file.
    # sys.path.append(REPO_DIR.as_posix())
    sys.path.append(LIMAP_DIR.as_posix())
    sys.path.append(HYPERSIM_DIR.as_posix())
