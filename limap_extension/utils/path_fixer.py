import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
LIMAP_DIR = REPO_DIR / "limap"
HYPERSIM_DIR = LIMAP_DIR / "runners" / "hypersim"
# DEEPLSD_DIR = LIMAP_DIR / "third-party" / "DeepLSD"
THIRDPARTY_DIR = LIMAP_DIR / "third-party"


def allow_limap_imports():
    """Adds the necessary paths to allow imports from the LIMAP and extension repositories"""
    # Assuming repository directory is already on the path since you're running this file.
    # sys.path.append(REPO_DIR.as_posix())
    sys.path.append(LIMAP_DIR.as_posix())
    sys.path.append(HYPERSIM_DIR.as_posix())
    # sys.path.append(DEEPLSD_DIR.as_posix())

    for path in THIRDPARTY_DIR.iterdir():
        if not path.is_dir():
            continue

        src_path = path / "src"
        if src_path.exists():
            sys.path.append(src_path.as_posix())
            # sys.path.append(path.as_posix())
        else:
            sys.path.append(path.as_posix())
