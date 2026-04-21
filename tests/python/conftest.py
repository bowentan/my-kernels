import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT / "python"
BUILD_PKG_ROOT = os.environ.get("MY_KERNELS_BUILD_PYTHON_ROOT")
build_pkg_root_str = None

if BUILD_PKG_ROOT:
    build_pkg_root = Path(BUILD_PKG_ROOT)
    if build_pkg_root.is_dir():
        build_pkg_root_str = str(build_pkg_root)
        if build_pkg_root_str not in sys.path:
            sys.path.insert(0, build_pkg_root_str)

if str(PKG_ROOT) not in sys.path:
    insert_at = 1 if build_pkg_root_str and sys.path and sys.path[0] == build_pkg_root_str else 0
    sys.path.insert(insert_at, str(PKG_ROOT))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
