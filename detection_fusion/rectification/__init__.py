# Re-export from root level gt_rectify.py for now
# TODO: Move gt_rectify.py content here in a future release

import sys
from pathlib import Path

# Add parent directory to path to import gt_rectify
_root_dir = Path(__file__).parent.parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

try:
    from gt_rectify import GTRectificationError, GTRectifier

    __all__ = ["GTRectifier", "GTRectificationError"]
except ImportError:
    GTRectifier = None
    GTRectificationError = None
    __all__ = []
