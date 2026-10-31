"""Single-object entry point for the shared GPU batch baker.

Usage: direct_bc7.py MESH.glb CAMS.json INPUT_BC7_DIRECTORY OUTPUT_DIRECTORY
"""

import os
from pathlib import Path
import sys

if len(sys.argv) != 5:
    raise SystemExit(__doc__)
os.execv(
    sys.executable,
    [
        sys.executable,
        str(Path(__file__).with_name("batch_bc7.py")),
        "--single",
        *sys.argv[1:],
    ],
)
