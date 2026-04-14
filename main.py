from pathlib import Path
import sys


root = Path(__file__).resolve().parent
if not (root / "leap71_wrapper.py").exists():
	root = root.parent

sys.path.insert(0, str(root))

from leap71_wrapper import Leap71Workspace


workspace = Leap71Workspace.discover(root)
stl_path = workspace.generate_helix_heatx()
print(f"Generated: {stl_path}")