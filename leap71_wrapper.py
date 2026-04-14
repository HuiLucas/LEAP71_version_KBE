"""Python wrapper for the LEAP71 examples in this workspace.

The wrapper uses pythonnet to load the compiled C# assembly directly, calls the
HelixHeatX task through PicoGK.Library.Go, and then loads the generated STL for
notebook-based inspection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import sys

import numpy as np
from pythonnet import load
from stl import mesh as stl_mesh


load("coreclr")

from System import Single
from System.Reflection import Assembly
from System.Threading import ThreadStart


class Leap71Workspace:
    """Small Python facade over the C# LEAP71 example workspace."""

    def __init__(self, workspace_root: str | Path, dotnet_exe: str = "dotnet") -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.project_root = self.workspace_root / "PicoGK_Examples-main"
        self.csproj_path = self.project_root / "PicoGKExamples.csproj"
        self.cs_output_dir = self.project_root / "bin" / "Debug" / "net9.0"
        self.native_dir = self.cs_output_dir / "runtimes" / "win-x64" / "native"
        self.cs_assembly_path = self.cs_output_dir / "PicoGKExamples.dll"
        self.examples_dir = self.workspace_root / "Examples"
        self.helix_stl_path = self.examples_dir / "HelixHeatX.STL"

        if not self.csproj_path.exists():
            raise FileNotFoundError(f"Could not find C# project: {self.csproj_path}")
        if not self.cs_assembly_path.exists():
            raise FileNotFoundError(
                f"Could not find built C# assembly: {self.cs_assembly_path}. Build the project first."
            )
        if not self.native_dir.exists():
            raise FileNotFoundError(
                f"Could not find PicoGK native runtime directory: {self.native_dir}"
            )

    @classmethod
    def discover(cls, start_path: Optional[str | Path] = None) -> "Leap71Workspace":
        """Find the workspace root by walking upward from a starting path."""

        current = Path(start_path or Path.cwd()).resolve()
        candidates = [current, *current.parents]

        for candidate in candidates:
            project_file = candidate / "PicoGK_Examples-main" / "PicoGKExamples.csproj"
            if project_file.exists():
                return cls(candidate)

        raise FileNotFoundError(
            "Could not locate the PicoGK_Examples-main project. Start the notebook from the workspace folder."
        )

    def _load_csharp_types(self) -> None:
        """Load the compiled C# assembly and its dependencies through pythonnet."""

        if str(self.cs_output_dir) not in sys.path:
            sys.path.insert(0, str(self.cs_output_dir))

        native_path = str(self.native_dir)
        if native_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = native_path + os.pathsep + os.environ.get("PATH", "")

        self._dll_dir_handle = os.add_dll_directory(str(self.native_dir))

        import clr  # type: ignore[import-not-found]

        clr.AddReference(str(self.cs_assembly_path))
        clr.AddReference(str(self.cs_output_dir / "PicoGK.dll"))

        pico_assembly = Assembly.LoadFrom(str(self.cs_output_dir / "PicoGK.dll"))
        app_assembly = Assembly.LoadFrom(str(self.cs_assembly_path))

        self._library_type = pico_assembly.GetType("PicoGK.Library")
        self._helix_type = app_assembly.GetType("Leap71.CoolCube.HelixHeatX")
        self._helix_task = self._helix_type.GetMethod("Task")
        self._go_method = self._library_type.GetMethod("Go")

    def run_helix_heatx(self, voxel_size: float = 0.5) -> Path:
        """Run the existing C# HelixHeatX example through pythonnet."""

        self._load_csharp_types()

        def run_task() -> None:
            self._helix_task.Invoke(None, None)

        thread_start = ThreadStart(run_task)
        self._go_method.Invoke(
            None,
            [Single(voxel_size), thread_start, str(self.examples_dir), "", "", "", False],
        )
        return self.helix_stl_path

    def generate_helix_heatx(self, voxel_size: float = 0.5) -> Path:
        """Run the C# example and return the generated STL path."""

        self.run_helix_heatx(voxel_size=voxel_size)
        if not self.helix_stl_path.exists():
            raise FileNotFoundError(f"HelixHeatX STL was not generated: {self.helix_stl_path}")
        return self.helix_stl_path

    @staticmethod
    def load_stl(stl_path: str | Path):
        """Load an STL file with numpy-stl."""

        return stl_mesh.Mesh.from_file(str(stl_path))

    @staticmethod
    def plot_stl(mesh_obj, ax=None, title: str = "HelixHeatX", max_faces: int = 8000):
        """Plot an STL mesh using matplotlib."""

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        vectors = mesh_obj.vectors
        if len(vectors) > max_faces:
            step = max(1, len(vectors) // max_faces)
            vectors = vectors[::step]

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        collection = Poly3DCollection(vectors, alpha=0.92)
        collection.set_facecolor((0.22, 0.45, 0.78, 0.92))
        collection.set_edgecolor((0.15, 0.15, 0.18, 0.12))
        ax.add_collection3d(collection)

        points = vectors.reshape(-1, 3)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        centers = (mins + maxs) / 2.0
        radius = spans.max() / 2.0

        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        return fig, ax
