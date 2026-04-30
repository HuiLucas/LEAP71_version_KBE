"""
leap71_bindings.py
==================
C# bindings for PicoGK and the LEAP71 ShapeKernel / LatticeLibrary via pythonnet.

All C# types required to recreate HelixHeatX entirely in Python are imported here
and re-exported as top-level names.  Helper utilities bridge the most common
Python ↔ .NET friction points: Vector3 construction, List<T>, scalar-vector
multiplication, and running code inside PicoGK's Library.Go environment.

Usage
-----
    import leap71_bindings as leap71

    # Create geometry
    v    = leap71.vec3(1.0, 0.0, 0.0)
    frame = leap71.LocalFrame(v)

    # Run inside PicoGK (headless – no viewer window)
    def my_task():
        lat = leap71.Lattice()
        lat.AddBeam(leap71.vec3(0,0,0), 1.0, leap71.vec3(10,0,0), 1.0)
        vox = leap71.Voxels(lat)
        leap71.Sh.ExportVoxelsToSTLFile(vox, "/path/to/out.stl")

    leap71.run_in_library(my_task, voxel_size=0.5)

Note
----
Build the C# project first:
    cd PicoGK_Examples-main && dotnet build
The compiled assemblies must exist under bin/Debug/net9.0/ before importing.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Callable, Iterable


# ── 1. Bootstrap pythonnet with the CoreCLR runtime ───────────────────────

from pythonnet import load as _pn_load
_pn_load("coreclr")


# ── 2. Resolve assembly paths ─────────────────────────────────────────────

_HERE         = Path(__file__).resolve().parent
_PROJECT      = _HERE / "PicoGK_Examples-main"
_BIN          = _PROJECT / "bin" / "Debug" / "net9.0"
_PICOGK_DLL   = _BIN / "PicoGK.dll"
_EXAMPLES_DLL = _BIN / "PicoGKExamples.dll"

_SYSTEM = platform.system().lower()
_MACHINE = platform.machine().lower()
if _SYSTEM == "windows":
    _RID = "win-x64"
elif _SYSTEM == "darwin" and _MACHINE in {"arm64", "aarch64"}:
    _RID = "osx-arm64"
elif _SYSTEM == "linux" and _MACHINE in {"x86_64", "amd64"}:
    _RID = "linux-x64"
else:
    _RID = None

if _RID is None:
    raise RuntimeError(
        "This PicoGK package does not include a native runtime for this platform "
        f"({platform.system()} / {platform.machine()}). The package currently "
        "contains win-x64 and osx-arm64 native binaries only, so the notebook "
        "must be run on Windows x64 or Apple Silicon macOS, or with a PicoGK "
        "build that provides libpicogk.1.7.so for Linux."
    )
loclinnat = os.environ.get("_LOCAL_LINUX_NATIVE", None)
if loclinnat != None and _RID == "linux-x64":
    _LOCAL_LINUX_NATIVE = Path(loclinnat)
else:
    _LOCAL_LINUX_NATIVE = Path.home() / "PicoGKRuntime" / "build" / "lib"
if _RID == "linux-x64" and (_LOCAL_LINUX_NATIVE / "picogk.so").exists():
    _NATIVE = _LOCAL_LINUX_NATIVE
else:
    _NATIVE = _BIN / "runtimes" / _RID / "native"

for _p in [_PROJECT, _BIN, _PICOGK_DLL, _EXAMPLES_DLL]:
    if not Path(_p).exists():
        raise FileNotFoundError(
            f"Required path not found: {_p}\n"
            "Build the C# project with 'dotnet build' before importing this module."
        )

if not _NATIVE.exists():
    raise FileNotFoundError(
        f"Required native runtime path not found: {_NATIVE}\n"
        "Build the C# project with 'dotnet build' before importing this module."
    )

if _RID == "linux-x64":
    _linux_lib = _NATIVE / "picogk.so"
    if not _linux_lib.exists():
        raise FileNotFoundError(
            f"Required Linux PicoGK runtime not found: {_linux_lib}\n"
            "Set up ~/PicoGKRuntime/build/lib/picogk.so or provide a PicoGK "
            "package with a linux-x64 native runtime."
        )
    for _alias in ("libpicogk.1.7.so", "picogk.1.7.so"):
        _target = _BIN / _alias
        if not _target.exists():
            try:
                _target.symlink_to(_linux_lib)
            except OSError:
                shutil.copy2(_linux_lib, _target)


# ── 3. Configure search paths and load the native PicoGK library ─────────

if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

_native_str = str(_NATIVE)
if _native_str not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _native_str + os.pathsep + os.environ.get("PATH", "")

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_native_str)

import clr  # type: ignore[import-not-found]  # noqa: E402
from System.Runtime.Loader import AssemblyLoadContext  # noqa: E402

AssemblyLoadContext.Default.LoadFromAssemblyPath(str(_PICOGK_DLL))
AssemblyLoadContext.Default.LoadFromAssemblyPath(str(_EXAMPLES_DLL))


# ── 4. .NET / System types ────────────────────────────────────────────────

from System import Single                           # noqa: E402
from System.Numerics import Vector3                 # noqa: E402
from System.Threading import ThreadStart            # noqa: E402
from System.Collections.Generic import List         # noqa: E402


# ── 5. PicoGK core types ──────────────────────────────────────────────────

from PicoGK import Library, Voxels, Lattice, Mesh   # noqa: E402


# ── 6. LEAP71 ShapeKernel types ───────────────────────────────────────────
#
#  All compiled into PicoGKExamples.dll from the source trees:
#    LEAP71_ShapeKernel-1.0.0/ShapeKernel/
#    LEAP71_LatticeLibrary-main/LatticeLibrary/
#
#  Namespace: Leap71.ShapeKernel

from Leap71.ShapeKernel import (                    # noqa: E402
    # Frames
    LocalFrame,
    # Base shapes
    BaseBox,
    BaseCylinder,
    # Lattice shapes
    LatticeManifold,
    # Splines
    TangentialControlSpline,
    # Static utility classes
    Sh,             # ShBasicFunctions / ShCombinedFunctions / ShExportFunctions /
                    # ShLatticeFunctions / ShVoxelFunctions / ShPreviewFunctions
    Cp,             # ColorPalette – colour string constants
    Uf,             # UsefulFormulas / SuperShapes
    VecOperations,  # vecGetCylPoint, vecRotateAroundZ, vecRotateAroundAxis,
                    #  vecTranslatePointOntoFrame, Normalize (extension), …
)


# ── 7. LEAP71 ConstructionModules ─────────────────────────────────────────
#
#  Compiled from src/ into PicoGKExamples.dll
#  Namespace: Leap71.ConstructionModules

from Leap71.ConstructionModules import (            # noqa: E402
    ScrewHole,
    ThreadCutter,
    ThreadReinforcement,
)


# ── 8. Convenient enum aliases ────────────────────────────────────────────

#: Uf.ESuperShape enum values: ROUND, HEX, QUAD, TRI
ESuperShape = Uf.ESuperShape

#: Sh.EExport enum values: STL, TGA, PNG, CSV, VDB, CLI
EExport = Sh.EExport


# ── 9. Frequently-used Vector3 constants ─────────────────────────────────

UNIT_X: Vector3 = Vector3.UnitX
UNIT_Y: Vector3 = Vector3.UnitY
UNIT_Z: Vector3 = Vector3.UnitZ
ZERO:   Vector3 = Vector3.Zero


# ── 10. Python helper functions ───────────────────────────────────────────


def vec3(x: float, y: float, z: float) -> Vector3:
    """Construct a System.Numerics.Vector3 from three Python floats."""
    return Vector3(Single(x), Single(y), Single(z))


def vmul(scalar, vec: Vector3) -> Vector3:
    """
    Scale *vec* by *scalar*.

    Handles both Python floats and C# Singles returned from Uf.fTransFixed etc.
    Uses Vector3.Multiply so operator-overload dispatch is unambiguous.
    """
    return Vector3.Multiply(Single(float(scalar)), vec)


def normalize(v: Vector3) -> Vector3:
    """
    Return the unit-length version of *v*.

    Calls VecOperations.Normalize, which is the extension method that the
    ShapeKernel defines on Vector3 (mirrors the C# (vec).Normalize() syntax).
    """
    return VecOperations.Normalize(v)


def cross(a: Vector3, b: Vector3) -> Vector3:
    """Cross product of two Vector3 values (delegates to System.Numerics)."""
    return Vector3.Cross(a, b)


def vox_combine_all(voxels: Iterable) -> Voxels:
    """
    Union-combine an iterable of Voxels into one by sequential addition.

    Note: deliberately does NOT call ``Voxels.voxCombineAll`` because pythonnet
    may wrap its C# return value in a tuple when the method internally uses
    ref/out semantics.  Sequential ``+`` is unambiguous and always returns a
    plain ``Voxels`` object.
    """
    result = None
    for v in voxels:
        result = v if result is None else result + v
    if result is None:
        raise ValueError("vox_combine_all received an empty iterable")
    return result


def export_voxels_to_stl(voxels: Voxels, stl_path: str | Path) -> None:
    """
    Export voxels as an STL without using ShapeKernel's logging wrapper.

    ``Sh.ExportVoxelsToSTLFile`` calls ``Library.Log`` and therefore requires
    ``Library.Go`` to have initialized successfully.  On Linux/headless
    notebooks that viewer bootstrap can block, while PicoGK's direct mesh
    export works fine.
    """
    mesh = voxels.mshAsMesh()
    mesh.SaveToStlFile(str(stl_path), Mesh.EStlUnit.MM, None, Single(1.0))

def export_voxels_to_vdb(voxels: Voxels, vdb_path: str | Path) -> None:
    """
    Export voxels as an OpenVDB file without using ShapeKernel's logging wrapper.

    ``Sh.ExportVoxelsToVDBFile`` calls ``Library.Log`` and therefore requires
    ``Library.Go`` to have initialized successfully.  On Linux/headless
    notebooks that viewer bootstrap can block, while PicoGK's direct VDB
    export works fine.
    """
    voxels.SaveToVdbFile(str(vdb_path), Single(1.0))


def run_in_library(
    task: Callable[[], None],
    voxel_size: float = 0.5,
    output_dir: str | Path | None = None,
    headless: bool = True,
) -> None:
    """
    Execute *task* inside PicoGK's Library.Go render environment.

    Library.Go blocks the calling thread until the task completes.

    Parameters
    ----------
    task       : zero-argument callable – the geometry construction body.
    voxel_size : voxel edge length in mm (default 0.5 mm).
    output_dir : directory where STL/screenshot files are written.
                 Defaults to <workspace>/Examples/ next to this file.
    headless   : kept for notebook call-site compatibility.  Preview/viewer
                 calls inside *task* should still be guarded by this flag.
    """
    if output_dir is None:
        output_dir = _HERE / "Examples"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if headless:
        task()
        return

    Library.Go(
        Single(voxel_size),   # fVoxelSizeMM
        ThreadStart(task),    # fnTask
        str(output_dir),      # strOutputFolder
    )
