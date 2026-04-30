
import os
import math
import sys
from pathlib import Path

import numpy as np

# ── set up X11 display environment for graphical windows ──────────────────
os.environ.setdefault('DISPLAY', ':2')
xdg_dir = f'/tmp/xdg-runtime-{os.getuid()}'
os.makedirs(xdg_dir, mode=0o700, exist_ok=True)
os.environ['XDG_RUNTIME_DIR'] = xdg_dir
os.environ.setdefault("GDK_BACKEND", "x11")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

# Try OpenSWR software renderer first (more stable than llvmpipe)
# If unavailable, fall back to llvmpipe
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'swr,llvmpipe'

# Aggressive workaround for Mesa shader caching bug that causes "free(): invalid pointer"
# Completely disable all caching mechanisms
os.environ['MESA_SHADER_CACHE'] = 'false'
os.environ['MESA_SHADER_CACHE_DIR'] = ''
os.environ['MESA_DISK_CACHE_DIR'] = ''
os.environ['ZSTD_NBTHREADS'] = '1'


# ── locate leap71_bindings.py (one directory above notebooks/) ────────────
REPO_ROOT = Path.cwd()
if not (REPO_ROOT / 'leap71_bindings.py').exists():
    REPO_ROOT = REPO_ROOT.parent
if not (REPO_ROOT / 'leap71_bindings.py').exists():
    raise FileNotFoundError('Cannot find leap71_bindings.py – run this notebook from the repo root')
sys.path.insert(0, str(REPO_ROOT))

import leap71_bindings as leap71

# ── pull the most-used names into this namespace for readability ──────────
from leap71_bindings import (
    # .NET / PicoGK core
    Single, Vector3, Voxels, Lattice, Library,
    # ShapeKernel
    LocalFrame, BaseBox, BaseCylinder, LatticeManifold,
    TangentialControlSpline,
    Sh, Cp, Uf, VecOperations,
    # ConstructionModules
    ScrewHole, ThreadCutter, ThreadReinforcement,
    # Vector3 constants
    UNIT_X, UNIT_Y, UNIT_Z, ZERO,
    # Enum aliases
    ESuperShape, EExport,
    # Python helpers
    vec3, vmul, normalize, cross, vox_combine_all, export_voxels_to_stl, run_in_library,
)

OUTPUT_DIR = REPO_ROOT / 'Examples'
OUTPUT_DIR.mkdir(exist_ok=True)
STL_PATH   = OUTPUT_DIR / 'HelixHeatX_Python.STL'

print('Bindings loaded successfully.')
print(f'Output will be written to: {OUTPUT_DIR}')
print(f'Display configured: DISPLAY={os.environ.get("DISPLAY")}, XDG_RUNTIME_DIR={os.environ.get("XDG_RUNTIME_DIR")}')

class HelixHeatX:
    """Pure-Python reimplementation of Leap71.CoolCube.HelixHeatX.

    All geometry is built by calling ShapeKernel / LatticeLibrary / PicoGK
    functions through pythonnet bindings defined in leap71_bindings.py.
    """

    # fluid-type constants (replaces the C# private enum EFluid)
    HOT  = 'HOT'
    COOL = 'COOL'

    # ─────────────────────────────────────────────────────────────────────
    # Constructor  (mirrors HelixHeatX.cs)
    # ─────────────────────────────────────────────────────────────────────

    def __init__(self):
        """Set up default parameters and local reference frames."""
        half_io_len = 75.0
        half_io_wid = 26.5

        self.m_first_inlet_frame   = LocalFrame(vec3(-half_io_len, -half_io_wid, 50), -UNIT_X)
        self.m_second_inlet_frame  = LocalFrame(vec3(-half_io_len,  half_io_wid, 50), -UNIT_X)
        self.m_first_outlet_frame  = LocalFrame(vec3( half_io_len, -half_io_wid, 50),  UNIT_X)
        self.m_second_outlet_frame = LocalFrame(vec3( half_io_len,  half_io_wid, 50),  UNIT_X)

        # centre of the helix at the bottom of the z-range
        self.m_centre_bottom_frame = LocalFrame(vec3(-50, 0, 50), UNIT_X, UNIT_Z)

        # outer bounding box (includes port protrusions)
        outer_box = BaseBox(
            LocalFrame(vec3(0, 0, -4)),
            107,
            2 * half_io_len + 24.0,
            104,
        )
        self.m_vox_bounding = outer_box.oConstructVoxels()

        self.m_plate_thickness = 3.5
        self.m_wall_thickness  = 0.8
        self.m_io_radius       = 7.0

    # ─────────────────────────────────────────────────────────────────────
    # Geometry helpers  (mirrors Misc.cs)
    # ─────────────────────────────────────────────────────────────────────

    def _vec_trafo(self, vec_pt: Vector3) -> Vector3:
        """Translate a point from the helix coordinate system to world space."""
        return VecOperations.vecTranslatePointOntoFrame(self.m_centre_bottom_frame, vec_pt)

    def _f_get_inner_radius(self, f_phi: float, f_lr: float) -> float:
        """Inner radius of the fluid channel cross-section (circular super-shape)."""
        return 10.0 * Uf.fGetSuperShapeRadius(Single(f_phi), ESuperShape.ROUND)

    def _f_get_outer_radius(self, f_phi: float, f_lr: float) -> float:
        """Outer radius of the fluid channel cross-section (square super-shape)."""
        return 50.0 * Uf.fGetSuperShapeRadius(Single(f_phi), ESuperShape.QUAD)

    def _add_centre_piece(self, vox_outer: Voxels) -> Voxels:
        """Add a thin cross-bar at the centre-bottom for print support."""
        box = BaseBox(self.m_centre_bottom_frame, 100, 20, 2)
        return vox_outer + box.oConstructVoxels()

    # ─────────────────────────────────────────────────────────────────────
    # Helical fluid voids  (mirrors HelicalVoids.cs + IOPipes.cs)
    # ─────────────────────────────────────────────────────────────────────

    def _get_helical_void(self, efluid: str):
        """Build the helical fluid channel for one fluid (HOT or COOL).

        Returns (vox_inner_volume, vox_splitters).
        """
        f_phi_start = math.pi if efluid == self.HOT else 0.0
        f_beam      = 0.5 * self.m_plate_thickness
        f_start_z   = 0.0
        f_end_z     = 100.0
        f_total_len = f_end_z - f_start_z
        f_inter     = self.m_wall_thickness

        n_turns  = int(f_total_len / (2.0 * self.m_plate_thickness + 2.0 * f_inter))
        f_turns  = n_turns - 0.5
        f_slope  = (f_turns * 2.0 * math.pi) / f_total_len

        lat_void  = Lattice()
        n_samples = int(f_total_len / 0.005)

        vec_first_pt1 = vec_first_pt2 = ZERO
        vec_last_pt1  = vec_last_pt2  = ZERO

        for i in range(n_samples):
            f_lr    = i / n_samples
            f_z     = f_start_z + f_lr * (f_end_z - f_start_z)
            f_phi   = f_phi_start + f_slope * (f_z - f_start_z)
            f_inner = float(self._f_get_inner_radius(f_phi, f_lr))
            f_outer = float(self._f_get_outer_radius(f_phi, f_lr)) - f_beam

            vec_pt1 = VecOperations.vecGetCylPoint(Single(f_inner), Single(f_phi), Single(f_z))
            vec_pt2 = VecOperations.vecGetCylPoint(Single(f_outer), Single(f_phi), Single(f_z))
            vec_pt1 = self._vec_trafo(vec_pt1)
            vec_pt2 = self._vec_trafo(vec_pt2)
            vec_pt3 = vec_pt1 + vmul(3.0, UNIT_Z)
            vec_pt4 = vec_pt2 + vmul(3.0, UNIT_Z)

            lat_void.AddBeam(vec_pt1, Single(f_beam), vec_pt2, Single(f_beam))
            lat_void.AddBeam(vec_pt1, Single(f_beam), vec_pt3, Single(0.2))
            lat_void.AddBeam(vec_pt2, Single(f_beam), vec_pt4, Single(0.2))

            if i == 0:
                vec_first_pt1, vec_first_pt2 = vec_pt1, vec_pt2
            if i == n_samples - 1:
                vec_last_pt1, vec_last_pt2 = vec_pt1, vec_pt2

        vox_helix                      = Voxels(lat_void)
        vox_inlet,  vox_in_split  = self._get_inlet( efluid, vec_first_pt1, vec_first_pt2, f_beam)
        vox_outlet, vox_out_split = self._get_outlet(efluid, vec_last_pt1,  vec_last_pt2,  f_beam)

        vox_inner     = vox_inlet + vox_outlet + vox_helix
        vox_splitters = vox_in_split + vox_out_split
        return vox_inner, vox_splitters

    def _get_inlet(self, efluid: str, vec_pt1_p: Vector3, vec_pt2_p: Vector3, f_beam: float):
        """Inlet pipe transition + internal splitter wall for one fluid.

        Returns (vox_inlet, vox_splitter).  Mirrors IOPipes.cs::GetInlet.
        """
        vec_end     = self.m_second_inlet_frame.vecGetPosition()
        vec_end_dir = -UNIT_X

        vec_len_dir   = normalize(vec_pt2_p - vec_pt1_p)
        vec_normal    = cross(UNIT_Y, vec_len_dir)
        vec_start_dir = cross(vec_len_dir, vec_normal)

        if efluid == self.HOT:
            vec_end       = self.m_first_inlet_frame.vecGetPosition()
            vec_normal    = cross(-UNIT_Z, vec_len_dir)
            vec_start_dir = cross(vec_len_dir, vec_normal)

        f_inlet_radius = self.m_io_radius
        vec_start      = vmul(0.5, vec_pt1_p + vec_pt2_p)
        vec_start_ori  = normalize(vec_pt2_p - vec_pt1_p)
        f_start_length = float((vec_pt2_p - vec_start).Length())

        o_spline = TangentialControlSpline(
            vec_start, vec_end, vec_start_dir, vec_end_dir,
            Single(20), Single(10)
        )

        lat_inlet    = Lattice()
        lat_splitter = Lattice()
        a_points     = list(o_spline.aGetPoints(500))
        n_pts        = len(a_points)

        for i, vec_pt in enumerate(a_points):
            f_lr    = i / n_pts
            f_beam2 = float(Uf.fTransFixed(Single(f_beam),         Single(f_inlet_radius), Single(f_lr)))
            f_len2  = float(Uf.fTransFixed(Single(f_start_length), Single(0.0),            Single(f_lr)))
            f_tip   = float(Uf.fTransFixed(Single(3.0),            Single(10.0),           Single(f_lr)))
            f_top   = float(Uf.fTransFixed(Single(0.4),            Single(1.0),            Single(f_lr)))

            vp1 = vec_pt - vmul(f_len2, vec_start_ori)
            vp2 = vec_pt + vmul(f_len2, vec_start_ori)

            if vp1.Z > vp2.Z:
                vp3 = vp1 + vmul(f_tip, UNIT_Z)
                lat_inlet.AddBeam(vp1, Single(f_beam2), vp3, Single(0.2))
                sp0 = vp2 - vmul(10.0, UNIT_Z)
            else:
                vp3 = vp2 + vmul(f_tip, UNIT_Z)
                lat_inlet.AddBeam(vp2, Single(f_beam2), vp3, Single(0.2))
                sp0 = vp1 - vmul(10.0, UNIT_Z)

            sp1 = vp3 + vmul(f_beam2,        UNIT_Z)
            sp2 = vp3 + vmul(f_beam2 + 5.0,  UNIT_Z)
            sp3 = vp3 + vmul(f_beam2 + 10.0, UNIT_Z)
            lat_splitter.AddBeam(sp0, Single(0.4),    sp1, Single(0.4))
            lat_splitter.AddBeam(sp1, Single(0.4),    sp2, Single(f_top))
            lat_splitter.AddBeam(sp2, Single(f_top),  sp3, Single(f_top))

            lat_inlet.AddBeam(vp1, Single(f_beam2), vp2, Single(f_beam2))

        vox_inlet    = Voxels(lat_inlet)
        vox_splitter = Voxels(lat_splitter) & vox_inlet
        return vox_inlet, vox_splitter

    def _get_outlet(self, efluid: str, vec_pt1_p: Vector3, vec_pt2_p: Vector3, f_beam: float):
        """Outlet pipe transition + internal splitter wall for one fluid.

        Returns (vox_outlet, vox_splitter).  Mirrors IOPipes.cs::GetOutlet.
        """
        vec_end     = self.m_second_outlet_frame.vecGetPosition()
        vec_end_dir = UNIT_X

        vec_len_dir   = normalize(vec_pt2_p - vec_pt1_p)
        vec_normal    = cross(UNIT_Y, vec_len_dir)
        vec_start_dir = cross(vec_len_dir, vec_normal)

        if efluid == self.HOT:
            vec_end       = self.m_first_outlet_frame.vecGetPosition()
            vec_normal    = cross(UNIT_Z, vec_len_dir)
            vec_start_dir = cross(vec_len_dir, vec_normal)

        f_inlet_radius = self.m_io_radius
        vec_start      = vmul(0.5, vec_pt1_p + vec_pt2_p)
        vec_start_ori  = normalize(vec_pt2_p - vec_pt1_p)
        f_start_length = float((vec_pt2_p - vec_start).Length())

        o_spline = TangentialControlSpline(
            vec_start, vec_end, vec_start_dir, vec_end_dir,
            Single(20), Single(10)
        )

        lat_outlet   = Lattice()
        lat_splitter = Lattice()
        a_points     = list(o_spline.aGetPoints(500))
        n_pts        = len(a_points)

        for i, vec_pt in enumerate(a_points):
            f_lr    = i / n_pts
            f_beam2 = float(Uf.fTransFixed(Single(f_beam),         Single(f_inlet_radius), Single(f_lr)))
            f_len2  = float(Uf.fTransFixed(Single(f_start_length), Single(0.0),            Single(f_lr)))
            f_tip   = float(Uf.fTransFixed(Single(3.0),            Single(10.0),           Single(f_lr)))
            f_top   = float(Uf.fTransFixed(Single(0.4),            Single(1.0),            Single(f_lr)))

            vp1 = vec_pt - vmul(f_len2, vec_start_ori)
            vp2 = vec_pt + vmul(f_len2, vec_start_ori)

            if vp1.Z > vp2.Z:
                vp3 = vp1 + vmul(f_tip, UNIT_Z)
                lat_outlet.AddBeam(vp1, Single(f_beam2), vp3, Single(0.2))
                sp0 = vp2 - vmul(10.0, UNIT_Z)
            else:
                vp3 = vp2 + vmul(f_tip, UNIT_Z)
                lat_outlet.AddBeam(vp2, Single(f_beam2), vp3, Single(0.2))
                sp0 = vp1 - vmul(10.0, UNIT_Z)

            sp1 = vp3 + vmul(f_beam2,        UNIT_Z)
            sp2 = vp3 + vmul(f_beam2 + 5.0,  UNIT_Z)
            sp3 = vp3 + vmul(f_beam2 + 10.0, UNIT_Z)
            lat_splitter.AddBeam(sp0, Single(0.4),   sp1, Single(0.4))
            lat_splitter.AddBeam(sp1, Single(0.4),   sp2, Single(f_top))
            lat_splitter.AddBeam(sp2, Single(f_top), sp3, Single(f_top))

            lat_outlet.AddBeam(vp1, Single(f_beam2), vp2, Single(f_beam2))

        vox_outlet   = Voxels(lat_outlet)
        vox_splitter = Voxels(lat_splitter) & vox_outlet
        return vox_outlet, vox_splitter

    # ─────────────────────────────────────────────────────────────────────
    # Internal fins  (mirrors InternalFins.cs)
    # ─────────────────────────────────────────────────────────────────────

    def _vox_get_turning_fins(self, efluid: str) -> Voxels:
        """Fins at the four 45°/135°/225°/315° corner positions of each turn."""
        f_phi_start = math.pi if efluid == self.HOT else 0.0
        f_wall = 0.4
        f_beam = 0.5 * f_wall
        f_start_z, f_end_z = 0.0, 100.0
        f_total = f_end_z - f_start_z
        f_inter = 0.8

        n_turns  = int(f_total / (2.0 * self.m_plate_thickness + 2.0 * f_inter))
        f_turns  = n_turns - 0.5
        f_slope  = (f_turns * 2.0 * math.pi) / f_total

        lat_fins  = Lattice()
        n_samples = int(f_total / 0.005)
        d_angle   = 20.0

        for i in range(n_samples):
            f_lr  = i / n_samples
            f_z   = f_start_z + f_lr * (f_end_z - f_start_z)
            f_phi = f_phi_start + f_slope * (f_z - f_start_z)
            f_phi_deg = (f_phi / math.pi * 180.0) % 360.0

            at_corner = (
                (45.0  - d_angle < f_phi_deg < 45.0  + d_angle) or
                (135.0 - d_angle < f_phi_deg < 135.0 + d_angle) or
                (225.0 - d_angle < f_phi_deg < 225.0 + d_angle) or
                (315.0 - d_angle < f_phi_deg < 315.0 + d_angle)
            )
            if not at_corner:
                continue

            n_fins = 20
            for j in range(n_fins):
                f_phi_fin = f_phi - (15.0 / 180.0 * math.pi) * math.cos(3.0 * (j / n_fins - 0.5))
                f_inner   = float(self._f_get_inner_radius(f_phi_fin, f_lr))
                f_outer   = float(self._f_get_outer_radius(f_phi_fin, f_lr)) - f_beam
                f_r       = f_inner + 5.0 + j / (n_fins - 1) * (f_outer - 10.0 - f_inner)

                vp1 = self._vec_trafo(VecOperations.vecGetCylPoint(
                    Single(f_r), Single(f_phi_fin), Single(f_z - 0.5 * self.m_plate_thickness)))
                vp2 = self._vec_trafo(VecOperations.vecGetCylPoint(
                    Single(f_r), Single(f_phi_fin), Single(f_z + 0.5 * self.m_plate_thickness)))

                # shift down by 1.5 mm for printability
                vp1 = vec3(vp1.X, vp1.Y, vp1.Z - 1.5)
                vp2 = vec3(vp2.X, vp2.Y, vp2.Z - 1.5)
                vp3 = vmul(0.5, vp1 + vp2) + vmul(3.0, UNIT_Z)

                lat_fins.AddBeam(vp1, Single(f_beam), vp3, Single(f_beam))
                lat_fins.AddBeam(vp2, Single(f_beam), vp3, Single(f_beam))

        return Voxels(lat_fins)

    def _vox_get_straight_fins(self, efluid: str) -> Voxels:
        """Fins along the straight 0°/180° sections; twisted fins at 90°/270°."""
        f_phi_start = math.pi if efluid == self.HOT else 0.0
        f_wall = 0.4
        f_beam = 0.5 * f_wall
        f_start_z, f_end_z = 0.0, 100.0
        f_total = f_end_z - f_start_z
        f_inter = 0.8

        n_turns  = int(f_total / (2.0 * self.m_plate_thickness + 2.0 * f_inter))
        f_turns  = n_turns - 0.5
        f_slope  = (f_turns * 2.0 * math.pi) / f_total

        lat_fins  = Lattice()
        n_samples = int(f_total / 0.005)
        d_angle   = 15.0
        n_fins    = 8

        for i in range(n_samples):
            f_lr  = i / n_samples
            f_z   = f_start_z + f_lr * (f_end_z - f_start_z)
            f_phi = f_phi_start + f_slope * (f_z - f_start_z)
            f_phi_deg = (f_phi / math.pi * 180.0) % 360.0

            at_straight = (
                (0.0   - d_angle < f_phi_deg < 0.0   + d_angle) or
                (360.0 - d_angle < f_phi_deg < 360.0 + d_angle) or
                (180.0 - d_angle < f_phi_deg < 180.0 + d_angle)
            )
            at_90  = (90.0  - d_angle < f_phi_deg < 90.0  + d_angle)
            at_270 = (270.0 - d_angle < f_phi_deg < 270.0 + d_angle)

            if not (at_straight or at_90 or at_270):
                continue

            for j in range(n_fins):
                f_phi_fin = f_phi - (15.0 / 180.0 * math.pi) * math.cos(3.0 * (j / n_fins - 0.5))
                f_inner   = float(self._f_get_inner_radius(f_phi_fin, f_lr))
                f_outer   = float(self._f_get_outer_radius(f_phi_fin, f_lr)) - f_beam
                f_r       = f_inner + 5.0 + j / (n_fins - 1) * (f_outer - 10.0 - f_inner)

                vp1 = self._vec_trafo(VecOperations.vecGetCylPoint(
                    Single(f_r), Single(f_phi_fin), Single(f_z - 0.5 * self.m_plate_thickness)))
                vp2 = self._vec_trafo(VecOperations.vecGetCylPoint(
                    Single(f_r), Single(f_phi_fin), Single(f_z + 0.5 * self.m_plate_thickness)))

                vp1 = vec3(vp1.X, vp1.Y, vp1.Z - 1.5)
                vp2 = vec3(vp2.X, vp2.Y, vp2.Z - 1.5)
                vp3 = vmul(0.5, vp1 + vp2)

                if at_90 or at_270:
                    # twist the fins for flow mixing (mirrors C# dTurnPhi logic)
                    ref_deg = 270.0 if at_270 else 90.0
                    d_turn  = (f_phi_deg - ref_deg + d_angle) / (2.0 * d_angle) * 2.0 * math.pi
                    vp1 = VecOperations.vecRotateAroundZ(vp1, Single(d_turn), vp3)
                    vp2 = VecOperations.vecRotateAroundZ(vp2, Single(d_turn), vp3)

                vp4 = vp3 + vmul(3.0, UNIT_Z)
                lat_fins.AddBeam(vp1, Single(f_beam), vp4, Single(f_beam))
                lat_fins.AddBeam(vp2, Single(f_beam), vp4, Single(f_beam))

        return Voxels(lat_fins)

    # ─────────────────────────────────────────────────────────────────────
    # Outer structure  (mirrors OuterStructure.cs)
    # ─────────────────────────────────────────────────────────────────────

    def _vox_get_outer_structure(self) -> Voxels:
        """Shell wall + oscillating reinforcement ribs that bound the cube."""
        f_total   = 100.0
        f_beam    = 1.0
        lat       = Lattice()
        side_phis = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]

        f_z = 0.0
        while f_z < f_total:
            f_lr = f_z / f_total
            for f_side in side_phis:
                for sign in (+1, -1):
                    f_phi   = f_side + sign * 0.25 * math.pi * math.cos(
                        2.0 * 2.0 * math.pi / f_total * f_z)
                    f_inner = float(self._f_get_outer_radius(f_phi, f_lr)) - 15.0
                    f_outer = float(self._f_get_outer_radius(f_phi, f_lr)) + 15.0
                    vp1 = self._vec_trafo(VecOperations.vecGetCylPoint(
                        Single(f_inner), Single(f_phi), Single(f_z)))
                    vp2 = self._vec_trafo(VecOperations.vecGetCylPoint(
                        Single(f_outer), Single(f_phi), Single(f_z)))
                    lat.AddBeam(vp1, Single(f_beam), vp2, Single(f_beam))
            f_z += 0.3

        vox = Voxels(lat)
        vox.OverOffset(5.0, 0.5)
        vox.Smoothen(1.0)
        vox = vox & self.m_vox_bounding
        return vox

    # ─────────────────────────────────────────────────────────────────────
    # Flange  (mirrors Flange.cs)
    # ─────────────────────────────────────────────────────────────────────

    def _get_flange(self):
        """Bottom mounting flange with six screw bosses and thread cutters.

        Returns (vox_flange, vox_screw_holes, vox_screw_cutter)  –  each is
        a plain Voxels object built by sequential union (no voxCombineAll).
        """
        f_core_r     = 5.0
        f_max_r      = 6.0
        f_cut_len    = 24.0
        f_thread_r   = 3.5
        f_thread_len = 2.0
        f_head_r     = 7.0
        f_head_len   = 10.0

        x_vals = [-60.0, 60.0]
        y_vals = [-38.0,  0.0, 38.0]

        flanges = []
        screws  = []
        cutters = []

        for fx in x_vals:
            for fy in y_vals:
                pt = vec3(fx, fy, 0.0)

                screw_hole = ScrewHole(
                    LocalFrame(pt + vmul(6.0, UNIT_Z)),
                    f_thread_len, f_thread_r, f_head_len, f_head_r,
                )
                screws.append(screw_hole.voxConstruct())

                cyl = BaseCylinder(LocalFrame(pt), 8.0, f_head_r + 5.0)
                flanges.append(cyl.oConstructVoxels())

                cutter = ThreadCutter(
                    LocalFrame(pt - vmul(10.0, UNIT_Z)),
                    f_cut_len, f_max_r, f_core_r, 1.3,
                )
                cutters.append(cutter.voxConstruct())

        # use our Python helper that chains Voxels + Voxels (no voxCombineAll)
        vox_flange      = vox_combine_all(flanges)
        vox_screw_holes = vox_combine_all(screws)
        vox_screw_cut   = vox_combine_all(cutters)
        return vox_flange, vox_screw_holes, vox_screw_cut

    # ─────────────────────────────────────────────────────────────────────
    # IO threads  (mirrors IOThreads.cs)
    # ─────────────────────────────────────────────────────────────────────

    def _vox_get_io_threads(self) -> Voxels:
        """Thread-reinforcement bosses on all four inlet/outlet pipe ends."""
        f_outer_r = 14.0
        f_length  = 12.0
        vec_shift = UNIT_Z

        source_frames = [
            self.m_first_inlet_frame,
            self.m_second_inlet_frame,
            self.m_first_outlet_frame,
            self.m_second_outlet_frame,
        ]
        vox_list = []
        for src in source_frames:
            fr = LocalFrame.oGetTranslatedFrame(src, vec_shift)
            fr = LocalFrame.oGetInvertFrame(fr, True, False)
            fr = LocalFrame.oGetTranslatedFrame(fr, vmul(-f_length, fr.vecGetLocalZ()))
            thread = ThreadReinforcement(fr, f_length, self.m_io_radius, f_outer_r)
            vox_list.append(thread.voxConstruct())

        return vox_combine_all(vox_list)

    # ─────────────────────────────────────────────────────────────────────
    # IO cuts  (mirrors IOCuts.cs)
    # ─────────────────────────────────────────────────────────────────────

    def _vox_get_io_cuts(self) -> Voxels:
        """Shapes that open all four port bore-holes and add entrance chamfers."""
        f_cut_r   = 2.5
        f_cut_len = 12.0

        port_frames = [
            self.m_first_inlet_frame,
            self.m_second_inlet_frame,
            self.m_first_outlet_frame,
            self.m_second_outlet_frame,
        ]
        vox_list = []

        for fr in port_frames:
            manifold = LatticeManifold(fr, f_cut_len, f_cut_r)
            vox_list.append(manifold.oConstructVoxels())

        # chamfer beams at the entrance of each port
        for fr in port_frames:
            tip_frame = LocalFrame.oGetTranslatedFrame(
                fr, vmul(f_cut_len + 2.0, fr.vecGetLocalZ()))
            p_tip  = tip_frame.vecGetPosition()
            p_back = p_tip - vmul(4.0, tip_frame.vecGetLocalZ())
            vox_list.append(Voxels(Sh.latFromBeam(p_tip, p_back, Single(7.0), Single(2.0), False)))

        return vox_combine_all(vox_list)

    # ─────────────────────────────────────────────────────────────────────
    # IO print-supports  (mirrors IOSupports.cs)
    # ─────────────────────────────────────────────────────────────────────

    def _vox_get_io_supports(self) -> Voxels:
        """Lattice support struts under each cantilevered port pipe."""
        f_min_beam = 1.0
        lat = Lattice()

        port_frames = [
            self.m_first_inlet_frame,
            self.m_first_outlet_frame,
            self.m_second_inlet_frame,
            self.m_second_outlet_frame,
        ]

        for fr in port_frames:
            px = float(fr.vecGetPosition().X)
            py = float(fr.vecGetPosition().Y)

            ang1 = -50.0 / 180.0 * math.pi
            if px > 0:
                ang1 = -ang1
            ang2 = -20.0 / 180.0 * math.pi
            if px > 0:
                ang2 = -ang2
            ang_in = 15.0 / 180.0 * math.pi
            if py > 0:
                ang_in = -ang_in

            vec_dir1 = VecOperations.vecRotateAroundAxis(-UNIT_Z, Single(ang1),   UNIT_Y, ZERO)
            vec_dir1 = VecOperations.vecRotateAroundAxis(vec_dir1, Single(ang_in), UNIT_X, ZERO)
            vec_dir2 = VecOperations.vecRotateAroundAxis(-UNIT_Z, Single(ang2),   UNIT_Y, ZERO)
            vec_dir2 = VecOperations.vecRotateAroundAxis(vec_dir2, Single(ang_in), UNIT_X, ZERO)

            for ds in range(30):
                f_lr    = ds / 30.0
                f_max_b = float(Uf.fTransFixed(
                    Single(self.m_io_radius + 6.0),
                    Single(self.m_io_radius + 2.0),
                    Single(f_lr)))
                d_h = (f_max_b - f_min_beam) / math.tan(30.0 / 180.0 * math.pi)

                vec_pt1  = fr.vecGetPosition() + vmul(10.0 - ds, fr.vecGetLocalZ())
                vec_kink = vec_pt1 + vmul(d_h, vec_dir2)
                dir1_z   = float(vec_dir1.Z)
                kink_z   = float(vec_kink.Z)
                t        = -kink_z / dir1_z if abs(dir1_z) > 1e-9 else 0.0
                vec_pt2  = vec_kink + vmul(t, vec_dir1)

                lat.AddBeam(vec_pt1, Single(f_max_b),   vec_kink, Single(f_min_beam))
                lat.AddBeam(vec_kink, Single(f_min_beam), vec_pt2, Single(f_min_beam))

        return Voxels(lat)

    # ─────────────────────────────────────────────────────────────────────
    # Print web  (mirrors PrintWeb.cs)
    # ─────────────────────────────────────────────────────────────────────

    def _vox_get_print_web(self) -> Voxels:
        """Horizontal grooves on the build-plate face for powder removal."""
        f_z    = -4.0
        f_beam = 0.8
        f_y    = 70.0
        lat    = Lattice()
        f_x    = 0.0
        while f_x <= 60.0:
            lat.AddBeam(vec3( f_x, -f_y, f_z), Single(f_beam), vec3( f_x, f_y, f_z), Single(f_beam))
            lat.AddBeam(vec3(-f_x, -f_y, f_z), Single(f_beam), vec3(-f_x, f_y, f_z), Single(f_beam))
            f_x += 10.0
        return Voxels(lat)

    # ─────────────────────────────────────────────────────────────────────
    # Full construction sequence  (mirrors HelixHeatX.cs::voxConstruct)
    # ─────────────────────────────────────────────────────────────────────

    def construct(self, output_dir: Path, headless: bool = True) -> Voxels:
        """Build the complete heat exchanger and export the STL.

        Must be called from inside a Library.Go task thread.

        Parameters
        ----------
        output_dir : directory where the STL is written.
        headless   : when False the PicoGK viewer is live and Sh.PreviewVoxels
                     calls are made after each major construction step,
                     mirroring the original C# voxConstruct sequence.
        """
        def preview(vox, colour=Cp.strRock, alpha=1.0):
            """No-op in headless mode; calls Sh.PreviewVoxels otherwise."""
            if not headless:
                Sh.PreviewVoxels(vox, colour, alpha)

        def screenshot(tag):
            """No-op in headless mode; requests a viewer screenshot otherwise."""
            if not headless:
                Library.oViewer().RequestScreenShot(
                    Sh.strGetExportPath(EExport.TGA, tag))

        screenshot('Screenshot_00')

        print('  building turning fins …')
        vox_hot_corner_fins  = self._vox_get_turning_fins(self.HOT)
        vox_cool_corner_fins = self._vox_get_turning_fins(self.COOL)
        vox_all_corner_fins  = vox_hot_corner_fins + vox_cool_corner_fins
        preview(vox_all_corner_fins, Cp.strWarning, 0.5)
        screenshot('Screenshot_01')

        print('  building straight fins …')
        vox_hot_straight  = self._vox_get_straight_fins(self.HOT)
        vox_cool_straight = self._vox_get_straight_fins(self.COOL)
        vox_all_straight  = vox_hot_straight + vox_cool_straight
        preview(vox_all_straight, Cp.strToothpaste, 0.5)
        screenshot('Screenshot_02')

        vox_fins = vox_all_corner_fins + vox_all_straight
        preview(vox_fins, Cp.strRock)

        print('  building outer structure …')
        vox_structure = self._vox_get_outer_structure()
        preview(vox_structure, Cp.strRock, 0.3)

        print('  building hot-fluid helical void …')
        vox_hot_void, vox_hot_split = self._get_helical_void(self.HOT)

        print('  building cool-fluid helical void …')
        vox_cool_void, vox_cool_split = self._get_helical_void(self.COOL)

        # enforce wall-thickness separation between the two fluid channels
        vox_hot_void  = vox_hot_void  - vox_cool_void.voxOffset(self.m_wall_thickness)
        vox_cool_void = vox_cool_void - vox_hot_void.voxOffset(self.m_wall_thickness)
        preview(vox_hot_void,  Cp.strPitaya, 0.5)
        screenshot('Screenshot_03')
        preview(vox_cool_void, Cp.strFrozen, 0.5)
        screenshot('Screenshot_04')

        vox_inner = vox_hot_void + vox_cool_void
        vox_split = vox_hot_split + vox_cool_split
        vox_outer = vox_inner.voxOffset(0.9)

        print('  building flange …')
        vox_flange, vox_screw_holes, _vox_flange_cut = self._get_flange()
        preview(vox_screw_holes,  Cp.strRed)
        preview(_vox_flange_cut,  Cp.strRed, 0.4)
        screenshot('Screenshot_05')

        vox_flange.Fillet(5.0)
        vox_flange.Smoothen(0.5)
        preview(vox_flange, Cp.strGray, 0.6)
        screenshot('Screenshot_06')

        print('  building IO supports …')
        vox_outer = vox_outer + vox_flange + self._vox_get_io_supports()
        vox_outer.Fillet(5.0)
        vox_outer.Smoothen(0.5)
        screenshot('Screenshot_07')

        vox_outer = self._add_centre_piece(vox_outer)
        vox_outer = vox_outer + vox_structure - vox_screw_holes
        vox_outer.ProjectZSlice(4.0, -4.0)
        vox_outer = vox_outer - self._vox_get_print_web()

        vox_result = vox_outer - vox_inner
        vox_result = vox_result + vox_fins + vox_split
        vox_result = vox_result & self.m_vox_bounding

        print('  building IO threads and cuts …')
        vox_result = vox_result + self._vox_get_io_threads() - self._vox_get_io_cuts()
        screenshot('Screenshot_08')

        if not headless:
            Library.oViewer().RemoveAllObjects()
        preview(vox_result, Cp.strRock)
        screenshot('Screenshot_09')

        stl_path = str(output_dir / 'HelixHeatX_Python.STL')
        print(f'  exporting STL → {stl_path}')
        export_voxels_to_stl(vox_result, stl_path)

        return vox_result


print('HelixHeatX class defined.')

import time

# ── toggle this to open the interactive PicoGK viewer ─────────────────────
HEADLESS   = False     # False → viewer window with step-by-step previews
VOXEL_SIZE = 0.5     # mm – match the original example

# Instantiate outside Library.Go so we can inspect parameters before the run
hx = HelixHeatX()
print(f'Plate thickness : {hx.m_plate_thickness} mm')
print(f'Wall thickness  : {hx.m_wall_thickness} mm')
print(f'IO radius       : {hx.m_io_radius} mm')
print()

t0 = time.time()
mode_label = 'headless' if HEADLESS else 'with viewer'
print(f'Starting construction  (voxel size = {VOXEL_SIZE} mm, mode = {mode_label}) …')

def _task():
    hx.construct(OUTPUT_DIR, headless=HEADLESS)

run_in_library(_task, voxel_size=VOXEL_SIZE, output_dir=OUTPUT_DIR, headless=HEADLESS)

elapsed = time.time() - t0
print(f'Done in {elapsed:.1f} s')
print(f'STL written to: {STL_PATH}')

#!pip install -U pyvista[jupyter] numpy-stl
try:
    import pyvista as pv
    from stl import mesh as stl_mesh
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "STL visualisation needs optional packages. Install them in the "
        "LEAP71 conda env with: pip install 'pyvista[jupyter]' numpy-stl"
    ) from exc
import numpy as np

pv.global_theme.allow_empty_mesh = True

# 'trame' for interactive rotate/zoom; 'static' for a plain PNG
PYVISTA_BACKEND = 'trame'
pv.set_jupyter_backend(PYVISTA_BACKEND)

if not STL_PATH.exists():
    raise FileNotFoundError(f'STL not found: {STL_PATH} - did Cell 3 complete?')

# Read the file using numpy-stl (which handles bad normals better)
stl_obj = stl_mesh.Mesh.from_file(str(STL_PATH))

# Convert numpy-stl mesh to PyVista PolyData
# stl_obj.vectors is (n_triangles, 3, 3) → reshape to flat vertices + triangle indices
vertices = stl_obj.vectors.reshape(-1, 3)
n_triangles = stl_obj.vectors.shape[0]
faces = np.column_stack([np.full(n_triangles, 3), 
                         np.arange(n_triangles * 3).reshape(-1, 3)])
mesh = pv.PolyData(vertices, faces)

# Recompute normals to fix any non-finite values
mesh = mesh.compute_normals()

print(f'Loaded {mesh.n_cells:,} triangles | bounds: {[round(b, 1) for b in mesh.bounds]}')

pl = pv.Plotter(notebook=True)
pl.add_mesh(
    mesh,
    color='#2e85cc',
    show_edges=False,
    smooth_shading=True,
    specular=0.6,
    specular_power=20,
    ambient=0.15,
)
pl.add_axes()
pl.set_background('#1a1a2e')
pl.camera_position = 'iso'
pl.show()


