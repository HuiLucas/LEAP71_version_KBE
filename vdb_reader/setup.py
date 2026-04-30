"""Build the vdb_numpy pybind11 extension in-place."""
from setuptools import setup, Extension
import pybind11
import subprocess, sys, os

OPENVDB_SRC  = "/workspace/PicoGKRuntime/openvdb/openvdb"
OPENVDB_BUILD = "/workspace/PicoGKRuntime/build"
PYBIND11_INC  = pybind11.get_include()

ext = Extension(
    "vdb_numpy",
    sources=["vdb_numpy.cpp"],
    include_dirs=[
        PYBIND11_INC,
        OPENVDB_SRC,                               # so <openvdb/*.h> resolves
        f"{OPENVDB_BUILD}/openvdb/openvdb/openvdb",  # generated version.h
        "/usr/include",
    ],
    library_dirs=[
        f"{OPENVDB_BUILD}/Lib",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
    ],
    libraries=["openvdb", "tbb", "boost_iostreams", "z"],
    extra_compile_args=[
        "-std=c++17",
        "-O2",
        "-DOPENVDB_ABI_VERSION_NUMBER=11",
        "-DOPENVDB_USE_BLOSC",
    ],
    extra_link_args=[
        "-Wl,-rpath,/usr/local/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-lblosc",
    ],
    language="c++",
)

setup(name="vdb_numpy", ext_modules=[ext])
