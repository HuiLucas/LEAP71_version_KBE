// Minimal pybind11 extension: read OpenVDB FloatGrid → numpy array
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/math/BBox.h>

#include <string>
#include <vector>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;

// Returns (dense_array [Z,Y,X], origin [x,y,z], voxel_size)
// where dense_array is float32, shape (nz,ny,nx), and
// origin is the world-space coordinate of voxel (0,0,0).
py::tuple read_vdb_as_numpy(const std::string& filepath, const std::string& grid_name = "")
{
    openvdb::initialize();

    openvdb::io::File file(filepath);
    file.open(false);  // false = don't delay loading

    openvdb::GridBase::Ptr base_grid;

    if (grid_name.empty()) {
        // pick the first FloatGrid
        for (openvdb::io::File::NameIterator it = file.beginName(); it != file.endName(); ++it) {
            base_grid = file.readGrid(it.gridName());
            if (openvdb::FloatGrid::Ptr fg = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid))
                break;
            base_grid.reset();
        }
    } else {
        base_grid = file.readGrid(grid_name);
    }
    file.close();

    if (!base_grid)
        throw std::runtime_error("No FloatGrid found in: " + filepath);

    openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
    if (!grid)
        throw std::runtime_error("Grid '" + base_grid->getName() + "' is not a FloatGrid");

    // Determine bounding box in index space
    openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    if (bbox.empty()) {
        // return empty array
        py::array_t<float> arr({0,0,0});
        return py::make_tuple(arr, py::make_tuple(0.0f,0.0f,0.0f), 1.0f);
    }

    openvdb::Coord mn = bbox.min();
    openvdb::Coord mx = bbox.max();

    int nx = mx.x() - mn.x() + 1;
    int ny = mx.y() - mn.y() + 1;
    int nz = mx.z() - mn.z() + 1;

    // Allocate dense array (Z, Y, X ordering → standard numpy)
    py::array_t<float> arr({nz, ny, nx});
    auto buf = arr.mutable_unchecked<3>();

    // Fill with background value first
    float bg = grid->background();
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
                buf(z, y, x) = bg;

    // Copy active voxels
    for (auto it = grid->cbeginValueOn(); it; ++it) {
        int ix = it.getCoord().x() - mn.x();
        int iy = it.getCoord().y() - mn.y();
        int iz = it.getCoord().z() - mn.z();
        buf(iz, iy, ix) = it.getValue();
    }

    // World-space origin of the min corner
    openvdb::Vec3d world_origin = grid->indexToWorld(mn);
    double voxel_size = grid->voxelSize()[0];

    return py::make_tuple(
        arr,
        py::make_tuple((float)world_origin.x(), (float)world_origin.y(), (float)world_origin.z()),
        (float)voxel_size
    );
}

// Returns list of grid names in the file
std::vector<std::string> list_grids(const std::string& filepath)
{
    openvdb::initialize();
    openvdb::io::File file(filepath);
    file.open(false);
    std::vector<std::string> names;
    for (auto it = file.beginName(); it != file.endName(); ++it)
        names.push_back(it.gridName());
    file.close();
    return names;
}

PYBIND11_MODULE(vdb_numpy, m) {
    m.doc() = "Minimal OpenVDB → numpy reader";
    m.def("read_vdb_as_numpy", &read_vdb_as_numpy,
          py::arg("filepath"), py::arg("grid_name") = "",
          "Read first FloatGrid from a VDB file into a dense numpy array.\n"
          "Returns (array[Z,Y,X], origin(x,y,z), voxel_size_mm)");
    m.def("list_grids", &list_grids, py::arg("filepath"),
          "List all grid names in a VDB file");
}
