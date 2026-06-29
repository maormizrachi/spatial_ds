# spatial_ds

A C++17 header-only library for spatial data structures.

## Overview

`spatial_ds` provides templated spatial data structures for efficient spatial queries in N-dimensional spaces. The library includes octrees, KD-trees, binary search trees, range trees, and geometric primitives, with optional MPI support for distributed octree partitioning.

## Features

- **OctTree** — templated 3D octree with point insertion, lookup, removal, and sphere range queries
- **DistributedOctTree** — MPI-distributed octree for rank-aware spatial queries
- **KDTree** — k-dimensional tree for multi-dimensional nearest-neighbor and range search
- **BinaryTree / AVLTree** — generic binary search tree and self-balancing AVL variant
- **RangeTree / GroupRangeTree** — d-dimensional range trees built on binary trees
- **GroupTree** — binary tree variant storing multiple values per node
- **Geometric primitives** — axis-aligned bounding boxes, spheres, and intersection utilities
- **SpatialDsError** — structured exception type with contextual key–value entries

## Architecture

| Component | Description |
|-----------|-------------|
| `OctTree/` | Templated 3D octree (`OctTree<T>`) for spatial indexing. Nodes carry bounding boxes; supports insert, find, remove, and sphere-based range queries. |
| `DistributedOctTree/` | Wraps a serial octree with MPI rank metadata (`OctTreeRankedValue`) for distributed domain decomposition. Queries return the set of ranks intersecting a sphere. |
| `KDTree/` | KD-tree (`KDTree<T, D>`) alternating partition axes across D dimensions. Supports insert, find, and sphere range queries. |
| `BinaryTree.hpp` | Generic binary search tree with subtree min/max tracking, successor/predecessor search, and customizable comparison. |
| `AVLTree/` | Self-balancing AVL tree extending `BinaryTree<T>`. |
| `RangeTree/` | Multi-dimensional range tree (`RangeTree<T>`) for fast orthogonal range queries, built recursively over coordinate dimensions. |
| `GroupTree/` | Binary tree storing up to N values per node, with splitting on overflow. |
| `GroupRangeTree/` | Combines `GroupTree` with recursive range-tree structure for grouped range queries. |
| `utils/BoundingBox.hpp` | Axis-aligned 3D bounding box with intersection and containment tests. |
| `utils/Sphere.hpp` | Sphere primitive defined by center and radius. |
| `utils/geometry.hpp` | Geometric utilities (e.g., sphere–box intersection). |
| `utils/raw_type.h` | Type trait helper (`is_raw_type_defined`) for extracting coordinate types from point wrappers. |
| `SpatialDsError.hpp` | Exception class extending `std::runtime_error` with attachable diagnostic entries. |

## Requirements

- **C++17** compiler
- **CMake 3.14+** (optional; for project integration)
- **MPI** (optional; required when including `DistributedOctTree/`)
- **Boost.Container** (optional; used by `DistributedOctTree/` for `flat_set` and `small_vector`)
- **Vector Class Library** (optional; enabled via `USE_VCL_VECTORIZATION` for SIMD-accelerated geometry)

## External Dependencies

When building with MPI support, the library depends on [cpp-MPI-utils](https://github.com/maormizrachi/cpp-MPI-utils) (`mpi_utils`) for serialization of bounding boxes, spheres, and ranked octree values. The consumer must add `mpi_utils` to the include path.

## Usage

Add the parent directory of `spatial_ds` to your include path so that headers resolve as `<spatial_ds/...>`:

```cmake
target_include_directories(my_target PRIVATE /path/to/parent)
```

```cpp
#include <spatial_ds/OctTree/OctTree.hpp>
#include <spatial_ds/utils/Sphere.hpp>

// Point type must support operator[], coord_type, and optionally Raw_type
struct Point3 {
    using coord_type = double;
    coord_type coords[3];
    coord_type operator[](int i) const { return coords[i]; }
};

using Tree = OctTree<Point3>;

int main() {
    Point3 ll{{0, 0, 0}}, ur{{1, 1, 1}};
    Tree tree(ll, ur);

    Point3 p{{0.5, 0.5, 0.5}};
    tree.insert(p);

    Sphere<Point3> query(p, 0.1);
    auto neighbors = tree.range(query);

    return tree.find(p) ? 0 : 1;
}
```

For distributed usage with MPI:

```cpp
#include <spatial_ds/DistributedOctTree/DistributedOctTree.hpp>

DistributedOctTree<Point3> distTree(&serialTree, /*detailedNodeInfo=*/false, MPI_COMM_WORLD);
auto ranks = distTree.getIntersectingRanks(Sphere<Point3>(center, radius));
```

## Error Handling

Operations that detect invalid state throw `SpatialDsError`, a subclass of `std::runtime_error`. Contextual details can be attached via `addEntry(name, value)` and retrieved with `getEntries()`:

```cpp
#include <spatial_ds/SpatialDsError.hpp>

try {
    // spatial_ds operation
} catch (const SpatialDsError &e) {
    std::cerr << e.getErrorMessage() << std::endl;
    for (const auto &[name, val] : e.getEntries()) {
        std::cerr << "  " << name << ": " << val << std::endl;
    }
}
```

## License

[License placeholder — specify license here.]
