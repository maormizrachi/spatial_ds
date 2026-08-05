cmake_minimum_required(VERSION 3.14)
project(spatial_ds LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Intel|IntelLLVM)$")
    add_compile_options(-fp-model precise)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only library -- no sources to compile.
# Consumer must add the parent of this directory to include paths
# so that #include <spatial_ds/...> resolves correctly.
#
# External dependencies (consumer must provide include paths):
#   - mpi_utils (cpp-MPI-utils)  -- required by MPI-dependent headers
