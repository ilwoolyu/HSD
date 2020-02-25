cmake_minimum_required(VERSION 3.5)

project(HSD)

set(HSD_VERSION_MAJOR 1)
set(HSD_VERSION_MINOR 2)
set(HSD_VERSION_PATCH 6)
set(HSD_VERSION
${HSD_VERSION_MAJOR}.${HSD_VERSION_MINOR}.${HSD_VERSION_PATCH})

set (CMAKE_CXX_STANDARD 11)

# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified.")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
	mark_as_advanced(CMAKE_BUILD_TYPE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(LAPACK REQUIRED)
find_package(MeshLib REQUIRED)
find_package(CUDA QUIET)
find_package(OpenMP QUIET)

# environmental variables
if (CUDA_FOUND)
	option(ENABLE_CUDA_BLAS "Enable CUDA BLAS for fast Levenberg-Marquardt optimization" ${CUDA_FOUND})
endif()

if (ENABLE_CUDA_BLAS)
	set(CUDA_NVCC_FLAGS	${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_61,code=sm_61)
else()
	find_package(BLAS REQUIRED)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopenblas")
endif()

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# add subdiretories
add_subdirectory(wrapper)
