set(CLI11_VER "1.9.1" CACHE STRING "CLI11 Version: see https://github.com/CLIUtils/CLI11/releases")
mark_as_advanced(CLI11_VER)
set(CLI11_DIRS ${CMAKE_BINARY_DIR}/CMakeFiles/CLI11)
file(DOWNLOAD https://github.com/CLIUtils/CLI11/releases/download/v${CLI11_VER}/CLI11.hpp ${CLI11_DIRS}/CLI11.hpp)
include_directories(${CLI11_DIRS})

include_directories(
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_SOURCE_DIR}/src/lapack
)

if (ENABLE_CUDA_BLAS)
	include_directories(${CMAKE_SOURCE_DIR}/src/cuda)
	cuda_add_executable(
		HSD
		${CMAKE_SOURCE_DIR}/src/HSD.cpp
		${CMAKE_SOURCE_DIR}/src/cuda/grad.cu
		${CMAKE_SOURCE_DIR}/wrapper/main.cpp ${HSD_WRAPEPR}
	)
	CUDA_ADD_CUBLAS_TO_TARGET(HSD)
	target_link_libraries(HSD ${CUDA_BLAS_LIBRARIES})
	target_compile_definitions(HSD PRIVATE _USE_CUDA_BLAS)
else()
	add_executable(
		HSD
		${CMAKE_SOURCE_DIR}/src/HSD.cpp
		${CMAKE_SOURCE_DIR}/wrapper/main.cpp
	)
endif()

if (OPENMP_FOUND)
    target_compile_definitions(HSD PRIVATE _USE_OPENMP)
endif()

option(BLAS_SYSV "Enable the BLAS linear solver without the inverse Hessian matrix." ON)
mark_as_advanced(BLAS_SYSV)
if (BLAS_SYSV)
   target_compile_definitions(HSD PRIVATE _USE_SYSV)
endif()
target_compile_definitions(HSD PRIVATE HSD_VERSION=\"v${HSD_VERSION}\")

mark_as_advanced(CLI11_VER)
target_link_libraries(HSD Mesh ${LAPACK_LIBRARIES})
