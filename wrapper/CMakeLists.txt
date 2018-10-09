include(${GenerateCLP_USE_FILE})

include_directories(
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_SOURCE_DIR}/src/lapack
)

generateCLP(HSD_WRAPEPR ${CMAKE_SOURCE_DIR}/wrapper/HSD.xml)

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
		${CMAKE_SOURCE_DIR}/wrapper/main.cpp ${HSD_WRAPEPR}
	)
endif()

if (OPENMP_FOUND)
    target_compile_definitions(HSD PRIVATE _USE_OPENMP)
endif()

target_link_libraries(HSD Mesh ${LAPACK_LIBRARIES})
