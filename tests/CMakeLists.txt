# # find OpenCL and CUDA
# find_package(OpenCL)
# find_package(CUDA)
# find_package(CUDAToolkit)

# message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")


# # Add the variables to the compile definitions
# add_compile_definitions(
#     $<$<BOOL:${CLE_CUDA}>:CLE_CUDA>
#     $<$<BOOL:${CLE_OPENCL}>:CLE_OPENCL>
# )

# # add target library
# add_library(${LIBRARY_NAME} STATIC ${SOURCES} ${HEADERS})


# # add OpenCL and CUDA include directories
# set(CLE_OPENCL 0)
# set(CLE_CUDA 0)
# if (OpenCL_FOUND)
#     set(CLE_OPENCL 1)
#     target_include_directories(${LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
#     target_link_libraries(${LIBRARY_NAME} PUBLIC ${OpenCL_LIBRARIES})
# endif()
# if (CUDA_FOUND)
#     set(CLE_CUDA 1)
#     set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#     set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
#     target_include_directories(${LIBRARY_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
#     # target_link_libraries(${LIBRARY_NAME} PUBLIC ${CUDA_LIBRARIES})
#     target_link_libraries(${LIBRARY_NAME} PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::OpenCL)
# endif()


add_executable(backend_test backend_test.cpp)
add_dependencies(backend_test ${LIBRARY_NAME})
# target_link_libraries(backend_test PRIVATE ${LIBRARY_NAME})
target_include_directories(backend_test PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
set_target_properties(backend_test PROPERTIES FOLDER "Tests")

add_executable(array_test array_test.cpp)
add_dependencies(array_test ${LIBRARY_NAME})
# target_link_libraries(array_test PRIVATE ${LIBRARY_NAME})
target_include_directories(array_test PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
set_target_properties(array_test PROPERTIES FOLDER "Tests")

add_executable(push_pull_test push_pull_test.cpp)
add_dependencies(push_pull_test ${LIBRARY_NAME})
# target_link_libraries(push_pull_test PRIVATE ${LIBRARY_NAME})
target_include_directories(push_pull_test PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
set_target_properties(push_pull_test PROPERTIES FOLDER "Tests")

add_test(NAME backend_test COMMAND backend_test)
add_test(NAME array_test COMMAND array_test)
add_test(NAME push_pull_test COMMAND push_pull_test)