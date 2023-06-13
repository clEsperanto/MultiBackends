cmake_minimum_required(VERSION 3.20)

project(CLIc VERSION 0.0.1) # GEM : GPU Execution Manager

# Branch or Release tag to retrive the ocl/cu files
set(CLESPERANTO_KERNEL_TAG clesperanto_kernels)

# C++ standard and extensions
set(CMAKE_CXX_STANDARD 17)           # C++17, global property instead of target property because of wrapper
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # C++17 is required (no fallback)
set(CMAKE_CXX_EXTENSIONS OFF)        # Don't use e.g. GNU extension (like -std=gnu++11) for portability

# set library name
if(NOT LIBRARY_NAME)
  set(LIBRARY_NAME ${PROJECT_NAME})
endif()

# set build as static or shared library (default: static)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
if (WIN32 AND BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# manage build type options (default: Release)
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(isMultiConfig)
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo" CACHE STRING "" FORCE)
  message(STATUS "CMAKE_CONFIGURATION_TYPES: ${CMAKE_CONFIGURATION_TYPES}")
  message(STATUS "CMAKE_GENERATOR: Multi-config")
else()
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
  message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
  message(STATUS "CMAKE_GENERATOR: Single-config")
endif()
set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELEASE_POSTFIX "")

# managed third party libraries
add_subdirectory(thirdparty)

# list project source files
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
set(INCLUDE_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/include)

# find GPU Framework (OpenCL, CUDA)
find_package(OpenCL)
find_package(CUDAToolkit)
if (NOT OpenCL_FOUND AND NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "No GPU framework found (OpenCL, CUDA). Please install one of them.")
endif()

# target library definition
add_library(${LIBRARY_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${KERNELS_HEADER_DIR}>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)        # force C++ linker
set_target_properties(${LIBRARY_NAME} PROPERTIES VERSION ${PROJECT_VERSION}) # set version
add_dependencies(${LIBRARY_NAME} generate_kernels)

# link GPU framework to the target
set(CLE_OPENCL OFF)
set(CLE_CUDA OFF)
if (OpenCL_FOUND)
    set(CLE_OPENCL ON)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} PUBLIC ${OpenCL_LIBRARIES})
endif()
if (CUDAToolkit_FOUND)
    set(CLE_CUDA ON)
    set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::OpenCL) 
endif()

# add the variables to the compile definitions for the source code
add_compile_definitions(
    $<$<BOOL:${CLE_CUDA}>:CLE_CUDA>
    $<$<BOOL:${CLE_OPENCL}>:CLE_OPENCL>
)

# enable testing
include(CTest)
add_subdirectory(tests)
