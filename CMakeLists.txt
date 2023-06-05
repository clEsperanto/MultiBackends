cmake_minimum_required(VERSION 3.20)

project(Cobra VERSION 0.0.1) 
# COBRA : CUDA OPENCL BACKEND RUNTIME API

# C++ standard and extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Don't use e.g. GNU extension (like -std=gnu++11) for portability

if(NOT LIBRARY_NAME)
  set(LIBRARY_NAME ${PROJECT_NAME})
endif()

# set build as static or shared
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
if (WIN32 AND BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Manage build type options (default: Release)
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
# Configurations tag to avoid compilation colliding
set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELEASE_POSTFIX "")

# list project source files
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
set(INCLUDE_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/include)

# find GPU Framework (OpenCL, CUDA)
find_package(OpenCL)
find_package(CUDAToolkit)
if (NOT OpenCL_FOUND AND NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "No GPU framework found (OpenCL, CUDA).")
endif()

# target definition
add_library(${LIBRARY_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${LIBRARY_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

# link GPU framework to the target
set(CLE_OPENCL OFF)
set(CLE_CUDA OFF)
if (OpenCL_FOUND)
    set(CLE_OPENCL ON)
    target_include_directories(${LIBRARY_NAME} INTERFACE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} INTERFACE ${OpenCL_LIBRARIES})
endif()
if (CUDAToolkit_FOUND)
    set(CLE_CUDA ON)
    set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${LIBRARY_NAME} INTERFACE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} INTERFACE CUDA::cudart CUDA::cuda_driver CUDA::OpenCL)
endif()

# Add the variables to the compile definitions
add_compile_definitions(
    $<$<BOOL:${CLE_CUDA}>:CLE_CUDA>
    $<$<BOOL:${CLE_OPENCL}>:CLE_OPENCL>
)

include(CTest)
add_subdirectory(tests)

# install headers
install(DIRECTORY ${INCLUDE_DIR_LIST} DESTINATION include)
