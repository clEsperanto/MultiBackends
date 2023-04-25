cmake_minimum_required(VERSION 3.20)

project(Backends)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Don't use e.g. GNU extension (like -std=gnu++11) for portability

if(NOT LIBRARY_NAME)
  set(LIBRARY_NAME ${PROJECT_NAME})
endif()

# glob all cpp file in src folder
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

# find OpenCL and CUDA
find_package(OpenCL REQUIRED)
find_package(CUDA REQUIRED)

# add target library
add_library(${LIBRARY_NAME} STATIC ${SOURCES} ${HEADERS})

# add include directories
target_include_directories(${LIBRARY_NAME} PUBLIC include)

# add OpenCL and CUDA include directories
if (OpenCL_FOUND)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} PUBLIC ${OpenCL_LIBRARIES})
endif()
if (CUDA_FOUND)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} PUBLIC ${CUDA_LIBRARIES})
endif()

set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)


include(CTest)
add_subdirectory(tests)