cmake_minimum_required(VERSION 3.20)

project(BackEnds)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# glob all cpp file in src folder
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

# find OpenCL and CUDA
find_package(OpenCL REQUIRED)
find_package(CUDA REQUIRED)

# add target library
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

# add OpenCL and CUDA include directories
if (OpenCL_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
endif()
if (CUDA_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

# add include directories
target_include_directories(${PROJECT_NAME} PUBLIC include)

# add OpenCL and CUDA libraries to target dependencies
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})

