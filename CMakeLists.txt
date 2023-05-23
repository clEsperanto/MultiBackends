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
find_package(OpenCL)
find_package(CUDA)
find_package(CUDAToolkit)


message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")


# Add the variables to the compile definitions
add_compile_definitions(
    $<$<BOOL:${CLE_CUDA}>:CLE_CUDA>
    $<$<BOOL:${CLE_OPENCL}>:CLE_OPENCL>
)

# add target library
add_library(${LIBRARY_NAME} STATIC ${SOURCES} ${HEADERS})

# add include directories
target_include_directories(${LIBRARY_NAME} PUBLIC include)

# add OpenCL and CUDA include directories
set(CLE_OPENCL 0)
set(CLE_CUDA 0)
if (OpenCL_FOUND)
    set(CLE_OPENCL 1)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} PUBLIC ${OpenCL_LIBRARIES})
endif()
if (CUDA_FOUND)
    set(CLE_CUDA 1)
    set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    # target_link_libraries(${LIBRARY_NAME} PUBLIC ${CUDA_LIBRARIES})
    target_link_libraries(${LIBRARY_NAME} PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::OpenCL)
endif()

# Add the variables to the compile definitions
add_compile_definitions(
    $<$<BOOL:${CLE_CUDA}>:CLE_CUDA>
    $<$<BOOL:${CLE_OPENCL}>:CLE_OPENCL>
)

set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)


include(CTest)
add_subdirectory(tests)

# Reading the preamble for both CUDA & OpenCL
file(DOWNLOAD "https://raw.githubusercontent.com/CherifMZ/clij-opencl-kernels/cuda_preamble/preamble/preamble.cu" "${CMAKE_CURRENT_BINARY_DIR}/preamble.cu")
file(DOWNLOAD "https://raw.githubusercontent.com/CherifMZ/clij-opencl-kernels/cuda_preamble/preamble/preamble.cl" "${CMAKE_CURRENT_BINARY_DIR}/preamble.cl")

# Set the input and output file path for both CUDA & OpenCL
set(INPUT_FILE_CUDA "${CMAKE_CURRENT_BINARY_DIR}/preamble.cu")
set(OUTPUT_FILE_CUDA "${CMAKE_CURRENT_BINARY_DIR}/output_cu.h")

set(INPUT_FILE_CL "${CMAKE_CURRENT_BINARY_DIR}/preamble.cl")
set(OUTPUT_FILE_CL "${CMAKE_CURRENT_BINARY_DIR}/output_cl.h")

# Read the content of the input file
file(READ "${INPUT_FILE_CUDA}" FILE_CONTENT_CUDA)
file(READ "${INPUT_FILE_CL}" FILE_CONTENT_CL)

# Split the contents into multiple string literals
string(REPLACE "\n" "\\n\"\n\"" FILE_CONTENT_CUDA "${FILE_CONTENT_CUDA}")

# Generate output.h file with the contents of preamble.cu split into string literals
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "#ifndef CUDA_PREAMBLE_H\n#define CUDA_PREAMBLE_H\n\n")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "namespace cudaKernel { \n")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "const char* preamble = \n")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\"")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "${FILE_CONTENT_CUDA}")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\\n\"")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\n\"\";")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\n } \n")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\n\n#endif // CUDA_PREAMBLE_H\n")

string(REPLACE "\n" "\\n\"\n\"" FILE_CONTENT_CL "${FILE_CONTENT_CL}")

# Generate output.h file with the contents of preamble.cl split into string literals
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "#ifndef OCL_PREAMBLE_H\n#define OCL_PREAMBLE_H\n\n")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "namespace oclKernel { \n")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "const char* preamble = \n")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\"")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "${FILE_CONTENT_CL}")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\\n\"")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\n\"\";")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\n } \n")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\n\n#endif // OCL_PREAMBLE_H\n")