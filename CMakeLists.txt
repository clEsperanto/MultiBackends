cmake_minimum_required(VERSION 3.20)

project(Backends)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Don't use e.g. GNU extension (like -std=gnu++11) for portability
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

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

FetchContent_Declare(OpenCL_Kernels
    GIT_REPOSITORY https://github.com/CherifMZ/clij-opencl-kernels.git
    GIT_TAG        cuda_preamble
    BUILD_ALWAYS OFF
    CONFIGURE_COMMAND ""
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernel_source"
)

if(NOT OpenCL_Kernels_POPULATED)
    FetchContent_Populate(OpenCL_Kernels)
endif()

# Set the destination folder paths
set(DESTINATION_FOLDER "${CMAKE_CURRENT_BINARY_DIR}/kernel_destination")

# Get a list of files in the source folder for preamble & kernels
file(GLOB SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/kernel_source/kernels/*.cl")
set(INPUT_FILE_CUDA "${CMAKE_CURRENT_BINARY_DIR}/kernel_source/preamble/preamble.cu")
set(INPUT_FILE_CL "${CMAKE_CURRENT_BINARY_DIR}/kernel_source/preamble/preamble.cl")

# Iterate over the source files of kernels
foreach(SOURCE_FILE ${SOURCE_FILES})
    # Get the file name without extension
    get_filename_component(FILE_NAME ${SOURCE_FILE} NAME_WE)
    
    # Use of uppercase
    string(TOUPPER "${FILE_NAME}" UPPERCASE_NAME)

    # Read the file content
    file(READ "${SOURCE_FILE}" FILE_CONTENT)

    string(REPLACE "\n" "\\n\"\n\"" FILE_CONTENT "${FILE_CONTENT}")

    file(WRITE "${DESTINATION_FOLDER}/${FILE_NAME}.h" "#ifndef" " ${UPPERCASE_NAME}_H\n" "#define" " ${UPPERCASE_NAME}_H\n\n")
    file(APPEND "${DESTINATION_FOLDER}/${FILE_NAME}.h" "namespace oclKernel { \n")
    file(APPEND "${DESTINATION_FOLDER}/${FILE_NAME}.h" "const char* ${FILE_NAME} = \n")
    file(APPEND "${DESTINATION_FOLDER}/${FILE_NAME}.h" "\"")
    file(APPEND "${DESTINATION_FOLDER}/${FILE_NAME}.h" "${FILE_CONTENT}")
    file(APPEND "${DESTINATION_FOLDER}/${FILE_NAME}.h" "\\n\"")
    file(APPEND "${DESTINATION_FOLDER}/${FILE_NAME}.h" "\n\"\";")
    file(APPEND "${DESTINATION_FOLDER}/${FILE_NAME}.h" "\n } \n")
    file(APPEND "${DESTINATION_FOLDER}/${FILE_NAME}.h" "\n\n#endif // " "${UPPERCASE_NAME}_H\n")

endforeach()

# Read CUDA preamble
file(READ "${INPUT_FILE_CUDA}" FILE_CONTENT_CUDA)
string(REPLACE "\n" "\\n\"\n\"" FILE_CONTENT_CUDA "${FILE_CONTENT_CUDA}")

file(WRITE "${DESTINATION_FOLDER}/cuda_preamble.h" "#ifndef CUDA_PREAMBLE_H\n#define CUDA_PREAMBLE_H\n\n")
file(APPEND "${DESTINATION_FOLDER}/cuda_preamble.h" "namespace cudaKernel { \n")
file(APPEND "${DESTINATION_FOLDER}/cuda_preamble.h" "const char* preamble = \n")
file(APPEND "${DESTINATION_FOLDER}/cuda_preamble.h" "\"")
file(APPEND "${DESTINATION_FOLDER}/cuda_preamble.h" "${FILE_CONTENT_CUDA}")
file(APPEND "${DESTINATION_FOLDER}/cuda_preamble.h" "\\n\"")
file(APPEND "${DESTINATION_FOLDER}/cuda_preamble.h" "\n\"\";")
file(APPEND "${DESTINATION_FOLDER}/cuda_preamble.h" "\n } \n")
file(APPEND "${DESTINATION_FOLDER}/cuda_preamble.h" "\n\n#endif // CUDA_PREAMBLE_H\n")

# Read OCL preamble
file(READ "${INPUT_FILE_CL}" INPUT_FILE_CL)
string(REPLACE "\n" "\\n\"\n\"" INPUT_FILE_CL "${INPUT_FILE_CL}")

file(WRITE "${DESTINATION_FOLDER}/ocl_preamble.h" "#ifndef OCL_PREAMBLE_H\n#define OCL_PREAMBLE_H\n\n")
file(APPEND "${DESTINATION_FOLDER}/ocl_preamble.h" "namespace oclKernel { \n")
file(APPEND "${DESTINATION_FOLDER}/ocl_preamble.h" "const char* preamble = \n")
file(APPEND "${DESTINATION_FOLDER}/ocl_preamble.h" "\"")
file(APPEND "${DESTINATION_FOLDER}/ocl_preamble.h" "${INPUT_FILE_CL}")
file(APPEND "${DESTINATION_FOLDER}/ocl_preamble.h" "\\n\"")
file(APPEND "${DESTINATION_FOLDER}/ocl_preamble.h" "\n\"\";")
file(APPEND "${DESTINATION_FOLDER}/ocl_preamble.h" "\n } \n")
file(APPEND "${DESTINATION_FOLDER}/ocl_preamble.h" "\n\n#endif // OCL_PREAMBLE_H\n")


























# # Reading the preamble for both CUDA & OpenCL
# file(DOWNLOAD "https://raw.githubusercontent.com/CherifMZ/clij-opencl-kernels/cuda_preamble/preamble/preamble.cu" "${CMAKE_CURRENT_BINARY_DIR}/preamble.cu")
# file(DOWNLOAD "https://raw.githubusercontent.com/CherifMZ/clij-opencl-kernels/cuda_preamble/preamble/preamble.cl" "${CMAKE_CURRENT_BINARY_DIR}/preamble.cl")
# file(DOWNLOAD "https://raw.githubusercontent.com/CherifMZ/clij-opencl-kernels/clesperanto_kernels/kernels/add_image_and_scalar.cl" "${CMAKE_CURRENT_BINARY_DIR}/add_image_and_scalar.cl")

# # Set the input file path for both CUDA & OpenCL
# set(INPUT_FILE_CUDA "${CMAKE_CURRENT_BINARY_DIR}/preamble.cu")
# set(INPUT_FILE_CL "${CMAKE_CURRENT_BINARY_DIR}/preamble.cl")

# # set the input file for a kernel
# set(INPUT_FILE_KERNEL "${CMAKE_CURRENT_BINARY_DIR}/add_image_and_scalar.cl")

# # Read the content of the input files
# file(READ "${INPUT_FILE_CUDA}" FILE_CONTENT_CUDA)
# file(READ "${INPUT_FILE_CL}" FILE_CONTENT_CL)
# file(READ "${INPUT_FILE_KERNEL}" FILE_CONTENT_KERNEL)

# # Cuda file
# string(REPLACE "\n" "\\n\"\n\"" FILE_CONTENT_CUDA "${FILE_CONTENT_CUDA}")

# # Split the contents into multiple string literals
# file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "#ifndef CUDA_PREAMBLE_H\n#define CUDA_PREAMBLE_H\n\n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "namespace cudaKernel { \n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "const char* preamble = \n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\"")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "${FILE_CONTENT_CUDA}")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\\n\"")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\n\"\";")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\n } \n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cu.h" "\n\n#endif // CUDA_PREAMBLE_H\n")


# # OpenCL file
# string(REPLACE "\n" "\\n\"\n\"" FILE_CONTENT_CL "${FILE_CONTENT_CL}")

# # Generate output.h file with the contents of preamble.cl split into string literals
# file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "#ifndef OCL_PREAMBLE_H\n#define OCL_PREAMBLE_H\n\n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "namespace oclKernel { \n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "const char* preamble = \n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\"")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "${FILE_CONTENT_CL}")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\\n\"")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\n\"\";")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\n } \n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_cl.h" "\n\n#endif // OCL_PREAMBLE_H\n")

# # Kenrel file
# string(REPLACE "\n" "\\n\"\n\"" FILE_CONTENT_KERNEL "${FILE_CONTENT_KERNEL}")

# file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/../include/output_add_image_and_scalar.h" "#ifndef OCL_ADD_IMAGE_AND_SCALAR_H\n#define OCL_ADD_IMAGE_AND_SCALAR_H\n\n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_add_image_and_scalar.h" "namespace oclKernel { \n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_add_image_and_scalar.h" "const char* add_image_and_scalar = \n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_add_image_and_scalar.h" "\"")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_add_image_and_scalar.h" "${FILE_CONTENT_KERNEL}")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_add_image_and_scalar.h" "\\n\"")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_add_image_and_scalar.h" "\n\"\";")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_add_image_and_scalar.h" "\n } \n")
# file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/../include/output_add_image_and_scalar.h" "\n\n#endif // OCL_ADD_IMAGE_AND_SCALAR_H\n")