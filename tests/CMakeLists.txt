
add_executable(backend_test backend_test.cpp)
add_dependencies(backend_test ${LIBRARY_NAME})
target_link_libraries(backend_test PRIVATE ${LIBRARY_NAME})
set_target_properties(backend_test PROPERTIES FOLDER "Tests")

add_executable(array_test array_test.cpp)
add_dependencies(array_test ${LIBRARY_NAME})
target_link_libraries(array_test PRIVATE ${LIBRARY_NAME})
set_target_properties(array_test PROPERTIES FOLDER "Tests")

add_executable(image_test image_test.cpp)
add_dependencies(image_test ${LIBRARY_NAME})
target_link_libraries(image_test PRIVATE ${LIBRARY_NAME})
set_target_properties(image_test PROPERTIES FOLDER "Tests")

add_executable(push_pull_test push_pull_test.cpp)
add_dependencies(push_pull_test ${LIBRARY_NAME})
target_link_libraries(push_pull_test PRIVATE ${LIBRARY_NAME})
set_target_properties(push_pull_test PROPERTIES FOLDER "Tests")

add_executable(defines_test defines_test.cpp)
add_dependencies(defines_test ${LIBRARY_NAME})
target_link_libraries(defines_test PRIVATE ${LIBRARY_NAME})
set_target_properties(defines_test PROPERTIES FOLDER "Tests")

add_executable(absolute_test absolute_test.cpp)
add_dependencies(absolute_test ${LIBRARY_NAME})
target_link_libraries(absolute_test PRIVATE ${LIBRARY_NAME})
set_target_properties(absolute_test PROPERTIES FOLDER "Tests")

add_executable(cuda_src_test cuda_src_test.cpp)
add_dependencies(cuda_src_test ${LIBRARY_NAME})
target_link_libraries(cuda_src_test PRIVATE ${LIBRARY_NAME})
set_target_properties(cuda_src_test PROPERTIES FOLDER "Tests")

add_test(NAME backend_test COMMAND backend_test)
add_test(NAME array_test COMMAND array_test)
add_test(NAME image_test COMMAND image_test)
add_test(NAME push_pull_test COMMAND push_pull_test)
add_test(NAME defines_test COMMAND defines_test)
add_test(NAME absolute_test COMMAND absolute_test)
add_test(NAME cuda_src_test COMMAND cuda_src_test)
