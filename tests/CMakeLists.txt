
add_executable(backend_test backend_test.cpp)
add_dependencies(backend_test ${LIBRARY_NAME})
target_link_libraries(backend_test PRIVATE ${LIBRARY_NAME})
set_target_properties(backend_test PROPERTIES FOLDER "Tests")

add_executable(array_test array_test.cpp)
add_dependencies(array_test ${LIBRARY_NAME})
target_link_libraries(array_test PRIVATE ${LIBRARY_NAME})
set_target_properties(array_test PROPERTIES FOLDER "Tests")

add_executable(absolute_test absolute_test.cpp)
add_dependencies(absolute_test ${LIBRARY_NAME})
target_link_libraries(absolute_test PRIVATE ${LIBRARY_NAME})
set_target_properties(absolute_test PROPERTIES FOLDER "Tests")

add_executable(gaussian_test gaussian_test.cpp)
add_dependencies(gaussian_test ${LIBRARY_NAME})
target_link_libraries(gaussian_test PRIVATE ${LIBRARY_NAME})
set_target_properties(gaussian_test PROPERTIES FOLDER "Tests")

add_executable(difference_of_gaussian_test difference_of_gaussian_test.cpp)
add_dependencies(difference_of_gaussian_test ${LIBRARY_NAME})
target_link_libraries(difference_of_gaussian_test PRIVATE ${LIBRARY_NAME})
set_target_properties(difference_of_gaussian_test PROPERTIES FOLDER "Tests")

add_test(NAME backend_test COMMAND backend_test)
add_test(NAME array_test COMMAND array_test)
add_test(NAME absolute_test COMMAND absolute_test)
add_test(NAME gaussian_test COMMAND gaussian_test)
add_test(NAME difference_of_gaussian_test COMMAND difference_of_gaussian_test)

