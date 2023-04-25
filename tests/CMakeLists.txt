

add_executable(backend_test backend_test.cpp)
add_dependencies(backend_test ${LIBRARY_NAME})
target_link_libraries(backend_test PRIVATE ${LIBRARY_NAME})
set_target_properties(backend_test PROPERTIES FOLDER "Tests")

add_test(NAME backend_test COMMAND backend_test)