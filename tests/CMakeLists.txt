
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG        v1.13.0
# )
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest))

add_executable(backend_test backend_test.cpp)
add_dependencies(backend_test ${LIBRARY_NAME})
target_link_libraries(backend_test PRIVATE ${LIBRARY_NAME})
set_target_properties(backend_test PROPERTIES FOLDER "Tests")

add_executable(array_test array_test.cpp)
add_dependencies(array_test ${LIBRARY_NAME})
target_link_libraries(array_test PRIVATE ${LIBRARY_NAME})
set_target_properties(array_test PROPERTIES FOLDER "Tests")

add_executable(push_pull_test push_pull_test.cpp)
add_dependencies(push_pull_test ${LIBRARY_NAME})
target_link_libraries(push_pull_test PRIVATE ${LIBRARY_NAME})
set_target_properties(push_pull_test PROPERTIES FOLDER "Tests")

add_test(NAME backend_test COMMAND backend_test)
add_test(NAME array_test COMMAND array_test)
add_test(NAME push_pull_test COMMAND push_pull_test)