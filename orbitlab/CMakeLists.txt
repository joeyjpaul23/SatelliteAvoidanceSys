cmake_minimum_required(VERSION 3.20)
project(OrbitLab VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(orbitlab_physics INTERFACE)
target_include_directories(orbitlab_physics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(orbitlab src/main.cpp)
target_link_libraries(orbitlab PRIVATE orbitlab_physics)
target_compile_options(orbitlab PRIVATE -Wall -Wextra -Wpedantic)

add_executable(orbitlab_tests tests/physics_tests.cpp)
target_link_libraries(orbitlab_tests PRIVATE orbitlab_physics)
target_compile_options(orbitlab_tests PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME physics_invariants COMMAND orbitlab_tests)

add_custom_target(generate
  COMMAND orbitlab ${CMAKE_CURRENT_SOURCE_DIR}/web/telemetry.js
  DEPENDS orbitlab
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Propagating research scenario and writing web/telemetry.js")
