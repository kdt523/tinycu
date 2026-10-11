cmake_minimum_required(VERSION 3.20)
project(tinycu VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(TINYCU_ENABLE_SANITIZERS "Build with ASan + UBSan" OFF)
set(TINYCU_BUILD_RUNTIME AUTO CACHE STRING "Build the CUDA Driver API runner (ON/OFF/AUTO)")

# Global flags: strict warnings, no FP contraction (interpreter must be bit-exact with GPU).
add_compile_options(-Wall -Wextra -Werror -pedantic -ffp-contract=off)

if(TINYCU_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer
                      -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- core library: everything in src/ except driver/ and runtime/ ----
file(GLOB_RECURSE TINYCU_CORE_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(FILTER TINYCU_CORE_SOURCES EXCLUDE REGEX "/src/(driver|runtime)/")
add_library(tinycu_core STATIC ${TINYCU_CORE_SOURCES})
target_include_directories(tinycu_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(tinycu_core PUBLIC TINYCU_VERSION="${PROJECT_VERSION}")

add_executable(tinycu src/driver/main.cpp)
target_link_libraries(tinycu PRIVATE tinycu_core)

# ---- optional GPU runner ----
if(TINYCU_BUILD_RUNTIME STREQUAL "AUTO")
  find_package(CUDAToolkit QUIET)
  set(_tinycu_runtime ${CUDAToolkit_FOUND})
elseif(TINYCU_BUILD_RUNTIME)
  find_package(CUDAToolkit REQUIRED)
  set(_tinycu_runtime ON)
else()
  set(_tinycu_runtime OFF)
endif()
if(_tinycu_runtime AND EXISTS ${PROJECT_SOURCE_DIR}/src/runtime/tinycu_run.cpp)
  add_executable(tinycu-run src/runtime/tinycu_run.cpp)
  target_link_libraries(tinycu-run PRIVATE tinycu_core CUDA::cuda_driver)
endif()
message(STATUS "TinyCU: GPU runtime runner = ${_tinycu_runtime}")

# ---- tests ----
include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)

  file(GLOB TINYCU_UNIT_TESTS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/unit/*.cpp)
  add_executable(tinycu_unit_tests ${TINYCU_UNIT_TESTS})
  target_link_libraries(tinycu_unit_tests PRIVATE tinycu_core GTest::gtest_main)
  gtest_discover_tests(tinycu_unit_tests)

  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_runner ${PROJECT_SOURCE_DIR}/tools/run_file_test.py)
  file(GLOB _golden CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/golden/*.tc)
  foreach(tc ${_golden})
    get_filename_component(_name ${tc} NAME_WE)
    add_test(NAME golden.${_name} COMMAND Python3::Interpreter ${_runner} golden $<TARGET_FILE:tinycu> ${tc})
  endforeach()
  file(GLOB _errors CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/errors/*.tc
       ${PROJECT_SOURCE_DIR}/tests/warnings/*.tc)
  foreach(tc ${_errors})
    get_filename_component(_name ${tc} NAME_WE)
    add_test(NAME error.${_name} COMMAND Python3::Interpreter ${_runner} error $<TARGET_FILE:tinycu> ${tc})
  endforeach()

  add_test(NAME cli_version COMMAND tinycu --version)
  set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "^tinycu 0\\.0\\.1")
endif()
