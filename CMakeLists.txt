cmake_minimum_required(VERSION 3.22)
project(VectorDB VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(VECTORDB_DISABLE_NEON "Disable ARM NEON optimizations" OFF)
option(VECTORDB_DISABLE_SIMD "Disable all SIMD optimizations" OFF)
option(VECTORDB_ENABLE_SANITIZERS "Enable ASan/UBSan in Debug builds" OFF)
option(VECTORDB_ENABLE_NATIVE_OPT "Enable -march=native in Release" OFF)

if (MSVC)
    add_compile_options(/W4 /WX /permissive-)
else ()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    add_compile_options($<$<CONFIG:Release>:-O3>)
    if (VECTORDB_ENABLE_NATIVE_OPT)
        add_compile_options($<$<CONFIG:Release>:-march=native>)
    endif ()
endif ()

if (VECTORDB_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE MATCHES Debug)
    if (NOT MSVC)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g -O0)
        add_link_options(-fsanitize=address,undefined)
    endif ()
endif ()

include(FetchContent)
include(GoogleTest)
enable_testing()

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_library(vectordb STATIC
        src/FlatIndex.cpp
        src/HnswIndex.cpp
)

target_include_directories(vectordb PUBLIC include)
if (VECTORDB_DISABLE_SIMD)
    target_compile_definitions(vectordb PUBLIC VECTORDB_DISABLE_SIMD=1)
elseif (VECTORDB_DISABLE_NEON)
    target_compile_definitions(vectordb PUBLIC VECTORDB_DISABLE_NEON=1)
endif ()


add_executable(vectordb_demo main.cpp)
target_link_libraries(vectordb_demo PRIVATE vectordb)

add_executable(vectordb_test tests/test_index.cpp)
target_link_libraries(vectordb_test PRIVATE vectordb gtest_main)
gtest_discover_tests(vectordb_test)

add_executable(vectordb_bench benchmarks/bench_index.cpp)
target_link_libraries(vectordb_bench PRIVATE vectordb benchmark::benchmark)