cmake_minimum_required(VERSION 3.22)
project(VectorDB VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/W4 /WX /permissive-)
else ()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    add_compile_options(-march=native -O3)
endif ()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    if (NOT MSVC)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g -O0)
        add_link_options(-fsanitize=address,undefined)
    endif ()
endif ()

include(FetchContent)

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


add_executable(vectordb_demo main.cpp)
target_link_libraries(vectordb_demo PRIVATE vectordb)

add_executable(vectordb_test tests/test_index.cpp)
target_link_libraries(vectordb_test PRIVATE vectordb gtest_main)

add_executable(vectordb_bench benchmarks/bench_index.cpp)
target_link_libraries(vectordb_bench PRIVATE vectordb benchmark::benchmark)