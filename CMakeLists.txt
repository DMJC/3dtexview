cmake_minimum_required(VERSION 3.16)
project(3dtexview LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
pkg_check_modules(EPOXY REQUIRED epoxy)
find_package(assimp REQUIRED)

add_executable(3dtexview
    src/main.cpp
)

target_include_directories(3dtexview PRIVATE
    ${GTK3_INCLUDE_DIRS}
    ${EPOXY_INCLUDE_DIRS}
)

target_compile_options(3dtexview PRIVATE
    ${GTK3_CFLAGS_OTHER}
    ${EPOXY_CFLAGS_OTHER}
)

target_link_libraries(3dtexview PRIVATE
    ${GTK3_LIBRARIES}
    ${EPOXY_LIBRARIES}
    assimp::assimp
)
