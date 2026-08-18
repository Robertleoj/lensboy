cmake_minimum_required(VERSION 3.15)
set(CMAKE_TOOLCHAIN_FILE
    "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    CACHE FILEPATH "vcpkg toolchain file")

project(lensboy_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Using ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    set(ENV{CCACHE_DEBUG} "1")

endif(CCACHE_FOUND)

if(APPLE)
    set(INSTALL_PATH "@loader_path")
else()
    set(INSTALL_PATH "$ORIGIN")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenMP REQUIRED)


pybind11_add_module(
    lensboy_bindings 
    MODULE 

    cpp_src/main.cpp
    cpp_src/calibrate_opencv.cpp
    cpp_src/calibrate_stereographic_opencv.cpp
    cpp_src/matching_spline_model.cpp
    cpp_src/pinhole_splined_fine_tune.cpp
    cpp_src/stereographic_splined_fine_tune.cpp
    cpp_src/normalize_pinhole_splined.cpp
    cpp_src/seeded_normalize.cpp
    cpp_src/lut_max_cell_error.cpp
)

target_compile_definitions(lensboy_bindings
    PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
    # PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF
)

target_link_libraries(
    lensboy_bindings
    PRIVATE

    Ceres::ceres
    Eigen3::Eigen
    fmt::fmt
    spdlog::spdlog
    OpenMP::OpenMP_CXX
)

if(APPLE)
    file(WRITE "${CMAKE_BINARY_DIR}/exported_symbols.txt" "_PyInit_lensboy_bindings\n")
    target_link_options(lensboy_bindings PRIVATE
        "-Wl,-exported_symbols_list,${CMAKE_BINARY_DIR}/exported_symbols.txt"
    )
elseif(UNIX)
    file(WRITE "${CMAKE_BINARY_DIR}/version_script.map"
        "{ global: PyInit_lensboy_bindings; local: *; };\n"
    )
    target_link_options(lensboy_bindings PRIVATE
        "-Wl,--version-script,${CMAKE_BINARY_DIR}/version_script.map"
    )
endif()

install(TARGETS lensboy_bindings DESTINATION lensboy)
