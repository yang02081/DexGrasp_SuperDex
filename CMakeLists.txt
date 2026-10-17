cmake_minimum_required(VERSION 3.10)
project(dexgrasp_zero_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env/bin)

get_filename_component(GRASPXL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
# The upstream layout places this repository below GraspXL/raisimGymTorch.
# Also support the workspace layout used by the SuperDex port, where both
# repositories are siblings.
if (NOT EXISTS "${GRASPXL_ROOT}/raisim")
    get_filename_component(_WORKSPACE_GRASPXL "${CMAKE_CURRENT_SOURCE_DIR}/../GraspXL" ABSOLUTE)
    if (EXISTS "${_WORKSPACE_GRASPXL}/raisim")
        set(GRASPXL_ROOT "${_WORKSPACE_GRASPXL}")
    endif()
endif()

set(PYBIND11_SOURCE_DIR "${GRASPXL_ROOT}/thirdParty/pybind11")
if (NOT EXISTS "${PYBIND11_SOURCE_DIR}/CMakeLists.txt")
    set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../thirdParty/pybind11")
endif()

# pybind11 2.9 bundled with the original project does not compile against
# Python 3.12. Prefer a compatible system package when available, then fall
# back to the upstream bundled source for older Python installations.
find_package(pybind11 2.10 CONFIG QUIET)
if (NOT pybind11_FOUND)
    add_subdirectory(${PYBIND11_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
endif()
list(APPEND CMAKE_PREFIX_PATH "${GRASPXL_ROOT}/thirdParty/Eigen3")
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

if (UNIX AND NOT APPLE)
    set(RAISIM_OS linux)
elseif (APPLE)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE RAISIM_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (RAISIM_ARCH STREQUAL "arm64")
        set(RAISIM_OS m1)
    else()
        set(RAISIM_OS mac)
    endif()
else()
    set(RAISIM_OS win32)
endif()

list(APPEND CMAKE_PREFIX_PATH "${GRASPXL_ROOT}/raisim/${RAISIM_OS}")
find_package(raisim CONFIG REQUIRED)

file(GLOB ENV_DIRECTORIES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/env/envs ${CMAKE_CURRENT_SOURCE_DIR}/env/envs/*)
foreach(environment_name ${ENV_DIRECTORIES})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env/envs/${environment_name})
        pybind11_add_module(${environment_name} NO_EXTRAS env/raisim_gym.cpp env/Yaml.cpp)
        target_link_libraries(${environment_name} PRIVATE raisim::raisim OpenMP::OpenMP_CXX)
        target_include_directories(${environment_name} PUBLIC
                ${EIGEN3_INCLUDE_DIRS}
                ${CMAKE_CURRENT_SOURCE_DIR}/env/envs/${environment_name}
                ${CMAKE_CURRENT_SOURCE_DIR}/env)
        target_compile_options(${environment_name} PRIVATE -fPIC -O3 -mno-avx2)
        target_compile_definitions(${environment_name} PRIVATE
                "-DRAISIMGYM_TORCH_ENV_NAME=${environment_name}"
                EIGEN_DONT_PARALLELIZE)
    endif()
endforeach()
