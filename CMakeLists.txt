cmake_minimum_required(VERSION 3.20)

project(kero-binary)

set(CMAKE_CXX_STANDARD 17)

# Config dir
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Configure Out-of-tree build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    find_package(MLIR REQUIRED CONFIG)

    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

    set(CMAKE_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

    list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
    list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)

    set(MLIR_ENABLE_BINDINGS_PYTHON ON)
    set(BACKEND_PACKAGE_STRING "LLVM" ${LLVM_PACKAGE_VERSION})
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Configure cmake
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build_tools/cmake)

include(AddMLIRPython)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(kero)
