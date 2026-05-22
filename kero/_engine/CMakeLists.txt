## Setting up nanobind for Python 3.12 Interpreter 
## So your virtual environment should be of 3.12 for now
## We will figure out what to do about fixed python version
set(DEV_MODULE Development.Module)

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

## There is chance I would want to remove this in future
## TODO: Cmake does not recognizes nanobind as of now without this command
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

## Stop adding version to shared objects
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

## Root for kero/_engine/_kero, this is where the MLIR Part will live
set(KERO_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_kero")

## Add the _kero in kero/__init__.py
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=_kero.")

################################################################################
# SOURCES
################################################################################

declare_mlir_python_sources(KeroEngineMLIRPythonSources)
declare_mlir_python_sources(KeroEngineMLIRPythonExtensions)

declare_mlir_python_sources(KeroEngineMLIRPythonSources.Dialects
    ROOT_DIR "${KERO_MLIR_PYTHON_ROOT_DIR}"
    ADD_TO_PARENT KeroEngineMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT KeroEngineMLIRPythonSources.Dialects
    ROOT_DIR "${KERO_MLIR_PYTHON_ROOT_DIR}"
    TD_FILE dialects/KeroBinds.td
    SOURCES dialects/db/__init__.py
    DIALECT_NAME db
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(KeroEngineMLIRPythonExtensions.Main
    MODULE_NAME _keroEngine
    ADD_TO_PARENT KeroEngineMLIRPythonExtensions
    PYTHON_BINDINGS_LIBRARY nanobind
    SOURCES KeroModule.cpp
    EMBED_CAPI_LINK_LIBS
        KeroCAPI ## add_mlir_public_c_api_library in lib/CAPI/
    PRIVATE_LINK_LIBS
      LLVMSupport
)

################################################################################
# Shared Library
################################################################################

set(_source_components
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    KeroEngineMLIRPythonSources
    KeroEngineMLIRPythonExtensions
)

add_mlir_python_common_capi_library(KeroEngineMLIRAggregateCAPI
    INSTALL_COMPONENT KeroEngineMLIRPythonModules
    INSTALL_DESTINATION python_packages/kero/_engine/_kero/_mlir_libs
    OUTPUT_DIRECTORY "${KERO_MLIR_PACKAGE_DIR}/kero/_engine/_kero/_mlir_libs"
    RELATIVE_INSTALL_ROOT ".."
    DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(KeroEngineMLIRPythonModules
    ROOT_PREFIX "${KERO_MLIR_PACKAGE_DIR}/kero/_engine/_kero"
    INSTALL_PREFIX "python_packages/kero/_engine/_kero"
    DECLARED_SOURCES ${_source_components}
    COMMON_CAPI_LINK_LIBS
        KeroEngineMLIRAggregateCAPI
)
