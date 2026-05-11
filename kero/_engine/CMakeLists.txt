set(DEV_MODULE Development.Module)

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(BUILD_SHARED_LIBS OFF)

nanobind_add_module(_keroEngine KeroModule.cpp ArrowBridge.cpp)

set(KERO_MLIR_PYTHON_ROOT_DIR "${CMAKE_SOURCE_DIR}/kero")

declare_mlir_python_sources(KeroEngineMLIRPythonSources)

declare_mlir_python_sources(KeroEngineMLIRPython.Dialects
    ROOT_DIR "${KERO_MLIR_PYTHON_ROOT_DIR}/_engine"
    ADD_TO_PARENT KeroEngineMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT KeroEngineMLIRPython.Dialects
    ROOT_DIR "${KERO_MLIR_PYTHON_ROOT_DIR}/_engine"
    TD_FILE _kero/dialects/KeroBinds.td
    SOURCES _kero/dialects/db/__init__.py
    DIALECT_NAME db
)

find_package(Arrow REQUIRED)

target_link_libraries(_keroEngine
    PRIVATE
    Arrow::arrow_shared
    KeroCAPI
)

target_include_directories(_keroEngine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(_keroEngine PUBLIC cxx_std_17)
install(TARGETS _keroEngine LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/kero)
