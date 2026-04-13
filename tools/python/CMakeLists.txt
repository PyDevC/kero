set(DEV_MODULE Development.Module)
find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(BUILD_SHARED_LIBS OFF)

nanobind_add_module(_kero KeroModule.cpp)

target_link_libraries(_kero
    PRIVATE
    DB
    DBToTensorPass
    MLIRIR
    MLIRPass
    MLIRSupport
)

target_include_directories(_kero PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(_kero PRIVATE cxx_std_17)
install(TARGETS _kero LIBRARY DESTINATION kero)
