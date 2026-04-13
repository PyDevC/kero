set(DEV_MODULE Development.Module)
find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_kero KeroModule.cpp)
target_link_libraries(_kero
    PRIVATE
    DB               # Your Dialect library
    DBToTensorPass   # Your Conversion library
    MLIRIR
    MLIRPass
)

target_include_directories(_kero PRIVATE ${PROJECT_SOURCE_DIR}/include)
