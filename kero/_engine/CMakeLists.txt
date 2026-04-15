find_package(Arrow REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_arrow_bridge ArrowBridge.cpp)

target_link_libraries(_arrow_bridge 
    PRIVATE 
    Arrow::arrow_shared
)

target_include_directories(_arrow_bridge PRIVATE ${PROJECT_SOURCE_DIR}/include)

install(TARGETS _arrow_bridge DESTINATION kero/_engine)
