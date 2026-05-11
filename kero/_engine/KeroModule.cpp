#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "kero-c/Dialect.h"
#include "kero-c/Registration.h"

namespace nb = nanobind;

NB_MODULE(_keroEngine, m) {
    KeroRegisterAllPasses();

    m.doc() = "Kero Engine for compilation and optimization of db IR";

    m.def(
        "register_dialect",
        [](MlirContext context, bool load) {
            MlirDialectHandle handle = mlirGetDialectHandle__db__(); // auto generated handle
            mlirDialectHandleRegisterDialect(handle, context);
            if (load) {
                mlirDialectHandleLoadDialect(handle, context);
            }
        },
        nb::arg("context"), nb::arg("load") = true);
}
