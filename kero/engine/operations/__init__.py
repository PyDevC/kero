from .operators import (
    Aggregation,
    Mathematical,
    Logical,
)

operators = {
    "Aggregation": Aggregation.operator.expand(), 
    "Mathematical": Mathematical.operator.expand(),
    "Logical": Logical.operator.expand(),
}
