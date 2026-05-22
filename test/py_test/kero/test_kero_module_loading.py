class TestEngineModuleLoading:
    def test_engine_is_namespace_package(self):
        from kero._engine._kero.dialects import db
        from kero._engine._kero.dialects.db import ScanOp

    def test_register_dialect_accepts_context(self):
        import kero._engine._kero._mlir_libs._keroEngine as _keroEngine
        from kero._engine._kero.ir import Context
        ctx = Context()
        _keroEngine.register_dialect(ctx)       # load=True by default
