"""kero/engine/compiler.py

MLIR compilation pipeline — Python side.

This module owns the :class:`Compiler` class, which is the thin Python
wrapper around the C++ ``KeroModule`` (``kero/_engine/KeroModule.cpp``).

Responsibilities (Developer Guide §2.2, §4):
- Create and hold the ``MLIRContext`` (via ``KeroModule``) for the engine
  lifetime.  **One context per engine** — never create a second one.
- Create the shared ``Module`` object that the Parser writes into.
- Accept an :class:`~kero.parser.ir_emitter.IRQueryMeta` from the Parser
  and invoke ``KeroModule.compile_cpu()`` to produce a
  :class:`CompiledQuery`.

The ``CompiledQuery`` dataclass mirrors the C++ struct of the same name
and carries everything the :class:`~kero.engine.executor.Executor` needs
to call the JIT-compiled function.

Language boundary note (Developer Guide §10.4):
Only :mod:`kero._engine` (i.e. ``KeroModule.cpp``) and
``kero._engine.executor`` cross the Python/C++ boundary.  This file
is pure Python; it delegates all MLIR work to the C++ extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pyarrow as pa

from kero._mlir import Module

# Import the C++ extension module.  At build time this becomes
# kero/_kero.so (or _kero.pyd on Windows).  The try/except allows
# importing the module in environments where the C++ extension has not
# been built yet (e.g. during documentation generation or early
# development phases where only the Python parser is being tested).
try:
    from kero import _kero                          # compiled extension
    _HAVE_KERO = True
except ImportError:  # pragma: no cover
    _kero = None     # type: ignore[assignment]
    _HAVE_KERO = False


# ---------------------------------------------------------------------------
# CompiledQuery
#
# Mirrors the C++ CompiledQuery struct (Developer Guide §5.6).
# The Executor reads every field here; nothing else should write to it
# after construction.
# ---------------------------------------------------------------------------

@dataclass
class CompiledQuery:
    """Everything the Executor needs to call the JIT-compiled function.

    Attributes
    ----------
    jit_fn:
        The underlying C++ ``_kero.CompiledQuery`` object.  The Executor
        calls ``jit_fn.call_jit(ptrs)`` to invoke the compiled kernel.
    table_name:
        Name of the queried table.  Used by the Executor to look up the
        Arrow Table.
    referenced_columns:
        Ordered list of column names corresponding to function arg1..N.
        **This is the argument-ordering contract** between the IR Emitter
        and the Executor (Developer Guide §10.1).
    output_schema:
        Arrow schema of the result table.  Used by the Executor to
        reconstruct an Arrow Table from the output memref.
    n_cols:
        Static column count (C dimension).  Matches the first dim of the
        table tensor in the lowered IR.
    ir_text:
        Original db-dialect MLIR text.  Kept for debugging; not used at
        runtime.
    """

    jit_fn: object                          # _kero.CompiledQuery (opaque)
    table_name: str
    referenced_columns: List[str]
    output_schema: pa.Schema
    n_cols: int
    ir_text: str = field(default="", repr=False)

    def __repr__(self) -> str:
        return (
            f"CompiledQuery(table={self.table_name!r}, "
            f"columns={self.referenced_columns}, "
            f"n_cols={self.n_cols})"
        )


# ---------------------------------------------------------------------------
# Compiler
#
# Owns the KeroModule (and through it the MLIRContext) for one KeroEngine
# instance.  Follows Developer Guide §2.3: "The Compiler creates the
# context and owns it for the lifetime of the engine."
# ---------------------------------------------------------------------------

class Compiler:
    """Thin Python wrapper around the C++ KeroModule compilation pipeline.

    One :class:`Compiler` instance is created per :class:`KeroEngine`
    instance and is shared across all queries on that engine.

    Parameters
    ----------
    None.  Construction is side-effect-free except for initialising
    the LLVM JIT targets (idempotent) and allocating the MLIRContext.

    Raises
    ------
    RuntimeError
        If the ``_kero`` C++ extension module is not available.
    """

    def __init__(self) -> None:
        if not _HAVE_KERO:
            raise RuntimeError(
                "Compiler: the '_kero' C++ extension is not available. "
                "Build the project with CMake before using the Compiler."
            )

        # The KeroModule owns the MLIRContext for this engine instance.
        # Developer Guide §2.3: "The Compiler creates the context and
        # owns it for the lifetime of the engine."
        self._km: _kero.KeroModule = _kero.KeroModule()

        # The shared Module is the handoff point between Parser and
        # Compiler.  The Parser writes func.func definitions into it;
        # compile() reads the accumulated IR text back out.
        self._module: Module = Module()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def module(self) -> Module:
        """The shared MLIR Module (inject into Parser via attach_module)."""
        return self._module

    def compile(self, meta) -> CompiledQuery:
        """Compile an emitted query into a JIT-callable object.

        This is the main entry point called by :class:`KeroEngine` after
        the Parser has emitted the db-dialect IR into the shared module.

        Parameters
        ----------
        meta:
            An :class:`~kero.parser.ir_emitter.IRQueryMeta` produced by
            the IR Emitter.  Carries the IR text, table name, column
            order, schema, and column count.

        Returns
        -------
        CompiledQuery
            Ready for the Executor to call.

        Raises
        ------
        RuntimeError
            If the MLIR pass pipeline or JIT compilation fails.
        """
        # Retrieve the complete module IR text.  The Parser wrote one or
        # more func.func definitions into self._module; get_ir() wraps
        # them in "module { ... }".
        ir_text = self._module.get_ir()

        # Invoke the C++ compilation pipeline:
        #   db-dialect → tensor/linalg/arith → bufferization → LLVM dialect
        #   → ExecutionEngine JIT → function pointer.
        cpp_cq: _kero.CompiledQuery = self._km.compile_cpu(
            ir_text,
            meta.table_name,
            meta.referenced_columns,
            meta.n_cols,
        )

        # Wrap in the Python CompiledQuery dataclass so the Executor has a
        # single, well-typed object to work with.
        return CompiledQuery(
            jit_fn=cpp_cq,
            table_name=meta.table_name,
            referenced_columns=meta.referenced_columns,
            output_schema=meta.output_schema,
            n_cols=meta.n_cols,
            ir_text=ir_text,
        )

    def verify_ir(self, ir_text: Optional[str] = None) -> str:
        """Parse and verify IR text without running the pass pipeline.

        Parameters
        ----------
        ir_text:
            The MLIR text to verify.  If *None*, uses the accumulated
            text from ``self.module.get_ir()``.

        Returns
        -------
        str
            ``""`` on success, or an error message string on failure.
        """
        text = ir_text if ir_text is not None else self._module.get_ir()
        return self._km.verify_ir(text)

    def lower_to_llvm_ir(self, ir_text: Optional[str] = None) -> str:
        """Run the full pipeline and return the resulting LLVM dialect IR.

        Diagnostic helper; production code uses :meth:`compile` instead.

        Parameters
        ----------
        ir_text:
            Source MLIR to lower.  Defaults to ``self.module.get_ir()``.

        Returns
        -------
        str
            The LLVM dialect IR as a string.

        Raises
        ------
        RuntimeError
            If parsing or any pass fails.
        """
        text = ir_text if ir_text is not None else self._module.get_ir()
        return self._km.lower_to_llvm_ir(text)

    def reset_module(self) -> None:
        """Clear all accumulated functions from the shared module.

        Call this between independent queries if the same Compiler /
        Module is being reused and old function definitions should not
        persist.  :class:`KeroEngine` calls this automatically if
        ``per_query=True`` was set on the Parser.
        """
        self._module.clear()

    def __repr__(self) -> str:
        return f"Compiler(module_fns={len(self._module)})"
