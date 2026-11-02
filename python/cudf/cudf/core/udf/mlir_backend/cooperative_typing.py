"""Typing for CooperativeArrayType: reduction attributes.

Grouped columns are modelled as ``CooperativeArrayType`` views. Only
scalar-returning reductions (min/max/sum/mean/var/std/idxmin/idxmax/corr/
count/size) are supported inside a JIT ``GroupBy.apply`` UDF; operators
applied directly to a grouped column are rejected (see below).
"""

from __future__ import annotations

from cudf.core.udf._ops import arith_ops, bitwise_ops, comparison_ops, unary_ops
from cudf.core.udf.mlir_backend.cooperative_types import CooperativeArrayType
from cudf.core.udf.utils import UDFError
from numba_cuda_mlir.extending import typing_registry as registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.typing import signature as nb_signature
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
)

_UDF_DOC_URL = "https://docs.rapids.ai/api/cudf/stable/cudf/guide-to-udfs/"


class CooperativeArray:
    """Sentinel for constructing CooperativeArrayType from array + size."""
    pass


# ---------------------------------------------------------------------------
# Constructor: CooperativeArray(array, size) -> CooperativeArrayType(dtype)
# ---------------------------------------------------------------------------

@registry.register_global(CooperativeArray)
class CooperativeArrayConstructorTemplate(AbstractTemplate):
    key = CooperativeArray

    def generic(self, args, kws):
        if kws:
            return None
        if len(args) == 2:
            arr, size = args
            if isinstance(arr, types.Array) and isinstance(size, types.Integer):
                return nb_signature(
                    CooperativeArrayType(arr.dtype), arr, size
                )
        elif len(args) == 3:
            arr, size, index = args
            if (
                isinstance(arr, types.Array)
                and isinstance(size, types.Integer)
                and isinstance(index, types.Array)
            ):
                return nb_signature(
                    CooperativeArrayType(arr.dtype), arr, size, index
                )


# ---------------------------------------------------------------------------
# Attribute typing: .size, .sum()
# ---------------------------------------------------------------------------

def _mean_like_result(dtype):
    # pandas mean/var/std of float32 stays float32; everything else -> float64.
    return types.float32 if dtype == types.float32 else types.float64


class CoopSumTemplate(AbstractTemplate):
    key = "CooperativeArray.sum"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            # pandas promotes integer sums to int64 (overflow-safe).
            retty = (
                types.int64
                if isinstance(self.this.dtype, types.Integer)
                else self.this.dtype
            )
            return nb_signature(retty, recvr=self.this)


class CoopMeanTemplate(AbstractTemplate):
    key = "CooperativeArray.mean"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(_mean_like_result(self.this.dtype), recvr=self.this)


class CoopMinTemplate(AbstractTemplate):
    key = "CooperativeArray.min"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(self.this.dtype, recvr=self.this)


class CoopMaxTemplate(AbstractTemplate):
    key = "CooperativeArray.max"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(self.this.dtype, recvr=self.this)


class CoopStdTemplate(AbstractTemplate):
    key = "CooperativeArray.std"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(_mean_like_result(self.this.dtype), recvr=self.this)


class CoopVarTemplate(AbstractTemplate):
    key = "CooperativeArray.var"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(_mean_like_result(self.this.dtype), recvr=self.this)


class CoopCorrTemplate(AbstractTemplate):
    key = "CooperativeArray.corr"

    def generic(self, args, kws):
        if len(args) != 1 or kws:
            return None
        other = args[0]
        if not isinstance(other, CooperativeArrayType):
            return None
        dt = self.this.dtype
        if not isinstance(dt, types.Integer):
            # Correlation of floating types is not yet supported (#13839).
            raise UDFError(
                f"Series.corr(Series) is not supported for "
                f"({dt}, {other.dtype}) within JIT GroupBy apply. "
                f"Supported features are listed at: {_UDF_DOC_URL}"
            )
        return nb_signature(types.float64, other, recvr=self.this)


class CoopIdxMaxTemplate(AbstractTemplate):
    key = "CooperativeArray.idxmax"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(types.int64, recvr=self.this)


class CoopIdxMinTemplate(AbstractTemplate):
    key = "CooperativeArray.idxmin"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(types.int64, recvr=self.this)



_SUPPORTED_DTYPES = (types.float64, types.float32, types.int64, types.int32)


@registry.register_attr
class CooperativeArrayAttr(AttributeTemplate):
    """Reduction attributes for every ``CooperativeArrayType`` instance.

    A single class-keyed template (like the legacy ``GroupAttr``) so that
    columns of unsupported dtypes - modelled with a ``Poison`` element type -
    raise a clear ``UDFError`` when used, while a UDF that never touches them
    still compiles.
    """

    key = CooperativeArrayType

    @staticmethod
    def _require_supported(mod):
        if isinstance(mod.dtype, types.Poison):
            raise UDFError(
                "Use of a column of an unsupported dtype detected within a "
                "GroupBy.apply() UDF. Only columns of the following dtypes may "
                "be used through the GroupBy.apply() JIT engine: "
                f"{[str(t) for t in _SUPPORTED_DTYPES]}"
            )

    def resolve_size(self, mod):
        self._require_supported(mod)
        return types.int64

    def resolve_sum(self, mod):
        self._require_supported(mod)
        return types.BoundFunction(CoopSumTemplate, mod)

    def resolve_mean(self, mod):
        self._require_supported(mod)
        return types.BoundFunction(CoopMeanTemplate, mod)

    def resolve_min(self, mod):
        self._require_supported(mod)
        return types.BoundFunction(CoopMinTemplate, mod)

    def resolve_max(self, mod):
        self._require_supported(mod)
        return types.BoundFunction(CoopMaxTemplate, mod)

    def resolve_std(self, mod):
        self._require_supported(mod)
        return types.BoundFunction(CoopStdTemplate, mod)

    def resolve_var(self, mod):
        self._require_supported(mod)
        return types.BoundFunction(CoopVarTemplate, mod)

    def resolve_corr(self, mod):
        self._require_supported(mod)
        return types.BoundFunction(CoopCorrTemplate, mod)

    def resolve_idxmax(self, mod):
        self._require_supported(mod)
        return types.BoundFunction(CoopIdxMaxTemplate, mod)

    def resolve_idxmin(self, mod):
        self._require_supported(mod)
        return types.BoundFunction(CoopIdxMinTemplate, mod)


_ca_f64 = CooperativeArrayType(types.float64)


# ---------------------------------------------------------------------------
# Unsupported operators on grouped columns.
#
# A JIT GroupBy.apply UDF may only produce scalar reduction results. Applying
# arithmetic, comparison, bitwise, or unary operators directly to a grouped
# column (e.g. ``group['x'] + group['y']`` or ``-group['x']``) has no scalar
# meaning and is rejected with a UDFError, matching the legacy backend. The
# array-returning operators are intentionally not implemented.
# ---------------------------------------------------------------------------

def _make_unsupported_op_template(op):
    class UnsupportedCoopOpTemplate(AbstractTemplate):
        key = op

        def generic(self, args, kws):
            # Only intercept when a grouped column participates; scalar
            # operations (e.g. on reduction results) type normally.
            if not any(isinstance(a, CooperativeArrayType) for a in args):
                return None
            fname = getattr(op, "__name__", str(op))
            sr_err = ", ".join("Series" for _ in args)
            raise UDFError(
                f"{fname}({sr_err}) is not supported by JIT GroupBy apply. "
                f"Supported features are listed at: {_UDF_DOC_URL}"
            )

    return UnsupportedCoopOpTemplate


for _op in (*arith_ops, *comparison_ops, *bitwise_ops, *unary_ops):
    registry.register_global(_op)(_make_unsupported_op_template(_op))


# ---------------------------------------------------------------------------
# Factory stub: cooperative_array_from_ptr(data_ptr, size) -> CooperativeArray
# ---------------------------------------------------------------------------

def _cooperative_array_from_ptr_stub(data_ptr, size):
    """Sentinel: construct a non-owning CooperativeArray from raw pointer and element count."""
    raise NotImplementedError("device-only")


class CoopFromPtrTemplate(AbstractTemplate):
    key = _cooperative_array_from_ptr_stub

    def generic(self, args, kws):
        if len(args) == 2 and not kws:
            if isinstance(args[0], types.Integer) and isinstance(args[1], types.Integer):
                return nb_signature(_ca_f64, types.int64, types.int64)


registry.register_global(
    _cooperative_array_from_ptr_stub,
    types.Function(CoopFromPtrTemplate),
)

cooperative_array_from_ptr = _cooperative_array_from_ptr_stub
