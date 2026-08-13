# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import numba
from numba import cuda, types
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    register_model,
    type_callable,
    typeof_impl,
)
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate, AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.np import numpy_support

from cudf.core.udf._ops import arith_ops, comparison_ops, unary_ops
from cudf.core.udf.utils import Row, UDFError

index_default_type = types.int64
group_size_type = types.int64
SUPPORTED_GROUPBY_NUMBA_TYPES = [
    types.int32,
    types.int64,
    types.float32,
    types.float64,
]
SUPPORTED_GROUPBY_NUMPY_TYPES = [
    numpy_support.as_dtype(dt) for dt in SUPPORTED_GROUPBY_NUMBA_TYPES
]

_UDF_DOC_URL = "https://docs.rapids.ai/api/cudf/stable/cudf/guide-to-udfs/"


class Group:
    """
    A piece of python code whose purpose is to be replaced
    during compilation. After being registered to GroupType,
    serves as a handle for instantiating GroupType objects
    in python code and accessing their attributes
    """

    pass


class GroupType(numba.types.Type):
    """
    Numba extension type carrying metadata associated with a single
    GroupBy group. This metadata ultimately is passed to the CUDA
    __device__ function which actually performs the work.
    """

    def __init__(self, group_scalar_type, index_type=index_default_type):
        if (
            group_scalar_type not in SUPPORTED_GROUPBY_NUMBA_TYPES
            and not isinstance(group_scalar_type, types.Poison)
        ):
            # A frame containing an column with an unsupported dtype
            # is calling groupby apply. Construct a GroupType with
            # a poisoned type so we can later error if this group is
            # used in the UDF body
            group_scalar_type = types.Poison(group_scalar_type)
        self.group_scalar_type = group_scalar_type
        self.index_type = index_type
        self.group_data_type = types.CPointer(group_scalar_type)
        self.group_size_type = group_size_type
        self.group_index_type = types.CPointer(index_type)
        super().__init__(
            name=f"Group({self.group_scalar_type}, {self.index_type})"
        )


class GroupByJITDataFrame(Row):
    pass


register_model(GroupByJITDataFrame)(models.RecordModel)


@typeof_impl.register(Group)
def typeof_group(val, c):
    """
    Tie Group and GroupType together such that when Numba
    sees usage of Group in raw python code, it knows to
    treat those usages as uses of GroupType
    """
    return GroupType(
        numba.np.numpy_support.from_dtype(val.dtype),
        numba.np.numpy_support.from_dtype(val.index_dtype),
    )


# The typing of the python "function" Group.__init__
# as it appears in python code
@type_callable(Group)
def type_group(context):
    def typer(group_data, size, index):
        if (
            isinstance(group_data, types.Array)
            and isinstance(size, types.Integer)
            and isinstance(index, types.Array)
        ):
            return GroupType(group_data.dtype, index.dtype)

    return typer


@register_model(GroupType)
class GroupModel(models.StructModel):
    """
    Model backing GroupType instances. See the link below for details.
    https://github.com/numba/numba/blob/main/numba/core/datamodel/models.py
    """

    def __init__(self, dmm, fe_type):
        members = [
            ("group_data", types.CPointer(fe_type.group_scalar_type)),
            ("size", group_size_type),
            ("index", types.CPointer(fe_type.index_type)),
        ]
        super().__init__(dmm, fe_type, members)


call_cuda_functions: dict[Any, Any] = {}


def _register_cuda_binary_reduction_caller(funcname, lty, rty, retty):
    cuda_func = cuda.declare_device(
        f"Block{funcname}_{lty}_{rty}",
        retty(types.CPointer(lty), types.CPointer(rty), group_size_type),
    )

    def caller(lhs, rhs, size):
        return cuda_func(lhs, rhs, size)

    call_cuda_functions.setdefault(funcname.lower(), {})

    type_key = retty, lty, rty
    call_cuda_functions[funcname.lower()][type_key] = caller


def _register_cuda_unary_reduction_caller(funcname, inputty, retty):
    cuda_func = cuda.declare_device(
        f"Block{funcname}_{inputty}",
        retty(types.CPointer(inputty), group_size_type),
    )

    def caller(data, size):
        return cuda_func(data, size)

    call_cuda_functions.setdefault(funcname.lower(), {})

    type_key = (retty, inputty)
    call_cuda_functions[funcname.lower()][type_key] = caller


def _register_cuda_idx_reduction_caller(funcname, inputty):
    cuda_func = cuda.declare_device(
        f"Block{funcname}_{inputty}",
        types.int64(
            types.CPointer(inputty),
            types.CPointer(index_default_type),
            group_size_type,
        ),
    )

    def caller(data, index, size):
        return cuda_func(data, index, size)

    # only support default index type right now
    type_key = (index_default_type, inputty)
    call_cuda_functions.setdefault(funcname.lower(), {})
    call_cuda_functions[funcname.lower()][type_key] = caller


class GroupOpBase(AbstractTemplate):
    def make_error_string(self, args):
        fname = self.key.__name__
        sr_err = ", ".join(["Series" for _ in range(len(args))])
        return (
            f"{fname}({sr_err}) is not supported by JIT GroupBy "
            f"apply. Supported features are listed at: {_UDF_DOC_URL}"
        )

    def generic(self, args, kws):
        # early exit to make sure typing doesn't fail for normal
        # non-group ops
        if not all(isinstance(arg, GroupType) for arg in args):
            return None
        # check if any groups are poisoned for this op
        for arg in args:
            if isinstance(arg.group_scalar_type, types.Poison):
                raise UDFError(
                    f"Use of a column of {arg.group_scalar_type.ty} detected "
                    "within UDF body. Only columns of the following dtypes "
                    "may be used through the GroupBy.apply() JIT engine: "
                    f"{[str(x) for x in SUPPORTED_GROUPBY_NUMPY_TYPES]}"
                )
        if funcs := call_cuda_functions.get(self.key.__name__):
            for sig in funcs.keys():
                if all(
                    arg.group_scalar_type == ty
                    for arg, ty in zip(args, sig, strict=True)
                ):
                    return nb_signature(sig[0], *args)
        raise UDFError(self.make_error_string(args))


class GroupAttrBase(AbstractTemplate):
    def make_error_string(self, args):
        fname = self.key.split(".")[-1]
        args = (self.this, *args)
        dtype_err = ", ".join([str(g.group_scalar_type) for g in args])
        sr_err = ", ".join(["Series" for _ in range(len(args) - 1)])
        return (
            f"Series.{fname}({sr_err}) is not supported for "
            f"({dtype_err}) within JIT GroupBy apply. To see "
            f"what's available, visit {_UDF_DOC_URL}"
        )

    def generic(self, args, kws):
        # earlystop to make sure typing doesn't fail for normal
        # non-group ops
        if not all(isinstance(arg, GroupType) for arg in args):
            return None
        # check if any groups are poisioned for this op
        for arg in (self.this, *args):
            if isinstance(arg.group_scalar_type, types.Poison):
                raise UDFError(
                    f"Use of a column of {arg.group_scalar_type.ty} detected "
                    "within UDAF body. Only columns of the following dtypes "
                    "may be used through the GroupBy.apply() JIT engine: "
                    f"{[str(x) for x in SUPPORTED_GROUPBY_NUMPY_TYPES]}"
                )
        fname = self.key.split(".")[-1]
        if funcs := call_cuda_functions.get(fname):
            for sig in funcs.keys():
                retty, selfty, *argtys = sig
                if self.this.group_scalar_type == selfty and all(
                    arg.group_scalar_type == ty
                    for arg, ty in zip(args, argtys, strict=True)
                ):
                    return nb_signature(retty, *args, recvr=self.this)
        raise UDFError(self.make_error_string(args))


class GroupUnaryAttrBase(GroupAttrBase):
    pass


class GroupBinaryAttrBase(GroupAttrBase):
    pass


def _make_unary_attr(funcname):
    class GroupUnaryReductionAttrTyping(GroupUnaryAttrBase):
        key = f"GroupType.{funcname}"

    def _attr(self, mod):
        return types.BoundFunction(
            GroupUnaryReductionAttrTyping,
            GroupType(mod.group_scalar_type, mod.index_type),
        )

    return _attr


def _create_reduction_attr(name, retty=None):
    class Attr(AbstractTemplate):
        key = name

    def generic(self, args, kws):
        return nb_signature(
            self.this.group_scalar_type if not retty else retty,
            recvr=self.this,
        )

    Attr.generic = generic

    def _attr(self, mod):
        return types.BoundFunction(
            Attr, GroupType(mod.group_scalar_type, mod.index_type)
        )

    return _attr


class GroupIdxMax(AbstractTemplate):
    key = "GroupType.idxmax"

    def generic(self, args, kws):
        return nb_signature(self.this.index_type, recvr=self.this)


class GroupIdxMin(AbstractTemplate):
    key = "GroupType.idxmin"

    def generic(self, args, kws):
        return nb_signature(self.this.index_type, recvr=self.this)


class GroupCorr(GroupBinaryAttrBase):
    key = "GroupType.corr"


class DataFrameAttributeTemplate(AttributeTemplate):
    def resolve(self, value, attr):
        raise UDFError(
            f"JIT GroupBy.apply() does not support DataFrame.{attr}(). "
        )


@cuda_registry.register_attr
class DataFrameAttr(DataFrameAttributeTemplate):
    key = GroupByJITDataFrame


@cuda_registry.register_attr
class GroupAttr(AttributeTemplate):
    key = GroupType

    resolve_max = _make_unary_attr("max")
    resolve_min = _make_unary_attr("min")
    resolve_sum = _make_unary_attr("sum")

    resolve_mean = _make_unary_attr("mean")
    resolve_var = _make_unary_attr("var")
    resolve_std = _make_unary_attr("std")

    resolve_size = _create_reduction_attr(
        "GroupType.size", retty=group_size_type
    )
    resolve_count = _create_reduction_attr(
        "GroupType.count", retty=types.int64
    )

    def resolve_idxmax(self, mod):
        return types.BoundFunction(
            GroupIdxMax, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_idxmin(self, mod):
        return types.BoundFunction(
            GroupIdxMin, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_corr(self, mod):
        return types.BoundFunction(
            GroupCorr, GroupType(mod.group_scalar_type, mod.index_type)
        )


for ty in SUPPORTED_GROUPBY_NUMBA_TYPES:
    _register_cuda_unary_reduction_caller("Max", ty, ty)
    _register_cuda_unary_reduction_caller("Min", ty, ty)
    _register_cuda_idx_reduction_caller("IdxMax", ty)
    _register_cuda_idx_reduction_caller("IdxMin", ty)

    if ty in types.integer_domain:
        _register_cuda_binary_reduction_caller("Corr", ty, ty, types.float64)


_register_cuda_unary_reduction_caller("Sum", types.int32, types.int64)
_register_cuda_unary_reduction_caller("Sum", types.int64, types.int64)
_register_cuda_unary_reduction_caller("Sum", types.float32, types.float32)
_register_cuda_unary_reduction_caller("Sum", types.float64, types.float64)


_register_cuda_unary_reduction_caller("Mean", types.int32, types.float64)
_register_cuda_unary_reduction_caller("Mean", types.int64, types.float64)
_register_cuda_unary_reduction_caller("Mean", types.float32, types.float32)
_register_cuda_unary_reduction_caller("Mean", types.float64, types.float64)

_register_cuda_unary_reduction_caller("Std", types.int32, types.float64)
_register_cuda_unary_reduction_caller("Std", types.int64, types.float64)
_register_cuda_unary_reduction_caller("Std", types.float32, types.float32)
_register_cuda_unary_reduction_caller("Std", types.float64, types.float64)

_register_cuda_unary_reduction_caller("Var", types.int32, types.float64)
_register_cuda_unary_reduction_caller("Var", types.int64, types.float64)
_register_cuda_unary_reduction_caller("Var", types.float32, types.float32)
_register_cuda_unary_reduction_caller("Var", types.float64, types.float64)


for attr in ("group_data", "index", "size"):
    make_attribute_wrapper(GroupType, attr, attr)


for op in arith_ops + comparison_ops + unary_ops:
    cuda_registry.register_global(op)(GroupOpBase)
