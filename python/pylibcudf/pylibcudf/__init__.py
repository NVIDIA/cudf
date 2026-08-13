# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# If libcudf was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libcudf
except ModuleNotFoundError:
    pass
else:
    libcudf.load_library()
    del libcudf

from . import (
    aggregation,
    binaryop,
    column_factories,
    concatenate,
    context,
    contiguous_split,
    copying,
    datetime,
    experimental,
    expressions,
    filling,
    groupby,
    hashing,
    interop,
    io,
    join,
    json,
    labeling,
    lists,
    merge,
    null_mask,
    nvtext,
    partitioning,
    prefetch,
    quantiles,
    reduce,
    replace,
    reshape,
    rolling,
    round,
    search,
    sorting,
    span,
    stream_compaction,
    strings,
    table_equality,
    traits,
    transform,
    transpose,
    types,
    unary,
    utilities,
    utils,
)
from .column import Column
from .gpumemoryview import gpumemoryview
from .scalar import Scalar
from .table import Table
from .types import DataType, MaskState, TypeId

__all__ = [
    "Column",
    "DataType",
    "MaskState",
    "Scalar",
    "Table",
    "TypeId",
    "aggregation",
    "binaryop",
    "column_factories",
    "concatenate",
    "context",
    "contiguous_split",
    "copying",
    "datetime",
    "experimental",
    "expressions",
    "filling",
    "gpumemoryview",
    "groupby",
    "hashing",
    "interop",
    "io",
    "join",
    "json",
    "labeling",
    "lists",
    "merge",
    "null_mask",
    "nvtext",
    "partitioning",
    "prefetch",
    "quantiles",
    "reduce",
    "replace",
    "reshape",
    "rolling",
    "round",
    "search",
    "sorting",
    "span",
    "stream_compaction",
    "strings",
    "table_equality",
    "traits",
    "transform",
    "transpose",
    "types",
    "unary",
    "utilities",
    "utils",
]
