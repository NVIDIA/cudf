# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

def fill(
    destination: Column,
    begin: int,
    end: int,
    value: Scalar,
    stream: CudaStreamLike | None = None,
) -> Column: ...
def fill_in_place(
    destination: Column,
    begin: int,
    end: int,
    value: Scalar,
    stream: CudaStreamLike | None = None,
) -> None: ...
def sequence(
    size: int, init: Scalar, step: Scalar, stream: CudaStreamLike | None = None
) -> Column: ...
def repeat(
    input_table: Table,
    count: Column | int,
    stream: CudaStreamLike | None = None,
) -> Table: ...
def calendrical_month_sequence(
    n: int, init: Scalar, months: int, stream: CudaStreamLike | None = None
) -> Column: ...
