# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.table import Table
from pylibcudf.types import NullEquality
from pylibcudf.utils import CudaStreamLike

def tables_equal(
    left: Table,
    right: Table,
    nulls_equal: NullEquality = NullEquality.EQUAL,
    stream: CudaStreamLike | None = None,
) -> bool: ...
