# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pylibcudf as plc

from cudf.core.column import ColumnBase
from cudf.core.column.column import _normalize_types_column
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.dataframe import DataFrame
from cudf.core.series import Series
from cudf.utils.dtypes import dtype_from_pylibcudf_column


def from_dlpack(pycapsule_obj) -> Series | DataFrame:
    """Converts from a DLPack tensor to a cuDF object.

    DLPack is an open-source memory tensor structure:
    `dmlc/dlpack <https://github.com/dmlc/dlpack>`_.

    This function takes a PyCapsule object which contains a pointer to
    a DLPack tensor as input, and returns a cuDF object. This function deep
    copies the data in the DLPack tensor into a cuDF object.

    Parameters
    ----------
    pycapsule_obj : PyCapsule
        Input DLPack tensor pointer which is encapsulated in a PyCapsule
        object.

    Returns
    -------
    A cuDF DataFrame or Series depending on if the input DLPack tensor is 1D
    or 2D.

    Notes
    -----
    cuDF from_dlpack() assumes column-major (Fortran order) input. If the input
    tensor is row-major, transpose it before passing it to this function.
    """
    plc_table = plc.interop.from_dlpack(pycapsule_obj)
    data = ColumnAccessor(
        dict(
            enumerate(
                (
                    ColumnBase.create(
                        _normalize_types_column(col),
                        dtype=dtype_from_pylibcudf_column(col),
                    )
                    for col in plc_table.columns()
                )
            )
        ),
        verify=False,
        rangeindex=True,
    )

    if len(data) == 1:
        return Series._from_data(data)
    else:
        return DataFrame._from_data(data)
