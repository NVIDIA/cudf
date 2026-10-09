# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pylibcudf as plc

from cudf.core.dataframe import DataFrame
from cudf.utils import ioutils


@ioutils.doc_read_avro()
def read_avro(
    filepath_or_buffer,
    columns=None,
    skiprows=None,
    num_rows=None,
    storage_options=None,
    filesystem=None,
):
    """{docstring}"""

    filepath_or_buffer = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        storage_options=storage_options,
        filesystem=filesystem,
    )
    filepath_or_buffer = ioutils._select_single_source(
        filepath_or_buffer, "read_avro"
    )

    num_rows = -1 if num_rows is None else num_rows
    skip_rows = 0 if skiprows is None else skiprows

    if not isinstance(num_rows, int) or num_rows < -1:
        raise TypeError("num_rows must be an int >= -1")
    if not isinstance(skip_rows, int) or skip_rows < 0:
        raise TypeError("skip_rows must be an int >= 0")

    options = (
        plc.io.avro.AvroReaderOptions.builder(
            plc.io.types.SourceInfo([filepath_or_buffer])
        )
        .skip_rows(skip_rows)
        .num_rows(num_rows)
        .build()
    )

    if columns is not None and len(columns) > 0:
        options.set_columns(columns)

    return DataFrame.from_pylibcudf(plc.io.avro.read_avro(options))
