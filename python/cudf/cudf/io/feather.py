# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

from pyarrow import feather

from cudf.core.dataframe import DataFrame
from cudf.utils import ioutils


@ioutils.doc_read_feather()
def read_feather(path, *args, **kwargs):
    """{docstring}"""

    warnings.warn(
        "Using CPU via PyArrow to read feather dataset, this may "
        "be GPU accelerated in the future"
    )
    # ``pyarrow.feather`` is deprecated as of PyArrow 24 but remains the
    # supported compatibility wrapper for the Feather-specific API.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pyarrow\.feather\.read_table is deprecated",
            category=FutureWarning,
        )
        pa_table = feather.read_table(path, *args, **kwargs)
    return DataFrame.from_arrow(pa_table)


@ioutils.doc_to_feather()
def to_feather(df, path, *args, **kwargs):
    """{docstring}"""
    warnings.warn(
        "Using CPU via PyArrow to write Feather dataset, this may "
        "be GPU accelerated in the future"
    )
    # Feather doesn't support using an index
    pa_table = df.to_arrow(preserve_index=False)
    # See the corresponding note in ``read_feather`` above.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pyarrow\.feather\.write_feather is deprecated",
            category=FutureWarning,
        )
        feather.write_feather(pa_table, path, *args, **kwargs)
