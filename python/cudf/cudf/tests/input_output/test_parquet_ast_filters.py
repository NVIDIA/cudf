# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression for the Parquet AST scalar guard; requires a CUDA GPU."""

import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cudf
import cudf.io.parquet as parquet
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "value",
    [
        "absolute",
        1,
        1.25,
        datetime.date(2026, 1, 1),
        datetime.timedelta(days=1),
    ],
)
def test_parquet_ast_supported_scalar(value):
    assert parquet._raise_for_unsupported_scalar_types(value) is value


@pytest.mark.parametrize("value", [object(), b"absolute", ["absolute"], None])
def test_parquet_ast_unsupported_scalar(value):
    with pytest.raises(NotImplementedError):
        parquet._raise_for_unsupported_scalar_types(value)


@pytest.mark.parametrize("low_memory", [False, True])
@pytest.mark.parametrize("project", [False, True])
def test_parquet_ast_string_filter_skips_dataset(
    tmp_path, monkeypatch, low_memory, project
):
    table = pa.table(
        {
            "image_url": [
                "same",
                "invalid",
                "same",
                "null",
                "case",
                "unsupported",
                "same",
                "missing",
            ],
            "page_url": [
                "page",
                None,
                None,
                None,
                "other",
                "page",
                "page",
                "page",
            ],
            "crawl_id": ["cc", "cc", None, "cc", None, "cc", "cc", "cc"],
            "url_status": [
                "absolute",
                "invalid",
                "resolved_relative",
                None,
                "absolute",
                "unsupported",
                "absolute",
                "missing",
            ],
        }
    )
    paths = [tmp_path / "first.parquet", tmp_path / "second.parquet"]
    for index, path in enumerate(paths):
        pq.write_table(table.slice(index * 4, 4), path, row_group_size=2)
    paths = [str(path) for path in paths]
    filters = [("url_status", "in", ["absolute", "resolved_relative"])]
    columns = ["image_url", "page_url", "crawl_id"] if project else None
    expected = cudf.DataFrame.from_arrow(
        pq.read_table(paths, columns=columns, filters=filters)
    )

    def reject_dataset_fallback(*args, **kwargs):
        pytest.fail("Supported string IN filter fell back to Arrow dataset")

    monkeypatch.setattr(parquet, "_process_dataset", reject_dataset_fallback)
    with cudf.option_context("io.parquet.low_memory", low_memory):
        actual = cudf.read_parquet(
            paths,
            columns=columns,
            filters=filters,
            categorical_partitions=None,
            use_pandas_metadata=False,
        )
    assert len(actual) == 4
    assert_eq(actual.reset_index(drop=True), expected.reset_index(drop=True))
