# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_table_eq

import rmm

import pylibcudf as plc


@pytest.fixture
def stats_mrs():
    upstream = rmm.mr.get_current_device_resource()
    return (
        rmm.mr.StatisticsResourceAdaptor(upstream),
        rmm.mr.StatisticsResourceAdaptor(upstream),
    )


def test_defaults_to_current_device_resource():
    current = rmm.mr.get_current_device_resource()
    mrs = plc.MemoryResources()
    assert mrs.output is current
    assert mrs.temporary is current


def test_attributes(stats_mrs):
    output, temporary = stats_mrs
    mrs = plc.MemoryResources(output=output, temporary=temporary)
    assert isinstance(mrs, rmm.mr.DeviceMemoryResource)
    assert mrs.output is output
    assert mrs.temporary is temporary
    assert plc.MemoryResources(output).temporary is (
        rmm.mr.get_current_device_resource()
    )


def test_allocates_from_output(stats_mrs):
    output, temporary = stats_mrs
    mrs = plc.MemoryResources(output, temporary)
    ptr = mrs.allocate(256)
    assert output.allocation_counts.current_bytes == 256
    assert temporary.allocation_counts.total_count == 0
    mrs.deallocate(ptr, 256)
    assert output.allocation_counts.current_bytes == 0


def test_output_and_temporary_resources(stats_mrs):
    output, temporary = stats_mrs
    num_rows = 1000
    pa_table = pa.table({"a": pa.array(range(num_rows), type=pa.int64())})
    pa_mask = pa.array([i % 2 == 0 for i in range(num_rows)])
    table = plc.Table.from_arrow(pa_table)
    mask = plc.Column.from_arrow(pa_mask)

    result = plc.stream_compaction.apply_boolean_mask(
        table, mask, mr=plc.MemoryResources(output, temporary)
    )

    assert_table_eq(pa_table.filter(pa_mask), result)
    assert output.allocation_counts.current_bytes > 0
    assert temporary.allocation_counts.total_count > 0
    assert temporary.allocation_counts.current_bytes == 0


def test_single_resource_is_output(stats_mrs):
    output, _ = stats_mrs
    pa_table = pa.table({"a": pa.array(range(100), type=pa.int64())})
    pa_mask = pa.array([i % 2 == 0 for i in range(100)])

    result = plc.stream_compaction.apply_boolean_mask(
        plc.Table.from_arrow(pa_table),
        plc.Column.from_arrow(pa_mask),
        mr=output,
    )

    assert_table_eq(pa_table.filter(pa_mask), result)
    assert output.allocation_counts.current_bytes > 0
