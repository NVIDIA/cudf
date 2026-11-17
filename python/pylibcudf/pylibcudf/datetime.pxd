# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.datetime cimport datetime_component, rounding_frequency
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column extract_datetime_component(
    Column input,
    datetime_component component,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column ceil_datetimes(
    Column input,
    rounding_frequency freq,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column floor_datetimes(
    Column input,
    rounding_frequency freq,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column round_datetimes(
    Column input,
    rounding_frequency freq,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column add_calendrical_months(
    Column timestamps,
    ColumnOrScalar months,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column day_of_year(
    Column input, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column is_leap_year(
    Column input, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column last_day_of_month(
    Column input, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column extract_quarter(
    Column input, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column days_in_month(
    Column input, object stream = *, DeviceMemoryResource mr = *
)
