# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cudf_polars_cucascade.libcucascade.version cimport (
    CUDF_VERSION_MAJOR,
    CUDF_VERSION_MINOR,
    CUDF_VERSION_PATCH,
)

__all__ = ["built_against_cudf_version"]

#: The (major, minor, patch) of the libcudf this extension was compiled against. It subclasses
#: libcudf and pylibcudf types, so it only works with the libcudf it was built against;
#: cudf-polars checks this before using it.
built_against_cudf_version = (CUDF_VERSION_MAJOR, CUDF_VERSION_MINOR, CUDF_VERSION_PATCH)
