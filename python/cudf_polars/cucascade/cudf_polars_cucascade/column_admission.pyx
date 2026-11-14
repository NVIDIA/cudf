# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cudf_polars_cucascade.libcucascade.column_admission cimport (
    column_stats as cpp_column_stats,
)

__all__ = ["ColumnStats"]


cdef class ColumnStats:
    """
    Admission accounting of one column, in bytes.

    A column is ranked by how often reads ask for all of it, and the cache keeps the
    columns that rank highest, up to its admission budget. Get them from
    :meth:`ScanCache.column_stats`.
    """
    __hash__ = None

    def __init__(self):
        raise ValueError(
            "ColumnStats cannot be constructed directly; use ScanCache.column_stats."
        )

    @staticmethod
    cdef ColumnStats from_libcucascade(cpp_column_stats stats):
        cdef ColumnStats result = ColumnStats.__new__(ColumnStats)
        result.c_obj = stats
        return result

    @property
    def name(self):
        """Name of the column, ``"<table>/<column>"``."""
        return self.c_obj.name.decode()

    @property
    def registered(self):
        """Size of the column in every registered file."""
        return self.c_obj.registered

    @property
    def requested(self):
        """Bytes of the column that reads asked for."""
        return self.c_obj.requested

    @property
    def admitted(self):
        """Requested bytes that the policy asked the cache to keep."""
        return self.c_obj.admitted

    @property
    def refused(self):
        """Requested bytes that the policy left uncached."""
        return self.c_obj.refused

    @property
    def admitted_footprint(self):
        """Cache footprint, in whole chunks, of the column's admitted pieces."""
        return self.c_obj.admitted_footprint

    @property
    def rank(self):
        """``requested / registered``: how often the column has been read in full."""
        return self.c_obj.rank

    def __repr__(self):
        return (
            f"ColumnStats(name={self.name!r}, registered={self.registered}, "
            f"requested={self.requested}, admitted={self.admitted}, "
            f"refused={self.refused})"
        )
