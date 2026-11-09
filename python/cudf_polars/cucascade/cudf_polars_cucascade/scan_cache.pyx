# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.span cimport span as std_span
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from pylibcudf.libcudf.io.datasource cimport datasource

from cudf_polars_cucascade.advising_datasource cimport AdvisingDatasource
from cudf_polars_cucascade.column_admission cimport ColumnStats
from cudf_polars_cucascade.libcucascade.column_admission cimport (
    column_stats as cpp_column_stats,
)
from cudf_polars_cucascade.libcucascade.scan_cache cimport (
    const_uint32_t,
    const_uint64_t,
    scan_cache as cpp_scan_cache,
)

__all__ = ["ScanCache", "ScanCacheConfig"]


cdef class ScanCacheConfig:
    """
    Settings of a :class:`ScanCache`.

    Parameters
    ----------
    cache_bytes : int
        Size of the pinned host cache. All of it is allocated when the cache is created.
    endpoint : str
        Object store endpoint, for example ``https://s3.us-east-1.amazonaws.com``.
    region : str
        Object store region.
    access_key : str
        Access key of the credentials.
    secret_key : str
        Secret key of the credentials.
    session_token : str, default ""
        Session token of the credentials, if they are temporary.
    block_bytes : int, default 262144
        Cache chunk size. 256 KiB measured best on S3.
    eviction_threshold : float, default 0.97
        Share of the pool in use at which the evictor starts. Leave headroom below
        1.0 for the pinned staging blocks of reads that are not cached.
    eviction_batch : float, default 0.02
        Share of the pool that one eviction pass frees.
    admission_budget : float, default 0.96
        Share of the cache that admitted columns may fill, counted in whole chunks.
    min_advise_bytes : int, default 262144
        Reads smaller than this are not cached. 0 caches nothing.
    reactors : int, default 16
        REST reactor threads.
    connections : int, default 128
        Connections each reactor drives.
    stall_seconds : int, default 5
        A request that moves under 64 KiB/s for this long is aborted and retried.
    pinning_threads : int, default 8
        Threads that pin the cache when it is created.
    numa_node : int or None, default None
        NUMA node to place the cache on. ``None`` (the default) uses the node of the current
        CUDA device. -1 leaves the placement to the CUDA driver, which pins one pool at a time and
        is slower.
    """
    def __init__(
        self,
        *,
        size_t cache_bytes,
        str endpoint,
        str region,
        str access_key,
        str secret_key,
        str session_token="",
        size_t block_bytes=262144,
        double eviction_threshold=0.97,
        double eviction_batch=0.02,
        double admission_budget=0.96,
        size_t min_advise_bytes=262144,
        size_t reactors=16,
        size_t connections=128,
        long stall_seconds=5,
        size_t pinning_threads=8,
        numa_node=None,
    ):
        self.c_obj.cache_bytes = cache_bytes
        self.c_obj.block_bytes = block_bytes
        self.c_obj.eviction_threshold = eviction_threshold
        self.c_obj.eviction_batch = eviction_batch
        self.c_obj.admission_budget = admission_budget
        self.c_obj.min_advise_bytes = min_advise_bytes
        self.c_obj.reactors = reactors
        self.c_obj.connections = connections
        self.c_obj.stall_seconds = stall_seconds
        self.c_obj.pinning_threads = pinning_threads
        if numa_node is not None:
            self.c_obj.numa_node = <int> numa_node
        self.c_obj.endpoint = endpoint.encode()
        self.c_obj.region = region.encode()
        self.c_obj.access_key = access_key.encode()
        self.c_obj.secret_key = secret_key.encode()
        self.c_obj.session_token = session_token.encode()


cdef class ScanCache:
    """
    A host-memory cache for the parquet reads of object-store scans.

    Assembles the pinned host cache, the REST io context, the column admission
    policy and a cache of opened objects. :meth:`open` returns a datasource for a
    pylibcudf reader. The cache is pinned when the object is created, which takes
    about 0.2 s per GiB.

    Parameters
    ----------
    config : ScanCacheConfig
        The settings of the cache.
    """
    def __init__(self, ScanCacheConfig config not None):
        with nogil:
            self.c_obj = make_unique[cpp_scan_cache](config.c_obj)

    cpdef AdvisingDatasource open(self, str path):
        """
        Open an object-store object for reading.

        The first open of a path resolves its size, version tag and footer with one
        request; later opens of it need none.

        Parameters
        ----------
        path : str
            An ``s3://`` path.

        Returns
        -------
        AdvisingDatasource
            The object, as a pylibcudf datasource.
        """
        cdef string c_path = path.encode()
        cdef unique_ptr[datasource] c_source
        with nogil:
            c_source = move(self.c_obj.get().open(c_path))
        return AdvisingDatasource.from_libcucascade(move(c_source), self)

    cpdef void register_layout(
        self,
        str path,
        str table,
        list columns,
        list column_of_chunk,
        list offsets,
        list sizes,
    ):
        """
        Register the column chunks of one file with the admission policy.

        The policy ranks the columns of the files that share a table, and caches
        the ones that rank highest.

        Parameters
        ----------
        path : str
            Path of the file.
        table : str
            Name shared by the files of one table.
        columns : list[str]
            Column names.
        column_of_chunk : list[int]
            For each column chunk, its index into ``columns``.
        offsets : list[int]
            For each column chunk, its byte offset in the file.
        sizes : list[int]
            For each column chunk, its size in bytes.
        """
        cdef string c_path = path.encode()
        cdef string c_table = table.encode()
        cdef vector[string] c_columns = [name.encode() for name in columns]
        cdef vector[uint32_t] c_column_of_chunk = column_of_chunk
        cdef vector[uint64_t] c_offsets = offsets
        cdef vector[uint64_t] c_sizes = sizes
        with nogil:
            self.c_obj.get().register_layout(
                c_path,
                c_table,
                c_columns,
                std_span[const_uint32_t](
                    c_column_of_chunk.data(), c_column_of_chunk.size()
                ),
                std_span[const_uint64_t](c_offsets.data(), c_offsets.size()),
                std_span[const_uint64_t](c_sizes.data(), c_sizes.size()),
            )

    cpdef list[ColumnStats] column_stats(self):
        """
        Per-column admission accounting.

        Returns
        -------
        list[ColumnStats]
            One entry per column seen.
        """
        cdef vector[cpp_column_stats] c_stats
        with nogil:
            c_stats = self.c_obj.get().stats()
        return [ColumnStats.from_libcucascade(stats) for stats in c_stats]

    cpdef str summary(self):
        """
        Counters of the prefetching cache (reads, hits, misses, evictions).

        Returns
        -------
        str
            The counters as text.
        """
        cdef string c_summary
        with nogil:
            c_summary = self.c_obj.get().summary()
        return c_summary.decode()
