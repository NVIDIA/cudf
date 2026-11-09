# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.optional cimport optional
from libcpp.span cimport span as std_span
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from pylibcudf.io.datasource cimport Datasource
from pylibcudf.libcudf.io.datasource cimport datasource

from cudf_polars_cucascade._exception_handler cimport cucascade_exception_handler

__all__ = [
    "AdvisingDatasource",
    "ColumnStats",
    "ScanCache",
    "ScanCacheOptions",
    "built_against_cudf_version",
]

ctypedef const uint32_t const_uint32_t
ctypedef const uint64_t const_uint64_t


cdef extern from "cudf/version_config.hpp" nogil:
    enum:
        CUDF_VERSION_MAJOR
        CUDF_VERSION_MINOR
        CUDF_VERSION_PATCH

#: The (major, minor, patch) of the libcudf this module was compiled against. It subclasses
#: libcudf and pylibcudf types, so it only works with the libcudf it was built against;
#: cudf-polars checks this before using it.
built_against_cudf_version = (CUDF_VERSION_MAJOR, CUDF_VERSION_MINOR, CUDF_VERSION_PATCH)


cdef extern from "cucascade/io/column_admission.hpp" namespace "cucascade::io" nogil:

    cdef cppclass column_stats:
        string name
        uint64_t registered
        uint64_t requested
        uint64_t admitted
        uint64_t refused
        uint64_t admitted_footprint
        double rank


cdef extern from "cucascade/io/scan_cache.hpp" namespace "cucascade::io" nogil:

    cdef cppclass scan_cache_options:
        scan_cache_options() except +cucascade_exception_handler
        size_t cache_bytes
        size_t block_bytes
        double eviction_threshold
        double eviction_batch
        double admission_budget
        size_t min_advise_bytes
        size_t reactors
        size_t connections
        long stall_seconds
        size_t pinning_threads
        optional[int] numa_node
        string endpoint
        string region
        string access_key
        string secret_key
        string session_token


cdef extern from "cucascade_datasource.hpp" namespace "cudf_polars_cucascade" nogil:

    cdef cppclass cucascade_datasource(datasource):
        size_t size() except +cucascade_exception_handler const


cdef extern from "scan_cache.hpp" namespace "cudf_polars_cucascade" nogil:

    cdef cppclass scan_cache:
        scan_cache(const scan_cache_options&) except +cucascade_exception_handler
        unique_ptr[datasource] open(
            const string& path
        ) except +cucascade_exception_handler
        void register_layout(
            const string& path,
            const string& table,
            const vector[string]& columns,
            std_span[const_uint32_t] column_of_chunk,
            std_span[const_uint64_t] offsets,
            std_span[const_uint64_t] sizes,
        ) except +cucascade_exception_handler
        vector[column_stats] stats() except +cucascade_exception_handler const
        string summary() except +cucascade_exception_handler const


cdef class ColumnStats:
    """
    Admission accounting of one column, in bytes.

    A column is ranked by how often reads ask for all of it, and the cache keeps the
    columns that rank highest, up to its admission budget.

    Not constructible directly: get them from :meth:`ScanCache.column_stats`.
    """
    cdef column_stats c_obj

    __hash__ = None

    def __init__(self):
        raise ValueError("Get ColumnStats from ScanCache.column_stats.")

    @staticmethod
    cdef ColumnStats from_cpp(column_stats stats):
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


cdef class AdvisingDatasource(Datasource):
    """
    An object-store object read through cuCascade's prefetching cache.

    Every sizeable read of it is advised to the cache, under the cache's column
    admission policy, before it runs. It is a :class:`pylibcudf.io.datasource.Datasource`,
    so any pylibcudf reader can read it.

    Not constructible directly: get them from :meth:`ScanCache.open`. A datasource
    keeps the cache that opened it alive.
    """
    cdef unique_ptr[datasource] c_obj
    cdef object owner

    def __init__(self):
        raise ValueError("Get an AdvisingDatasource from ScanCache.open.")

    @staticmethod
    cdef AdvisingDatasource from_cpp(unique_ptr[datasource] source, object owner):
        cdef AdvisingDatasource result = AdvisingDatasource.__new__(AdvisingDatasource)
        result.c_obj = move(source)
        result.owner = owner
        return result

    cdef datasource* get_datasource(self) except * nogil:
        return self.c_obj.get()

    cpdef size_t nbytes(self):
        """
        Size of the object.

        Returns
        -------
        int
            Size in bytes.
        """
        return (<cucascade_datasource*> self.c_obj.get()).size()


cdef class ScanCacheOptions:
    """
    Settings of a :class:`ScanCache`.

    Parameters
    ----------
    cache_bytes
        Size of the pinned host cache. All of it is allocated when the cache is created.
    endpoint
        Object store endpoint, for example ``https://s3.us-east-1.amazonaws.com``.
    region
        Object store region.
    access_key
        Access key of the credentials.
    secret_key
        Secret key of the credentials.
    session_token
        Session token of the credentials, if they are temporary.
    block_bytes
        Cache chunk size. 256 KiB measured best on S3.
    eviction_threshold
        Share of the pool in use at which the evictor starts. Leave headroom below
        1.0 for the pinned staging blocks of reads that are not cached.
    eviction_batch
        Share of the pool that one eviction pass frees.
    admission_budget
        Share of the cache that admitted columns may fill, counted in whole chunks.
    min_advise_bytes
        Reads smaller than this are not cached. 0 caches nothing.
    reactors
        REST reactor threads.
    connections
        Connections each reactor drives.
    stall_seconds
        A request that moves under 64 KiB/s for this long is aborted and retried.
    pinning_threads
        Threads that pin the cache when it is created.
    numa_node
        NUMA node to place the cache on. ``None`` (the default) uses the node of the current
        CUDA device. -1 leaves the placement to the CUDA driver, which pins one pool at a time and
        is slower.
    """
    cdef scan_cache_options c_obj

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
    options
        The settings of the cache.
    """
    cdef unique_ptr[scan_cache] c_obj

    def __init__(self, ScanCacheOptions options not None):
        with nogil:
            self.c_obj = make_unique[scan_cache](options.c_obj)

    cpdef AdvisingDatasource open(self, str path):
        """
        Open an object-store object for reading.

        The first open of a path resolves its size, version tag and footer with one
        request; later opens of it need none.

        Parameters
        ----------
        path
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
        return AdvisingDatasource.from_cpp(move(c_source), self)

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
        path
            Path of the file.
        table
            Name shared by the files of one table.
        columns
            Column names.
        column_of_chunk
            For each column chunk, its index into ``columns``.
        offsets
            For each column chunk, its byte offset in the file.
        sizes
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

    cpdef list column_stats(self):
        """
        Per-column admission accounting.

        Returns
        -------
        list[ColumnStats]
            One entry per column seen.
        """
        cdef vector[column_stats] c_stats
        with nogil:
            c_stats = self.c_obj.get().stats()
        return [ColumnStats.from_cpp(stats) for stats in c_stats]

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
