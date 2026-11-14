# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional cuCascade-backed reads of remote parquet files."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

import pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "configure_cucascade",
    "register_layouts",
    "register_layouts_from_sources",
    "wrap_sources",
]

_MAX_OPEN_THREADS = 32
_EXTENSION = "cudf_polars_cucascade"

_lock = threading.Lock()
_cache: Any = None
_registered: set[str] = set()


def _installed_cudf_version() -> tuple[int, int] | None:
    """The (major, minor) of the libcudf in use, from the installed pylibcudf, or None."""
    for distribution in ("pylibcudf", "libcudf"):
        try:
            installed = Version(importlib.metadata.version(distribution))
        except (importlib.metadata.PackageNotFoundError, InvalidVersion):
            continue
        return installed.major, installed.minor
    return None


def _load_extension() -> types.ModuleType:
    """Import the extension and check it was built against the libcudf that is in use."""
    try:
        module = importlib.import_module(_EXTENSION)
    except ImportError as err:
        raise ImportError(
            f"cucascade_cache_bytes is set but the {_EXTENSION} extension cannot be "
            "imported. It is built from source, against the libcudf of the environment it "
            "runs in. From the root of the cudf repository, run\n\n"
            "    pip install --no-build-isolation --no-deps ./python/cudf_polars/cucascade\n\n"
            "See python/cudf_polars/cucascade/README.md."
        ) from err
    built = tuple(module.built_against_cudf_version[:2])
    running = _installed_cudf_version()
    if running is not None and built != running:
        raise ImportError(
            f"{_EXTENSION} was built against libcudf {built[0]}.{built[1]}, but libcudf "
            f"{running[0]}.{running[1]} is installed. It subclasses libcudf types, so it only "
            "works with the libcudf it was built against: rebuild it with the command in "
            "python/cudf_polars/cucascade/README.md."
        )
    return module


def configure_cucascade(cache_bytes: int) -> None:
    """
    Create the cuCascade cache for this process.

    The cache is process-global and is created once, so a later call with a
    different size has no effect.

    Parameters
    ----------
    cache_bytes
        Size of the host-memory cache in bytes. The memory is pinned here, at
        engine start, so the first query does not pay for it. 0 leaves the
        cache off.
    """
    global _cache  # noqa: PLW0603
    if cache_bytes == 0:
        return
    with _lock:
        if _cache is not None:
            return
        module = _load_extension()
        region = (
            os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
            or "us-east-1"
        )
        _cache = module.ScanCache(
            module.ScanCacheConfig(
                cache_bytes=cache_bytes,
                endpoint=f"https://s3.{region}.amazonaws.com",
                region=region,
                access_key=os.environ["AWS_ACCESS_KEY_ID"],
                secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                session_token=os.environ.get("AWS_SESSION_TOKEN", ""),
            )
        )


def _is_remote(path: str) -> bool:
    return "://" in str(path)


def _layout(info: Any) -> tuple[list[str], list[int], list[int], list[int]]:
    """
    Byte ranges of every column chunk of one parquet file.

    Parameters
    ----------
    info
        An object with ``path``, ``size`` and the parsed ``file_metadata``.

    Returns
    -------
    Column names, then for each column chunk its column index, offset and size.
    """
    file_metadata = info.file_metadata
    source = plc.io.SourceInfo([plc.io.types.FilepathSource(info.path, info.size)])

    def options(columns: list[str] | None) -> Any:
        opts = plc.io.parquet.ParquetReaderOptions.builder(source).build()
        if columns is not None:
            opts.set_column_names(columns)
        return opts

    def reader(opts: Any) -> Any:
        return plc.io.experimental.HybridScanReader.from_metadata(
            plc.io.experimental.HybridScanMetadata.from_parquet_metadata(
                file_metadata, opts
            )
        )

    base = options(None)
    row_groups = reader(base).all_row_groups(base)
    columns = list(file_metadata.columnchunk_metadata.keys())
    col_index: list[int] = []
    offsets: list[int] = []
    sizes: list[int] = []
    # One call with every column selected returns the chunks row-group-major.
    all_opts = options(columns)
    ranges = reader(all_opts).all_column_chunks_byte_ranges(row_groups, all_opts)
    if len(ranges) == len(row_groups) * len(columns):
        for k, rng in enumerate(ranges):
            col_index.append(k % len(columns))
            offsets.append(rng.offset)
            sizes.append(rng.size)
    else:  # nested columns: one call per column
        for i, name in enumerate(columns):
            opts = options([name])
            for rng in reader(opts).all_column_chunks_byte_ranges(row_groups, opts):
                col_index.append(i)
                offsets.append(rng.offset)
                sizes.append(rng.size)
    return columns, col_index, offsets, sizes


def register_layouts(infos: Sequence[Any]) -> None:
    """
    Tell the cache where each column chunk of the given files lives.

    The cache uses the layout to rank columns by how often queries read them and
    to admit only the hot ones. A file whose layout cannot be computed is read
    without column-aware admission.

    Parameters
    ----------
    infos
        Objects with ``path``, ``size`` and the parsed ``file_metadata``.
    """
    if _cache is None:
        return
    for info in infos:
        path = str(info.path)
        if not _is_remote(path):
            continue
        with _lock:
            if path in _registered:
                continue
            _registered.add(path)
        try:
            columns, col_index, offsets, sizes = _layout(info)
        except Exception:
            continue
        _cache.register_layout(
            path, path.rsplit("/", 1)[0], columns, col_index, offsets, sizes
        )


def register_layouts_from_sources(paths: Sequence[str], sources: Sequence[Any]) -> None:
    """
    Register the layout of files whose footers the engine did not read.

    Parameters
    ----------
    paths
        File paths.
    sources
        The sources returned by :func:`wrap_sources` for ``paths``.
    """
    if _cache is None:
        return
    with _lock:
        todo = [
            (p, s)
            for p, s in zip(paths, sources, strict=True)
            if str(p) not in _registered and _is_remote(str(p))
        ]
    if not todo:
        return
    try:
        footers = plc.io.parquet_metadata.read_parquet_footers(
            plc.io.types.SourceInfo([s for _, s in todo])
        )
    except Exception:
        return
    register_layouts(
        [
            types.SimpleNamespace(path=str(p), size=s.nbytes(), file_metadata=fm)
            for (p, s), fm in zip(todo, footers, strict=True)
        ]
    )


def wrap_sources(paths: Sequence[str]) -> list[Any] | None:
    """
    Return cuCascade-backed data sources for ``paths``.

    Opening a file the first time resolves its size and footer with one request;
    later opens of it need none.

    Parameters
    ----------
    paths
        File paths.

    Returns
    -------
    One source per path, or None when the cache is off or any path is not remote,
    in which case the caller uses the standard read path. The caller must keep the
    returned list alive for as long as the sources are read.
    """
    if _cache is None or not paths or not all(_is_remote(p) for p in paths):
        return None
    if len(paths) == 1:
        return [_cache.open(str(paths[0]))]
    # Opening a source (size lookup plus footer read) releases the GIL.
    with ThreadPoolExecutor(min(_MAX_OPEN_THREADS, len(paths))) as pool:
        return list(pool.map(lambda p: _cache.open(str(p)), paths))
