# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for normalizing file paths and URIs."""

from __future__ import annotations

__all__ = ["strip_file_uri"]


def strip_file_uri(path: str) -> str:
    """
    Convert a ``file:`` URI to a plain local filesystem path.

    libcudf and KvikIO treat anything that looks like ``<scheme>://...`` as a
    remote URL, so local paths expressed as ``file:`` URIs must have the scheme
    removed before they are handed to a reader.

    Parameters
    ----------
    path
        A path or URI. ``file:///abs/path`` and ``file:/abs/path`` (the form
        written by some Iceberg catalogs) both become ``/abs/path``. Paths
        with any other scheme (``s3://``, ``gs://``, ``abfs://``, ...) and
        plain paths are returned unchanged.

    Returns
    -------
    The local path, or ``path`` unchanged if it is not a ``file:`` URI.

    Notes
    -----
    Polars rejects ``file://`` URIs with a non-empty host, so
    ``file://host/foo/path`` is not handled specially.
    """
    # Like Polars, only a lowercase ``file:`` scheme is recognized.
    if not path.startswith("file:"):
        return path
    rest = path[5:]
    if rest.startswith("//"):
        # file:///foo/path is treated as a local path.
        return rest[2:]
    if rest.startswith("/"):
        # file:/foo/path
        return rest
    return path
