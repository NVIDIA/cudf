# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-aware assertions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars import GPUEngine
from polars.testing.asserts import assert_frame_equal

from cudf_polars.dsl.translate import Translator
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    from cudf_polars.typing import CollectKwargs


__all__: list[str] = [
    "assert_gpu_result_equal",
    "assert_ir_translation_raises",
    "assert_sink_ir_translation_raises",
    "assert_sink_result_equal",
]


def assert_gpu_result_equal(
    lazydf: pl.LazyFrame,
    *,
    engine: GPUEngine,
    collect_kwargs: CollectKwargs | None = None,
    polars_collect_kwargs: CollectKwargs | None = None,
    check_row_order: bool = True,
    check_column_order: bool = True,
    check_dtypes: bool = True,
    check_exact: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    categorical_as_str: bool = False,
) -> None:
    """
    Assert that collection of a lazyframe on GPU produces correct results.

    Parameters
    ----------
    lazydf
        frame to collect.
    engine
        Custom GPU engine configuration.
    collect_kwargs
        Common keyword arguments to pass to collect for both polars CPU and
        cudf-polars.
        Useful for controlling optimization settings.
    polars_collect_kwargs
        Keyword arguments to pass to collect for execution on polars CPU.
        Overrides kwargs in collect_kwargs.
        Useful for controlling optimization settings.
    check_row_order
        Expect rows to be in same order
    check_column_order
        Expect columns to be in same order
    check_dtypes
        Expect dtypes to match
    check_exact
        Require exact equality for floats, if `False` compare using
        rtol and atol.
    rtol
        Relative tolerance for float comparisons
    atol
        Absolute tolerance for float comparisons
    categorical_as_str
        Decat categoricals to strings before comparing

    Raises
    ------
    AssertionError
        If the GPU and CPU collection do not match.
    NotImplementedError
        If GPU collection failed in some way.
    """
    gpu_kwargs = collect_kwargs or {}
    cpu_kwargs = gpu_kwargs | (polars_collect_kwargs or {})

    # These keywords are correct, but mypy doesn't see that.
    # the 'misc' is for 'error: Keywords must be strings'
    expect = lazydf.collect(**cpu_kwargs)  # type: ignore[call-overload]  # (kwargs assembled dynamically)
    got = lazydf.collect(**gpu_kwargs, engine=engine)  # type: ignore[call-overload]  # (kwargs assembled dynamically)
    # In multi-rank SPMD mode each rank holds only its local slice; gather the
    # full result on every rank so each rank can compare against the CPU result.
    if (
        engine.config.get("executor_options", {}).get("cluster") == "spmd"
    ):  # pragma: no cover
        from cudf_polars.engine.spmd import (
            SPMDEngine,
            allgather_polars_dataframe,
        )

        assert isinstance(engine, SPMDEngine)
        if engine.nranks > 1:
            got = allgather_polars_dataframe(engine=engine, local_df=got, op_id=0)

    assert_kwargs_bool: dict[str, bool] = {
        "check_row_order": check_row_order,
        "check_column_order": check_column_order,
        "check_dtypes": check_dtypes,
        "check_exact": check_exact,
        "categorical_as_str": categorical_as_str,
    }

    tol_kwargs: dict[str, float] = {"rel_tol": rtol, "abs_tol": atol}

    # the type checker errors with:
    # Argument 4 to "assert_frame_equal" has incompatible type "**dict[str, float]"; expected "bool"  [arg-type]
    # which seems to be a bug in the type checker / type annotations.
    assert_frame_equal(expect, got, **assert_kwargs_bool, **tol_kwargs)  # type: ignore[arg-type]


def assert_ir_translation_raises(
    q: pl.LazyFrame, engine: pl.GPUEngine, *exceptions: type[Exception]
) -> None:
    """
    Assert that translation of a query raises an exception.

    Parameters
    ----------
    q
        Query to translate.
    engine
        GPU engine configuration to use during translation.
    exceptions
        Exceptions that one expects might be raised.

    Returns
    -------
    None
        If translation successfully raised the specified exceptions.

    Raises
    ------
    AssertionError
       If the specified exceptions were not raised.
    """
    translator = Translator(q._ldf.visit(), engine)
    translator.translate_ir()
    if errors := translator.errors:
        for err in errors:
            assert any(isinstance(err, err_type) for err_type in exceptions), (
                f"Translation DID NOT RAISE {exceptions}. The following "
                f"errors were seen instead: {errors}"
            )
        return
    else:
        raise AssertionError(f"Translation DID NOT RAISE {exceptions}")


def _resolve_sink_format(path: Path) -> str:
    """Returns valid sink format for assert utilities."""
    suffix = path.suffix.lower()
    supported_ext = {
        ".csv": "csv",
        ".pq": "parquet",
        ".parquet": "parquet",
        ".json": "ndjson",
        ".ndjson": "ndjson",
    }
    if suffix not in supported_ext:
        raise ValueError(f"Unsupported file format: {suffix}")
    return supported_ext[suffix]


def assert_sink_result_equal(
    lazydf: pl.LazyFrame,
    path: str | Path,
    *,
    engine: GPUEngine,
    read_kwargs: dict | None = None,
    write_kwargs: dict | None = None,
) -> None:
    """
    Assert that writing a LazyFrame via sink produces the same output.

    Parameters
    ----------
    lazydf
        The LazyFrame to sink.
    path
        The file path to use. Suffix must be one of:
        '.csv', '.parquet', '.pq', '.json', '.ndjson'.
    engine
        The GPU engine to use for the sink operation.
    read_kwargs
        Optional keyword arguments to pass to the corresponding `pl.read_*` function.
    write_kwargs
        Optional keyword arguments to pass to the corresponding `sink_*` function.

    Raises
    ------
    AssertionError
        If the outputs from CPU and GPU sink differ.
    ValueError
        If the file extension is not one of the supported formats.
    """
    path = Path(path)
    read_kwargs = read_kwargs or {}
    write_kwargs = write_kwargs or {}

    fmt = _resolve_sink_format(path)

    cpu_path = path.with_name(f"{path.stem}_cpu{path.suffix}")
    gpu_path = path.with_name(f"{path.stem}_gpu{path.suffix}")

    sink_fn = getattr(lazydf, f"sink_{fmt}")
    read_fn = getattr(pl, f"read_{fmt}")

    sink_fn(cpu_path, **write_kwargs)
    sink_fn(gpu_path, engine=engine, **write_kwargs)

    expected = read_fn(cpu_path, **read_kwargs)
    # the multi-partition executor might produce multiple files, one per partition.
    if (
        isinstance(engine, GPUEngine)
        and ConfigOptions.from_polars_engine(engine).executor.name == "streaming"
        and gpu_path.is_dir()
    ):  # pragma: no cover
        result = read_fn(gpu_path.joinpath("*"), **read_kwargs)
    else:
        result = read_fn(gpu_path, **read_kwargs)

    assert_frame_equal(expected, result)


def assert_sink_ir_translation_raises(
    lazydf: pl.LazyFrame,
    path: str | Path,
    engine: pl.GPUEngine,
    write_kwargs: dict,
    *exceptions: type[Exception],
) -> None:
    """
    Assert that translation of a sink query raises an exception.

    Parameters
    ----------
    lazydf
        The LazyFrame to sink.
    path
        The file path. Must have one of the supported suffixes.
    engine
        GPU engine configuration to use during translation.
    write_kwargs
        Keyword arguments to pass to the `sink_*` method.
    *exceptions
        One or more expected exception types that should be raised during translation.

    Raises
    ------
    AssertionError
        If translation does not raise any of the expected exceptions.
        If an exception occurs before translation begins.
    ValueError
        If the file extension is not one of the supported formats.
    """
    path = Path(path)
    fmt = _resolve_sink_format(path)

    try:
        lazy_sink = getattr(lazydf, f"sink_{fmt}")(
            path,
            engine=engine,
            lazy=True,
            **write_kwargs,
        )
    except Exception as e:
        raise AssertionError(
            f"Sink function raised an exception before translation: {e}"
        ) from e

    assert_ir_translation_raises(lazy_sink, engine, *exceptions)
