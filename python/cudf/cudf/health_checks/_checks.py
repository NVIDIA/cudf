#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""cuDF health checks for rapids doctor."""

_INSTALL_DOCS = "https://docs.rapids.ai/install/"


def import_check(verbose=False, **kwargs):
    """Check that cuDF can be imported.

    On failure, use the RAPIDS install docs.
    """
    try:
        import cudf
    except ImportError as e:
        raise ImportError(
            "cuDF could not be imported. Install cuDF with conda or pip as "
            f"described at {_INSTALL_DOCS}"
        ) from e
    if verbose:
        return f"cuDF {cudf.__version__} is available"


def functional_check(verbose=False, **kwargs):
    """Check that a basic groupby/aggregation runs and matches expected values."""
    import cudf

    df = cudf.DataFrame({"a": [1, 2, 2], "b": [10, 20, 30]})
    res = df.groupby("a", as_index=False).agg({"b": "sum"})

    if len(res) != 2:
        raise AssertionError(
            f"Expected 2 group rows after groupby/agg, got {len(res)}"
        )

    got_a = res["a"].to_pandas().tolist()
    expected_a = [1, 2]
    if got_a != expected_a:
        raise AssertionError(
            f"groupby key column mismatch: got {got_a}, expected {expected_a}"
        )

    got_b = res["b"].to_pandas().tolist()
    expected_b = [10, 50]
    if got_b != expected_b:
        raise AssertionError(
            f"groupby sum mismatch: got {got_b}, expected {expected_b}"
        )

    if verbose:
        return "cuDF groupby/agg succeeded"


def _is_libnvvm_missing_error(exc: OSError) -> bool:
    """True for the common missing-libnvvm loader failure."""
    msg = str(exc)
    return (
        "libnvvm.so" in msg
        and "cannot open shared object file" in msg
        and "No such file or directory" in msg
    )


def functional_numba_check(verbose=False, **kwargs):
    """Exercise ``Series.apply`` (Numba path).

    Only the specific ``libnvvm.so`` missing-library ``OSError`` is rewritten
    with install guidance; other ``OSError``s propagate unchanged.
    """
    import cudf

    s = cudf.Series(["a", "aa", "b"])
    try:
        out = s.apply(lambda x: len(x))
    except OSError as e:
        if _is_libnvvm_missing_error(e):
            raise OSError(
                (
                    "cuDF Series.apply failed: libnvvm.so could not be loaded. "
                    "This is likely due to a missing CUDA toolkit or the CUDA toolkit not being on the dynamic linker path. "
                    f"Please follow the installation instructions carefully: {_INSTALL_DOCS}"
                )
            ) from e
        raise

    assert list(out.to_pandas()) == [1, 2, 1]

    if verbose:
        return "cuDF Series.apply (Numba path) succeeded"
