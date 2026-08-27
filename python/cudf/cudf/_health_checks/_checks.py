#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""cuDF health checks for rapids doctor. Docs https://github.com/rapidsai/rapids-cli"""

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
    import pandas as pd

    import cudf

    data = {"a": [1, 2, 2], "b": [10, 20, 30]}
    res_pd = pd.DataFrame(data).groupby("a", as_index=False).agg({"b": "sum"})
    res_cu = (
        cudf.DataFrame(data).groupby("a", as_index=False).agg({"b": "sum"})
    )

    got = res_cu.to_pandas()

    # cuDF groupby does not guarantee the same row order as pandas; align for
    # comparison by sorting on all result columns.
    got = got.sort_values(by=list(got.columns)).reset_index(drop=True)
    expected = res_pd.sort_values(by=list(res_pd.columns)).reset_index(
        drop=True
    )

    if not got.equals(expected):
        raise AssertionError(
            "cuDF groupby/agg result does not match pandas after aligning order:\n"
            f"cuDF (as pandas):\n{got}\n\npandas:\n{expected}"
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


def _is_nvjitlink_version_mismatch_error(exc: ValueError) -> bool:
    """True for the common incompatible-nvJitLink failure."""
    cause = exc.__cause__
    if cause is None:
        return False

    msg = str(cause)
    return (
        "nvJitLink error log:" in msg
        and "may need newer version of nvJitLink library" in msg
    )


def functional_numba_check(verbose=False, **kwargs):
    """Exercise ``Series.apply`` (Numba path).

    Specific CUDA library failures are rewritten with installation guidance;
    other exceptions propagate unchanged.
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
    except ValueError as e:
        if _is_nvjitlink_version_mismatch_error(e):
            raise ValueError(
                (
                    "cuDF Series.apply failed because nvJitLink is incompatible "
                    "with this cuDF installation. Ensure that "
                    "the installed nvidia-nvjitlink-cu* package is compatible "
                    "with the CUDA toolkit version required by cuDF; another package may be pinning an "
                    "older version. Please follow the installation instructions "
                    f"carefully: {_INSTALL_DOCS}"
                )
            ) from e
        raise

    assert list(out.to_pandas()) == [1, 2, 1]

    if verbose:
        return "cuDF Series.apply (Numba path) succeeded"
