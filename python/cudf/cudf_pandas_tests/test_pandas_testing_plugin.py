# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("fail_first", [False, True])
def test_plotting_options_do_not_leak_between_tests(
    pytester, accelerated, fail_first
):
    pytest.importorskip("matplotlib")
    pytester.makeconftest(
        """
        from importlib import import_module

        import matplotlib
        import matplotlib.pyplot as plt
        import pytest

        restore_plotting_options = import_module(
            "cudf.pandas.scripts.pandas-testing-plugin"
        ).restore_plotting_options

        @pytest.fixture(autouse=True)
        def mpl_cleanup():
            with matplotlib.rc_context():
                matplotlib.use("template")
                yield
            plt.close("all")
        """
    )
    pytester.makepyfile(
        f"""
        import pandas as pd

        def test_change_plotting_options():
            pd.plotting.plot_params["x_compat"] = True
            pd.plotting.plot_params["custom"] = "changed"
            assert {not fail_first!r}

        def test_datetime_plot_uses_default_options():
            assert pd.plotting.plot_params["x_compat"] is False
            assert "custom" not in pd.plotting.plot_params
            series = pd.Series(
                [1, 2, 3], index=pd.date_range("2020-01-01", periods=3)
            )
            ax = series.plot()
            assert ax.freq == "D"
        """
    )
    args = ["-q"]
    if accelerated:
        args.extend(["-p", "cudf.pandas"])
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=2 - fail_first, failed=int(fail_first))
