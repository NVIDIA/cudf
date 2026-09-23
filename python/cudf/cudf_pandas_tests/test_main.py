# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tempfile
import textwrap


def _run_python(*, cudf_pandas, command):
    executable = "python "
    if cudf_pandas:
        executable += "-m cudf.pandas "
    return subprocess.check_output(
        executable + command,
        shell=True,
        text=True,
        encoding="utf-8",
    )


def test_run_cudf_pandas_with_script():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
        code = textwrap.dedent(
            """
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3]})
            print(df['a'].sum())
            """
        )
        f.write(code)
        f.flush()

        res = _run_python(cudf_pandas=True, command=f.name)
        expect = _run_python(cudf_pandas=False, command=f.name)

    assert res != ""
    assert res == expect


def test_run_cudf_pandas_line_profile_preserves_file(tmp_path):
    # GH #23010: ``--line-profile`` runs an instrumented copy of the script
    # from a temp file. ``__file__`` must still refer to the original script so
    # scripts that locate sibling files relative to ``__file__`` keep working.
    (tmp_path / "sibling.txt").write_text("hello-sibling")
    script = tmp_path / "script.py"
    script.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            sibling = Path(__file__).resolve().parent / "sibling.txt"
            print("FILE:", __file__)
            print("CONTENT:", sibling.read_text())
            """
        )
    )

    res = _run_python(cudf_pandas=True, command=f"--line-profile {script}")
    assert "CONTENT: hello-sibling" in res
    assert str(script) in res


def test_run_cudf_pandas_with_script_with_cmd_args():
    input_args_and_code = """-c 'import pandas as pd; df = pd.DataFrame({"a": [1, 2, 3]}); print(df["a"].sum())'"""

    res = _run_python(cudf_pandas=True, command=input_args_and_code)
    expect = _run_python(cudf_pandas=False, command=input_args_and_code)

    assert res != ""
    assert res == expect


def test_run_cudf_pandas_with_script_with_cmd_args_check_cudf():
    """Verify that cudf is active with -m cudf.pandas."""
    input_args_and_code = """-c 'import pandas as pd; print(pd)'"""

    res = _run_python(cudf_pandas=True, command=input_args_and_code)
    expect = _run_python(cudf_pandas=False, command=input_args_and_code)

    assert "cudf" in res
    assert "<module 'pandas' from" in expect


def test_cudf_pandas_script_repl():
    def start_repl_process(cmd):
        return subprocess.Popen(
            cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def get_repl_output(process, commands):
        for command in commands:
            process.stdin.write(command)
            process.stdin.flush()
        return process.communicate()

    p1 = start_repl_process("python -m cudf.pandas")
    p2 = start_repl_process("python")
    commands = [
        "import pandas as pd\n",
        "print(pd.Series(range(2)).sum())\n",
        "print(pd.Series(range(5)).sum())\n",
        "import sys\n",
        "print(pd.Series(list('abcd')), file=sys.stderr)\n",
    ]

    res = get_repl_output(p1, commands)
    expect = get_repl_output(p2, commands)

    # Check stdout
    assert res[0] != ""
    assert res[0] == expect[0]

    # Check stderr
    assert res[1] != ""
    assert res[1] == expect[1]

    p1.kill()
    p2.kill()
