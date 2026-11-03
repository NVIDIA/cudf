# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Usage:

python -m cudf.pandas <script.py> <args>
python -m cudf.pandas -m module <args>
"""

import argparse
import code
import runpy
import sys
import tempfile
import types
from contextlib import contextmanager

from . import install
from .profiler import Profiler, lines_with_profiling


def _run_instrumented_as_main(code_path, file_path):
    """Execute the script at ``code_path`` as ``__main__``, exposing
    ``file_path`` as ``__file__``.

    The line profiler runs an *instrumented copy* of the user's script written
    to a temporary file. Compiling with ``code_path`` as the code object's
    filename keeps per-line profiling output and tracebacks pointed at that
    instrumented source, while ``__file__`` (and ``sys.argv[0]``, set by the
    caller) continue to refer to the original script. This lets scripts that
    locate sibling files relative to ``__file__`` work under ``--line-profile``
    just as they do without it (GH #23010).
    """
    with open(code_path) as f:
        code_obj = compile(f.read(), code_path, "exec")
    main_module = types.ModuleType("__main__")
    main_module.__file__ = file_path
    main_module.__loader__ = None
    main_module.__spec__ = None
    saved_main = sys.modules["__main__"]
    sys.modules["__main__"] = main_module
    try:
        exec(code_obj, main_module.__dict__)
    finally:
        sys.modules["__main__"] = saved_main


@contextmanager
def profile(function_profile, line_profile, fn):
    if fn is None and (line_profile or function_profile):
        raise RuntimeError("Enabling the profiler requires a script name.")
    if line_profile:
        with open(fn) as f:
            lines = f.readlines()

        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".py") as f:
            f.write(lines_with_profiling(lines, function_profile).encode())
            f.seek(0)

            yield f.name
    elif function_profile:
        with Profiler() as profiler:
            yield fn
        profiler.print_per_function_stats()
    else:
        yield fn


def main():
    parser = argparse.ArgumentParser(
        prog="python -m cudf.pandas",
        description=(
            "Run a Python script with Pandas Accelerator Mode enabled. "
            "In Pandas Accelerator Mode, all imports of pandas will "
            "automatically use GPU accelerated cuDF equivalents where "
            "possible."
        ),
    )

    parser.add_argument(
        "-m",
        dest="module",
        nargs=1,
    )
    parser.add_argument(
        "-c",
        dest="cmd",
        nargs=1,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Perform per-function profiling of this script.",
    )
    parser.add_argument(
        "--line-profile",
        action="store_true",
        help="Perform per-line profiling of this script.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass on to the script",
    )

    args = parser.parse_args()

    if args.cmd:
        f = tempfile.NamedTemporaryFile(mode="w+b", suffix=".py")
        f.write(args.cmd[0].encode())
        f.seek(0)
        args.args.insert(0, f.name)

    install()

    script_name = args.args[0] if len(args.args) > 0 else None
    with profile(args.profile, args.line_profile, script_name) as fn:
        if script_name is not None:
            args.args[0] = fn
        if args.module:
            (module,) = args.module
            # run the module passing the remaining arguments
            # as if it were run with python -m <module> <args>
            sys.argv[:] = [module, *args.args]  # not thread safe?
            runpy.run_module(module, run_name="__main__")
        elif len(args.args) >= 1:
            # Remove ourself from argv and continue
            sys.argv[:] = args.args
            if args.line_profile and script_name is not None:
                # ``fn`` (args.args[0]) is an instrumented copy of the script in
                # a temporary file. Run it so that ``__file__`` and
                # ``sys.argv[0]`` still refer to the original script.
                sys.argv[0] = script_name
                _run_instrumented_as_main(args.args[0], script_name)
            else:
                runpy.run_path(args.args[0], run_name="__main__")
        else:
            if sys.stdin.isatty():
                banner = f"Python {sys.version} on {sys.platform}"
                site_import = not sys.flags.no_site
                if site_import:
                    cprt = 'Type "help", "copyright", "credits" or "license" for more information.'
                    banner += "\n" + cprt
            else:
                # Don't show prompts or banners if stdin is not a TTY
                sys.ps1 = ""
                sys.ps2 = ""
                banner = ""

            # Launch an interactive interpreter
            code.interact(banner=banner, exitmsg="")


if __name__ == "__main__":
    main()
