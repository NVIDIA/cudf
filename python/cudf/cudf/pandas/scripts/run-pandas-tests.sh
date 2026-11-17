#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Pandas unit tests with cudf.pandas.
#
# Usage:
#   run-pandas-tests.sh <pytest args> <path to pandas tests (optional)>
#
# Examples
# Run a single test
#   run-pandas-tests.sh -n auto -v tests/groupby/test_groupby_dropna.py
# Run all tests
#   run-pandas-tests.sh --tb=line --report-log=log.json
#
# This script creates a `pandas-testing` directory if it doesn't exist
#

set -euo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Grab the Pandas source corresponding to the version
# of Pandas installed.
PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)")

echo "Running Pandas tests for version ${PANDAS_VERSION}"
mkdir -p pandas-testing
cd pandas-testing

if [ ! -d "pandas" ]; then
    git clone https://github.com/pandas-dev/pandas --depth=1 -b "v${PANDAS_VERSION}" pandas
fi

if [ ! -d "pandas-tests" ]; then
    # Copy just the tests out of the Pandas source tree.
    # Not exactly sure why this is needed but Pandas
    # imports fail if we don't do this:
    mkdir -p pandas-tests
    cp -r pandas/pandas/tests pandas-tests/
    # Some tests navigate up from tests/ and reference data files in sibling
    # directories (e.g. tests/io/formats/style/test_html.py looks for
    # io/formats/templates/ five levels above the test file).  Copy those
    # non-test data directories so that the expected relative paths resolve.
    mkdir -p pandas-tests/io/formats
    cp -r pandas/pandas/io/formats/templates pandas-tests/io/formats/templates
    # directory layout requirement
    # conftest.py
    # pyproject.toml
    # tests/
    cp pandas/pandas/conftest.py pandas-tests/conftest.py
    # Vendored from pandas/pyproject.toml
    cat > pandas-tests/pyproject.toml << \EOF
[tool.pytest.ini_options]
xfail_strict = true
markers = [
  "single_cpu: tests that should run on a single cpu only",
  "slow: mark a test as slow",
  "network: mark a test as network",
  "db: tests requiring a database (mysql or postgres)",
  "clipboard: mark a pd.read_clipboard test",
  "arm_slow: mark a test as slow for arm64 architecture",
  "skip_ubsan: Tests known to fail UBSAN check",
  "fails_arm_wheels: Tests known to fail on arm64 wheels",
]
EOF

    # Substitute `pandas.tests` with a relative import.
    # This will depend on the location of the test module relative to
    # the pandas-tests directory.
    for hit in $(find pandas-tests -iname '*.py' -print0 | xargs -0 grep "pandas.tests" | cut -d ":" -f 1 | sort | uniq); do
        # Get the relative path to the test module
        test_module=$(echo "$hit" | cut -d "/" -f 2-)
        # Count directory separators to find how many levels to go up.
        # The sed pattern replaces "pandas.tests." (including the trailing dot)
        # with N dots, where N equals the number of path components in test_module.
        # E.g. tests/test_col.py (1 slash) -> 1 dot, tests/frame/f.py (2 slashes) -> 2 dots.
        num_dirs="${test_module//[^\/]/}"
        num_dirs="${#num_dirs}"
        # Build exactly num_dirs dots (0 dots is valid for num_dirs=0).
        relative_import=""
        for _i in $(seq 1 $num_dirs); do
            relative_import="${relative_import}."
        done
        # Replace "pandas.tests." (including trailing dot) so the replacement
        # dots are not combined with an extra dot left by the old pattern.
        sed -i "s/pandas\.tests\./${relative_import}/g" "$hit"
    done
fi

# Run the tests
cd pandas-tests/


PANDAS_CI="1" python -m pytest \
    -p cudf.pandas \
    -p cudf.pandas.scripts.pandas-testing-plugin \
    --import-mode=importlib \
    -m "not slow and not single_cpu and not db and not network" \
    --disable-warnings \
    "$@"

mv ./*.json ..
cd ..
echo "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
