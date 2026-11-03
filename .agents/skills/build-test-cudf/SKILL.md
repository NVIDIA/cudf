---
name: build-test-cudf
description: Use this skill to build and test code changes inside a cudf devcontainer.
---

## Check if we are in devcontainer

Detect cudf devcontainer usage by checking if the username is `coder`. If not, ignore everything else in this file and skip.

## Building cudf

cudf packages must be built in this order as needed (each depends on the previous):

1. C++ library (libcudf)
2. Python wheel wrapper for the C++ library
3. Cython bindings (compiles `.pyx` files)
4. high-level Python packages

When rebuilding a package, ensure its dependencies are already built. Since we are in cudf devcontainer, DO NOT run the `build.sh` script or install any packages yourself. Instead, always use the `build-*`, `test-*`, `rapids-*`, `clean-*` scripts located at `/usr/bin`.

### Building libcudf

Whenever building libcudf from scratch (CMake has not been run before), run:

```bash
build-cudf-cpp -j0
```

For incremental builds, you can also just navigate to `cpp/build/latest` and run:

```bash
ninja
```

#### CMake options

Both `build-cudf-cpp` and `configure-cudf-cpp` accept CMake `-D` options directly as arguments. See `cpp/CMakeLists.txt` for a full list of available CMake options


```bash
# Default option
build-cudf-cpp -j0 -DBUILD_BENCHMARKS=ON
# Multiple options can be combined
build-cudf-cpp -j0 -DBUILD_BENCHMARKS=ON -DBUILD_TESTS=OFF
```

Similarly, configure without building:

```bash
configure-cudf-cpp -DBUILD_BENCHMARKS=ON
```

### Building python wheel wrapper

```bash
build-libcudf-python
```

### Cython bindings (pylibcudf)

```bash
build-pylibcudf-python
```

### High-level Python packages

```bash
build-cudf-python        # cudf-python
build-cudf-polars-python # cudf-polars
```

## Clean up

Similar to build instructions, we can use `clean-cudf-xxx` scripts also located at `/usr/bin`. Use `clean-cudf` to clean everything.

## Build error handling

If there are build errors, cleaning before building will usually resolve problems. If we run into a fatal CMake error while building libcudf indicating packages/version mismatch, update the environment using the following script. If the error persists, use `--force` flag with the script

```bash
rapids-make-${PYTHON_PACKAGE_MANAGER}-env
```

Make sure to run `clean-cudf` after running this for a fresh subsequent build.

## Running Google tests

Make sure that libcudf has been built before running any of these. Discover all Google test name binaries at `cpp/build/latest/gtests` and run relevant Google tests using `test-cudf-xxx` script located at `/usr/bin`.

```bash
test-cudf-cpp -j10             # all tests 10 parallel jobs
test-cudf-cpp -R <NAME>_TEST   # specific test suite
cd cpp/build/latest/gtests && <NAME>_TEST --gtest_filter="<pattern>" # Run tests matching the <pattern> from the <NAME>_TEST
```

## Running Pytests

These instructions are applicable for running tests for pylibcudf and high-level python packages. Make sure that the package has been built before running its pytests. Pylibcudf tests can be discovered at: `python/pylibcudf/tests` and subfolders. Run them using:

```bash
pytest python/pylibcudf/tests/<subfolder>/test_<name>.py # specific pylibcudf test suite
```

cudf-python pytests can be discovered at: `python/cudf/cudf/tests` and subfolders. Run them using:

```bash
test-cudf-python # run all cudf-python Pytests
pytest python/cudf/cudf/tests/<subfolder>/test_<name>.py # specific cudf-python test suite
```

Similarly, cudf-polars pytests can be discovered at: `python/cudf_polars/tests` and subfolders. Run them using:

```bash
test-cudf-polars-python
```
