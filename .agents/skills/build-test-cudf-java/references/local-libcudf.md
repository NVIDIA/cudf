# Build JNI against a local libcudf build

Use this path when the Java/JNI changes require a libcudf source build.

## Verify Java-required libcudf build flags

Read `java/README.md` (section "Build From Source") for the current required libcudf flags and verify every one in `CMakeCache.txt`.

Set the build directory to the libcudf build you intend to use. The repository default is shown here:

```bash
export CUDF_ROOT="$(git rev-parse --show-toplevel)"
export CUDF_CPP_BUILD_DIR="${CUDF_CPP_BUILD_DIR:-$CUDF_ROOT/cpp/build}"
```

Also inspect the CUDA architecture, per-thread default stream setting, and shared/static library setting; these determine the matching JNI configuration:

```bash
grep -E \
  'CMAKE_CUDA_ARCHITECTURES|CUDF_USE_PER_THREAD_DEFAULT_STREAM|BUILD_SHARED_LIBS' \
  "$CUDF_CPP_BUILD_DIR/CMakeCache.txt"
```

If any required flag is missing, rebuild libcudf with the [`build-test-cudf` skill](../../build-test-cudf/SKILL.md), passing the current flags from `java/README.md`. Ignore flags that CMake reports as unused.

## Configure and build JNI

Point the Java build at the libcudf source and build directories, not an installed package. JNI CMake prefers `CUDF_INSTALL_DIR` when it is set, so unset it to prevent an installed libcudf from overriding `CUDF_CPP_BUILD_DIR`:

```bash
unset CUDF_INSTALL_DIR
```

Match these JNI settings to the libcudf build:

- Set `CUDF_USE_PER_THREAD_DEFAULT_STREAM=ON` only when libcudf was built with `CUDF_USE_PER_THREAD_DEFAULT_STREAM=ON`.
- Set `CUDF_JNI_LIBCUDF_STATIC=ON` for a static libcudf build and `OFF` for a shared build.
- Keep `USE_GDS=OFF` unless the test environment supports GDS and GDS tests are required.

For example, a static PTDS build like the Java CI configuration uses:

```bash
export MVN_COMMON_OPTS="${MVN_COMMON_OPTS:-} \
  -DCUDF_CPP_BUILD_DIR=$CUDF_CPP_BUILD_DIR \
  -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON \
  -DCUDF_JNI_LIBCUDF_STATIC=ON \
  -DUSE_GDS=OFF"
```

Add `-DCMAKE_CUDA_ARCHITECTURES=<architecture>` matching the value recorded from libcudf's CMakeCache.txt; if omitted, the pom defaults to RAPIDS (all architectures), which is slower.

When switching libcudf providers or changing JNI CMake configuration, remove the cached JNI artifacts before building:

```bash
cd "$CUDF_ROOT/java"
rm -rf target/cmake-build target/classes
mvn install $MVN_COMMON_OPTS -DskipTests
```

For incremental source/JNI rebuilds with the same provider and configuration, preserve the native build cache and run only:

```bash
mvn install $MVN_COMMON_OPTS -DskipTests
```

Do not remove the entire `target/` directory unless a full clean build is necessary. Preserving `target/cmake-build` avoids repeating the slow native dependency build.

Continue with [Running Java tests](../SKILL.md#running-java-tests).
