# Build JNI against an installed nightly libcudf

Use this path for Java/JNI-only changes when rebuilding libcudf from source is unnecessary. It validates against a compatible nightly package; it does not reproduce a release build.

## Select and install the nightly

Identify the libcudf commit required by the branch and the package series from `VERSION`. For a pull request, use its base commit or merge base unless the changes explicitly require a newer libcudf commit:

```bash
REQUIRED_COMMIT="$(git merge-base HEAD upstream/main)"
LIBCUDF_SERIES="$(cut -d. -f1,2 "$(git rev-parse --show-toplevel)/VERSION")"
```

Inspect available nightly versions and build strings for that series:

```bash
conda search --override-channels \
  -c rapidsai-nightly -c conda-forge -c nvidia \
  "libcudf=${LIBCUDF_SERIES}.*"
```

The build string ends in the first eight characters of the source commit, as defined by `conda/recipes/libcudf/recipe.yaml`. Prefer a build ending in the first eight characters of `REQUIRED_COMMIT`; if it is unavailable, use `git merge-base --is-ancestor <PACKAGE_COMMIT> "$REQUIRED_COMMIT"` to select the nearest available ancestor. Match the CUDA major version used by the devcontainer. Use an existing conda environment, or create one before installing the package.

Let conda solve libcudf's declared dependencies instead of pinning its transitive libraries manually. Preview the solution, then install the same exact libcudf build:

```bash
conda install --dry-run -y --override-channels \
  -c rapidsai-nightly -c conda-forge -c nvidia \
  'libcudf=<EXACT_VERSION>=<EXACT_BUILD>' \
  'cuda-version=<CUDA_MAJOR>.*'

conda install -y --override-channels \
  -c rapidsai-nightly -c conda-forge -c nvidia \
  'libcudf=<EXACT_VERSION>=<EXACT_BUILD>' \
  'cuda-version=<CUDA_MAJOR>.*'
```

## Configure and build JNI

Activate the environment and point CMake and the runtime linker at its installed libcudf:

```bash
conda activate <ENVIRONMENT>

unset CUDF_ROOT CUDF_CPP_BUILD_DIR
export CUDF_INSTALL_DIR="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CUDF_INSTALL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

nvcc --version
```

Installed nightly packages are shared-library builds, so configure JNI accordingly:

```bash
export MVN_COMMON_OPTS="${MVN_COMMON_OPTS:-} \
  -DCUDF_JNI_LIBCUDF_STATIC=OFF \
  -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=OFF \
  -DUSE_GDS=OFF"
```

Remove cached JNI artifacts when switching libcudf providers, then rebuild only Java and JNI:

```bash
cd "$(git rev-parse --show-toplevel)/java"
rm -rf target/cmake-build target/classes
mvn install $MVN_COMMON_OPTS -DskipTests
```

## Verify symbol resolution

Confirm that CMake selected the installed package and that every JNI dependency resolves:

```bash
grep -E 'CUDF_INSTALL_DIR|cudf_DIR|CUDF_USE_PER_THREAD_DEFAULT_STREAM|CUDF_JNI_LIBCUDF_STATIC|USE_GDS' target/cmake-build/CMakeCache.txt
ldd -r target/cmake-build/libcudfjni.so
```

The POM packages the libcudf resolved by `ldd`, so verify that it comes from `CUDF_INSTALL_DIR`. Its transitive shared-library dependencies are not bundled; keep `$CUDF_INSTALL_DIR/lib` on `LD_LIBRARY_PATH` while running tests. If resolution points at another environment, clean `target/cmake-build` and rebuild with the intended conda environment active.

Continue with [Running Java tests](../SKILL.md#running-java-tests).
