# cudf-polars-cucascade

An optional extension that gives cudf-polars a host-memory cache for remote parquet reads, built
on [cuCascade](https://github.com/NVIDIA/cuCascade)'s REST (S3) backend and prefetching cache. It
is used when the `cucascade_cache_bytes` executor option is set (env
`CUDF_POLARS__EXECUTOR__CUCASCADE_CACHE_BYTES`); with the option unset, cudf-polars never imports
it.

## Why it is built from source

The extension contains a `cudf::io::datasource` subclass and Cython that subclasses pylibcudf's
`Datasource`. Both depend on the exact layout of libcudf's and pylibcudf's types, so they must be
compiled against the libcudf and pylibcudf that will run them: an extension built against one
libcudf can crash with another. For that reason it is built here, from this repository, in the
environment where cudf-polars runs, and no package of it is published.

cuCascade is fetched at a pinned commit (`cmake/get_cucascade.cmake`) and compiled into the
extension as static libraries. Only its io core is built, which has no libcudf dependency. Nothing
of cuCascade is installed, and there is no libcucascade package to install or to keep in step with
libcudf.

## Building

You need an environment that has cudf-polars's own build requirements (libcudf, pylibcudf, a CUDA
toolkit, Cython, scikit-build-core, cython-cmake) and what cuCascade's io core needs: liburing,
libcurl, OpenSSL and libnuma (and their headers), and network access to GitHub at configure time
(for cuCascade, rapids-cmake and the header-only `concurrentqueue`).

From the root of the repository:

```bash
pip install --no-build-isolation --no-deps ./python/cudf_polars/cucascade
```

The build uses the libcudf that `find_package(cudf)` finds, which should be the one installed in
the environment. To build against a local checkout of cuCascade, for example while changing it:

```bash
CMAKE_ARGS="-DCPM_cuCascade_SOURCE=/path/to/cuCascade" \
  pip install --no-build-isolation --no-deps ./python/cudf_polars/cucascade
```

Rebuild it whenever libcudf or pylibcudf in the environment changes.

## Using it

```bash
export CUDF_POLARS__EXECUTOR__CUCASCADE_CACHE_BYTES=$((100 * 1024**3))   # 100 GiB, pinned at engine start
```

If the option is set and the extension is not installed, cudf-polars raises an error that says how
to build it.

## Testing

```bash
pytest python/cudf_polars/cucascade/tests
```

The tests need a GPU (the cache is pinned host memory) and no network.

## Layout

```
cucascade/
├── CMakeLists.txt, pyproject.toml     build with scikit-build-core, like pylibcudf (abi3)
├── cmake/get_cucascade.cmake          fetches and builds cuCascade at a pinned commit
├── cudf_polars_cucascade/
│   ├── cpp/                           cucascade_datasource (a cudf::io::datasource) and scan_cache
│   ├── libcucascade/                  Cython declarations of the C++ classes, like pylibcudf/libcudf
│   ├── scan_cache.pyx, .pxd, .pyi     ScanCache and ScanCacheConfig
│   ├── advising_datasource.*          AdvisingDatasource, a pylibcudf Datasource
│   ├── column_admission.*             ColumnStats
│   ├── build_info.*                   the libcudf version the extension was built against
│   └── exception_handler.pxd          maps cuCascade's exceptions to Python's
└── tests/
```

Unlike `pylibcudf`, which links an installed libcudf, this package compiles its (small) C++ in
the same build: it is not a published package, so there is no separate libcucascade to link.

The wrapper holds no logic. What to cache, how reads are advised to the cache, and the cache itself
are in cuCascade, with its own tests. This package gives that to pylibcudf's readers as a
`Datasource`.
