# Where existing code lives

Search these directories on `$REMOTE/main`, nearest first, before adding new code.

| Directory | What is there |
|---|---|
| The touched file's own directory or package, `cpp/src/<module>/` | Private helpers of the module you are changing |
| `cpp/include/cudf/<module>/` | Public APIs |
| `cpp/include/cudf/detail/`, `cpp/include/cudf/<module>/detail/` | Internal APIs shared across modules |
| `cpp/include/cudf/utilities/`, `cpp/include/cudf/detail/utilities/`, `cpp/src/utilities/` | General utilities |
| `cpp/src/io/utilities/` | Shared I/O helpers |
| `cpp/include/cudf_test/`, `cpp/benchmarks/common/` | Test and benchmark helpers |
| `python/cudf/cudf/utils/`, `python/cudf_polars/cudf_polars/utils/` | Python helpers; pylibcudf usually already binds the libcudf API |
| CCCL: `cub/cub/`, `thrust/thrust/`, `libcudacxx/include/cuda/` | Parallel algorithms, block- and warp-level primitives, and device-side math, bit, atomic, and iterator utilities |
| cuco: `include/cuco/` | Hash tables and filters |

Any configured cudf build has the CCCL and cuco sources under `<build dir>/_deps/cccl-src` and `<build dir>/_deps/cuco-src`.

```bash
git grep -n '<operation or symbol>' $REMOTE/main -- <directory>
grep -rln '<operation or symbol>' <cccl-src>/cub/cub <cccl-src>/thrust/thrust <cccl-src>/libcudacxx/include/cuda <cuco-src>/include/cuco
```
