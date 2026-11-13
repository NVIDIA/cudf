# Where existing code lives

Look in this order: the same directory or package and its private helpers (`cpp/src/<module>/`, `utilities` and `common` folders, sibling Python modules, `conftest.py`), then libcudf's shared headers, then CCCL.

| New code does | Look first in |
|---|---|
| Reduce, scan, select or partition, histogram, sort, merge, search, run-length encode, top-k, segmented versions, batched copies | CUB `cub/device/device_*.cuh`, then Thrust `thrust/*.h` |
| Block- or warp-level work inside a kernel, shared-memory staging, async copies, raw intrinsics or inline PTX | CUB `cub/block/`, `cub/warp/`; libcu++ `cuda/warp`, `cuda/barrier` (`cuda::memcpy_async`); cudf `cpp/src/io/utilities/block_utils.cuh` |
| Integer math, rounding, alignment, overflow, bit tricks, popcount | libcu++ `cuda/cmath`, `cuda/numeric`, `cuda/memory`, `cuda/std/bit`; cudf `detail/utilities/integer_utils.hpp`, `detail/integral_math.cuh`, `utilities/bit.hpp` |
| Atomics, iterators, functors, execution policies | libcu++ `cuda/atomic`, `cuda/iterator`, `cuda/functional`; cudf `detail/utilities/device_atomics.cuh`, `detail/iterator.cuh`, `detail/indexalator.cuh`, `detail/offsets_iterator.cuh`, `detail/utilities/device_operators.cuh`, `detail/utilities/execution_env.hpp` |
| Counting, building, or copying null masks | cudf `detail/null_mask.hpp` and `.cuh`, `detail/valid_if.cuh` |
| Row comparison, row hashing, type dispatch | cudf `detail/row_operator/`, `hashing/detail/`, `utilities/type_dispatcher.hpp` |
| Group or run boundaries of sorted keys | cudf `cpp/src/groupby/common/groupby_helper_group_offsets.cuh` |
| Hash tables, filters | cuco `cuco/static_{set,map,multiset,multimap}.cuh`, `cuco/bloom_filter.cuh`, `cuco/hash_functions.cuh`; cudf `detail/cuco_helpers.hpp` |
| Gather, scatter, conditional copy, sizes to offsets | cudf `detail/algorithms/`, `detail/gather.cuh`, `scatter.cuh`, `copy_if.cuh`, `copy_if_else.cuh`, `copy_range.cuh`, `sizes_to_offsets_iterator.cuh` |
| Buffers, host-device copies, grid sizing | cudf `detail/utilities/vector_factories.hpp`, `buffer_factories.hpp`, `cuda_memcpy.hpp`, `batched_memcpy.hpp`, `batched_memset.hpp`, `grid_1d.cuh` |
| Strings, lists, text parsing | cudf `strings/detail/`, `lists/detail/`, `cpp/src/io/utilities/` |
| Test and benchmark helpers | `cudf_test/type_lists.hpp`, `column_wrapper.hpp`, `iterator_utilities.hpp`, `random.hpp`, `memory_resource_utilities.hpp`; `cpp/benchmarks/common/generate_input.hpp`, `random_distribution_factory.cuh`, `nvbench_utilities.hpp` |
| Python helpers | The touched file's own package first; then `python/cudf/cudf/utils/`, `python/cudf/cudf/core/_internals/`, `python/cudf_polars/cudf_polars/utils/`; pylibcudf usually already binds the libcudf API |

CCCL and cuco sources: any configured cudf build has them under `<build dir>/_deps/cccl-src` and `<build dir>/_deps/cuco-src`. Otherwise clone NVIDIA/cccl and NVIDIA/cuCollections shallowly, or use the docs at https://nvidia.github.io/cccl/.

```bash
grep -rln '<operation or symbol>' <cccl-src>/cub/cub <cccl-src>/thrust/thrust <cccl-src>/libcudacxx/include/cuda <cuco-src>/include/cuco
```
