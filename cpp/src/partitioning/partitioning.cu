/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fixed_width.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/utilities/alignment.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/thread/thread_load.cuh>
#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/type_traits>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {
namespace {
namespace cg = cooperative_groups;

// A 1,024-thread CTA provides 32 warp-sized groups for flushing partitions concurrently.
constexpr size_type PREFERRED_STAGED_SCATTER_BLOCK_SIZE = 1024;
// A 512-thread CTA halves the per-iteration shared-memory footprint when 1,024 threads cannot fit.
constexpr size_type FALLBACK_STAGED_SCATTER_BLOCK_SIZE = 512;
// Map-only CTAs process eight rows per thread because they stage no fixed-width payload values.
constexpr size_type GATHER_MAP_ROWS_PER_THREAD = 8;

/** @brief Fraction of L2 capacity available to one concurrently active fixed-width column batch. */
constexpr double L2_CACHE_USAGE_FRACTION = 1.0;

/** @brief Number of bytes in an L2 cache sector. */
constexpr std::uint64_t L2_SECTOR_BYTES = 32;

// Scatter launch configuration
constexpr size_type SCATTER_BLOCK_SIZE      = 256;
constexpr size_type SCATTER_ROWS_PER_THREAD = 1;

/**
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses the modulo operation.
 */
template <typename hash_value_t>
class modulo_partitioner {
 public:
  modulo_partitioner(size_type num_partitions) : divisor{num_partitions} {}

  __device__ size_type operator()(hash_value_t hash_value) const { return hash_value % divisor; }

 private:
  size_type const divisor;
};

template <typename T>
bool is_power_two(T number)
{
  return (0 == (number & (number - 1)));
}

/**
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses a bitwise mask. Only works when num_partitions is a power of 2.
 *
 * For n % d, if d is a power of two, then it can be computed more efficiently
 * via a single bitwise AND as: n & (d - 1)
 */
template <typename hash_value_t>
class bitwise_partitioner {
 public:
  bitwise_partitioner(size_type num_partitions) : mask{(num_partitions - 1)}
  {
    assert(is_power_two(num_partitions));
  }

  __device__ size_type operator()(hash_value_t hash_value) const
  {
    return hash_value & mask;  // hash_value & (num_partitions - 1)
  }

 private:
  size_type const mask;
};

/**
 * @brief Maps a type identifier to its fixed-width type, or to `void` for unsupported types.
 *
 * @tparam Id Type identifier to map
 */
template <cudf::type_id Id>
struct dispatch_fixed_width_type {
  using dispatched_type = id_to_type<Id>;  ///< Type represented by `Id`

  /// Fixed-width dispatched type, or `void` when `dispatched_type` is not fixed-width
  using type =
    cuda::std::conditional_t<cudf::is_fixed_width<dispatched_type>(), dispatched_type, void>;
};

/**
 * @brief Hashes rows by directly hashing and combining their fixed-width elements.
 *
 * @tparam Hash Element hash functor template
 * @tparam HasNulls Whether any key column may contain nulls
 */
template <template <typename> class Hash, bool HasNulls>
class fixed_width_row_hasher {
 public:
  /**
   * @brief Constructs a row hasher for fixed-width key columns.
   *
   * @param keys Device view of the key columns
   * @param seed Initial hash seed applied to every element hash
   */
  fixed_width_row_hasher(table_device_view keys, uint32_t seed) : _keys{keys}, _seed{seed} {}

  /**
   * @brief Hashes one row while preserving generic row-hash column ordering and combination.
   *
   * @param row Row index to hash
   * @return Combined hash value for the row
   */
  __device__ hash_value_type operator()(size_type row) const
  {
    auto hash = hash_element(_keys.column(0), row);
    for (size_type column_index = 1; column_index < _keys.num_columns(); ++column_index) {
      hash =
        cudf::hashing::detail::hash_combine(hash, hash_element(_keys.column(column_index), row));
    }
    return hash;
  }

 private:
  /// Null-checking policy selected at compile time
  using nullate_type = cuda::std::conditional_t<HasNulls, nullate::YES, nullate::NO>;

  /**
   * @brief Hashes one key element using the shared row element hasher.
   *
   * @param column Fixed-width key column containing the element
   * @param row Row index of the element
   * @return Hash value for the element
   */
  __device__ hash_value_type hash_element(column_device_view const& column, size_type row) const
  {
    auto const hasher =
      detail::row::hash::element_hasher<Hash, nullate_type>{nullate_type{}, _seed};
    return cudf::type_dispatcher<dispatch_fixed_width_type>(column.type(), hasher, column, row);
  }

  table_device_view _keys;  ///< Device view of the fixed-width key columns
  uint32_t _seed;           ///< Initial seed applied to each element hash
};

/**
 * @brief Hashes rows and records their partition and CTA-local offset metadata.
 *
 * Each CTA also writes its partition histogram in partition-major order. A single exclusive scan
 * of these counts later produces both the CTA output offsets and the global partition starts.
 *
 * @param[in] the_hasher Hasher whose rows will be partitioned
 * @param[in] num_partitions The number of partitions to divide the rows into
 * @param[in] the_partitioner Functor that maps a row hash to a partition number
 * @param[out] partition_metadata Partition number and CTA-local offset for each row
 * @param[out] block_partition_sizes Partition sizes for each CTA in partition-major order
 */
template <class row_hasher_t, typename partitioner_type, typename PartitionMetadataView>
CUDF_KERNEL void compute_row_partition_numbers(row_hasher_t the_hasher,
                                               size_type const num_partitions,
                                               partitioner_type const the_partitioner,
                                               PartitionMetadataView const __grid_constant__
                                                 partition_metadata,
                                               size_type* __restrict__ block_partition_sizes)
{
  // Accumulate histogram of the size of each partition in shared memory
  extern __shared__ size_type shared_partition_sizes[];

  auto tid          = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  // Initialize local histogram
  thread_index_type partition_number = threadIdx.x;
  while (partition_number < num_partitions) {
    shared_partition_sizes[partition_number] = 0;
    partition_number += blockDim.x;
  }

  __syncthreads();

  // Compute the hash value for each row, store it to the array of hash values
  // and compute the partition to which the hash value belongs and increment
  // the shared memory counter for that partition
  while (tid < partition_metadata.size()) {
    auto const row_number                = static_cast<size_type>(tid);
    hash_value_type const row_hash_value = the_hasher(row_number);

    size_type const partition_number = the_partitioner(row_hash_value);

    auto const partition_offset =
      atomicAdd(&(shared_partition_sizes[partition_number]), size_type(1));
    partition_metadata.store(row_number, partition_number, partition_offset);

    tid += stride;
  }

  __syncthreads();

  // Flush shared memory histogram to global memory
  partition_number = threadIdx.x;
  while (partition_number < num_partitions) {
    size_type const block_partition_size = shared_partition_sizes[partition_number];

    // Record the size of this partition in this block
    size_type const write_location        = partition_number * gridDim.x + blockIdx.x;
    block_partition_sizes[write_location] = block_partition_size;
    partition_number += blockDim.x;
  }
}

/**
 * @brief  Given an array of partition numbers, computes the final output
 location for each element in the output such that all rows with the same
 partition are contiguous in memory.
 *
 * @param partition_metadata Partition metadata for each row
 * @param[out] row_output_locations The final output location for each row
 * @param num_partitions The number of partitions
 * @param[out] block_partition_offsets Array that holds the offset of each
 partition for each thread block,
 * i.e., { {block0 partition0 offset, block1 partition0 offset, ...},
         {block0 partition1 offset, block1 partition1 offset, ...},
         ...
         {block0 partition(num_partitions-1) offset, block1
 partition(num_partitions -1) offset, ...} }
 */
template <typename PartitionMetadataView>
CUDF_KERNEL void compute_row_output_locations(PartitionMetadataView partition_metadata,
                                              size_type* __restrict__ row_output_locations,
                                              size_type const num_partitions,
                                              size_type const* __restrict__ block_partition_offsets)
{
  // Shared array that holds the offset of this blocks partitions in
  // global memory
  extern __shared__ size_type shared_partition_offsets[];

  // Initialize array of this blocks offsets from global array
  thread_index_type partition_number = threadIdx.x;
  while (partition_number < num_partitions) {
    shared_partition_offsets[partition_number] =
      block_partition_offsets[partition_number * gridDim.x + blockIdx.x];
    partition_number += blockDim.x;
  }
  __syncthreads();

  auto tid          = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  // Get each row's partition number and output location by incrementing the
  // block's offset counter for that partition number.
  while (tid < partition_metadata.size()) {
    auto const row_number = static_cast<size_type>(tid);
    // Get partition number of this row
    size_type const partition_number = partition_metadata.partition(row_number);

    // Get output location based on partition number by incrementing the
    // corresponding partition offset for this block
    size_type const row_output_location =
      atomicAdd(&(shared_partition_offsets[partition_number]), size_type(1));

    row_output_locations[row_number] = row_output_location;

    tid += stride;
  }
}

/**
 * @brief Describes one fixed-width column copied by the staged-scatter kernel.
 */
struct fixed_width_column_descriptor {
  std::uint8_t const* input;  ///< Start of the sliced input column data
  std::uint8_t* output;       ///< Start of the output column data
  size_type element_width;    ///< Element width in bytes
};

/**
 * @brief Describes consecutive fixed-width columns copied by one staged-scatter launch.
 */
struct fixed_width_copy_batch {
  size_type first_column;       ///< Index of the first column descriptor
  size_type num_columns;        ///< Number of column descriptors
  size_type max_element_width;  ///< Widest element in the batch
};

/**
 * @brief Groups payload columns by the operation used to materialize them.
 */
struct partition_column_groups {
  std::vector<size_type> fixed_width_indices;     ///< Columns handled by the staged-scatter kernel
  std::vector<size_type> variable_width_indices;  ///< Columns materialized with gather
  size_type max_element_width{};                  ///< Widest staged-scatter element in bytes
};

/**
 * @brief Groups payload columns for copying fixed-width columns together or gathering.
 *
 * @param input Input table whose columns will be materialized
 * @return Fixed-width and variable-width column indices and the maximum element width
 */
partition_column_groups group_columns(table_view const& input)
{
  partition_column_groups groups;
  for (size_type index = 0; index < input.num_columns(); ++index) {
    auto const& column = input.column(index);

    // The staged kernel dispatches by physical storage width. Accept the power-of-two widths
    // up to its largest supported word, uint4.
    if (cudf::is_fixed_width(column.type())) {
      auto const width = cudf::size_of(column.type());
      if (std::has_single_bit(width) && width <= sizeof(uint4)) {
        groups.fixed_width_indices.push_back(index);
        groups.max_element_width =
          std::max(groups.max_element_width, static_cast<size_type>(width));
        continue;
      }
    }

    // Materialize variable-width and unsupported fixed-width columns with gather.
    groups.variable_width_indices.push_back(index);
  }
  return groups;
}

/**
 * @brief Helper function used to decide whether staged-scatter materialization requires a gather
 * map.
 *
 * @param column_groups Fixed-width and gathered column groups
 * @param fixed_width_input Fixed-width columns handled by staged copying
 * @return true if gathered columns or fixed-width validity masks require the map
 */
bool requires_gather_map(partition_column_groups const& column_groups,
                         table_view const& fixed_width_input)
{
  return !column_groups.variable_width_indices.empty() || has_nulls(fixed_width_input);
}

/**
 * @brief Returns a row's index within its CTA-local row tile.
 *
 * @param iteration Grid-stride iteration processed by the current thread
 * @return CTA-local row index
 */
__device__ size_type local_row_index(size_type iteration)
{
  return iteration * static_cast<size_type>(blockDim.x) + static_cast<size_type>(threadIdx.x);
}

/**
 * @brief Returns the global row processed by a thread in a grid-stride iteration.
 *
 * @param iteration Grid-stride iteration processed by the current thread
 * @return Global input row index
 */
__device__ thread_index_type global_row_index(size_type iteration)
{
  return cudf::detail::grid_1d::global_thread_id() +
         static_cast<thread_index_type>(iteration) * cudf::detail::grid_1d::grid_stride();
}

/**
 * @brief Computes exclusive CTA-local offsets from one CTA's partition counts.
 *
 * Produces `local_partition_offsets[p]` as the sum of the counts for partitions preceding `p`.
 *
 * @tparam BlockSize Number of threads in the CTA
 * @param block Cooperative group representing the CTA
 * @param block_partition_sizes Partition-major counts for all CTAs
 * @param[out] local_partition_offsets Exclusive partition offsets for this CTA
 * @param num_partitions Number of partitions
 */
template <size_type BlockSize>
__device__ void scan_partition_counts(cg::thread_block const& block,
                                      size_type const* block_partition_sizes,
                                      size_type* local_partition_offsets,
                                      size_type num_partitions)
{
  using block_scan = cub::BlockScan<size_type, BlockSize>;
  __shared__ typename block_scan::TempStorage scan_storage;
  __shared__ size_type tile_prefix;

  if (block.thread_rank() == 0) {
    local_partition_offsets[0] = 0;  // Initialize start of offset array
    tile_prefix                = 0;
  }
  block.sync();

  // Process one partition per thread and carry the total across CTA-sized tiles.
  for (size_type tile = 0; tile < num_partitions; tile += BlockSize) {
    auto const partition = tile + static_cast<size_type>(block.thread_rank());
    auto const count =
      partition < num_partitions
        ? block_partition_sizes[static_cast<std::size_t>(partition) * gridDim.x + blockIdx.x]
        : size_type{0};
    auto const prefix = tile_prefix;
    size_type partition_end;
    size_type tile_total;
    block_scan(scan_storage).InclusiveSum(count, partition_end, tile_total);

    if (partition < num_partitions) {
      local_partition_offsets[partition + 1] = prefix + partition_end;
    }
    if (block.thread_rank() == 0) { tile_prefix = prefix + tile_total; }
    block.sync();
  }
}

/**
 * @brief Prepares the shared routing state used to copy partitioned values.
 *
 * @tparam BlockSize Number of threads in the CTA
 * @tparam PartitionMetadataView Device-accessible partition metadata view
 * @param block Cooperative group representing the CTA
 * @param num_rows Number of input rows
 * @param num_partitions Number of partitions
 * @param rows_per_thread Number of grid-stride iterations per thread
 * @param partition_metadata Partition and CTA-local offset for every row
 * @param block_partition_sizes Partition counts for every CTA
 * @param scanned_block_partition_sizes Exclusive prefix sum of CTA partition counts in
 * partition-major order; each entry is the global output offset for one CTA partition
 * @param[out] local_slots CTA-local destination slot for every row
 * @param[out] local_partition_offsets CTA-local partition offsets
 * @param[out] global_partition_offsets Output offset for each CTA partition
 */
template <size_type BlockSize, typename PartitionMetadataView>
__device__ void prepare_partition_copy(cg::thread_block const& block,
                                       size_type num_rows,
                                       size_type num_partitions,
                                       size_type rows_per_thread,
                                       PartitionMetadataView partition_metadata,
                                       size_type const* block_partition_sizes,
                                       size_type const* scanned_block_partition_sizes,
                                       size_type* local_slots,
                                       size_type* local_partition_offsets,
                                       size_type* global_partition_offsets)
{
  scan_partition_counts<BlockSize>(
    block, block_partition_sizes, local_partition_offsets, num_partitions);

  // Load this CTA's global output start for each partition.
  for (size_type partition = block.thread_rank(); partition < num_partitions;
       partition += block.size()) {
    global_partition_offsets[partition] =
      scanned_block_partition_sizes[static_cast<std::size_t>(partition) * gridDim.x + blockIdx.x];
  }

  // Convert each row's partition-local offset into its slot in the CTA payload buffer.
  for (size_type iteration = 0; iteration < rows_per_thread; ++iteration) {
    auto const global_row = global_row_index(iteration);
    if (global_row < static_cast<thread_index_type>(num_rows)) {
      auto const row = static_cast<size_type>(global_row);
      size_type partition;
      size_type offset;
      partition_metadata.load(row, partition, offset);
      local_slots[local_row_index(iteration)] = local_partition_offsets[partition] + offset;
    }
  }
  block.sync();
}

/**
 * @brief Copies values through a CTA-local partitioned payload buffer.
 *
 * @tparam InputIterator Random-access iterator over input values
 * @tparam OutputIterator Random-access iterator over partitioned output values
 * @tparam Word Shared-memory value type
 * @param block Cooperative group representing the CTA
 * @param input Input values in row order
 * @param output Output values in partitioned row order
 * @param payload Shared-memory payload buffer
 * @param local_slots CTA-local destination slot for every row
 * @param num_rows Number of input rows
 * @param num_partitions Number of partitions
 * @param rows_per_thread Number of grid-stride iterations per thread
 * @param local_partition_offsets CTA-local partition offsets
 * @param global_partition_offsets Output offset for each CTA partition
 */
template <typename InputIterator, typename OutputIterator, typename Word>
__device__ void copy_partitioned_values(cg::thread_block const& block,
                                        InputIterator input,
                                        OutputIterator output,
                                        Word* payload,
                                        size_type const* local_slots,
                                        size_type num_rows,
                                        size_type num_partitions,
                                        size_type rows_per_thread,
                                        size_type const* local_partition_offsets,
                                        size_type const* global_partition_offsets)
{
  // Input values are consumed once, so avoid retaining their cache lines in L2.
  for (size_type iteration = 0; iteration < rows_per_thread; ++iteration) {
    auto const global_row = global_row_index(iteration);
    if (global_row < static_cast<thread_index_type>(num_rows)) {
      auto const row                                   = static_cast<size_type>(global_row);
      payload[local_slots[local_row_index(iteration)]] = cub::ThreadLoad<cub::LOAD_CS>(input + row);
    }
  }
  block.sync();

  auto const thread_rank = static_cast<size_type>(block.thread_rank());
  auto const block_size  = static_cast<size_type>(block.size());
  for (size_type partition = thread_rank / cudf::detail::warp_size; partition < num_partitions;
       partition += block_size / cudf::detail::warp_size) {
    auto const partition_size =
      local_partition_offsets[partition + 1] - local_partition_offsets[partition];
    auto const output_offset = global_partition_offsets[partition];
    for (size_type offset = thread_rank % cudf::detail::warp_size; offset < partition_size;
         offset += cudf::detail::warp_size) {
      output[output_offset + offset] = payload[local_partition_offsets[partition] + offset];
    }
  }
  block.sync();
}

/**
 * @brief Byte offsets within the staged-scatter dynamic shared-memory buffer.
 *
 * The buffer contains the partitioned payload, one destination slot per CTA row, one local
 * offset per partition plus the terminal offset, and one global offset per partition.
 *
 * @tparam BlockSize Number of threads in the CTA
 */
template <size_type BlockSize>
struct staged_scatter_smem {
  CUDF_HOST_DEVICE constexpr staged_scatter_smem(size_type num_partitions,
                                                 size_type rows_per_thread,
                                                 size_type element_width)
  {
    auto const rows_per_block      = static_cast<std::size_t>(BlockSize) * rows_per_thread;
    local_slots_offset             = rows_per_block * element_width;
    local_partition_offsets_offset = local_slots_offset + rows_per_block * sizeof(size_type);
    global_partition_offsets_offset =
      local_partition_offsets_offset +
      (static_cast<std::size_t>(num_partitions) + 1) * sizeof(size_type);
    bytes = global_partition_offsets_offset +
            static_cast<std::size_t>(num_partitions) * sizeof(size_type);
  }

  std::size_t local_slots_offset;               ///< Byte offset of the CTA-local row slots
  std::size_t local_partition_offsets_offset;   ///< Byte offset of the CTA-local partition offsets
  std::size_t global_partition_offsets_offset;  ///< Byte offset of the global partition offsets
  std::size_t bytes;                            ///< Total dynamic shared-memory size in bytes
};

/**
 * @brief Copies the fixed-width payload columns described by `descriptors` in one kernel launch.
 *
 * Each CTA computes the partitioned output position of every row it owns. For each descriptor, it
 * stages the column values in partition order in shared memory and writes each partition as a
 * contiguous output range. The row-to-output mapping is computed once and reused across columns.
 *
 * @tparam BlockSize Number of threads in the CTA
 * @tparam PartitionMetadataView Device-accessible partition metadata view
 * @param columns Fixed-width input/output column descriptors
 * @param batches Consecutive ranges of column descriptors
 * @param num_rows Number of input rows
 * @param num_partitions Number of partitions
 * @param rows_per_thread Number of grid-stride iterations per thread
 * @param partition_metadata Partition and CTA-local offset for every row
 * @param block_partition_sizes Partition counts for every CTA
 * @param scanned_block_partition_sizes Exclusive prefix sum of CTA partition counts in
 * partition-major order; each entry is the global output offset for one CTA partition
 */
template <size_type BlockSize, typename PartitionMetadataView>
CUDF_KERNEL void copy_fixed_width_columns(fixed_width_column_descriptor const* columns,
                                          fixed_width_copy_batch const* batches,
                                          size_type num_rows,
                                          size_type num_partitions,
                                          size_type rows_per_thread,
                                          PartitionMetadataView partition_metadata,
                                          size_type const* block_partition_sizes,
                                          size_type const* scanned_block_partition_sizes)
{
  extern __shared__ __align__(16)
    std::uint8_t shared_memory[];  // align to the maximum supported element width
  auto const batch         = batches[blockIdx.y];
  auto const batch_columns = columns + batch.first_column;
  auto const smem =
    staged_scatter_smem<BlockSize>{num_partitions, rows_per_thread, batch.max_element_width};
  auto* payload     = shared_memory;
  auto* local_slots = reinterpret_cast<size_type*>(shared_memory + smem.local_slots_offset);
  auto* local_partition_offsets =
    reinterpret_cast<size_type*>(shared_memory + smem.local_partition_offsets_offset);
  auto* global_partition_offsets =
    reinterpret_cast<size_type*>(shared_memory + smem.global_partition_offsets_offset);

  auto const block = cg::this_thread_block();
  prepare_partition_copy<BlockSize>(block,
                                    num_rows,
                                    num_partitions,
                                    rows_per_thread,
                                    partition_metadata,
                                    block_partition_sizes,
                                    scanned_block_partition_sizes,
                                    local_slots,
                                    local_partition_offsets,
                                    global_partition_offsets);

  // Copy each column through the shared payload buffer, reusing the row slots and partition
  // offsets.
  for (size_type column_index = 0; column_index < batch.num_columns; ++column_index) {
    auto const descriptor = batch_columns[column_index];
    /** @brief Copies one descriptor using its physical storage-word type. */
    auto const copy_column = [&]<typename Word>() {
      copy_partitioned_values(block,
                              reinterpret_cast<Word const*>(descriptor.input),
                              reinterpret_cast<Word*>(descriptor.output),
                              reinterpret_cast<Word*>(payload),
                              local_slots,
                              num_rows,
                              num_partitions,
                              rows_per_thread,
                              local_partition_offsets,
                              global_partition_offsets);
    };

    switch (descriptor.element_width) {
      case 1: copy_column.template operator()<std::uint8_t>(); break;
      case 2: copy_column.template operator()<std::uint16_t>(); break;
      case 4: copy_column.template operator()<std::uint32_t>(); break;
      case 8: copy_column.template operator()<std::uint64_t>(); break;
      case 16: copy_column.template operator()<uint4>(); break;
      default: CUDF_UNREACHABLE("Unsupported element width in fixed-width partition copy");
    }
  }
}

/**
 * @brief Computes an output-to-input gather map with coalesced partition writes.
 *
 * @tparam BlockSize Number of threads in the CTA
 * @tparam PartitionMetadataView Device-accessible partition metadata view
 * @param num_rows Number of input rows
 * @param num_partitions Number of partitions
 * @param rows_per_thread Number of grid-stride iterations per thread
 * @param partition_metadata Partition and CTA-local offset for every row
 * @param block_partition_sizes Partition counts for every CTA
 * @param scanned_block_partition_sizes Exclusive prefix sum of CTA partition counts in
 * partition-major order; each entry is the global output offset for one CTA partition
 * @param[out] gather_map Output-to-input row mapping
 */
template <size_type BlockSize, typename PartitionMetadataView>
CUDF_KERNEL void compute_gather_map(size_type num_rows,
                                    size_type num_partitions,
                                    size_type rows_per_thread,
                                    PartitionMetadataView partition_metadata,
                                    size_type const* block_partition_sizes,
                                    size_type const* scanned_block_partition_sizes,
                                    size_type* gather_map)
{
  extern __shared__ __align__(16) std::uint8_t shared_memory[];
  auto const smem =
    staged_scatter_smem<BlockSize>{num_partitions, rows_per_thread, sizeof(size_type)};
  auto* payload     = reinterpret_cast<size_type*>(shared_memory);
  auto* local_slots = reinterpret_cast<size_type*>(shared_memory + smem.local_slots_offset);
  auto* local_partition_offsets =
    reinterpret_cast<size_type*>(shared_memory + smem.local_partition_offsets_offset);
  auto* global_partition_offsets =
    reinterpret_cast<size_type*>(shared_memory + smem.global_partition_offsets_offset);

  auto const block = cg::this_thread_block();
  prepare_partition_copy<BlockSize>(block,
                                    num_rows,
                                    num_partitions,
                                    rows_per_thread,
                                    partition_metadata,
                                    block_partition_sizes,
                                    scanned_block_partition_sizes,
                                    local_slots,
                                    local_partition_offsets,
                                    global_partition_offsets);

  copy_partitioned_values(block,
                          cuda::counting_iterator<size_type>{0},
                          gather_map,
                          payload,
                          local_slots,
                          num_rows,
                          num_partitions,
                          rows_per_thread,
                          local_partition_offsets,
                          global_partition_offsets);
}

/**
 * @brief Configures a kernel for maximum dynamic shared memory and returns its launch capacity.
 *
 * This opts the kernel specialization into the device's maximum dynamic shared
 * memory for subsequent launches before occupancy is evaluated. CUDA and configuration failures
 * are propagated. A zero result indicates that the requested CTA size is not supported by this
 * kernel specialization.
 *
 * @tparam Kernel CUDA kernel pointer type
 * @param kernel Kernel whose dynamic shared-memory limit is configured and queried
 * @param block_size Requested CTA size
 * @return Available dynamic shared memory in bytes, or zero when the CTA cannot launch
 */
template <typename Kernel>
std::size_t configure_dynamic_shared_memory(Kernel kernel, size_type block_size)
{
  cudaFuncAttributes attributes{};
  CUDF_CUDA_TRY(cudaFuncGetAttributes(&attributes, reinterpret_cast<void const*>(kernel)));
  if (attributes.maxThreadsPerBlock < block_size) { return 0; }

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  int opt_in_limit{};
  CUDF_CUDA_TRY(
    cudaDeviceGetAttribute(&opt_in_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  auto const statically_allocated = static_cast<std::size_t>(attributes.sharedSizeBytes);
  CUDF_EXPECTS(statically_allocated <= static_cast<std::size_t>(opt_in_limit),
               "Kernel static shared memory exceeds the device opt-in limit");
  auto const max_dynamic_shared_memory =
    static_cast<std::size_t>(opt_in_limit) - statically_allocated;

  CUDF_CUDA_TRY(cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel),
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(max_dynamic_shared_memory)));

  int active_blocks{};
  CUDF_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, block_size, 0));
  if (active_blocks == 0) { return 0; }

  std::size_t available{};
  CUDF_CUDA_TRY(cudaOccupancyAvailableDynamicSMemPerBlock(&available, kernel, 1, block_size));
  return std::min(available, max_dynamic_shared_memory);
}

/**
 * @brief Groups consecutive fixed-width columns into batches sized for L2 residency.
 *
 * The estimate includes the row metadata and output sectors used by all concurrently active CTAs.
 * Each column adds its output data and one possible boundary sector per non-empty partition.
 * Limiting this footprint reduces the scattered DRAM reloads that increase row activations,
 * read/write switches, and load latency. Input data uses streaming loads and is not included. The
 * smaller partition-count and descriptor arrays are also not included.
 *
 * The estimate assumes active CTAs have consecutive blockIdx.x values.
 *
 * @tparam BlockSize Number of threads in the staged-scatter CTA
 * @tparam PartitionMetadataView Device-accessible partition metadata view
 * @param columns Fixed-width columns in input order
 * @param grid_size Number of staged-scatter CTAs
 * @param rows_per_thread Number of rows processed by each thread
 * @param num_partitions Number of output partitions
 * @param max_element_width Widest staged column, which determines launch occupancy
 * @return Consecutive column batches
 */
template <size_type BlockSize, typename PartitionMetadataView>
std::vector<fixed_width_copy_batch> make_fixed_width_copy_batches(table_view const& columns,
                                                                  size_type grid_size,
                                                                  size_type rows_per_thread,
                                                                  size_type num_partitions,
                                                                  size_type max_element_width)
{
  if (columns.num_columns() == 0) { return {}; }

  static_assert(std::is_same_v<PartitionMetadataView, detail::partition_metadata::packed_view> ||
                std::is_same_v<PartitionMetadataView, detail::partition_metadata::default_view>);
  // This makes the output sector count additive across columns.
  static_assert(BlockSize % L2_SECTOR_BYTES == 0);

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDF_CUDA_TRY(cudaGetDeviceProperties(&properties, device));

  // Limit one batch to this many L2 sectors.
  auto const l2_sector_budget =
    static_cast<std::uint64_t>(static_cast<double>(std::max(properties.l2CacheSize, 0)) *
                               L2_CACHE_USAGE_FRACTION) /
    L2_SECTOR_BYTES;

  // Every CTA reserves enough shared memory for the widest column.
  auto const shared_memory_bytes =
    staged_scatter_smem<BlockSize>{num_partitions, rows_per_thread, max_element_width}.bytes;
  int active_ctas_per_sm{};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_ctas_per_sm,
    &copy_fixed_width_columns<BlockSize, PartitionMetadataView>,
    BlockSize,
    shared_memory_bytes));
  CUDF_EXPECTS(active_ctas_per_sm > 0,
               "Fixed-width copy batch has no active CTAs per multiprocessor");

  // Count the CTAs and rows that can be active at the same time.
  auto const ctas_per_active_wave =
    std::min<std::uint64_t>(static_cast<std::uint64_t>(grid_size),
                            static_cast<std::uint64_t>(active_ctas_per_sm) *
                              static_cast<std::uint64_t>(properties.multiProcessorCount));
  auto const rows_per_active_wave = ctas_per_active_wave * static_cast<std::uint64_t>(BlockSize) *
                                    static_cast<std::uint64_t>(rows_per_thread);

  // The packed layout stores the partition and local offset in one value. The default layout
  // stores them in two values.
  constexpr auto row_metadata_bytes_per_row =
    std::is_same_v<PartitionMetadataView, detail::partition_metadata::packed_view>
      ? sizeof(size_type)
      : 2 * sizeof(size_type);
  auto const row_metadata_sectors =
    rows_per_active_wave * row_metadata_bytes_per_row / L2_SECTOR_BYTES;
  auto const output_sector_budget =
    l2_sector_budget > row_metadata_sectors ? l2_sector_budget - row_metadata_sectors : 0;

  // A column writes at most one output run per non-empty partition.
  auto const max_output_runs_per_column =
    std::min<std::uint64_t>(static_cast<std::uint64_t>(num_partitions), rows_per_active_wave);

  std::vector<fixed_width_copy_batch> batches;
  std::uint64_t batch_output_sectors{};
  size_type column_index{};
  std::for_each(columns.begin(), columns.end(), [&](column_view const& column) {
    auto const element_width = static_cast<size_type>(cudf::size_of(column.type()));
    auto const column_output_sectors =
      (rows_per_active_wave / L2_SECTOR_BYTES) * static_cast<std::uint64_t>(element_width) +
      max_output_runs_per_column;

    if (batches.empty() || batch_output_sectors + column_output_sectors > output_sector_budget) {
      // A batch always contains at least one column, even when that column exceeds the budget.
      batches.push_back({column_index, 1, element_width});
      batch_output_sectors = column_output_sectors;
    } else {
      auto& batch = batches.back();
      ++batch.num_columns;
      batch.max_element_width = std::max(batch.max_element_width, element_width);
      batch_output_sectors += column_output_sectors;
    }
    ++column_index;
  });
  return batches;
}

/**
 * @brief Returns the maximum rows per thread supported by the staged-scatter path.
 *
 * @tparam BlockSize Number of threads in the staged-scatter CTA
 * @tparam Hasher Device row hasher type
 * @tparam Partitioner Hash-to-partition mapping type
 * @tparam PartitionMetadataView Device-accessible partition metadata view
 * @param num_partitions Number of partitions
 * @param input Table whose columns will be materialized
 * @param column_groups Fixed-width and variable-width column groups
 * @return Rows per thread, or 0 when a required kernel cannot launch
 */
template <size_type BlockSize,
          typename Hasher,
          typename Partitioner,
          typename PartitionMetadataView>
size_type staged_rows_per_thread(size_type num_partitions,
                                 table_view const& input,
                                 partition_column_groups const& column_groups)
{
  auto const histogram_bytes = static_cast<std::size_t>(num_partitions) * sizeof(size_type);
  auto const metadata_kernel =
    &compute_row_partition_numbers<Hasher, Partitioner, PartitionMetadataView>;
  // Metadata generation requires one shared counter for every partition.
  if (histogram_bytes > configure_dynamic_shared_memory(metadata_kernel, BlockSize)) { return 0; }

  auto const max_rows_per_thread = [num_partitions](
                                     auto kernel, size_type element_width, size_type upper_bound) {
    auto const available_smem_bytes = configure_dynamic_shared_memory(kernel, BlockSize);
    auto const fixed_smem_bytes =
      staged_scatter_smem<BlockSize>{num_partitions, 0, element_width}.bytes;
    if (fixed_smem_bytes >= available_smem_bytes) { return size_type{0}; }

    auto const bytes_per_iteration =
      staged_scatter_smem<BlockSize>{num_partitions, 1, element_width}.bytes - fixed_smem_bytes;
    return std::min(
      upper_bound,
      static_cast<size_type>((available_smem_bytes - fixed_smem_bytes) / bytes_per_iteration));
  };

  auto rows_per_thread = GATHER_MAP_ROWS_PER_THREAD;
  // Staged payloads must fit using the widest fixed-width element.
  if (!column_groups.fixed_width_indices.empty()) {
    rows_per_thread =
      max_rows_per_thread(&copy_fixed_width_columns<BlockSize, PartitionMetadataView>,
                          column_groups.max_element_width,
                          std::numeric_limits<size_type>::max());
    if (rows_per_thread == 0) { return 0; }
  }

  // Variable-width payloads and staged columns containing nulls require a gather map built from
  // the routing data. All-valid masks can be allocated directly.
  auto const fixed_width_input = input.select(column_groups.fixed_width_indices);
  if (requires_gather_map(column_groups, fixed_width_input)) {
    rows_per_thread = max_rows_per_thread(
      &compute_gather_map<BlockSize, PartitionMetadataView>, sizeof(size_type), rows_per_thread);
    if (rows_per_thread == 0) { return 0; }
  }

  return rows_per_thread;
}

/**
 * @brief Hash-partitions by building row destinations in global memory and scattering the input.
 */
template <typename Hasher>
std::pair<std::unique_ptr<table>, std::vector<size_type>> partition_global_scatter(
  table_view const& input,
  size_type num_rows,
  size_type num_partitions,
  Hasher hasher,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_partitions < std::numeric_limits<size_type>::max(),
               "num_partitions exceeds cudf's supported limit");

  auto row_partition_numbers = rmm::device_uvector<size_type>(num_rows, stream);

  // Compute partition number for each row
  if (is_power_two(num_partitions)) {
    auto const partitioner = bitwise_partitioner<hash_value_type>(num_partitions);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(num_rows),
                      row_partition_numbers.begin(),
                      [hasher, partitioner] __device__(size_type row) -> size_type {
                        return partitioner(hasher(row));
                      });
  } else {
    auto const partitioner = modulo_partitioner<hash_value_type>(num_partitions);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(num_rows),
                      row_partition_numbers.begin(),
                      [hasher, partitioner] __device__(size_type row) -> size_type {
                        return partitioner(hasher(row));
                      });
  }

  // Build histogram via cub::DeviceHistogram::HistogramEven.
  // HistogramEven writes num_partitions bins; the extra element is used by the exclusive scan
  // below to produce the total row count as the last offset. Zero-initialize to avoid UB.
  auto histogram = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    num_partitions + 1, stream, cudf::get_current_device_resource_ref());
  {
    auto const num_levels  = num_partitions + 1;
    auto const lower_level = size_type{0};
    auto const upper_level = num_partitions;

    std::size_t temp_storage_bytes{};
    cub::DeviceHistogram::HistogramEven(nullptr,
                                        temp_storage_bytes,
                                        row_partition_numbers.data(),
                                        histogram.data(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        num_rows,
                                        stream.value());
    rmm::device_buffer temp_storage(temp_storage_bytes, stream);
    cub::DeviceHistogram::HistogramEven(temp_storage.data(),
                                        temp_storage_bytes,
                                        row_partition_numbers.data(),
                                        histogram.data(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        num_rows,
                                        stream.value());
  }

  // Exclusive scan on histogram to get partition offsets.
  // histogram has num_partitions+1 elements; after scan, histogram[num_partitions] = num_rows.
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         histogram.begin(),
                         histogram.end(),
                         histogram.begin());

  // Copy partition offsets to pinned host memory asynchronously
  auto const pinned_offsets = cudf::detail::make_pinned_vector_async(histogram, stream);

  // Build scatter map: atomically increment partition offsets
  rmm::device_uvector<size_type> scatter_map(num_rows, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    row_partition_numbers.begin(),
    row_partition_numbers.end(),
    scatter_map.begin(),
    [offsets = histogram.data()] __device__(auto partition_number) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> ref(offsets[partition_number]);
      return ref.fetch_add(1, cuda::memory_order_relaxed);
    });

  // Scatter input rows into partitioned output
  auto output = detail::scatter(input, scatter_map, input, stream, mr);

  stream.synchronize();  // Pinned async D2H copy must finish before returning host vec

  // Convert pinned host_vector to std::vector for the return type
  auto partition_offsets = std::vector<size_type>(pinned_offsets.begin(), pinned_offsets.end());

  return std::pair{std::move(output), std::move(partition_offsets)};
}

/**
 * @brief Partitions a table by reusing each row's destination while copying its columns.
 *
 * Fixed-width columns are copied together using the CTA-local row mapping. Remaining columns and
 * validity masks are materialized using gather maps built from the same partition metadata.
 *
 * @tparam BlockSize Number of threads in the staged-scatter CTA
 * @tparam PartitionMetadataView Device-accessible partition metadata view type
 * @tparam Hasher Device-callable row hasher type
 * @tparam Partitioner Hash-to-partition mapping type
 * @param input Table whose rows are reordered into partitions
 * @param num_rows Number of rows to partition
 * @param num_partitions Number of output partitions
 * @param hasher Row hasher used to select partitions
 * @param partitioner Functor that maps row hashes to partition identifiers
 * @param partition_metadata Device-accessible partition metadata view
 * @param rows_per_thread Grid-stride iterations processed by each thread
 * @param column_groups Fixed-width and variable-width column groups
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for output allocations
 * @return Partitioned table and the starting offset of each partition
 */
template <size_type BlockSize,
          typename PartitionMetadataView,
          typename Hasher,
          typename Partitioner>
std::pair<std::unique_ptr<table>, std::vector<size_type>> partition_staged_scatter(
  table_view const& input,
  size_type num_rows,
  size_type num_partitions,
  Hasher hasher,
  Partitioner partitioner,
  PartitionMetadataView partition_metadata,
  size_type rows_per_thread,
  partition_column_groups const& column_groups,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Cover the input with CTAs using the selected number of rows per thread.
  auto const grid_size =
    static_cast<size_type>(cudf::detail::grid_1d{num_rows, BlockSize, rows_per_thread}.num_blocks);
  auto const histogram_bytes = static_cast<std::size_t>(num_partitions) * sizeof(size_type);
  auto const num_cta_partition_counts =
    static_cast<std::size_t>(grid_size) * static_cast<std::size_t>(num_partitions);

  // Allocate one count for every CTA/partition pair and pinned storage for the final partition
  // starts copied back to the host.
  auto block_partition_sizes = rmm::device_uvector<size_type>(num_cta_partition_counts, stream);
  auto scanned_block_partition_sizes =
    rmm::device_uvector<size_type>(num_cta_partition_counts, stream);
  auto host_partition_offsets =
    cudf::detail::make_pinned_vector_async<size_type>(num_partitions + 1, stream);
  host_partition_offsets[num_partitions] = num_rows;

  // Hash every row and record its partition plus its offset within the CTA-local partition. Each
  // CTA also writes the partition counts consumed by the prefix scan below.
  compute_row_partition_numbers<<<grid_size, BlockSize, histogram_bytes, stream.value()>>>(
    hasher, num_partitions, partitioner, partition_metadata, block_partition_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // Counts are partition-major: all CTA counts for partition 0, then all CTA counts for partition
  // 1, and so on. The scan therefore provides both each CTA's output offset and every partition's
  // global start.
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         block_partition_sizes.begin(),
                         block_partition_sizes.end(),
                         scanned_block_partition_sizes.begin());

  // Copy the first CTA offset for every partition. The source pitch skips one partition's
  // complete set of CTA offsets, while the destination advances by one element.
  CUDF_CUDA_TRY(cudaMemcpy2DAsync(host_partition_offsets.data(),
                                  sizeof(size_type),
                                  scanned_block_partition_sizes.data(),
                                  static_cast<std::size_t>(grid_size) * sizeof(size_type),
                                  sizeof(size_type),
                                  num_partitions,
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));

  // Split the input into columns copied by the staged kernel and columns materialized by gather.
  // A gather map is needed only for gathered data or fixed-width masks containing nulls.
  auto const fixed_width_input    = input.select(column_groups.fixed_width_indices);
  auto const variable_width_input = input.select(column_groups.variable_width_indices);
  auto const gather_map_required  = requires_gather_map(column_groups, fixed_width_input);

  // Allocate each staged output and describe its sliced input and output data buffers. All-valid
  // nullable columns receive their output masks directly and do not require mask gathering.
  std::vector<std::unique_ptr<column>> fixed_width_outputs;
  fixed_width_outputs.reserve(column_groups.fixed_width_indices.size());
  if (!column_groups.fixed_width_indices.empty()) {
    auto const batches = make_fixed_width_copy_batches<BlockSize, PartitionMetadataView>(
      fixed_width_input,
      grid_size,
      rows_per_thread,
      num_partitions,
      column_groups.max_element_width);
    CUDF_EXPECTS(batches.size() <= std::numeric_limits<std::uint16_t>::max(),
                 "Fixed-width column batch count exceeds the CUDA grid y limit");

    // Pack the logically separate column and batch arrays into one allocation and upload.
    auto const columns_bytes =
      column_groups.fixed_width_indices.size() * sizeof(fixed_width_column_descriptor);
    auto const batches_bytes  = batches.size() * sizeof(fixed_width_copy_batch);
    auto const metadata_bytes = columns_bytes + batches_bytes;
    auto const padded_bytes   = metadata_bytes + alignof(fixed_width_column_descriptor) - 1;
    auto host_metadata = cudf::detail::make_pinned_vector_async<std::uint8_t>(padded_bytes, stream);
    auto* host_columns =
      cudf::detail::align_ptr_for_type<fixed_width_column_descriptor>(host_metadata.data());
    auto* host_batches = reinterpret_cast<fixed_width_copy_batch*>(
      reinterpret_cast<std::uint8_t*>(host_columns) + columns_bytes);

    for (std::size_t descriptor_index = 0;
         descriptor_index < column_groups.fixed_width_indices.size();
         ++descriptor_index) {
      auto const column_index = column_groups.fixed_width_indices[descriptor_index];
      auto const& source      = input.column(column_index);
      auto const output_mask_state =
        source.nullable() && !source.has_nulls() ? mask_state::ALL_VALID : mask_state::UNALLOCATED;
      auto output =
        cudf::make_fixed_width_column(source.type(), source.size(), output_mask_state, stream, mr);
      auto output_view         = output->mutable_view();
      auto const element_width = static_cast<size_type>(cudf::size_of(source.type()));

      host_columns[descriptor_index] =
        fixed_width_column_descriptor{static_cast<std::uint8_t const*>(source.head()) +
                                        static_cast<std::size_t>(source.offset()) * element_width,
                                      static_cast<std::uint8_t*>(output_view.head()),
                                      element_width};
      fixed_width_outputs.push_back(std::move(output));
    }
    std::copy(batches.begin(), batches.end(), host_batches);

    auto device_metadata =
      rmm::device_buffer(padded_bytes, stream, cudf::get_current_device_resource_ref());
    auto* device_columns =
      cudf::detail::align_ptr_for_type<fixed_width_column_descriptor>(device_metadata.data());
    auto* device_batches = reinterpret_cast<fixed_width_copy_batch*>(
      reinterpret_cast<std::uint8_t*>(device_columns) + columns_bytes);
    cudf::detail::cuda_memcpy_async<std::uint8_t>(
      device_span<std::uint8_t>{reinterpret_cast<std::uint8_t*>(device_columns), metadata_bytes},
      host_span<std::uint8_t const>{reinterpret_cast<std::uint8_t const*>(host_columns),
                                    metadata_bytes},
      stream);

    auto const copy_shared_memory =
      staged_scatter_smem<BlockSize>{
        num_partitions, rows_per_thread, column_groups.max_element_width}
        .bytes;
    // The target GPUs rasterize CTAs along x before advancing y. Mapping rows to x and column
    // batches to y keeps each batch contiguous in launch order, approximating separate launches.
    auto const copy_grid =
      dim3{static_cast<unsigned int>(grid_size), static_cast<unsigned int>(batches.size())};
    copy_fixed_width_columns<BlockSize, PartitionMetadataView>
      <<<copy_grid, BlockSize, copy_shared_memory, stream.value()>>>(
        device_columns,
        device_batches,
        num_rows,
        num_partitions,
        rows_per_thread,
        partition_metadata,
        block_partition_sizes.data(),
        scanned_block_partition_sizes.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // Build one output-to-input map shared by all gathered columns and staged validity masks.
  auto gather_map = rmm::device_uvector<size_type>(gather_map_required ? num_rows : 0, stream);
  if (gather_map_required) {
    auto const shared_memory =
      staged_scatter_smem<BlockSize>{num_partitions, rows_per_thread, sizeof(size_type)}.bytes;
    compute_gather_map<BlockSize, PartitionMetadataView>
      <<<grid_size, BlockSize, shared_memory, stream.value()>>>(
        num_rows,
        num_partitions,
        rows_per_thread,
        partition_metadata,
        block_partition_sizes.data(),
        scanned_block_partition_sizes.data(),
        gather_map.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // Fixed-width data was copied directly, so use the map only for masks containing nulls.
  if (has_nulls(fixed_width_input)) {
    detail::gather_bitmask(fixed_width_input,
                           gather_map.begin(),
                           fixed_width_outputs,
                           detail::gather_bitmask_op::DONT_CHECK,
                           stream,
                           mr);
  }

  // Gather all remaining columns as one table so strings, nested columns, dictionaries, and their
  // validity masks consume the same output-to-input map.
  std::vector<std::unique_ptr<column>> variable_width_outputs;
  if (!column_groups.variable_width_indices.empty()) {
    auto gathered = cudf::detail::gather(variable_width_input,
                                         gather_map,
                                         out_of_bounds_policy::DONT_CHECK,
                                         negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);

    variable_width_outputs = gathered->release();
  }

  // Fixed-width and gathered columns were materialized separately. Place each output column back
  // at its original input column index.
  std::vector<std::unique_ptr<column>> output_columns(input.num_columns());
  for (std::size_t index = 0; index < column_groups.fixed_width_indices.size(); ++index) {
    output_columns[column_groups.fixed_width_indices[index]] =
      std::move(fixed_width_outputs[index]);
  }
  for (std::size_t index = 0; index < column_groups.variable_width_indices.size(); ++index) {
    output_columns[column_groups.variable_width_indices[index]] =
      std::move(variable_width_outputs[index]);
  }

  stream.synchronize();
  auto partition_offsets =
    std::vector<size_type>(host_partition_offsets.begin(), host_partition_offsets.end());
  return {std::make_unique<table>(std::move(output_columns), num_rows),
          std::move(partition_offsets)};
}

/**
 * @brief Partitions a table by scattering each input row to its computed output location.
 *
 * @tparam PartitionMetadataView Device-accessible partition metadata view type
 * @tparam Hasher Device-callable row hasher type
 * @tparam Partitioner Hash-to-partition mapping type
 * @param input Table whose rows are reordered into partitions
 * @param num_rows Number of rows to partition
 * @param num_partitions Number of output partitions
 * @param hasher Row hasher used to select partitions
 * @param partitioner Functor that maps row hashes to partition identifiers
 * @param partition_metadata Device-accessible partition metadata view
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for output allocations
 * @return Partitioned table and the starting offset of each partition
 */
template <typename PartitionMetadataView, typename Hasher, typename Partitioner>
std::pair<std::unique_ptr<table>, std::vector<size_type>> partition_scatter(
  table_view const& input,
  size_type num_rows,
  size_type num_partitions,
  Hasher hasher,
  Partitioner partitioner,
  PartitionMetadataView partition_metadata,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const grid_size = static_cast<size_type>(
    cudf::detail::grid_1d{num_rows, SCATTER_BLOCK_SIZE, SCATTER_ROWS_PER_THREAD}.num_blocks);
  auto const histogram_bytes = static_cast<std::size_t>(num_partitions) * sizeof(size_type);
  auto const num_cta_partition_counts =
    static_cast<std::size_t>(grid_size) * static_cast<std::size_t>(num_partitions);

  // The metadata CTA owns every row it visits. Its histogram records how many rows it contributes
  // to each partition, and each row stores its offset within that CTA-local partition.
  auto block_partition_sizes = rmm::device_uvector<size_type>(num_cta_partition_counts, stream);
  auto scanned_block_partition_sizes =
    rmm::device_uvector<size_type>(num_cta_partition_counts, stream);
  auto host_partition_offsets =
    cudf::detail::make_pinned_vector_async<size_type>(num_partitions + 1, stream);
  host_partition_offsets[num_partitions] = num_rows;

  compute_row_partition_numbers<<<grid_size, SCATTER_BLOCK_SIZE, histogram_bytes, stream.value()>>>(
    hasher, num_partitions, partitioner, partition_metadata, block_partition_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // Counts are partition-major: all CTA counts for partition 0, then all CTA counts for partition
  // 1, and so on. The scan therefore provides both each CTA's output offset and every partition's
  // global start.
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         block_partition_sizes.begin(),
                         block_partition_sizes.end(),
                         scanned_block_partition_sizes.begin());

  // Copy the first CTA offset for every partition. The source pitch skips one partition's
  // complete set of CTA offsets, while the destination advances by one element.
  CUDF_CUDA_TRY(cudaMemcpy2DAsync(host_partition_offsets.data(),
                                  sizeof(size_type),
                                  scanned_block_partition_sizes.data(),
                                  static_cast<std::size_t>(grid_size) * sizeof(size_type),
                                  sizeof(size_type),
                                  num_partitions,
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));

  auto row_output_locations = rmm::device_uvector<size_type>(num_rows, stream);
  compute_row_output_locations<<<grid_size, SCATTER_BLOCK_SIZE, histogram_bytes, stream.value()>>>(
    partition_metadata,
    row_output_locations.data(),
    num_partitions,
    scanned_block_partition_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  auto output = detail::scatter(input, row_output_locations, input, stream, mr);
  stream.synchronize();  // Async partition-offset copy must complete
  auto partition_offsets =
    std::vector<size_type>(host_partition_offsets.begin(), host_partition_offsets.end());
  return {std::move(output), std::move(partition_offsets)};
}

/**
 * @brief Allocates partition metadata and runs one staged-scatter configuration.
 *
 * @tparam BlockSize Number of threads in each staged-scatter CTA
 * @tparam PartitionMetadataView Metadata representation used by the staged-scatter kernels
 * @tparam Hasher Device-callable row hasher type
 * @tparam Partitioner Hash-to-partition mapping type
 * @param input Table whose rows are reordered into partitions
 * @param num_rows Number of rows to partition
 * @param num_partitions Number of output partitions
 * @param hasher Row hasher used to select partitions
 * @param partitioner Functor that maps row hashes to partition identifiers
 * @param rows_per_thread Number of input rows processed by each thread
 * @param column_groups Fixed-width and variable-width column groups
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for output allocations
 * @return Partitioned table and the starting offset of each partition
 */
template <size_type BlockSize,
          typename PartitionMetadataView,
          typename Hasher,
          typename Partitioner>
std::pair<std::unique_ptr<table>, std::vector<size_type>> run_staged_scatter(
  table_view const& input,
  size_type num_rows,
  size_type num_partitions,
  Hasher hasher,
  Partitioner partitioner,
  size_type rows_per_thread,
  partition_column_groups const& column_groups,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static_assert(std::is_same_v<PartitionMetadataView, detail::partition_metadata::packed_view> ||
                std::is_same_v<PartitionMetadataView, detail::partition_metadata::default_view>);

  if constexpr (std::is_same_v<PartitionMetadataView, detail::partition_metadata::packed_view>) {
    auto packed_metadata = rmm::device_uvector<std::uint32_t>(num_rows, stream);
    auto const partition_bits =
      detail::partition_metadata::ceil_log2(static_cast<std::uint64_t>(num_partitions));
    auto const metadata = detail::partition_metadata::packed_view{
      device_span<std::uint32_t>{packed_metadata}, partition_bits};
    return partition_staged_scatter<BlockSize>(input,
                                               num_rows,
                                               num_partitions,
                                               hasher,
                                               partitioner,
                                               metadata,
                                               rows_per_thread,
                                               column_groups,
                                               stream,
                                               mr);
  } else {
    auto row_partitions        = rmm::device_uvector<size_type>(num_rows, stream);
    auto row_partition_offsets = rmm::device_uvector<size_type>(num_rows, stream);
    auto const metadata        = detail::partition_metadata::default_view{
      device_span<size_type>{row_partitions}, device_span<size_type>{row_partition_offsets}};
    return partition_staged_scatter<BlockSize>(input,
                                               num_rows,
                                               num_partitions,
                                               hasher,
                                               partitioner,
                                               metadata,
                                               rows_per_thread,
                                               column_groups,
                                               stream,
                                               mr);
  }
}

/**
 * @brief Partitions rows using the first supported execution path.
 *
 * Staged scatter first uses packed metadata with the preferred CTA size, then retries that CTA size
 * with default metadata. A smaller CTA is used only as a shared-memory fallback and always uses
 * default metadata. When staged scatter is unavailable, the function tries shared-memory scatter
 * before using the global-memory implementation.
 *
 * @tparam Hasher Device-callable row hasher type
 * @tparam Partitioner Hash-to-partition mapping type
 * @param input Table whose rows are reordered into partitions
 * @param num_rows Number of rows to partition
 * @param num_partitions Number of output partitions
 * @param hasher Row hasher used to select partitions
 * @param partitioner Functor that maps row hashes to partition identifiers
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for output allocations
 * @return Partitioned table and the starting offset of each partition
 */
template <typename Hasher, typename Partitioner>
std::pair<std::unique_ptr<table>, std::vector<size_type>> dispatch_partition_impl(
  table_view const& input,
  size_type num_rows,
  size_type num_partitions,
  Hasher hasher,
  Partitioner partitioner,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const column_groups = group_columns(input);

  using packed_metadata_view  = detail::partition_metadata::packed_view;
  using default_metadata_view = detail::partition_metadata::default_view;

  // Prefer the fixed-width-optimized execution shape and packed metadata.
  if (auto const rows_per_thread =
        staged_rows_per_thread<PREFERRED_STAGED_SCATTER_BLOCK_SIZE,
                               Hasher,
                               Partitioner,
                               packed_metadata_view>(num_partitions, input, column_groups);
      rows_per_thread != 0 &&
      detail::partition_metadata::pick_layout(
        num_partitions, PREFERRED_STAGED_SCATTER_BLOCK_SIZE * rows_per_thread) ==
        detail::partition_metadata::layout::PACKED32) {
    return run_staged_scatter<PREFERRED_STAGED_SCATTER_BLOCK_SIZE, packed_metadata_view>(
      input,
      num_rows,
      num_partitions,
      hasher,
      partitioner,
      rows_per_thread,
      column_groups,
      stream,
      mr);
  }

  // Use separate partition and offset arrays with the 1,024-thread CTA.
  if (auto const rows_per_thread =
        staged_rows_per_thread<PREFERRED_STAGED_SCATTER_BLOCK_SIZE,
                               Hasher,
                               Partitioner,
                               default_metadata_view>(num_partitions, input, column_groups);
      rows_per_thread != 0) {
    return run_staged_scatter<PREFERRED_STAGED_SCATTER_BLOCK_SIZE, default_metadata_view>(
      input,
      num_rows,
      num_partitions,
      hasher,
      partitioner,
      rows_per_thread,
      column_groups,
      stream,
      mr);
  }

  // Use a 512-thread CTA to reduce the staged shared-memory footprint.
  if (auto const rows_per_thread =
        staged_rows_per_thread<FALLBACK_STAGED_SCATTER_BLOCK_SIZE,
                               Hasher,
                               Partitioner,
                               default_metadata_view>(num_partitions, input, column_groups);
      rows_per_thread != 0) {
    return run_staged_scatter<FALLBACK_STAGED_SCATTER_BLOCK_SIZE, default_metadata_view>(
      input,
      num_rows,
      num_partitions,
      hasher,
      partitioner,
      rows_per_thread,
      column_groups,
      stream,
      mr);
  }

  auto const scatter_layout = detail::partition_metadata::pick_layout(
    num_partitions, SCATTER_BLOCK_SIZE * SCATTER_ROWS_PER_THREAD);
  auto const scatter_histogram_bytes = static_cast<std::size_t>(num_partitions) * sizeof(size_type);
  auto const scatter_max_smem_bytes =
    []<typename PartitionMetadataView>(cuda::std::type_identity<PartitionMetadataView>) {
      return std::min(configure_dynamic_shared_memory(
                        &compute_row_partition_numbers<Hasher, Partitioner, PartitionMetadataView>,
                        SCATTER_BLOCK_SIZE),
                      configure_dynamic_shared_memory(
                        &compute_row_output_locations<PartitionMetadataView>, SCATTER_BLOCK_SIZE));
    };

  if (scatter_layout == detail::partition_metadata::layout::PACKED32 &&
      scatter_histogram_bytes <=
        scatter_max_smem_bytes(
          cuda::std::type_identity<detail::partition_metadata::packed_view>{})) {
    auto packed_metadata = rmm::device_uvector<std::uint32_t>(num_rows, stream);
    auto const partition_bits =
      detail::partition_metadata::ceil_log2(static_cast<std::uint64_t>(num_partitions));
    auto const metadata = detail::partition_metadata::packed_view{
      device_span<std::uint32_t>{packed_metadata}, partition_bits};
    return partition_scatter(
      input, num_rows, num_partitions, hasher, partitioner, metadata, stream, mr);
  } else if (scatter_histogram_bytes <=
             scatter_max_smem_bytes(
               cuda::std::type_identity<detail::partition_metadata::default_view>{})) {
    auto row_partitions        = rmm::device_uvector<size_type>(num_rows, stream);
    auto row_partition_offsets = rmm::device_uvector<size_type>(num_rows, stream);
    auto const metadata        = detail::partition_metadata::default_view{
      device_span<size_type>{row_partitions}, device_span<size_type>{row_partition_offsets}};
    return partition_scatter(
      input, num_rows, num_partitions, hasher, partitioner, metadata, stream, mr);
  } else {
    return partition_global_scatter(input, num_rows, num_partitions, hasher, stream, mr);
  }
}

/**
 * @brief Selects a power-of-two or modulo partitioner before launch selection.
 *
 * @tparam Hasher Device-callable row hasher type
 * @param input Table whose rows are reordered into partitions
 * @param num_rows Number of rows to partition
 * @param num_partitions Number of output partitions
 * @param hasher Row hasher used to select partitions
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for output allocations
 * @return Partitioned table and the starting offset of each partition
 */
template <typename Hasher>
std::pair<std::unique_ptr<table>, std::vector<size_type>> dispatch_hash_partition_table(
  table_view const& input,
  size_type num_rows,
  size_type num_partitions,
  Hasher hasher,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (is_power_two(num_partitions)) {
    auto const partitioner = bitwise_partitioner<hash_value_type>{num_partitions};
    return dispatch_partition_impl(
      input, num_rows, num_partitions, hasher, partitioner, stream, mr);
  }
  auto const partitioner = modulo_partitioner<hash_value_type>{num_partitions};
  return dispatch_partition_impl(input, num_rows, num_partitions, hasher, partitioner, stream, mr);
}

/**
 * @brief Hash-partitions a table using direct element hashing when all keys are fixed-width.
 *
 * Falls back to the generic row hasher for unsupported key types.
 *
 * @tparam Hash Element hash functor template
 * @tparam HasNulls Whether any key column may contain nulls; must be `true` when keys have nulls
 * @param input Table whose rows are reordered into partitions
 * @param table_to_hash Key columns used to select a partition for each row
 * @param num_partitions Number of output partitions
 * @param seed Initial hash seed
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for output allocations
 * @return Partitioned table and the starting offset of each partition
 */
template <template <typename> class Hash, bool HasNulls>
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition_table(
  table_view const& input,
  table_view const& table_to_hash,
  size_type num_partitions,
  uint32_t seed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = table_to_hash.num_rows();
  if (detail::is_fixed_width_partition_compatible(table_to_hash)) {
    auto key_device_view = table_device_view::create(table_to_hash, stream);
    auto const hasher    = fixed_width_row_hasher<Hash, HasNulls>{*key_device_view, seed};
    return dispatch_hash_partition_table(input, num_rows, num_partitions, hasher, stream, mr);
  }

  auto const row_hasher = detail::row::hash::row_hasher(table_to_hash, stream);
  auto const hasher     = row_hasher.device_hasher<Hash>(nullate::DYNAMIC{HasNulls}, seed);
  return dispatch_hash_partition_table(input, num_rows, num_partitions, hasher, stream, mr);
}

struct dispatch_map_type {
  /**
   * @brief Partitions the table `t` according to the `partition_map`.
   *
   * Algorithm:
   * - Compute the histogram of the size each partition
   * - Compute the exclusive scan of the histogram to get the offset for each
   * partition in the final partitioned output
   * - Use a transform iterator to materialize the scatter map of the rows from
   * `t` into the final output.
   *
   * @note JH: It would likely be more efficient to avoid the atomic increments
   * in the transform iterator. It would probably be faster to compute a
   * per-thread block histogram and compute an exclusive scan of all of the
   * per-block histograms (like in hash partition). But I'm purposefully trying
   * to reduce memory pressure by avoiding intermediate materializations. Plus,
   * atomics resolve in L2 and should be pretty fast since all the offsets will
   * fit in L2.
   *
   */
  template <typename MapType>
  std::pair<std::unique_ptr<table>, std::vector<size_type>> operator()(
    table_view const& t,
    column_view const& partition_map,
    size_type num_partitions,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
    requires(is_index_type<MapType>())
  {
    // Build a histogram of the number of rows in each partition
    rmm::device_uvector<size_type> histogram(num_partitions + 1, stream);
    std::size_t temp_storage_bytes{};
    std::size_t const num_levels = num_partitions + 1;
    size_type const lower_level  = 0;
    size_type const upper_level  = num_partitions;
    cub::DeviceHistogram::HistogramEven(nullptr,
                                        temp_storage_bytes,
                                        partition_map.begin<MapType>(),
                                        histogram.data(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        partition_map.size(),
                                        stream.value());

    rmm::device_buffer temp_storage(temp_storage_bytes, stream);

    cub::DeviceHistogram::HistogramEven(temp_storage.data(),
                                        temp_storage_bytes,
                                        partition_map.begin<MapType>(),
                                        histogram.data(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        partition_map.size(),
                                        stream.value());

    // `histogram` was created with an extra entry at the end such that an
    // exclusive scan will put the total number of rows at the end
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           histogram.begin(),
                           histogram.end(),
                           histogram.begin());

    // Copy offsets to host before the transform below modifies the histogram
    auto const partition_offsets = cudf::detail::make_std_vector(histogram, stream);

    // Unfortunately need to materialize the scatter map because
    // `detail::scatter` requires multiple passes through the iterator
    rmm::device_uvector<size_type> scatter_map(partition_map.size(), stream);

    // For each `partition_map[i]`, atomically increment the corresponding
    // partition offset to determine `i`s location in the output
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      partition_map.begin<MapType>(),
                      partition_map.end<MapType>(),
                      scatter_map.begin(),
                      [offsets = histogram.data()] __device__(auto partition_number) {
                        return atomicAdd(&offsets[partition_number], 1);
                      });

    // Scatter the rows into their partitions
    auto scattered = detail::scatter(t, scatter_map, t, stream, mr);

    return std::pair{std::move(scattered), std::move(partition_offsets)};
  }

  template <typename MapType, typename... Args>
  std::pair<std::unique_ptr<table>, std::vector<size_type>> operator()(Args&&...) const
    requires(not is_index_type<MapType>())
  {
    CUDF_FAIL("Unexpected, non-integral partition map.");
  }
};
}  // namespace

namespace detail {
namespace {

/**
 * @brief This hash function simply returns the input value cast to the
 * result_type of the functor.
 */
template <typename Key>
struct IdentityHash {
  using result_type                         = uint32_t;
  CUDF_HOST_DEVICE constexpr IdentityHash() = default;
  CUDF_HOST_DEVICE constexpr IdentityHash(uint32_t) {}

  template <typename return_type = result_type>
  CUDF_HOST_DEVICE constexpr return_type operator()(Key const& key) const
    requires(!cuda::std::is_arithmetic_v<Key>)
  {
    CUDF_UNREACHABLE("IdentityHash does not support this data type");
  }

  template <typename return_type = result_type>
  CUDF_HOST_DEVICE constexpr return_type operator()(Key const& key) const
    requires(cuda::std::is_arithmetic_v<Key>)
  {
    return static_cast<result_type>(key);
  }
};

template <template <typename> class hash_function>
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  table_view const& table_to_hash,
  int num_partitions,
  uint32_t seed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Return empty result if there are no partitions or nothing to hash
  if (num_partitions <= 0 || input.num_rows() == 0 || table_to_hash.num_columns() == 0) {
    return std::pair{empty_like(input), std::vector<size_type>(num_partitions + 1, 0)};
  }

  if constexpr (std::is_same_v<hash_function<void>, cudf::detail::IdentityHash<void>>) {
    for (auto const& c : table_to_hash) {
      CUDF_EXPECTS(is_numeric(c.type()), "IdentityHash does not support this data type");
    }
  }
  if (has_nested_nulls(table_to_hash)) {
    return hash_partition_table<hash_function, true>(
      input, table_to_hash, num_partitions, seed, stream, mr);
  } else {
    return hash_partition_table<hash_function, false>(
      input, table_to_hash, num_partitions, seed, stream, mr);
  }
}
}  // namespace

std::pair<std::unique_ptr<table>, std::vector<size_type>> partition(
  table_view const& t,
  column_view const& partition_map,
  size_type num_partitions,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(t.num_rows() == partition_map.size(),
               "Size mismatch between table and partition map.");
  CUDF_EXPECTS(not partition_map.has_nulls(), "Unexpected null values in partition_map.");

  if (num_partitions == 0 or t.num_rows() == 0) {
    // The output offsets vector must have size `num_partitions + 1` as per documentation.
    return std::pair{empty_like(t), std::vector<size_type>(num_partitions + 1, 0)};
  }

  return cudf::type_dispatcher(
    partition_map.type(), dispatch_map_type{}, t, partition_map, num_partitions, stream, mr);
}

std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  table_view const& keys,
  int num_partitions,
  hash_id hash_function,
  uint32_t seed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(keys.num_columns() == 0 || input.num_rows() == keys.num_rows(),
               "Input table and key table must have same number of rows, or key table should "
               "have no columns.",
               std::invalid_argument);
  switch (hash_function) {
    case (hash_id::HASH_IDENTITY):
      return hash_partition<detail::IdentityHash>(input, keys, num_partitions, seed, stream, mr);
    case (hash_id::HASH_MURMUR3):
      return hash_partition<cudf::hashing::detail::MurmurHash3_x86_32>(
        input, keys, num_partitions, seed, stream, mr);
    default: CUDF_FAIL("Unsupported hash function in hash_partition");
  }
}
}  // namespace detail

// Partition based on hash values
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  std::vector<size_type> const& columns_to_hash,
  int num_partitions,
  hash_id hash_function,
  uint32_t seed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::hash_partition(
    input, input.select(columns_to_hash), num_partitions, hash_function, seed, stream, mr);
}

std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  table_view const& keys,
  int num_partitions,
  hash_id hash_function,
  uint32_t seed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::hash_partition(input, keys, num_partitions, hash_function, seed, stream, mr);
}

// Partition based on an explicit partition map
std::pair<std::unique_ptr<table>, std::vector<size_type>> partition(
  table_view const& t,
  column_view const& partition_map,
  size_type num_partitions,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::partition(t, partition_map, num_partitions, stream, mr);
}

}  // namespace cudf
