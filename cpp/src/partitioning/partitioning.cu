/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/execution_env.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_histogram.cuh>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cudf {
namespace {
// Number of threads per block of the shared memory hash partition
constexpr size_type BLOCK_SIZE = 1024;
// `row_location` limits partitions and block-local partition offsets to 16 bits
constexpr size_type MAX_ROWS_PER_BLOCK = 1 << 16;

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
 * @brief The partition of a row and the row's offset in that partition of its thread block.
 */
struct alignas(4) row_location {
  cuda::std::uint16_t partition;  ///< Partition of the row
  cuda::std::uint16_t offset;     ///< Offset of the row in its partition of the thread block
};

/**
 * @brief Computes which partition each row of a device_table will belong to
 based on hashing each row, and applying a partition function to the hash value.
   Records the size of each partition for each thread block.
 *
 * Each thread block processes `blockDim.x * rows_per_thread` consecutive rows.
 *
 * @param[in] the_hasher The hasher whose rows will be partitioned
 * @param[in] num_rows The number of rows in the table
 * @param[in] num_partitions The number of partitions to divide the rows into
 * @param[in] rows_per_thread The number of rows processed by each thread
 * @param[in] the_partitioner The functor that maps a rows hash value to a
 partition number
 * @param[out] row_locations Array that holds the partition of each row and the
 row's offset in that partition of the thread block
 * @param[out] block_partition_sizes Array that holds the size of each partition
 for each block,
 * i.e., { {block0 partition0 size, block1 partition0 size, ...},
         {block0 partition1 size, block1 partition1 size, ...},
         ...
         {block0 partition(num_partitions-1) size, block1
 partition(num_partitions -1) size, ...} }
 */
template <class row_hasher_t, typename partitioner_type>
CUDF_KERNEL void __launch_bounds__(BLOCK_SIZE)
  compute_row_partition_numbers(row_hasher_t the_hasher,
                                size_type const num_rows,
                                size_type const num_partitions,
                                size_type const rows_per_thread,
                                partitioner_type const the_partitioner,
                                row_location* __restrict__ row_locations,
                                size_type* __restrict__ block_partition_sizes)
{
  // Accumulate histogram of the size of each partition in shared memory
  extern __shared__ size_type shared_partition_sizes[];

  // Initialize local histogram
  thread_index_type partition_number = threadIdx.x;
  while (partition_number < num_partitions) {
    shared_partition_sizes[partition_number] = 0;
    partition_number += blockDim.x;
  }

  __syncthreads();

  // Compute the hash value for each row, compute the partition to which the
  // hash value belongs and increment the shared memory counter for that
  // partition to get the row's offset in the partition
  auto const block_start =
    static_cast<thread_index_type>(blockIdx.x) * blockDim.x * rows_per_thread;
  for (size_type i = 0; i < rows_per_thread; ++i) {
    auto const tid = block_start + static_cast<thread_index_type>(i) * blockDim.x + threadIdx.x;
    if (tid >= num_rows) { break; }
    auto const row_number                = static_cast<size_type>(tid);
    hash_value_type const row_hash_value = the_hasher(row_number);

    size_type const partition_number = the_partitioner(row_hash_value);

    auto const row_offset = atomicAdd(&(shared_partition_sizes[partition_number]), size_type(1));
    row_locations[row_number] = {static_cast<cuda::std::uint16_t>(partition_number),
                                 static_cast<cuda::std::uint16_t>(row_offset)};
  }

  __syncthreads();

  // Flush shared memory histogram to global memory
  partition_number = threadIdx.x;
  while (partition_number < num_partitions) {
    // Record the size of this partition in this block
    auto const write_location             = partition_number * gridDim.x + blockIdx.x;
    block_partition_sizes[write_location] = shared_partition_sizes[partition_number];
    partition_number += blockDim.x;
  }
}

/**
 * @brief Describes a fixed-width column copied by `copy_block_partitions`.
 */
struct column_copy_descriptor {
  void const* input;       ///< Input data of the first row
  void* output;            ///< Output data
  size_type element_size;  ///< Size of each element in bytes
};

/**
 * @brief Returns the byte offset of the staged values in the shared memory of
 * `copy_block_partitions`, which follow the shared memory slot of each row and the shared and
 * global offsets of each partition.
 *
 * @param num_partitions The number of partitions
 * @param rows_per_block The number of rows processed by each thread block
 * @return The byte offset of the staged values
 */
[[nodiscard]] CUDF_HOST_DEVICE constexpr std::size_t staged_values_offset(size_type num_partitions,
                                                                          size_type rows_per_block)
{
  return cuda::round_up(
    (static_cast<std::size_t>(rows_per_block) + 2 * static_cast<std::size_t>(num_partitions) + 1) *
      sizeof(size_type),
    alignof(uint4));
}

/**
 * @brief Copies fixed-width columns to their partitions in the output and optionally computes the
 * gather map of the partitioned output.
 *
 * Each thread block stages its rows of a column in shared memory ordered by partition, then
 * copies each partition to the output with consecutive threads writing consecutive rows. The
 * shared memory slot of each row is computed once and reused for all columns.
 *
 * @param[in] columns The fixed-width columns to copy
 * @param[out] gather_map The input row of each output row, or empty if it is not needed
 * @param[in] num_rows The number of rows in the table
 * @param[in] num_partitions The number of partitions
 * @param[in] rows_per_thread The number of rows processed by each thread
 * @param[in] row_locations The partition of each row and the row's offset in that partition of
 * the thread block
 * @param[in] block_partition_offsets The output offset of each partition for each block, laid out
 * like the `block_partition_sizes` of `compute_row_partition_numbers`
 */
CUDF_KERNEL void __launch_bounds__(BLOCK_SIZE)
  copy_block_partitions(device_span<column_copy_descriptor const> columns,
                        device_span<size_type> gather_map,
                        size_type const num_rows,
                        size_type const num_partitions,
                        size_type const rows_per_thread,
                        row_location const* __restrict__ row_locations,
                        size_type const* __restrict__ block_partition_offsets)
{
  extern __shared__ __align__(alignof(uint4)) cuda::std::byte shared_memory[];
  auto const rows_per_block    = BLOCK_SIZE * rows_per_thread;
  auto row_slots               = reinterpret_cast<size_type*>(shared_memory);
  auto partition_offset_shared = row_slots + rows_per_block;
  auto partition_offset_global = partition_offset_shared + num_partitions + 1;
  auto block_output = shared_memory + staged_values_offset(num_partitions, rows_per_block);

  using BlockScan = cub::BlockScan<size_type, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Calculate the offset in shared memory of each partition in this thread block. The size of a
  // partition in this block is the distance to the next output offset.
  auto const num_offsets = static_cast<thread_index_type>(num_partitions) * gridDim.x;
  auto prefix_op         = cub::BlockScanRunningPrefixOp<size_type, cuda::std::plus<>>{0, {}};
  for (size_type tile = 0; tile < num_partitions; tile += BLOCK_SIZE) {
    auto const ipartition = tile + static_cast<size_type>(threadIdx.x);
    size_type partition_size{0};
    if (ipartition < num_partitions) {
      auto const index = static_cast<thread_index_type>(ipartition) * gridDim.x + blockIdx.x;
      auto const next_offset =
        index + 1 < num_offsets ? block_partition_offsets[index + 1] : num_rows;
      partition_offset_global[ipartition] = block_partition_offsets[index];
      partition_size                      = next_offset - block_partition_offsets[index];
    }
    size_type partition_offset;
    BlockScan(temp_storage).ExclusiveSum(partition_size, partition_offset, prefix_op);
    if (ipartition < num_partitions) { partition_offset_shared[ipartition] = partition_offset; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { partition_offset_shared[num_partitions] = prefix_op.running_total; }

  __syncthreads();

  // Calculate the shared memory slot of each row in this thread block
  auto const block_start = static_cast<thread_index_type>(blockIdx.x) * rows_per_block;
  for (size_type i = 0; i < rows_per_thread; ++i) {
    auto const row = block_start + i * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) { break; }
    auto const location = row_locations[row];
    row_slots[i * BLOCK_SIZE + threadIdx.x] =
      partition_offset_shared[location.partition] + location.offset;
  }

  __syncthreads();

  auto const copy_partitions = [&]<typename T>(auto input_iter, T* __restrict__ output_buf) {
    auto const block_values = reinterpret_cast<T*>(block_output);

    // Fetch the input data to shared memory
    for (size_type i = 0; i < rows_per_thread; ++i) {
      auto const row = block_start + i * BLOCK_SIZE + threadIdx.x;
      if (row >= num_rows) { break; }
      block_values[row_slots[i * BLOCK_SIZE + threadIdx.x]] = input_iter[row];
    }

    __syncthreads();

    // Copy data from shared memory to output using 32 threads for each partition
    for (size_type ipartition = threadIdx.x / cudf::detail::warp_size; ipartition < num_partitions;
         ipartition += BLOCK_SIZE / cudf::detail::warp_size) {
      size_type const nelements_partition =
        partition_offset_shared[ipartition + 1] - partition_offset_shared[ipartition];

      for (size_type row_offset = threadIdx.x % cudf::detail::warp_size;
           row_offset < nelements_partition;
           row_offset += cudf::detail::warp_size) {
        output_buf[partition_offset_global[ipartition] + row_offset] =
          block_values[partition_offset_shared[ipartition] + row_offset];
      }
    }

    __syncthreads();
  };

  auto const copy_column = [&]<typename T>(column_copy_descriptor const& column) {
    copy_partitions(static_cast<T const*>(column.input), static_cast<T*>(column.output));
  };
  for (auto const& column : columns) {
    switch (column.element_size) {
      case 1: copy_column.template operator()<cuda::std::uint8_t>(column); break;
      case 2: copy_column.template operator()<cuda::std::uint16_t>(column); break;
      case 4: copy_column.template operator()<cuda::std::uint32_t>(column); break;
      case 8: copy_column.template operator()<cuda::std::uint64_t>(column); break;
      case 16: copy_column.template operator()<uint4>(column); break;
      default: CUDF_UNREACHABLE("Unsupported element size");
    }
  }

  if (not gather_map.empty()) {
    copy_partitions(cuda::counting_iterator<size_type>{0}, gather_map.data());
  }
}

/**
 * @brief Allows `kernel` to use all of the shared memory of a thread block.
 *
 * @param kernel The kernel
 * @param device The device running `kernel`
 * @return The dynamic shared memory size available to `kernel` in bytes
 */
template <typename Kernel>
std::size_t enable_max_dynamic_shared_memory(Kernel kernel, cuda::device_ref device)
{
  cudaFuncAttributes attributes;
  CUDF_CUDA_TRY(cudaFuncGetAttributes(&attributes, kernel));
  auto const size =
    cuda::device_attributes::max_shared_memory_per_block_optin(device) - attributes.sharedSizeBytes;
  CUDF_CUDA_TRY(cudaFuncSetAttribute(
    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(size)));
  return size;
}

/**
 * @brief Returns the number of rows each thread processes in the shared memory hash partition,
 * or 0 if it does not apply.
 *
 * Each thread block stages its rows of one column at a time in shared memory, so the widest
 * staged element bounds the number of rows. Using as many rows as fit lengthens the runs of
 * consecutive output rows of each partition, as long as every multiprocessor still gets a thread
 * block. The shared memory hash partition does not apply when a thread block has fewer rows than
 * partitions, since the block partition offsets would then outnumber the rows.
 *
 * @param num_rows The number of rows
 * @param num_partitions The number of partitions
 * @param element_size The size in bytes of the widest staged element
 * @param shared_memory_size The dynamic shared memory size of `copy_block_partitions` in bytes
 * @param device The device running the hash partition
 * @return The number of rows per thread, or 0 if the shared memory hash partition does not apply
 */
[[nodiscard]] size_type shared_memory_rows_per_thread(size_type num_rows,
                                                      size_type num_partitions,
                                                      std::size_t element_size,
                                                      std::size_t shared_memory_size,
                                                      cuda::device_ref device)
{
  auto const partitions_size = staged_values_offset(num_partitions, 0);
  if (shared_memory_size <= partitions_size + alignof(uint4)) { return 0; }
  auto const rows_per_block = std::min<std::size_t>(
    (shared_memory_size - partitions_size - alignof(uint4)) / (element_size + sizeof(size_type)),
    MAX_ROWS_PER_BLOCK);
  auto const rows_per_thread = std::min(
    static_cast<size_type>(rows_per_block / BLOCK_SIZE),
    cuda::ceil_div(num_rows, cuda::device_attributes::multiprocessor_count(device) * BLOCK_SIZE));
  return BLOCK_SIZE * rows_per_thread >= num_partitions ? rows_per_thread : 0;
}

/**
 * @brief Hash-partition by staging the rows of each thread block in shared memory.
 *
 * Fixed-width columns are copied to their partitions through shared memory. Other columns and
 * the null masks of fixed-width columns are gathered with a gather map computed the same way.
 */
template <typename Hasher, typename Partitioner>
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition_table_shared_memory(
  table_view const& input,
  size_type num_partitions,
  Hasher hasher,
  Partitioner partitioner,
  size_type rows_per_thread,
  std::size_t element_size,
  cuda::device_ref device,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  auto const num_rows       = input.num_rows();
  auto const rows_per_block = BLOCK_SIZE * rows_per_thread;
  auto const grid_size      = cuda::ceil_div(num_rows, rows_per_block);
  auto const temp_mr        = mr.get_temporary_mr();

  // Fixed-width columns are copied through shared memory, other columns are gathered
  std::vector<size_type> fixed_width_indices;
  std::vector<size_type> gathered_indices;
  for (size_type i = 0; i < input.num_columns(); ++i) {
    (is_fixed_width(input.column(i).type()) ? fixed_width_indices : gathered_indices).push_back(i);
  }
  auto const fixed_width_input     = input.select(fixed_width_indices);
  auto const fixed_width_has_nulls = has_nulls(fixed_width_input);
  auto const needs_gather_map      = not gathered_indices.empty() or fixed_width_has_nulls;

  std::vector<std::unique_ptr<column>> output_cols(input.num_columns());
  std::vector<std::unique_ptr<column>> fixed_width_output;
  cuda::device_buffer<size_type> gather_map(
    stream, temp_mr, needs_gather_map ? num_rows : 0, cuda::no_init);
  auto partition_offsets =
    cudf::detail::make_pinned_vector_async<size_type>(num_partitions + 1, stream);
  partition_offsets.back() = num_rows;
  {
    cuda::device_buffer<row_location> row_locations(stream, temp_mr, num_rows, cuda::no_init);

    // Holds the size of each partition for each block, and then its output offset
    cuda::device_buffer<size_type> block_partition_offsets(
      stream, temp_mr, static_cast<std::size_t>(grid_size) * num_partitions, cuda::no_init);

    auto const kernel = compute_row_partition_numbers<Hasher, Partitioner>;
    enable_max_dynamic_shared_memory(kernel, device);
    kernel<<<grid_size, BLOCK_SIZE, num_partitions * sizeof(size_type), stream.get()>>>(
      hasher,
      num_rows,
      num_partitions,
      rows_per_thread,
      partitioner,
      row_locations.data(),
      block_partition_offsets.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    // Compute exclusive scan of all blocks' partition sizes in-place to determine
    // the starting point for each blocks portion of each partition in the output
    thrust::exclusive_scan(cudf::detail::exec_policy_nosync(stream, mr),
                           block_partition_offsets.begin(),
                           block_partition_offsets.end(),
                           block_partition_offsets.begin());

    // The offset of each partition in the output is the offset of its portion in block 0
    CUDF_CUDA_TRY(cudaMemcpy2DAsync(partition_offsets.data(),
                                    sizeof(size_type),
                                    block_partition_offsets.data(),
                                    grid_size * sizeof(size_type),
                                    sizeof(size_type),
                                    num_partitions,
                                    cudaMemcpyDefault,
                                    stream.get()));

    auto const shared_memory_size =
      staged_values_offset(num_partitions, rows_per_block) + rows_per_block * element_size;
    auto const launch_copy = [&](device_span<column_copy_descriptor const> columns,
                                 device_span<size_type> map) {
      copy_block_partitions<<<grid_size, BLOCK_SIZE, shared_memory_size, stream.get()>>>(
        columns,
        map,
        num_rows,
        num_partitions,
        rows_per_thread,
        row_locations.data(),
        block_partition_offsets.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    };

    // Gather the other columns before allocating the fixed-width output columns, so that the
    // temporary memory of the gather does not add to them
    if (needs_gather_map) { launch_copy({}, gather_map); }
    if (not gathered_indices.empty()) {
      auto gathered = cudf::detail::gather(input.select(gathered_indices),
                                           gather_map,
                                           out_of_bounds_policy::DONT_CHECK,
                                           negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           mr)
                        ->release();
      for (std::size_t i = 0; i < gathered_indices.size(); ++i) {
        output_cols[gathered_indices[i]] = std::move(gathered[i]);
      }
    }

    // Columns without nulls do not need their null masks gathered
    auto fixed_width_copies = cudf::detail::make_pinned_vector_async<column_copy_descriptor>(
      fixed_width_input.num_columns(), stream);
    for (size_type i = 0; i < fixed_width_input.num_columns(); ++i) {
      auto const& col = fixed_width_input.column(i);
      auto const mask_state =
        col.nullable() and not col.has_nulls() ? mask_state::ALL_VALID : mask_state::UNALLOCATED;
      auto output           = make_fixed_width_column(col.type(), num_rows, mask_state, stream, mr);
      auto const size       = static_cast<size_type>(size_of(col.type()));
      fixed_width_copies[i] = {
        static_cast<cuda::std::byte const*>(col.head()) + col.offset() * size,
        output->mutable_view().head(),
        size};
      fixed_width_output.push_back(std::move(output));
    }

    // Copy the columns in batches whose output for all concurrently running thread blocks fits
    // in the L2 cache, so that the output sectors shared by consecutive blocks are combined in
    // the L2 cache instead of in DRAM
    auto const rows_in_flight =
      static_cast<std::size_t>(
        std::min(grid_size, cuda::device_attributes::multiprocessor_count(device))) *
      rows_per_block;
    auto const batch_row_size =
      static_cast<std::size_t>(cuda::device_attributes::l2_cache_size(device)) / rows_in_flight;
    auto const d_fixed_width_copies =
      cudf::detail::make_device_buffer_async(fixed_width_copies, stream, mr.as_temporary());
    for (std::size_t begin = 0, end = 0; begin < fixed_width_copies.size(); begin = end) {
      for (std::size_t row_size = 0;
           end < fixed_width_copies.size() and
           (end == begin or row_size + fixed_width_copies[end].element_size <= batch_row_size);
           ++end) {
        row_size += fixed_width_copies[end].element_size;
      }
      launch_copy(
        device_span<column_copy_descriptor const>{d_fixed_width_copies}.subspan(begin, end - begin),
        {});
    }
  }

  if (fixed_width_has_nulls) {
    // Handle bitmask using gather to take advantage of ballot_sync
    detail::gather_bitmask(fixed_width_input,
                           gather_map.begin(),
                           fixed_width_output,
                           detail::gather_bitmask_op::DONT_CHECK,
                           stream,
                           mr);
  }
  for (std::size_t i = 0; i < fixed_width_indices.size(); ++i) {
    output_cols[fixed_width_indices[i]] = std::move(fixed_width_output[i]);
  }

  stream.sync();  // Async D2H copy must finish before returning host vec
  return std::pair{std::make_unique<table>(std::move(output_cols), num_rows),
                   std::vector<size_type>(partition_offsets.begin(), partition_offsets.end())};
}

/**
 * @brief Hash-partition using global memory when the shared memory hash partition does not apply.
 */
template <typename Hasher>
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition_table_global_memory(
  table_view const& input,
  size_type num_rows,
  size_type num_partitions,
  Hasher hasher,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_EXPECTS(num_partitions < std::numeric_limits<size_type>::max(),
               "num_partitions exceeds cudf's supported limit");
  auto const temp_mr = mr.get_temporary_mr();

  cuda::device_buffer<size_type> row_partition_numbers(stream, temp_mr, num_rows, cuda::no_init);

  // Compute partition number for each row
  if (is_power_two(num_partitions)) {
    auto const partitioner = bitwise_partitioner<hash_value_type>(num_partitions);
    thrust::transform(cudf::detail::exec_policy_nosync(stream, mr),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(num_rows),
                      row_partition_numbers.begin(),
                      [hasher, partitioner] __device__(size_type row) -> size_type {
                        return partitioner(hasher(row));
                      });
  } else {
    auto const partitioner = modulo_partitioner<hash_value_type>(num_partitions);
    thrust::transform(cudf::detail::exec_policy_nosync(stream, mr),
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
  auto histogram = cudf::detail::make_zeroed_device_buffer_async<size_type>(
    num_partitions + 1, stream, mr.as_temporary());
  CUDF_CUDA_TRY(cub::DeviceHistogram::HistogramEven(row_partition_numbers.data(),
                                                    histogram.data(),
                                                    num_partitions + 1,
                                                    size_type{0},
                                                    num_partitions,
                                                    num_rows,
                                                    cudf::detail::make_env(stream, mr)));

  // Exclusive scan on histogram to get partition offsets.
  // histogram has num_partitions+1 elements; after scan, histogram[num_partitions] = num_rows.
  thrust::exclusive_scan(cudf::detail::exec_policy_nosync(stream, mr),
                         histogram.begin(),
                         histogram.end(),
                         histogram.begin());

  // Copy partition offsets to pinned host memory asynchronously
  auto const pinned_offsets = cudf::detail::make_pinned_vector_async(histogram, stream);

  // Build scatter map: atomically increment partition offsets
  cuda::device_buffer<size_type> scatter_map(stream, temp_mr, num_rows, cuda::no_init);
  thrust::transform(
    cudf::detail::exec_policy_nosync(stream, mr),
    row_partition_numbers.begin(),
    row_partition_numbers.end(),
    scatter_map.begin(),
    [offsets = histogram.data()] __device__(size_type partition_number) -> size_type {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> ref(offsets[partition_number]);
      return ref.fetch_add(1, cuda::memory_order_relaxed);
    });

  // Scatter input rows into partitioned output
  auto output = detail::scatter(input, scatter_map, input, stream, mr.get_output_mr());

  stream.sync();  // Pinned async D2H copy must finish before returning host vec

  // Convert pinned host_vector to std::vector for the return type
  auto partition_offsets = std::vector<size_type>(pinned_offsets.begin(), pinned_offsets.end());

  return std::pair{std::move(output), std::move(partition_offsets)};
}

// NOTE hash_has_nulls must be true if table_to_hash has nulls
template <template <typename> class hash_function, bool hash_has_nulls>
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition_table(
  table_view const& input,
  table_view const& table_to_hash,
  size_type num_partitions,
  uint32_t seed,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  auto const num_rows = table_to_hash.num_rows();
  auto const temp_mr  = mr.get_temporary_mr();

  auto const row_hasher = detail::row::hash::row_hasher(table_to_hash, stream, temp_mr);
  auto const hasher =
    row_hasher.device_hasher<hash_function>(nullate::DYNAMIC{hash_has_nulls}, seed);

  // Shared memory stages fixed-width values and gather map entries
  auto element_size = sizeof(size_type);
  for (auto const& col : input) {
    if (is_fixed_width(col.type())) { element_size = std::max(element_size, size_of(col.type())); }
  }
  int dev;
  CUDF_CUDA_TRY(cudaGetDevice(&dev));
  auto const device = cuda::device_ref{dev};
  auto const rows_per_thread =
    shared_memory_rows_per_thread(num_rows,
                                  num_partitions,
                                  element_size,
                                  enable_max_dynamic_shared_memory(copy_block_partitions, device),
                                  device);
  if (rows_per_thread == 0) {
    return hash_partition_table_global_memory(input, num_rows, num_partitions, hasher, stream, mr);
  }

  auto const partition = [&](auto partitioner) {
    return hash_partition_table_shared_memory(input,
                                              num_partitions,
                                              hasher,
                                              partitioner,
                                              rows_per_thread,
                                              element_size,
                                              device,
                                              stream,
                                              mr);
  };
  // If the number of partitions is a power of two, we can compute the partition
  // number of each row more efficiently with bitwise operations
  if (is_power_two(num_partitions)) {
    return partition(bitwise_partitioner<hash_value_type>(num_partitions));
  }
  return partition(modulo_partitioner<hash_value_type>(num_partitions));
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
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const
    requires(is_index_type<MapType>())
  {
    // Build a histogram of the number of rows in each partition
    cuda::device_buffer<size_type> histogram(
      stream, cudf::get_current_device_resource_ref(), num_partitions + 1, cuda::no_init);
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
                                        stream.get());

    cuda::device_buffer<std::byte> temp_storage(
      stream, cudf::get_current_device_resource_ref(), temp_storage_bytes, cuda::no_init);

    cub::DeviceHistogram::HistogramEven(temp_storage.data(),
                                        temp_storage_bytes,
                                        partition_map.begin<MapType>(),
                                        histogram.data(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        partition_map.size(),
                                        stream.get());

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
    cuda::device_buffer<size_type> scatter_map(
      stream, cudf::get_current_device_resource_ref(), partition_map.size(), cuda::no_init);

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
  cuda::stream_ref stream,
  cudf::memory_resources mr)
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
  cuda::stream_ref stream,
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

std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(table_view const& input,
                                                                         table_view const& keys,
                                                                         int num_partitions,
                                                                         hash_id hash_function,
                                                                         uint32_t seed,
                                                                         cuda::stream_ref stream,
                                                                         cudf::memory_resources mr)
{
  CUDF_EXPECTS(
    keys.num_columns() == 0 || input.num_rows() == keys.num_rows(),
    "Input table and key table must have same number of rows, or key table should have no columns.",
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
  cuda::stream_ref stream,
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
  cuda::stream_ref stream,
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
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::partition(t, partition_map, num_partitions, stream, mr);
}

}  // namespace cudf
