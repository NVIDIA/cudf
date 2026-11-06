/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partition_fixed_width.cuh"
#include "partition_identity.cuh"
#include "partition_map_plan.cuh"
#include "partition_string_plan.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_histogram.cuh>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/stream>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace cudf {
namespace {
// Launch configuration for optimized hash partition
constexpr size_type OPTIMIZED_BLOCK_SIZE                     = 512;
constexpr size_type OPTIMIZED_ROWS_PER_THREAD                = 8;
constexpr size_type ELEMENTS_PER_THREAD                      = 2;
constexpr size_type THRESHOLD_FOR_OPTIMIZED_PARTITION_KERNEL = 1024;

// Launch configuration for fallback hash partition
constexpr size_type FALLBACK_BLOCK_SIZE      = 256;
constexpr size_type FALLBACK_ROWS_PER_THREAD = 1;

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
 * @brief Computes which partition each row of a device_table will belong to
 based on hashing each row, and applying a partition function to the hash value.
   Records the size of each partition for each thread block as well as the
 global size of each partition across all thread blocks.
 *
 * @param[in] the_hasher The hasher whose rows will be partitioned
 * @param[in] num_rows The number of rows in the table
 * @param[in] num_partitions The number of partitions to divide the rows into
 * @param[in] the_partitioner The functor that maps a rows hash value to a
 partition number
 * @param[out] row_partition_numbers Array that holds which partition each row
 belongs to
 * @param[out] row_partition_offset Array that holds the offset of each row in
 its partition of
 * the thread block
 * @param[out] block_partition_sizes Array that holds the size of each partition
 for each block,
 * i.e., { {block0 partition0 size, block1 partition0 size, ...},
         {block0 partition1 size, block1 partition1 size, ...},
         ...
         {block0 partition(num_partitions-1) size, block1
 partition(num_partitions -1) size, ...} }
 * @param[out] global_partition_sizes The number of rows in each partition.
 */
template <class row_hasher_t, typename partitioner_type>
CUDF_KERNEL void compute_row_partition_numbers(row_hasher_t the_hasher,
                                               size_type const num_rows,
                                               size_type const num_partitions,
                                               partitioner_type const the_partitioner,
                                               size_type* __restrict__ row_partition_numbers,
                                               size_type* __restrict__ row_partition_offset,
                                               size_type* __restrict__ block_partition_sizes,
                                               size_type* __restrict__ global_partition_sizes)
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
  while (tid < num_rows) {
    auto const row_number                = static_cast<size_type>(tid);
    hash_value_type const row_hash_value = the_hasher(row_number);

    size_type const partition_number = the_partitioner(row_hash_value);

    row_partition_numbers[row_number] = partition_number;

    row_partition_offset[row_number] =
      atomicAdd(&(shared_partition_sizes[partition_number]), size_type(1));

    tid += stride;
  }

  __syncthreads();

  // Flush shared memory histogram to global memory
  partition_number = threadIdx.x;
  while (partition_number < num_partitions) {
    size_type const block_partition_size = shared_partition_sizes[partition_number];

    // Update global size of each partition
    atomicAdd(&global_partition_sizes[partition_number], block_partition_size);

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
 * @param row_partition_numbers The array that records the partition number for
 each row
 * @param num_rows The number of rows
 * @param num_partitions THe number of partitions
 * @param[out] block_partition_offsets Array that holds the offset of each
 partition for each thread block,
 * i.e., { {block0 partition0 offset, block1 partition0 offset, ...},
         {block0 partition1 offset, block1 partition1 offset, ...},
         ...
         {block0 partition(num_partitions-1) offset, block1
 partition(num_partitions -1) offset, ...} }
 */
CUDF_KERNEL void compute_row_output_locations(size_type* __restrict__ row_partition_numbers,
                                              size_type const num_rows,
                                              size_type const num_partitions,
                                              size_type* __restrict__ block_partition_offsets)
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

  // Get each row's partition number, and get it's output location by
  // incrementing block's offset counter for that partition number
  // and store the row's output location in-place
  while (tid < num_rows) {
    auto const row_number = static_cast<size_type>(tid);
    // Get partition number of this row
    size_type const partition_number = row_partition_numbers[row_number];

    // Get output location based on partition number by incrementing the
    // corresponding partition offset for this block
    size_type const row_output_location =
      atomicAdd(&(shared_partition_offsets[partition_number]), size_type(1));

    // Store the row's output location in-place
    row_partition_numbers[row_number] = row_output_location;

    tid += stride;
  }
}

/**
 * @brief Move one column from the input table to the hashed table.
 *
 * @param[in] input_iter Iterator over the input column data
 * @param[out] output_buf Preallocated data buffer of the column in the output
 * table
 * @param[in] num_rows The number of rows in each column
 * @param[in] num_partitions The number of partitions to divide the rows into
 * @param[in] row_partition_numbers Array that holds which partition each row
 * belongs to
 * @param[in] row_partition_offset Array that holds the offset of each row in
 * its partition of the thread block.
 * @param[in] block_partition_sizes Array that holds the size of each partition
 * for each block
 * @param[in] scanned_block_partition_sizes The scan of block_partition_sizes
 */
template <typename InputIter, typename DataType>
CUDF_KERNEL void copy_block_partitions(InputIter input_iter,
                                       DataType* __restrict__ output_buf,
                                       size_type const num_rows,
                                       size_type const num_partitions,
                                       size_type const* __restrict__ row_partition_numbers,
                                       size_type const* __restrict__ row_partition_offset,
                                       size_type const* __restrict__ block_partition_sizes,
                                       size_type const* __restrict__ scanned_block_partition_sizes)
{
  extern __shared__ char shared_memory[];
  auto block_output = reinterpret_cast<DataType*>(shared_memory);
  auto partition_offset_shared =
    reinterpret_cast<size_type*>(block_output + OPTIMIZED_BLOCK_SIZE * OPTIMIZED_ROWS_PER_THREAD);
  auto partition_offset_global = partition_offset_shared + num_partitions + 1;

  using BlockScan = cub::BlockScan<size_type, OPTIMIZED_BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // use ELEMENTS_PER_THREAD=2 to support up to 1024 partitions
  size_type temp_histo[ELEMENTS_PER_THREAD];

  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    if (ELEMENTS_PER_THREAD * threadIdx.x + i < num_partitions) {
      temp_histo[i] =
        block_partition_sizes[blockIdx.x + (ELEMENTS_PER_THREAD * threadIdx.x + i) * gridDim.x];
    } else {
      temp_histo[i] = 0;
    }
  }

  __syncthreads();

  BlockScan(temp_storage).InclusiveSum(temp_histo, temp_histo);

  __syncthreads();

  if (threadIdx.x == 0) { partition_offset_shared[0] = 0; }

  // Calculate the offset in shared memory of each partition in this thread
  // block
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    if (ELEMENTS_PER_THREAD * threadIdx.x + i < num_partitions) {
      partition_offset_shared[ELEMENTS_PER_THREAD * threadIdx.x + i + 1] = temp_histo[i];
    }
  }

  // Fetch the offset in the output buffer of each partition in this thread
  // block
  for (thread_index_type ipartition = threadIdx.x; ipartition < num_partitions;
       ipartition += blockDim.x) {
    partition_offset_global[ipartition] =
      scanned_block_partition_sizes[ipartition * gridDim.x + blockIdx.x];
  }

  __syncthreads();

  // Fetch the input data to shared memory
  for (auto tid = cudf::detail::grid_1d::global_thread_id(); tid < num_rows;
       tid += cudf::detail::grid_1d::grid_stride()) {
    auto const row_number      = static_cast<size_type>(tid);
    size_type const ipartition = row_partition_numbers[row_number];

    block_output[partition_offset_shared[ipartition] + row_partition_offset[row_number]] =
      input_iter[row_number];
  }

  __syncthreads();

  // Copy data from shared memory to output using 32 threads for each partition
  constexpr int nthreads_partition = 32;
  static_assert(OPTIMIZED_BLOCK_SIZE % nthreads_partition == 0,
                "BLOCK_SIZE must be divisible by number of threads");

  for (size_type ipartition = threadIdx.x / nthreads_partition; ipartition < num_partitions;
       ipartition += OPTIMIZED_BLOCK_SIZE / nthreads_partition) {
    size_type const nelements_partition =
      partition_offset_shared[ipartition + 1] - partition_offset_shared[ipartition];

    for (size_type row_offset = threadIdx.x % nthreads_partition; row_offset < nelements_partition;
         row_offset += nthreads_partition) {
      output_buf[partition_offset_global[ipartition] + row_offset] =
        block_output[partition_offset_shared[ipartition] + row_offset];
    }
  }
}

template <typename InputIter, typename OutputIter>
void copy_block_partitions_impl(InputIter const input,
                                OutputIter output,
                                size_type num_rows,
                                size_type num_partitions,
                                size_type const* row_partition_numbers,
                                size_type const* row_partition_offset,
                                size_type const* block_partition_sizes,
                                size_type const* scanned_block_partition_sizes,
                                size_type grid_size,
                                cuda::stream_ref stream)
{
  // We need 3 chunks of shared memory:
  // 1. BLOCK_SIZE * ROWS_PER_THREAD elements of size_type for copying to output
  // 2. num_partitions + 1 elements of size_type for per-block partition offsets
  // 3. num_partitions + 1 elements of size_type for global partition offsets
  int const smem = OPTIMIZED_BLOCK_SIZE * OPTIMIZED_ROWS_PER_THREAD * sizeof(*output) +
                   (num_partitions + 1) * sizeof(size_type) * 2;

  copy_block_partitions<<<grid_size, OPTIMIZED_BLOCK_SIZE, smem, stream.get()>>>(
    input,
    output,
    num_rows,
    num_partitions,
    row_partition_numbers,
    row_partition_offset,
    block_partition_sizes,
    scanned_block_partition_sizes);
  CUDF_CUDA_TRY(cudaGetLastError());
}

rmm::device_uvector<size_type> compute_gather_map(size_type num_rows,
                                                  size_type num_partitions,
                                                  size_type const* row_partition_numbers,
                                                  size_type const* row_partition_offset,
                                                  size_type const* block_partition_sizes,
                                                  size_type const* scanned_block_partition_sizes,
                                                  size_type grid_size,
                                                  cuda::stream_ref stream)
{
  auto sequence = cuda::counting_iterator<cudf::size_type>{0};
  rmm::device_uvector<size_type> gather_map(num_rows, stream);

  copy_block_partitions_impl(sequence,
                             gather_map.begin(),
                             num_rows,
                             num_partitions,
                             row_partition_numbers,
                             row_partition_offset,
                             block_partition_sizes,
                             scanned_block_partition_sizes,
                             grid_size,
                             stream);

  return gather_map;
}

struct copy_block_partitions_dispatcher {
  template <typename DataType>
  constexpr static bool is_copy_block_supported()
  {
    // The shared-memory used for fixed-width types in the copy_block_partitions_impl function
    // will be too large for any DataType greater than int64_t.
    return is_fixed_width<DataType>() && (sizeof(DataType) <= sizeof(int64_t));
  }

  template <typename DataType, CUDF_ENABLE_IF(is_copy_block_supported<DataType>())>
  std::unique_ptr<column> operator()(column_view const& input,
                                     size_type const num_partitions,
                                     size_type const* row_partition_numbers,
                                     size_type const* row_partition_offset,
                                     size_type const* block_partition_sizes,
                                     size_type const* scanned_block_partition_sizes,
                                     size_type grid_size,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    rmm::device_buffer output(input.size() * sizeof(DataType), stream, mr);

    copy_block_partitions_impl(input.data<DataType>(),
                               static_cast<DataType*>(output.data()),
                               input.size(),
                               num_partitions,
                               row_partition_numbers,
                               row_partition_offset,
                               block_partition_sizes,
                               scanned_block_partition_sizes,
                               grid_size,
                               stream);

    return std::make_unique<column>(input.type(),
                                    input.size(),
                                    std::move(output),
                                    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                    0);
  }

  template <typename DataType, CUDF_ENABLE_IF(not is_copy_block_supported<DataType>())>
  std::unique_ptr<column> operator()(column_view const& input,
                                     size_type const num_partitions,
                                     size_type const* row_partition_numbers,
                                     size_type const* row_partition_offset,
                                     size_type const* block_partition_sizes,
                                     size_type const* scanned_block_partition_sizes,
                                     size_type grid_size,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    // Use move_to_output_buffer to create an equivalent gather map
    auto gather_map = compute_gather_map(input.size(),
                                         num_partitions,
                                         row_partition_numbers,
                                         row_partition_offset,
                                         block_partition_sizes,
                                         scanned_block_partition_sizes,
                                         grid_size,
                                         stream);

    auto gather_table = cudf::detail::gather(cudf::table_view({input}),
                                             gather_map,
                                             out_of_bounds_policy::DONT_CHECK,
                                             cudf::negative_index_policy::NOT_ALLOWED,
                                             stream,
                                             mr);
    return std::move(gather_table->release().front());
  }
};

/**
 * @brief Hash-partition using global memory when partition count exceeds shared memory capacity.
 */
template <typename Hasher>
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition_table_global_memory(
  table_view const& input,
  size_type num_rows,
  size_type num_partitions,
  Hasher hasher,
  cuda::stream_ref stream,
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
                                        stream.get());
    cuda::device_buffer<std::byte> temp_storage(
      stream, cudf::get_current_device_resource_ref(), temp_storage_bytes, cuda::no_init);
    cub::DeviceHistogram::HistogramEven(temp_storage.data(),
                                        temp_storage_bytes,
                                        row_partition_numbers.data(),
                                        histogram.data(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        num_rows,
                                        stream.get());
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
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = table_to_hash.num_rows();
  auto const temp_mr  = cudf::get_current_device_resource_ref();

  auto const row_hasher = detail::row::hash::row_hasher(table_to_hash, stream, temp_mr);
  auto const hasher =
    row_hasher.device_hasher<hash_function>(nullate::DYNAMIC{hash_has_nulls}, seed);

  // Check whether the per-block shared memory histograms fit in shared memory
  int dev;
  CUDF_CUDA_TRY(cudaGetDevice(&dev));
  auto const fits_in_shared_memory =
    static_cast<std::size_t>(num_partitions) <
    cuda::device_attributes::max_shared_memory_per_block(cuda::device_ref{dev}) / sizeof(size_type);

  if (!fits_in_shared_memory) {
    return hash_partition_table_global_memory(input, num_rows, num_partitions, hasher, stream, mr);
  }

  bool const use_optimization{num_partitions <= THRESHOLD_FOR_OPTIMIZED_PARTITION_KERNEL};
  auto const block_size = use_optimization ? OPTIMIZED_BLOCK_SIZE : FALLBACK_BLOCK_SIZE;
  auto const rows_per_thread =
    use_optimization ? OPTIMIZED_ROWS_PER_THREAD : FALLBACK_ROWS_PER_THREAD;
  auto const rows_per_block = block_size * rows_per_thread;

  std::size_t const grid_size = util::div_rounding_up_safe(num_rows, rows_per_block);

  // Allocate array to hold which partition each row belongs to
  auto row_partition_numbers = rmm::device_uvector<size_type>(num_rows, stream, temp_mr);

  // Array to hold the size of each partition computed by each block
  //  i.e., { {block0 partition0 size, block1 partition0 size, ...},
  //          {block0 partition1 size, block1 partition1 size, ...},
  //          ...
  //          {block0 partition(num_partitions-1) size, block1
  //          partition(num_partitions -1) size, ...} }
  auto block_partition_sizes =
    rmm::device_uvector<size_type>(grid_size * num_partitions, stream, temp_mr);

  auto scanned_block_partition_sizes =
    rmm::device_uvector<size_type>(grid_size * num_partitions, stream, temp_mr);

  // Holds the total number of rows in each partition
  auto global_partition_sizes =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(num_partitions, stream, temp_mr);

  auto row_partition_offset =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(num_rows, stream, temp_mr);

  // If the number of partitions is a power of two, we can compute the partition
  // number of each row more efficiently with bitwise operations
  if (is_power_two(num_partitions)) {
    // Determines how the mapping between hash value and partition number is
    // computed
    using partitioner_type = bitwise_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and
    // performing a partitioning operator on the hash value. Also computes the
    // number of rows in each partition both for each thread block as well as
    // across all blocks
    compute_row_partition_numbers<<<grid_size,
                                    block_size,
                                    num_partitions * sizeof(size_type),
                                    stream.get()>>>(hasher,
                                                    num_rows,
                                                    num_partitions,
                                                    partitioner_type(num_partitions),
                                                    row_partition_numbers.data(),
                                                    row_partition_offset.data(),
                                                    block_partition_sizes.data(),
                                                    global_partition_sizes.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    // Determines how the mapping between hash value and partition number is
    // computed
    using partitioner_type = modulo_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and
    // performing a partitioning operator on the hash value. Also computes the
    // number of rows in each partition both for each thread block as well as
    // across all blocks
    compute_row_partition_numbers<<<grid_size,
                                    block_size,
                                    num_partitions * sizeof(size_type),
                                    stream.get()>>>(hasher,
                                                    num_rows,
                                                    num_partitions,
                                                    partitioner_type(num_partitions),
                                                    row_partition_numbers.data(),
                                                    row_partition_offset.data(),
                                                    block_partition_sizes.data(),
                                                    global_partition_sizes.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // Compute exclusive scan of all blocks' partition sizes in-place to determine
  // the starting point for each blocks portion of each partition in the output
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         block_partition_sizes.begin(),
                         block_partition_sizes.end(),
                         scanned_block_partition_sizes.data());

  // Compute exclusive scan of size of each partition to determine offset
  // location of each partition in final output.
  // TODO This can be done independently on a separate stream
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         global_partition_sizes.begin(),
                         global_partition_sizes.end(),
                         global_partition_sizes.begin());

  // Copy the result of the exclusive scan to the output offsets array
  // to indicate the starting point for each partition in the output
  auto partition_offsets = cudf::detail::make_std_vector(global_partition_sizes, stream);
  // Add the total row count as the last offset to make the vector num_partitions + 1 in size
  partition_offsets.push_back(num_rows);

  // When the number of partitions is less than a threshold, we can apply an
  // optimization using shared memory to copy values to the output buffer.
  // Otherwise, fallback to using scatter.
  if (use_optimization) {
    std::vector<std::unique_ptr<column>> output_cols(input.num_columns());

    // Copy input to output by partition per column
    std::transform(input.begin(), input.end(), output_cols.begin(), [&](auto const& col) {
      return cudf::type_dispatcher<dispatch_storage_type>(col.type(),
                                                          copy_block_partitions_dispatcher{},
                                                          col,
                                                          num_partitions,
                                                          row_partition_numbers.data(),
                                                          row_partition_offset.data(),
                                                          block_partition_sizes.data(),
                                                          scanned_block_partition_sizes.data(),
                                                          grid_size,
                                                          stream,
                                                          mr);
    });

    if (has_nested_nulls(input)) {
      // Use copy_block_partitions to compute a gather map
      auto gather_map = compute_gather_map(num_rows,
                                           num_partitions,
                                           row_partition_numbers.data(),
                                           row_partition_offset.data(),
                                           block_partition_sizes.data(),
                                           scanned_block_partition_sizes.data(),
                                           grid_size,
                                           stream);

      // Handle bitmask using gather to take advantage of ballot_sync
      detail::gather_bitmask(
        input, gather_map.begin(), output_cols, detail::gather_bitmask_op::DONT_CHECK, stream, mr);
    }

    stream.sync();  // Async D2H copy must finish before returning host vec
    return std::pair{std::make_unique<table>(std::move(output_cols), num_rows),
                     std::move(partition_offsets)};
  } else {
    // Compute a scatter map from input to output such that the output rows are
    // sorted by partition number
    auto row_output_locations{row_partition_numbers.data()};
    auto scanned_block_partition_sizes_ptr{scanned_block_partition_sizes.data()};
    compute_row_output_locations<<<grid_size,
                                   block_size,
                                   num_partitions * sizeof(size_type),
                                   stream.get()>>>(
      row_output_locations, num_rows, num_partitions, scanned_block_partition_sizes_ptr);
    CUDF_CUDA_TRY(cudaGetLastError());

    // Use the resulting scatter map to materialize the output
    auto output = detail::scatter(input, row_partition_numbers, input, stream, mr);

    stream.sync();  // Async D2H copy must finish before returning host vec
    return std::pair{std::move(output), std::move(partition_offsets)};
  }
}

constexpr size_type MAP_PARTITION_BLOCK_SIZE       = 256;
constexpr size_type MAP_PARTITION_ITEMS_PER_THREAD = 4;
constexpr size_type MAP_PARTITION_TILE_SIZE =
  MAP_PARTITION_BLOCK_SIZE * MAP_PARTITION_ITEMS_PER_THREAD;
constexpr size_type MAP_PARTITION_MAX_BINS     = 256;
constexpr size_type MAP_PARTITION_WARP_SIZE    = 32;
constexpr size_type MAP_PARTITION_NUM_WARPS    = MAP_PARTITION_BLOCK_SIZE / MAP_PARTITION_WARP_SIZE;
constexpr size_type MAP_PARTITION_GATHER_TILES = 2;
static_assert(MAP_PARTITION_MAX_BINS <= MAP_PARTITION_BLOCK_SIZE,
              "Partition-base initialization requires one thread per partition");

/**
 * @brief Build warp-local histograms and optionally compute each row's warp-local rank.
 *
 * Each warp owns a histogram. Matching lanes elect one writer for each bin, avoiding
 * atomic updates even when many rows belong to the same partition. Contiguous tiles
 * per warp keep the resulting gather map in input order within each partition.
 */
template <bool ComputeRanks, typename MapType>
__device__ void partition_map_warp_histogram(
  MapType const* partition_map,
  size_type num_rows,
  size_type num_partitions,
  cuda::std::size_t tile_begin,
  size_type (&histogram)[MAP_PARTITION_NUM_WARPS][MAP_PARTITION_MAX_BINS],
  size_type (&ranks)[MAP_PARTITION_ITEMS_PER_THREAD],
  size_type (&partitions)[MAP_PARTITION_ITEMS_PER_THREAD])
{
  constexpr auto warp_size = MAP_PARTITION_WARP_SIZE;
  auto const lane          = threadIdx.x % warp_size;
  auto const warp          = threadIdx.x / warp_size;
  for (auto p = lane; p < num_partitions; p += warp_size) {
    histogram[warp][p] = 0;
  }
  __syncwarp();

#pragma unroll
  for (size_type item = 0; item < MAP_PARTITION_ITEMS_PER_THREAD; ++item) {
    auto const row =
      tile_begin + warp * warp_size * MAP_PARTITION_ITEMS_PER_THREAD + item * warp_size + lane;
    auto const valid  = row < num_rows;
    auto const active = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      auto const partition = static_cast<size_type>(partition_map[row]);
      auto const peers     = __match_any_sync(active, partition);
      auto const base      = histogram[warp][partition];
      if constexpr (ComputeRanks) {
        partitions[item] = partition;
        ranks[item]      = base + cuda::std::popcount(peers & ((1u << lane) - 1u));
      }
      // All readers of a bin finish before its elected writer updates it.
      __syncwarp(active);
      if (lane == __ffs(peers) - 1) {
        histogram[warp][partition] = base + cuda::std::popcount(peers);
      }
    }
    __syncwarp();
  }
  __syncthreads();
}

template <typename MapType>
CUDF_KERNEL void partition_map_local_ranks(MapType const* partition_map,
                                           size_type num_rows,
                                           size_type num_partitions,
                                           uint16_t* local_ranks,
                                           size_type* block_counts)
{
  constexpr size_type warp_size = 32;
  constexpr size_type num_warps = MAP_PARTITION_BLOCK_SIZE / warp_size;
  __shared__ size_type histogram[num_warps][MAP_PARTITION_MAX_BINS];
  auto const lane = threadIdx.x % warp_size;
  auto const warp = threadIdx.x / warp_size;
  size_type ranks[MAP_PARTITION_ITEMS_PER_THREAD];
  size_type partitions[MAP_PARTITION_ITEMS_PER_THREAD];
  partition_map_warp_histogram<true>(
    partition_map,
    num_rows,
    num_partitions,
    static_cast<cuda::std::size_t>(blockIdx.x) * MAP_PARTITION_TILE_SIZE,
    histogram,
    ranks,
    partitions);

  for (auto partition = threadIdx.x; partition < num_partitions;
       partition += MAP_PARTITION_BLOCK_SIZE) {
    size_type total = 0;
#pragma unroll
    for (size_type w = 0; w < num_warps; ++w) {
      auto const count        = histogram[w][partition];
      histogram[w][partition] = total;
      total += count;
    }
    block_counts[static_cast<cuda::std::size_t>(partition) * gridDim.x + blockIdx.x] = total;
  }
  __syncthreads();

#pragma unroll
  for (size_type item = 0; item < MAP_PARTITION_ITEMS_PER_THREAD; ++item) {
    auto const row = static_cast<cuda::std::size_t>(blockIdx.x) * MAP_PARTITION_TILE_SIZE +
                     warp * warp_size * MAP_PARTITION_ITEMS_PER_THREAD + item * warp_size + lane;
    if (row < num_rows) {
      // A rank is at most 1023, so two bytes per row are sufficient.
      local_ranks[row] = static_cast<uint16_t>(histogram[warp][partitions[item]] + ranks[item]);
    }
  }
}

/**
 * @brief Count two consecutive microtiles without materializing per-row ranks.
 */
template <typename MapType>
CUDF_KERNEL void partition_map_gather_counts(MapType const* partition_map,
                                             size_type num_rows,
                                             size_type num_partitions,
                                             size_type* block_counts)
{
  __shared__ size_type histogram[MAP_PARTITION_NUM_WARPS][MAP_PARTITION_MAX_BINS];
  size_type total = 0;
  for (size_type tile = 0; tile < MAP_PARTITION_GATHER_TILES; ++tile) {
    size_type ranks[MAP_PARTITION_ITEMS_PER_THREAD];
    size_type partitions[MAP_PARTITION_ITEMS_PER_THREAD];
    auto const tile_begin =
      (static_cast<cuda::std::size_t>(blockIdx.x) * MAP_PARTITION_GATHER_TILES + tile) *
      MAP_PARTITION_TILE_SIZE;
    partition_map_warp_histogram<false>(
      partition_map, num_rows, num_partitions, tile_begin, histogram, ranks, partitions);
    if (threadIdx.x < num_partitions) {
#pragma unroll
      for (size_type warp = 0; warp < MAP_PARTITION_NUM_WARPS; ++warp) {
        total += histogram[warp][threadIdx.x];
      }
    }
    // Every histogram reader must finish before the next microtile resets it.
    __syncthreads();
  }
  if (threadIdx.x < num_partitions) {
    block_counts[static_cast<cuda::std::size_t>(threadIdx.x) * gridDim.x + blockIdx.x] = total;
  }
}

/**
 * @brief Recompute microtile ranks and emit an output-to-input gather map.
 *
 * Each block carries its partition bases from the first microtile into the second,
 * preserving source locality without a global per-row rank buffer.
 */
template <typename MapType>
CUDF_KERNEL void partition_map_gather_scatter(MapType const* partition_map,
                                              size_type const* block_offsets,
                                              size_type num_rows,
                                              size_type num_partitions,
                                              size_type* gather_map,
                                              size_type* partition_offsets)
{
  __shared__ size_type histogram[MAP_PARTITION_NUM_WARPS][MAP_PARTITION_MAX_BINS];
  __shared__ size_type partition_bases[MAP_PARTITION_MAX_BINS];
  if (threadIdx.x < num_partitions) {
    auto const partition_base    = static_cast<cuda::std::size_t>(threadIdx.x) * gridDim.x;
    partition_bases[threadIdx.x] = block_offsets[partition_base + blockIdx.x];
    if (blockIdx.x == 0) { partition_offsets[threadIdx.x] = block_offsets[partition_base]; }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) { partition_offsets[num_partitions] = num_rows; }
  __syncthreads();

  auto const lane = threadIdx.x % MAP_PARTITION_WARP_SIZE;
  auto const warp = threadIdx.x / MAP_PARTITION_WARP_SIZE;
  for (size_type tile = 0; tile < MAP_PARTITION_GATHER_TILES; ++tile) {
    size_type ranks[MAP_PARTITION_ITEMS_PER_THREAD];
    size_type partitions[MAP_PARTITION_ITEMS_PER_THREAD];
    auto const tile_begin =
      (static_cast<cuda::std::size_t>(blockIdx.x) * MAP_PARTITION_GATHER_TILES + tile) *
      MAP_PARTITION_TILE_SIZE;
    partition_map_warp_histogram<true>(
      partition_map, num_rows, num_partitions, tile_begin, histogram, ranks, partitions);

    size_type next_base = 0;
    if (threadIdx.x < num_partitions) {
      size_type total = 0;
#pragma unroll
      for (size_type w = 0; w < MAP_PARTITION_NUM_WARPS; ++w) {
        auto const count          = histogram[w][threadIdx.x];
        histogram[w][threadIdx.x] = total;
        total += count;
      }
      next_base = partition_bases[threadIdx.x] + total;
    }
    __syncthreads();

#pragma unroll
    for (size_type item = 0; item < MAP_PARTITION_ITEMS_PER_THREAD; ++item) {
      auto const row = tile_begin +
                       warp * MAP_PARTITION_WARP_SIZE * MAP_PARTITION_ITEMS_PER_THREAD +
                       item * MAP_PARTITION_WARP_SIZE + lane;
      if (row < num_rows) {
        auto const partition = partitions[item];
        auto const destination =
          partition_bases[partition] + histogram[warp][partition] + ranks[item];
        gather_map[destination] = static_cast<size_type>(row);
      }
    }
    __syncthreads();
    if (threadIdx.x < num_partitions) { partition_bases[threadIdx.x] = next_base; }
    __syncthreads();
  }
}

template <typename MapType>
std::vector<size_type> make_partition_gather_map(column_view const& partition_map,
                                                 size_type num_partitions,
                                                 size_type* gather_map,
                                                 cuda::stream_ref stream)
{
  auto const num_rows = partition_map.size();
  auto const num_blocks =
    (num_rows - 1) / (MAP_PARTITION_GATHER_TILES * MAP_PARTITION_TILE_SIZE) + 1;
  auto const temp_mr = cudf::get_current_device_resource_ref();
  cuda::device_buffer<size_type> block_offsets(
    stream, temp_mr, static_cast<std::size_t>(num_blocks) * num_partitions, cuda::no_init);
  cuda::device_buffer<size_type> offsets(stream, temp_mr, num_partitions + 1, cuda::no_init);
  partition_map_gather_counts<<<num_blocks, MAP_PARTITION_BLOCK_SIZE, 0, stream.get()>>>(
    partition_map.begin<MapType>(), num_rows, num_partitions, block_offsets.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         block_offsets.begin(),
                         block_offsets.end(),
                         block_offsets.begin());
  partition_map_gather_scatter<<<num_blocks, MAP_PARTITION_BLOCK_SIZE, 0, stream.get()>>>(
    partition_map.begin<MapType>(),
    block_offsets.data(),
    num_rows,
    num_partitions,
    gather_map,
    offsets.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  auto const pinned_offsets = cudf::detail::make_pinned_vector(offsets, stream);
  return {pinned_offsets.begin(), pinned_offsets.end()};
}

template <typename MapType, typename InputIterator, typename OutputIterator>
CUDF_KERNEL void partition_map_scatter(MapType const* partition_map,
                                       uint16_t const* local_ranks,
                                       size_type const* block_offsets,
                                       size_type num_rows,
                                       size_type num_blocks,
                                       size_type num_partitions,
                                       InputIterator input,
                                       OutputIterator output,
                                       size_type* partition_offsets)
{
  __shared__ size_type partition_bases[MAP_PARTITION_MAX_BINS];
  if (threadIdx.x < num_partitions) {
    auto const partition_base    = static_cast<cuda::std::size_t>(threadIdx.x) * num_blocks;
    partition_bases[threadIdx.x] = block_offsets[partition_base + blockIdx.x];
    if (blockIdx.x == 0) { partition_offsets[threadIdx.x] = block_offsets[partition_base]; }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) { partition_offsets[num_partitions] = num_rows; }
  __syncthreads();

  // Reuse each block's partition bases across all rows in its tile.
#pragma unroll
  for (size_type item = 0; item < MAP_PARTITION_ITEMS_PER_THREAD; ++item) {
    auto const row = static_cast<cuda::std::size_t>(blockIdx.x) * MAP_PARTITION_TILE_SIZE +
                     item * MAP_PARTITION_BLOCK_SIZE + threadIdx.x;
    if (row < num_rows) {
      auto const partition   = static_cast<size_type>(partition_map[row]);
      auto const destination = partition_bases[partition] + local_ranks[row];
      output[destination]    = input[row];
    }
  }
}

template <typename MapType, typename InputIterator, typename OutputIterator>
std::vector<size_type> scatter_with_block_ranks(column_view const& partition_map,
                                                size_type num_partitions,
                                                InputIterator input,
                                                OutputIterator output,
                                                cuda::stream_ref stream)
{
  auto const num_rows   = partition_map.size();
  auto const num_blocks = (num_rows - 1) / MAP_PARTITION_TILE_SIZE + 1;
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  std::vector<size_type> partition_offsets;
  {
    cuda::device_buffer<uint16_t> local_ranks(stream, temp_mr, num_rows, cuda::no_init);
    cuda::device_buffer<size_type> block_offsets(
      stream, temp_mr, static_cast<std::size_t>(num_blocks) * num_partitions, cuda::no_init);
    cuda::device_buffer<size_type> offsets(stream, temp_mr, num_partitions + 1, cuda::no_init);
    partition_map_local_ranks<<<num_blocks, MAP_PARTITION_BLOCK_SIZE, 0, stream.get()>>>(
      partition_map.begin<MapType>(),
      num_rows,
      num_partitions,
      local_ranks.data(),
      block_offsets.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    // Partition-major layout gives both the partition base and preceding blocks'
    // row counts in one scan, without an atomic cursor for each partition.
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                           block_offsets.begin(),
                           block_offsets.end(),
                           block_offsets.begin());
    partition_map_scatter<<<num_blocks, MAP_PARTITION_BLOCK_SIZE, 0, stream.get()>>>(
      partition_map.begin<MapType>(),
      local_ranks.data(),
      block_offsets.data(),
      num_rows,
      num_blocks,
      num_partitions,
      input,
      output,
      offsets.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    auto const pinned_offsets = cudf::detail::make_pinned_vector(offsets, stream);
    partition_offsets.assign(pinned_offsets.begin(), pinned_offsets.end());
  }
  return partition_offsets;
}

template <typename MapType>
struct partition_single_column {
  template <typename T>
  std::pair<std::unique_ptr<table>, std::vector<size_type>> operator()(
    column_view const& input,
    column_view const& partition_map,
    size_type num_partitions,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const
  {
    if constexpr (cudf::is_fixed_width<T>()) {
      auto output  = cudf::allocate_like(input, mask_allocation_policy::NEVER, stream, mr);
      auto offsets = scatter_with_block_ranks<MapType>(
        partition_map, num_partitions, input.begin<T>(), output->mutable_view().begin<T>(), stream);
      std::vector<std::unique_ptr<column>> columns;
      columns.push_back(std::move(output));
      return {std::make_unique<table>(std::move(columns)), std::move(offsets)};
    } else {
      CUDF_FAIL("Expected a fixed-width column");
    }
  }
};

template <typename MapType>
std::pair<std::unique_ptr<table>, std::vector<size_type>> partition_with_block_ranks(
  table_view const& input,
  column_view const& partition_map,
  size_type num_partitions,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // A single fixed-width column with no nulls can be scattered directly, avoiding the
  // gather map and the subsequent gather of its payload.
  if (input.num_columns() == 1 && is_fixed_width(input.column(0).type()) &&
      (!input.column(0).nullable() || input.column(0).null_count() == 0)) {
    return type_dispatcher<dispatch_storage_type>(input.column(0).type(),
                                                  partition_single_column<MapType>{},
                                                  input.column(0),
                                                  partition_map,
                                                  num_partitions,
                                                  stream,
                                                  mr);
  }
  cuda::device_buffer<size_type> gather_map(
    stream, cudf::get_current_device_resource_ref(), input.num_rows(), cuda::no_init);
  auto partition_offsets =
    make_partition_gather_map<MapType>(partition_map, num_partitions, gather_map.data(), stream);
  auto output =
    cudf::detail::gather(input,
                         device_span<size_type const>{gather_map.data(), gather_map.size()},
                         out_of_bounds_policy::DONT_CHECK,
                         negative_index_policy::NOT_ALLOWED,
                         stream,
                         mr);
  return {std::move(output), std::move(partition_offsets)};
}

/**
 * @brief Compute partition boundaries without materializing a row permutation.
 */
template <typename MapType>
std::vector<size_type> make_map_partition_offsets(column_view const& partition_map,
                                                  size_type num_partitions,
                                                  cuda::stream_ref stream)
{
  rmm::device_uvector<size_type> offsets(static_cast<std::size_t>(num_partitions) + 1, stream);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(offsets.data() + num_partitions, 0, sizeof(size_type), stream.get()));
  std::size_t bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    bytes,
                                                    partition_map.begin<MapType>(),
                                                    offsets.data(),
                                                    num_partitions + 1,
                                                    size_type{0},
                                                    num_partitions,
                                                    partition_map.size(),
                                                    stream.get()));
  {
    rmm::device_buffer scratch(bytes, stream);
    CUDF_CUDA_TRY(cub::DeviceHistogram::HistogramEven(scratch.data(),
                                                      bytes,
                                                      partition_map.begin<MapType>(),
                                                      offsets.data(),
                                                      num_partitions + 1,
                                                      size_type{0},
                                                      num_partitions,
                                                      partition_map.size(),
                                                      stream.get()));
  }
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         offsets.begin(),
                         offsets.end(),
                         offsets.begin());
  auto const pinned_offsets = cudf::detail::make_pinned_vector(offsets, stream);
  return {pinned_offsets.begin(), pinned_offsets.end()};
}

/**
 * @brief Handle cases whose output values do not require a permutation.
 *
 * Admission uses CPU metadata only. For a single row, the one map value is read
 * solely to construct the host partition boundaries required by the API.
 */
template <typename MapType>
std::optional<std::pair<std::unique_ptr<table>, std::vector<size_type>>> try_partition_trivial(
  table_view const& input,
  column_view const& partition_map,
  size_type num_partitions,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (num_partitions <= 0 || num_partitions == std::numeric_limits<size_type>::max()) {
    return std::nullopt;
  }
  auto const rows    = input.num_rows();
  auto const uniform = std::all_of(input.begin(), input.end(), [](auto const& col) {
    return cudf::detail::partition_identity::is_uniform(col);
  });
  if (num_partitions != 1 && rows != 1 && !uniform) { return std::nullopt; }

  std::vector<size_type> offsets;
  if (num_partitions == 1) {
    offsets = {0, rows};
  } else if (rows == 1) {
    auto const value = cudf::detail::make_std_vector(
      device_span<MapType const>{partition_map.begin<MapType>(), 1}, stream)[0];
    offsets.resize(static_cast<std::size_t>(num_partitions) + 1);
    for (std::size_t p = 0; p < offsets.size(); ++p) {
      offsets[p] = p > static_cast<std::size_t>(value) ? rows : 0;
    }
  } else {
    offsets = make_map_partition_offsets<MapType>(partition_map, num_partitions, stream);
  }
  if (!uniform) {
    return std::pair{cudf::detail::partition_identity::copy(input, stream, mr), std::move(offsets)};
  }

  std::vector<std::unique_ptr<column>> columns;
  columns.reserve(input.num_columns());
  for (auto const& col : input) {
    columns.push_back(cudf::detail::partition_identity::uniform_column(col, stream, mr));
  }
  return std::pair{std::make_unique<table>(std::move(columns), rows), std::move(offsets)};
}

struct partition_gather_output {
  size_type* indices;
  __device__ void operator()(size_type source, size_type destination) const
  {
    indices[destination] = source;
  }
};

struct partition_plan {
  cudf::detail::partition_map_plan::algorithm method;
  bool direct;
  bool reordered;
};

/**
 * Choose a plan using only cached host metadata. Null-mask presence is not a
 * discriminator: a nullable column with null_count() == 0 uses the non-null path.
 * The large-input crossover is measured over different partition skews and widths.
 */
std::optional<partition_plan> select_partition_plan(table_view const& input, size_type partitions)
{
  namespace plans = cudf::detail::partition_map_plan;
  if (partitions <= 1 || partitions > MAP_PARTITION_MAX_BINS ||
      input.num_rows() < MAP_PARTITION_TILE_SIZE) {
    return std::nullopt;
  }
  auto const fixed = std::all_of(
    input.begin(), input.end(), [](auto const& col) { return is_fixed_width(col.type()); });
  auto const nullable =
    std::any_of(input.begin(), input.end(), [](auto const& col) { return col.null_count() > 0; });
  std::size_t bytes_per_row = 0;
  if (fixed) {
    for (auto const& col : input) {
      bytes_per_row += cudf::size_of(col.type());
    }
  }
  constexpr size_type large_input = 1'000'000;
  bool const large                = input.num_rows() >= large_input;
  bool const dense_nulls =
    fixed && nullable && large && std::all_of(input.begin(), input.end(), [&](auto const& col) {
      return int64_t{col.null_count()} * 4 >= int64_t{input.num_rows()} * 3;
    });
  // With few valid rows, skip payload reads and reorder writes without a row map.
  if (dense_nulls && input.num_columns() > 2) {
    return partition_plan{plans::algorithm::recompute, true, true};
  }
  if (partitions == 2) {
    bool const direct = fixed && input.num_columns() <= 2 &&
                        (input.num_columns() == 1 || !nullable || !large || bytes_per_row >= 16);
    partition_plan plan{plans::algorithm::binary, direct, false};
    if (fixed && !nullable && large) {
      plan.direct = true;
      if (input.num_columns() > 2) {
        // Reordered output pays off for a few wide columns and removes saved ranks.
        bool const reorder = input.num_columns() <= 4 && bytes_per_row >= 32;
        plan.method        = reorder ? plans::algorithm::recompute : plans::algorithm::saved;
        plan.reordered     = reorder;
      } else if (bytes_per_row > 4) {
        plan.method    = plans::algorithm::recompute;
        plan.reordered = input.num_columns() == 1 && bytes_per_row == 16;
      }
    }
    return plan;
  }
  if (fixed && large) {
    if (input.num_columns() == 1 ||
        (input.num_columns() == 2 && (!nullable || bytes_per_row >= 16))) {
      // Saved ranks avoid recomputing sparse nullable two-column tables; the
      // direct sink still removes the larger N-entry gather map.
      auto const method = nullable && input.num_columns() == 2 && !dense_nulls
                            ? plans::algorithm::saved
                          : partitions >= 16 ? plans::algorithm::reserve
                                             : plans::algorithm::recompute;
      return partition_plan{method, true, false};
    }
    if (!nullable) {
      // A one-tile prefix is more robust than supertiles for narrow multi-column
      // output with uniform partition maps, while retaining direct-output savings.
      return partition_plan{plans::algorithm::recompute, true, true};
    }
  }
  return std::nullopt;
}

struct dispatch_map_type {
  /**
   * @brief Partitions the table `t` according to the `partition_map`.
   *
   * For a small partition count and at least one full tile, compute warp-local
   * ranks and scan block histograms to scatter a single fixed-width column or build a
   * gather map without per-row global atomics.
   *
   * Otherwise:
   * - Compute the histogram of the size each partition
   * - Compute the exclusive scan of the histogram to get the offset for each
   * partition in the final partitioned output
   * - Use a transform iterator to materialize the scatter map of the rows from
   * `t` into the final output.
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
    if (auto result =
          try_partition_trivial<MapType>(t, partition_map, num_partitions, stream, mr)) {
      return std::move(*result);
    }
    // All-null and CPU-known empty-string columns are invariant. Build them
    // after the reduced call has released its row map and other temporaries.
    auto const rows = t.num_rows();
    auto const omit = [](column_view const& col) {
      return cudf::detail::partition_identity::is_uniform(col);
    };
    if (std::any_of(t.begin(), t.end(), omit)) {
      std::vector<column_view> reduced;
      for (auto const& col : t) {
        if (!omit(col)) { reduced.push_back(col); }
      }
      auto [result, offsets] = this->template operator()<MapType>(
        table_view{reduced}, partition_map, num_partitions, stream, mr);
      auto owned = result->release();
      std::vector<std::unique_ptr<column>> columns;
      columns.reserve(t.num_columns());
      auto next = owned.begin();
      for (auto const& col : t) {
        columns.push_back(omit(col)
                            ? cudf::detail::partition_identity::uniform_column(col, stream, mr)
                            : std::move(*next++));
      }
      return {std::make_unique<table>(std::move(columns), rows), std::move(offsets)};
    }
    if (auto result = cudf::detail::partition_strings_plan::partition<MapType>(
          t, partition_map, num_partitions, stream, mr)) {
      return std::move(*result);
    }
    if (auto plan = select_partition_plan(t, num_partitions)) {
      auto launch = [&](auto sink) {
        return cudf::detail::partition_map_plan::run(partition_map.begin<MapType>(),
                                                     t.num_rows(),
                                                     num_partitions,
                                                     plan->method,
                                                     sink,
                                                     plan->reordered,
                                                     stream);
      };
      if (plan->direct) { return cudf::detail::partition_fixed::partition(t, launch, stream, mr); }
      rmm::device_uvector<size_type> map(t.num_rows(), stream);
      auto offsets = launch(partition_gather_output{map.data()});
      auto output  = cudf::detail::gather(t,
                                         device_span<size_type const>{map.data(), map.size()},
                                         out_of_bounds_policy::DONT_CHECK,
                                         negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);
      return {std::move(output), std::move(offsets)};
    }
    if (num_partitions > 1 && num_partitions <= MAP_PARTITION_MAX_BINS &&
        t.num_rows() >= MAP_PARTITION_TILE_SIZE) {
      return partition_with_block_ranks<MapType>(t, partition_map, num_partitions, stream, mr);
    }

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
  cuda::stream_ref stream,
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

std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  table_view const& keys,
  int num_partitions,
  hash_id hash_function,
  uint32_t seed,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
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
