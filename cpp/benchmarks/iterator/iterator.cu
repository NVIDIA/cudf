/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/column_wrapper.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/iterator>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <nvbench/nvbench.cuh>

// -----------------------------------------------------------------------------
template <typename InputIterator, typename OutputIterator, typename T>
inline auto reduce_by_cub(OutputIterator result, InputIterator d_in, int num_items, T init)
{
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(
    nullptr, temp_storage_bytes, d_in, result, num_items, cudf::DeviceSum{}, init);

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(temp_storage_bytes, cudf::get_default_stream());

  // Run reduction
  cub::DeviceReduce::Reduce(
    d_temp_storage.data(), temp_storage_bytes, d_in, result, num_items, cudf::DeviceSum{}, init);

  return temp_storage_bytes;
}

// -----------------------------------------------------------------------------
template <typename T>
void raw_stream_bench_cub(cudf::column_view& col, rmm::device_uvector<T>& result)
{
  // std::cout << "raw stream cub: " << "\t";

  T init{0};
  auto begin    = col.data<T>();
  int num_items = col.size();

  reduce_by_cub(result.begin(), begin, num_items, init);
};

template <typename T, bool has_null>
void iterator_bench_cub(cudf::column_view& col, rmm::device_uvector<T>& result)
{
  // std::cout << "iterator cub " << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";

  T init{0};
  auto d_col    = cudf::column_device_view::create(col);
  int num_items = col.size();
  if (has_null) {
    auto begin = cudf::detail::make_null_replacement_iterator(*d_col, init);
    reduce_by_cub(result.begin(), begin, num_items, init);
  } else {
    auto begin = d_col->begin<T>();
    reduce_by_cub(result.begin(), begin, num_items, init);
  }
}

// -----------------------------------------------------------------------------
template <typename T>
void raw_stream_bench_thrust(cudf::column_view& col, rmm::device_uvector<T>& result)
{
  // std::cout << "raw stream thust: " << "\t\t";

  T init{0};
  auto d_in  = col.data<T>();
  auto d_end = d_in + col.size();
  thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{});
}

template <typename T, bool has_null>
void iterator_bench_thrust(cudf::column_view& col, rmm::device_uvector<T>& result)
{
  // std::cout << "iterator thust " << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";

  T init{0};
  auto d_col = cudf::column_device_view::create(col);
  if (has_null) {
    auto d_in  = cudf::detail::make_null_replacement_iterator(*d_col, init);
    auto d_end = d_in + col.size();
    thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{});
  } else {
    auto d_in  = d_col->begin<T>();
    auto d_end = d_in + col.size();
    thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{});
  }
}

// -----------------------------------------------------------------------------

void bench_iterator_cub_raw(nvbench::state& state)
{
  using T                = double;
  auto const column_size = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto num_gen           = cuda::counting_iterator<cudf::size_type>{0};

  cudf::test::fixed_width_column_wrapper<T> wrap_hasnull_F(num_gen, num_gen + column_size);
  cudf::column_view hasnull_F = wrap_hasnull_F;

  // Initialize dev_result
  auto dev_result = cudf::detail::make_zeroed_device_uvector<T>(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  auto const data_size = column_size * sizeof(T);
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { raw_stream_bench_cub<T>(hasnull_F, dev_result); });
}

void bench_iterator_cub_iter(nvbench::state& state)
{
  using T                = double;
  auto const column_size = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto num_gen           = cuda::counting_iterator<cudf::size_type>{0};

  cudf::test::fixed_width_column_wrapper<T> wrap_hasnull_F(num_gen, num_gen + column_size);
  cudf::column_view hasnull_F = wrap_hasnull_F;

  // Initialize dev_result
  auto dev_result = cudf::detail::make_zeroed_device_uvector<T>(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  auto const data_size = column_size * sizeof(T);
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { iterator_bench_cub<T, false>(hasnull_F, dev_result); });
}

void bench_iterator_thrust_raw(nvbench::state& state)
{
  using T                = double;
  auto const column_size = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto num_gen           = cuda::counting_iterator<cudf::size_type>{0};

  cudf::test::fixed_width_column_wrapper<T> wrap_hasnull_F(num_gen, num_gen + column_size);
  cudf::column_view hasnull_F = wrap_hasnull_F;

  // Initialize dev_result
  auto dev_result = cudf::detail::make_zeroed_device_uvector<T>(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  auto const data_size = column_size * sizeof(T);
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { raw_stream_bench_thrust<T>(hasnull_F, dev_result); });
}

void bench_iterator_thrust_iter(nvbench::state& state)
{
  using T                = double;
  auto const column_size = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto num_gen           = cuda::counting_iterator<cudf::size_type>{0};

  cudf::test::fixed_width_column_wrapper<T> wrap_hasnull_F(num_gen, num_gen + column_size);
  cudf::column_view hasnull_F = wrap_hasnull_F;

  // Initialize dev_result
  auto dev_result = cudf::detail::make_zeroed_device_uvector<T>(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  auto const data_size = column_size * sizeof(T);
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { iterator_bench_thrust<T, false>(hasnull_F, dev_result); });
}

NVBENCH_BENCH(bench_iterator_cub_raw)
  .set_name("iterator_cub_raw")
  .add_int64_axis("num_rows", {1000, 10000, 100000, 1000000, 10000000});

NVBENCH_BENCH(bench_iterator_cub_iter)
  .set_name("iterator_cub_iter")
  .add_int64_axis("num_rows", {1000, 10000, 100000, 1000000, 10000000});

NVBENCH_BENCH(bench_iterator_thrust_raw)
  .set_name("iterator_thrust_raw")
  .add_int64_axis("num_rows", {1000, 10000, 100000, 1000000, 10000000});

NVBENCH_BENCH(bench_iterator_thrust_iter)
  .set_name("iterator_thrust_iter")
  .add_int64_axis("num_rows", {1000, 10000, 100000, 1000000, 10000000});
