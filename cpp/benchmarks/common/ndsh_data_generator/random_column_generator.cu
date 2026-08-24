/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "random_column_generator.hpp"

#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/strings/detail/strings_children.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <string>

namespace cudf::datagen {

namespace {

// Functor for generating random strings
struct random_string_generator {
  char* chars;
  thrust::default_random_engine engine;
  thrust::uniform_int_distribution<unsigned char> char_dist;

  CUDF_HOST_DEVICE random_string_generator(char* c) : chars(c), char_dist(44, 122) {}

  __device__ void operator()(cuda::std::tuple<int64_t, int64_t> str_begin_end)
  {
    auto begin = cuda::std::get<0>(str_begin_end);
    auto end   = cuda::std::get<1>(str_begin_end);
    engine.discard(begin);
    for (auto i = begin; i < end; ++i) {
      auto ch = char_dist(engine);
      if (i == end - 1 && ch >= '\x7F') ch = ' ';  // last element ASCII only.
      if (ch >= '\x7F')                            // x7F is at the top edge of ASCII
        chars[i++] = '\xC4';                       // these characters are assigned two bytes
      chars[i] = static_cast<char>(ch + (ch >= '\x7F'));
    }
  }
};

// Functor for generating random numbers
template <typename T>
struct random_number_generator {
  T lower;
  T upper;

  CUDF_HOST_DEVICE random_number_generator(T lower, T upper) : lower(lower), upper(upper) {}

  __device__ T operator()(int64_t const idx) const
  {
    if constexpr (cudf::is_integral<T>()) {
      thrust::default_random_engine engine;
      thrust::uniform_int_distribution<T> dist(lower, upper);
      engine.discard(idx);
      return dist(engine);
    } else {
      thrust::default_random_engine engine;
      thrust::uniform_real_distribution<T> dist(lower, upper);
      engine.discard(idx);
      return dist(engine);
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> generate_random_string_column(cudf::size_type lower,
                                                            cudf::size_type upper,
                                                            cudf::size_type num_rows,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  CUDF_BENCHMARK_RANGE();
  auto offsets_begin = cudf::detail::make_counting_transform_iterator(
    0, random_number_generator<cudf::size_type>(lower, upper));
  auto [offsets_column, computed_bytes] = cudf::strings::detail::make_offsets_child_column(
    offsets_begin, offsets_begin + num_rows, stream, mr);
  rmm::device_uvector<char> chars(computed_bytes, stream);

  auto const offset_itr =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // We generate the strings in parallel into the `chars` vector using the
  // offsets vector generated above.
  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     cuda::make_zip_iterator(cuda::std::make_tuple(offset_itr, offset_itr + 1)),
                     num_rows,
                     random_string_generator(chars.data()));

  return cudf::make_strings_column(
    num_rows, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

template <typename T>
std::unique_ptr<cudf::column> generate_random_numeric_column(T lower,
                                                             T upper,
                                                             cudf::size_type num_rows,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  CUDF_BENCHMARK_RANGE();
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_to_id<T>()}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  cudf::size_type begin = 0;
  cudf::size_type end   = num_rows;
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator{begin},
                    cuda::counting_iterator{end},
                    col->mutable_view().begin<T>(),
                    random_number_generator<T>(lower, upper));
  return col;
}

template std::unique_ptr<cudf::column> generate_random_numeric_column<int8_t>(
  int8_t lower,
  int8_t upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> generate_random_numeric_column<int16_t>(
  int16_t lower,
  int16_t upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> generate_random_numeric_column<cudf::size_type>(
  cudf::size_type lower,
  cudf::size_type upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> generate_random_numeric_column<double>(
  double lower,
  double upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> generate_primary_key_column(cudf::scalar const& start,
                                                          cudf::size_type num_rows,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  CUDF_BENCHMARK_RANGE();
  return cudf::sequence(num_rows, start, stream, mr);
}

std::unique_ptr<cudf::column> generate_repeat_string_column(std::string const& value,
                                                            cudf::size_type num_rows,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  CUDF_BENCHMARK_RANGE();
  auto const scalar = cudf::string_scalar(value);
  return cudf::make_column_from_scalar(scalar, num_rows, stream, mr);
}

std::unique_ptr<cudf::column> generate_random_string_column_from_set(
  cudf::host_span<char const* const> set,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_BENCHMARK_RANGE();
  // Build a gather map of random strings to choose from
  // The size of the string sets always fits within 16-bit integers
  auto const indices =
    generate_primary_key_column(cudf::numeric_scalar<int16_t>(0), set.size(), stream, mr);
  auto const keys       = cudf::test::strings_column_wrapper(set.begin(), set.end()).release();
  auto const gather_map = cudf::table_view({indices->view(), keys->view()});

  // Build a column of random keys to gather from the set
  auto const gather_keys =
    generate_random_numeric_column<int16_t>(0, set.size() - 1, num_rows, stream, mr);

  // Perform the gather operation
  auto const gathered_table = cudf::gather(
    gather_map, gather_keys->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto gathered_table_columns = gathered_table->release();
  return std::move(gathered_table_columns[1]);
}

template <typename T>
std::unique_ptr<cudf::column> generate_repeat_sequence_column(T seq_length,
                                                              bool zero_indexed,
                                                              cudf::size_type num_rows,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  CUDF_BENCHMARK_RANGE();
  auto pkey =
    generate_primary_key_column(cudf::numeric_scalar<cudf::size_type>(0), num_rows, stream, mr);
  auto repeat_seq_zero_indexed = cudf::binary_operation(pkey->view(),
                                                        cudf::numeric_scalar<T>(seq_length),
                                                        cudf::binary_operator::MOD,
                                                        cudf::data_type{cudf::type_to_id<T>()},
                                                        stream,
                                                        mr);
  if (zero_indexed) { return repeat_seq_zero_indexed; }
  return cudf::binary_operation(repeat_seq_zero_indexed->view(),
                                cudf::numeric_scalar<T>(1),
                                cudf::binary_operator::ADD,
                                cudf::data_type{cudf::type_to_id<T>()},
                                stream,
                                mr);
}

template std::unique_ptr<cudf::column> generate_repeat_sequence_column<int8_t>(
  int8_t seq_length,
  bool zero_indexed,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> generate_repeat_sequence_column<cudf::size_type>(
  cudf::size_type seq_length,
  bool zero_indexed,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::datagen
