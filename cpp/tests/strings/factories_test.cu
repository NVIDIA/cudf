/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/utility>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cstring>
#include <vector>

struct StringsFactoriesTest : public cudf::test::BaseFixture {};

using string_pair = cuda::std::pair<char const*, cudf::size_type>;

TEST_F(StringsFactoriesTest, CreateColumnFromPair)
{
  std::vector<char const*> h_test_strings{"the quick brown fox jumps over the lazy dog",
                                          "the fat cat lays next to the other accénted cat",
                                          "a slow moving turtlé cannot catch the bird",
                                          "which can be composéd together to form a more complete",
                                          "thé result does not include the value in the sum in",
                                          "",
                                          nullptr,
                                          "absent stop words"};

  cudf::size_type memsize = 0;
  for (auto itr = h_test_strings.begin(); itr != h_test_strings.end(); ++itr)
    memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
  cudf::size_type count = (cudf::size_type)h_test_strings.size();
  thrust::host_vector<char> h_buffer(memsize);
  rmm::device_uvector<char> d_buffer(memsize, cudf::get_default_stream());
  thrust::host_vector<string_pair> strings(count);
  thrust::host_vector<cudf::size_type> h_offsets(count + 1);
  cudf::size_type offset = 0;
  cudf::size_type nulls  = 0;
  h_offsets[0]           = 0;
  for (cudf::size_type idx = 0; idx < count; ++idx) {
    char const* str = h_test_strings[idx];
    if (!str) {
      strings[idx] = string_pair{nullptr, 0};
      nulls++;
    } else {
      auto length = (cudf::size_type)strlen(str);
      memcpy(h_buffer.data() + offset, str, length);
      strings[idx] = string_pair{d_buffer.data() + offset, length};
      offset += length;
    }
    h_offsets[idx + 1] = offset;
  }
  auto d_strings = cudf::detail::make_device_uvector(
    strings, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(cudaMemcpy(d_buffer.data(), h_buffer.data(), memsize, cudaMemcpyDefault));
  auto column = cudf::make_strings_column(d_strings);
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_id::STRING});
  EXPECT_EQ(column->null_count(), nulls);
  if (nulls) {
    EXPECT_TRUE(column->nullable());
    EXPECT_TRUE(column->has_nulls());
  }
  EXPECT_EQ(1, column->num_children());
  EXPECT_NE(nullptr, column->view().head());

  cudf::strings_column_view strings_view(column->view());
  EXPECT_EQ(strings_view.size(), count);
  EXPECT_EQ(strings_view.offsets().size(), count + 1);
  EXPECT_EQ(strings_view.chars_size(cudf::get_default_stream()), memsize);

  // check string data
  cudf::test::strings_column_wrapper expected(
    h_test_strings.begin(),
    h_test_strings.end(),
    cudf::test::iterators::nulls_from_nullptrs(h_test_strings));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column->view(), expected);
}

TEST_F(StringsFactoriesTest, CreateColumnFromOffsets)
{
  std::vector<char const*> h_test_strings{"the quick brown fox jumps over the lazy dog",
                                          "the fat cat lays next to the other accénted cat",
                                          "a slow moving turtlé cannot catch the bird",
                                          "which can be composéd together to form a more complete",
                                          "thé result does not include the value in the sum in",
                                          "",
                                          nullptr,
                                          "absent stop words"};

  cudf::size_type memsize = 0;
  for (auto itr = h_test_strings.begin(); itr != h_test_strings.end(); ++itr)
    memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
  cudf::size_type count = (cudf::size_type)h_test_strings.size();
  std::vector<char> h_buffer(memsize);
  std::vector<cudf::size_type> h_offsets(count + 1);
  cudf::size_type offset         = 0;
  h_offsets[0]                   = offset;
  cudf::bitmask_type h_null_mask = 0;
  cudf::size_type null_count     = 0;
  for (cudf::size_type idx = 0; idx < count; ++idx) {
    h_null_mask     = (h_null_mask << 1);
    char const* str = h_test_strings[idx];
    if (str) {
      auto length = (cudf::size_type)strlen(str);
      memcpy(h_buffer.data() + offset, str, length);
      offset += length;
      h_null_mask |= 1;
    } else
      null_count++;
    h_offsets[idx + 1] = offset;
  }

  std::vector<cudf::bitmask_type> h_nulls{h_null_mask};
  auto d_buffer = cudf::detail::make_device_uvector(
    h_buffer, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto d_offsets = std::make_unique<cudf::column>(
    cudf::detail::make_device_uvector(
      h_offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref()),
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
    0);
  auto d_nulls = cuda::device_buffer<uint8_t>{
    cudf::get_default_stream(), cudf::get_current_device_resource_ref(), h_nulls};
  auto column = cudf::make_strings_column(
    count, std::move(d_offsets), d_buffer.release(), null_count, std::move(d_nulls));
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_id::STRING});
  EXPECT_EQ(column->null_count(), null_count);
  EXPECT_EQ(1, column->num_children());
  EXPECT_NE(nullptr, column->view().head());

  cudf::strings_column_view strings_view(column->view());
  EXPECT_EQ(strings_view.size(), count);
  EXPECT_EQ(strings_view.offsets().size(), count + 1);
  EXPECT_EQ(strings_view.chars_size(cudf::get_default_stream()), memsize);

  // check string data
  auto h_chars_data = cudf::detail::make_std_vector(
    cudf::device_span<char const>(strings_view.chars_begin(cudf::get_default_stream()),
                                  strings_view.chars_size(cudf::get_default_stream())),
    cudf::get_default_stream());
  auto h_offsets_data = cudf::detail::make_std_vector(
    cudf::device_span<cudf::size_type const>(
      strings_view.offsets().data<cudf::size_type>() + strings_view.offset(),
      strings_view.size() + 1),
    cudf::get_default_stream());
  EXPECT_EQ(memcmp(h_buffer.data(), h_chars_data.data(), h_buffer.size()), 0);
  EXPECT_EQ(
    memcmp(h_offsets.data(), h_offsets_data.data(), h_offsets.size() * sizeof(cudf::size_type)), 0);
}

TEST_F(StringsFactoriesTest, CreateScalar)
{
  std::string value = "test string";
  auto s            = cudf::make_string_scalar(value);
  auto string_s     = static_cast<cudf::string_scalar*>(s.get());

  EXPECT_EQ(string_s->to_string(), value);
  EXPECT_TRUE(string_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

TEST_F(StringsFactoriesTest, EmptyStringsColumn)
{
  auto d_chars   = rmm::device_uvector<char>(0, cudf::get_default_stream());
  auto d_offsets = std::make_unique<cudf::column>(
    cudf::detail::make_zeroed_device_uvector<cudf::size_type>(
      1, cudf::get_default_stream(), cudf::get_current_device_resource_ref()),
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
    0);
  auto d_nulls = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED);

  auto results =
    cudf::make_strings_column(0, std::move(d_offsets), d_chars.release(), 0, std::move(d_nulls));
  cudf::test::expect_column_empty(results->view());

  rmm::device_uvector<string_pair> d_strings{0, cudf::get_default_stream()};
  results = cudf::make_strings_column(d_strings);
  cudf::test::expect_column_empty(results->view());
}

namespace {

struct string_view_to_pair {
  __device__ string_pair operator()(cuda::std::pair<cudf::string_view, bool> const& p)
  {
    return (p.second) ? string_pair{p.first.data(), p.first.size_bytes()} : string_pair{nullptr, 0};
  }
};
}  // namespace

TEST_F(StringsFactoriesTest, StringPairWithNullsAndEmpty)
{
  cudf::test::strings_column_wrapper data(
    {"", "this", "is", "", "a", "", "column", "of", "strings", "", ""},
    {0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1});

  auto d_column = cudf::column_device_view::create(data);
  rmm::device_uvector<string_pair> pairs(d_column->size(), cudf::get_default_stream());
  thrust::transform(rmm::exec_policy_nosync(cudf::get_default_stream()),
                    d_column->pair_begin<cudf::string_view, true>(),
                    d_column->pair_end<cudf::string_view, true>(),
                    pairs.data(),
                    string_view_to_pair{});

  auto result = cudf::make_strings_column(pairs);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), data);
}

struct StringsBatchConstructionTest : public cudf::test::BaseFixture {};

TEST_F(StringsBatchConstructionTest, EmptyColumns)
{
  auto constexpr num_columns = 10;
  auto const stream          = cudf::get_default_stream();

  auto const d_string_pairs = rmm::device_uvector<string_pair>{0, stream};
  auto const input          = std::vector<cudf::device_span<string_pair const>>(
    num_columns, {d_string_pairs.data(), d_string_pairs.size()});
  auto const output = cudf::make_strings_column_batch(input, stream);

  auto const expected_col = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  for (auto const& col : output) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col->view(), col->view());
  }
}

TEST_F(StringsBatchConstructionTest, AllNullsColumns)
{
  auto constexpr num_columns = 10;
  auto constexpr num_rows    = 100;
  auto const stream          = cudf::get_default_stream();

  auto d_string_pairs = rmm::device_uvector<string_pair>{num_rows, stream};
  thrust::uninitialized_fill_n(rmm::exec_policy_nosync(stream),
                               d_string_pairs.data(),
                               d_string_pairs.size(),
                               string_pair{nullptr, 0});
  auto const input = std::vector<cudf::device_span<string_pair const>>(
    num_columns, {d_string_pairs.data(), d_string_pairs.size()});
  auto const output = cudf::make_strings_column_batch(input, stream);

  auto const expected_col = cudf::make_strings_column(d_string_pairs);
  for (auto const& col : output) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col->view(), col->view());
  }
}

namespace {

struct index_to_pair {
  int const num_test_strings;
  char const* d_chars;
  std::size_t const* d_offsets;
  int const* is_null;

  __device__ string_pair operator()(cudf::size_type idx)
  {
    auto const data_idx = idx % num_test_strings;
    return {is_null[data_idx] ? nullptr : d_chars + d_offsets[data_idx],
            static_cast<cudf::size_type>(d_offsets[data_idx + 1] - d_offsets[data_idx])};
  }
};

}  // namespace

TEST_F(StringsBatchConstructionTest, CreateColumnsFromPairs)
{
  auto constexpr num_columns  = 10;
  auto constexpr max_num_rows = 1000;
  auto const stream           = cudf::get_default_stream();
  auto const mr               = cudf::get_current_device_resource_ref();

  std::vector<char const*> h_test_strings{"the quick brown fox jumps over the lazy dog",
                                          "the fat cat lays next to the other accénted cat",
                                          "a slow moving turtlé cannot catch the bird",
                                          "which can be composéd together to form a more complete",
                                          "thé result does not include the value in the sum in",
                                          "",
                                          nullptr,
                                          "absent stop words"};
  auto const num_test_strings = static_cast<int>(h_test_strings.size());

  std::vector<std::size_t> h_offsets(num_test_strings + 1, 0);
  for (int i = 0; i < num_test_strings; ++i) {
    h_offsets[i + 1] = h_offsets[i] + (h_test_strings[i] ? strlen(h_test_strings[i]) : 0);
  }

  std::vector<char> h_chars(h_offsets.back());
  std::vector<int> is_null(num_test_strings, 0);
  for (int i = 0; i < num_test_strings; ++i) {
    if (h_test_strings[i]) {
      memcpy(h_chars.data() + h_offsets[i], h_test_strings[i], strlen(h_test_strings[i]));
    } else {
      is_null[i] = 1;
    }
  }

  auto const d_offsets = cudf::detail::make_device_uvector_async(h_offsets, stream, mr);
  auto const d_chars   = cudf::detail::make_device_uvector_async(h_chars, stream, mr);
  auto const d_is_null = cudf::detail::make_device_uvector_async(is_null, stream, mr);

  std::vector<rmm::device_uvector<string_pair>> d_input;
  std::vector<cudf::device_span<string_pair const>> input;
  d_input.reserve(num_columns);
  input.reserve(num_columns);

  for (int col_idx = 0; col_idx < num_columns; ++col_idx) {
    // Columns have sizes increase from `max_num_rows / num_columns` to `max_num_rows`.
    auto const num_rows =
      static_cast<int>(static_cast<double>(col_idx + 1) / num_columns * max_num_rows);

    auto string_pairs = rmm::device_uvector<string_pair>(num_rows, stream);
    thrust::tabulate(
      rmm::exec_policy_nosync(stream),
      string_pairs.begin(),
      string_pairs.end(),
      index_to_pair{num_test_strings, d_chars.begin(), d_offsets.begin(), d_is_null.begin()});

    d_input.emplace_back(std::move(string_pairs));
    input.emplace_back(d_input.back());
  }

  auto const output = cudf::make_strings_column_batch(input, stream, mr);

  for (std::size_t i = 0; i < num_columns; ++i) {
    auto const string_pairs = input[i];
    auto const expected     = cudf::make_strings_column(string_pairs, stream, mr);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), output[i]->view());
  }
}

// The test below requires a huge amount of memory, thus it is disabled by default.
TEST_F(StringsBatchConstructionTest, DISABLED_CreateLongStringsColumns)
{
  auto constexpr num_columns = 2;
  auto const stream          = cudf::get_default_stream();
  auto const mr              = cudf::get_current_device_resource_ref();

  std::vector<char const*> h_test_strings{"the quick brown fox jumps over the lazy dog",
                                          "the fat cat lays next to the other accénted cat",
                                          "a slow moving turtlé cannot catch the bird",
                                          "which can be composéd together to form a more complete",
                                          "thé result does not include the value in the sum in",
                                          "",
                                          nullptr,
                                          "absent stop words"};
  auto const num_test_strings = static_cast<int>(h_test_strings.size());

  std::vector<std::size_t> h_offsets(num_test_strings + 1, 0);
  for (int i = 0; i < num_test_strings; ++i) {
    h_offsets[i + 1] = h_offsets[i] + (h_test_strings[i] ? strlen(h_test_strings[i]) : 0);
  }

  std::vector<char> h_chars(h_offsets.back());
  std::vector<int> is_null(num_test_strings, 0);
  for (int i = 0; i < num_test_strings; ++i) {
    if (h_test_strings[i]) {
      memcpy(h_chars.data() + h_offsets[i], h_test_strings[i], strlen(h_test_strings[i]));
    } else {
      is_null[i] = 1;
    }
  }

  auto const d_offsets = cudf::detail::make_device_uvector_async(h_offsets, stream, mr);
  auto const d_chars   = cudf::detail::make_device_uvector_async(h_chars, stream, mr);
  auto const d_is_null = cudf::detail::make_device_uvector_async(is_null, stream, mr);

  // If we create a column by repeating h_test_strings by `max_cycles` times,
  // we will have it size around (1.5*INT_MAX) bytes.
  auto const max_cycles = static_cast<int>(static_cast<int64_t>(std::numeric_limits<int>::max()) *
                                           1.5 / h_offsets.back());

  std::vector<rmm::device_uvector<string_pair>> d_input;
  std::vector<cudf::device_span<string_pair const>> input;
  d_input.reserve(num_columns);
  input.reserve(num_columns);

  for (int col_idx = 0; col_idx < num_columns; ++col_idx) {
    // Columns have sizes increase from `max_cycles * num_test_strings / num_columns` to
    // `max_cycles * num_test_strings`.
    auto const num_rows = static_cast<int>(static_cast<double>(col_idx + 1) / num_columns *
                                           max_cycles * num_test_strings);

    auto string_pairs = rmm::device_uvector<string_pair>(num_rows, stream);
    thrust::tabulate(
      rmm::exec_policy_nosync(stream),
      string_pairs.begin(),
      string_pairs.end(),
      index_to_pair{num_test_strings, d_chars.begin(), d_offsets.begin(), d_is_null.begin()});

    d_input.emplace_back(std::move(string_pairs));
    input.emplace_back(d_input.back());
  }

  auto const output = cudf::make_strings_column_batch(input, stream, mr);

  for (std::size_t i = 0; i < num_columns; ++i) {
    auto const string_pairs = input[i];
    auto const expected     = cudf::make_strings_column(string_pairs, stream, mr);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), output[i]->view());
  }
}
