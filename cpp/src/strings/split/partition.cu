/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/split/partition.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <vector>

namespace cudf {
namespace strings {
namespace detail {
using string_index_pair = cuda::std::pair<char const*, size_type>;

namespace {
//
// Partition splits the string at the first occurrence of delimiter, and returns 3 elements
// containing the part before the delimiter, the delimiter itself, and the part after the delimiter.
// If the delimiter is not found, return 3 elements containing the string itself, followed by two
// empty strings.
//
// strs = ["abcde", nullptr, "a_bc_def", "a__bc", "_ab_cd", "ab_cd_"]
// results = partition(strs,"_")
//     col0  col1    col2
// 0  abcde  ""      ""
// 1  null   null    null
// 2  a      _       bc_déf
// 3  a      _      _bc
// 4  ""     _       ab_cd
// 5  ab     _       cd_
//
struct partition_fn {
  column_device_view const d_strings;    // strings to split
  string_view const d_delimiter;         // delimiter for split
  string_index_pair* d_indices_left{};   // the three
  string_index_pair* d_indices_delim{};  // output columns
  string_index_pair* d_indices_right{};  // amigos

  partition_fn(column_device_view const& d_strings,
               string_view const& d_delimiter,
               rmm::device_uvector<string_index_pair>& indices_left,
               rmm::device_uvector<string_index_pair>& indices_delim,
               rmm::device_uvector<string_index_pair>& indices_right)
    : d_strings(d_strings),
      d_delimiter(d_delimiter),
      d_indices_left(indices_left.data()),
      d_indices_delim(indices_delim.data()),
      d_indices_right(indices_right.data())
  {
  }

  __device__ void set_null_entries(size_type idx)
  {
    if (d_indices_left) {
      d_indices_left[idx]  = string_index_pair{nullptr, 0};
      d_indices_delim[idx] = string_index_pair{nullptr, 0};
      d_indices_right[idx] = string_index_pair{nullptr, 0};
    }
  }

  __device__ size_type check_delimiter(size_type idx,
                                       string_view const& d_str,
                                       string_view::const_iterator& itr)
  {
    size_type offset = itr.byte_offset();
    size_type pos    = -1;
    if (d_delimiter.empty()) {
      if (*itr <= ' ')  // whitespace delimited
        pos = offset;
    } else {
      auto bytes = cuda::std::min(d_str.size_bytes() - offset, d_delimiter.size_bytes());
      if (d_delimiter.compare(d_str.data() + offset, bytes) == 0) pos = offset;
    }
    if (pos >= 0)  // delimiter found, set results
    {
      d_indices_left[idx] = string_index_pair{d_str.data(), offset};
      if (d_delimiter.empty()) {
        d_indices_delim[idx] = string_index_pair{d_str.data() + offset, 1};
        ++offset;
      } else {
        d_indices_delim[idx] = string_index_pair{d_delimiter.data(), d_delimiter.size_bytes()};
        offset += d_delimiter.size_bytes();
      }
      d_indices_right[idx] = string_index_pair{d_str.data() + offset, d_str.size_bytes() - offset};
    }
    return pos;
  }

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      set_null_entries(idx);
      return;
    }
    string_view d_str = d_strings.element<string_view>(idx);
    size_type pos     = -1;
    for (auto itr = d_str.begin(); (pos < 0) && (itr < d_str.end()); ++itr)
      pos = check_delimiter(idx, d_str, itr);
    if (pos < 0)  // delimiter not found
    {
      d_indices_left[idx]  = string_index_pair{d_str.data(), d_str.size_bytes()};
      d_indices_delim[idx] = string_index_pair{"", 0};  // two empty
      d_indices_right[idx] = string_index_pair{"", 0};  // strings added
    }
  }
};

//
// This follows most of the same logic as partition above except that the delimiter
// search starts from the end of each string. Also, if no delimiter is found the
// resulting array includes two empty strings followed by the original string.
//
// strs = ["abcde", nullptr, "a_bc_def", "a__bc", "_ab_cd", "ab_cd_"]
// results = rpartition(strs,"_")
//     col0  col1   col2
// 0  ""     ""     abcde
// 1  null   null   null
// 2  a_bc   _      déf
// 3  a_     _      bc
// 4  ab     _      cd
// 5  ab_cd  _      ""
//
struct rpartition_fn : public partition_fn {
  rpartition_fn(column_device_view const& d_strings,
                string_view const& d_delimiter,
                rmm::device_uvector<string_index_pair>& indices_left,
                rmm::device_uvector<string_index_pair>& indices_delim,
                rmm::device_uvector<string_index_pair>& indices_right)
    : partition_fn(d_strings, d_delimiter, indices_left, indices_delim, indices_right)
  {
  }

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      set_null_entries(idx);
      return;
    }
    string_view d_str = d_strings.element<string_view>(idx);
    size_type pos     = -1;
    auto itr          = d_str.end();
    while ((pos < 0) && (d_str.begin() < itr)) {
      --itr;
      pos = check_delimiter(idx, d_str, itr);
    }
    if (pos < 0)  // delimiter not found
    {
      d_indices_left[idx]  = string_index_pair{"", 0};  // two empty
      d_indices_delim[idx] = string_index_pair{"", 0};  // strings
      d_indices_right[idx] = string_index_pair{d_str.data(), d_str.size_bytes()};
    }
  }
};

}  // namespace

std::unique_ptr<table> partition(strings_column_view const& strings,
                                 string_scalar const& delimiter,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");
  auto strings_count = strings.size();
  if (strings_count == 0) return std::make_unique<table>(std::vector<std::unique_ptr<column>>());
  auto strings_column = column_device_view::create(strings.parent(), stream);
  string_view d_delimiter(delimiter.data(), delimiter.size());
  auto left_indices  = rmm::device_uvector<string_index_pair>(strings_count, stream);
  auto delim_indices = rmm::device_uvector<string_index_pair>(strings_count, stream);
  auto right_indices = rmm::device_uvector<string_index_pair>(strings_count, stream);
  partition_fn partitioner(
    *strings_column, d_delimiter, left_indices, delim_indices, right_indices);

  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     strings_count,
                     partitioner);
  std::vector<std::unique_ptr<column>> results;
  results.emplace_back(make_strings_column(left_indices, stream, mr));
  results.emplace_back(make_strings_column(delim_indices, stream, mr));
  results.emplace_back(make_strings_column(right_indices, stream, mr));
  return std::make_unique<table>(std::move(results));
}

std::unique_ptr<table> rpartition(strings_column_view const& strings,
                                  string_scalar const& delimiter,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");
  auto strings_count = strings.size();
  if (strings_count == 0) return std::make_unique<table>(std::vector<std::unique_ptr<column>>());
  auto strings_column = column_device_view::create(strings.parent(), stream);
  string_view d_delimiter(delimiter.data(), delimiter.size());
  auto left_indices  = rmm::device_uvector<string_index_pair>(strings_count, stream);
  auto delim_indices = rmm::device_uvector<string_index_pair>(strings_count, stream);
  auto right_indices = rmm::device_uvector<string_index_pair>(strings_count, stream);
  rpartition_fn partitioner(
    *strings_column, d_delimiter, left_indices, delim_indices, right_indices);
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     strings_count,
                     partitioner);

  std::vector<std::unique_ptr<column>> results;
  results.emplace_back(make_strings_column(left_indices, stream, mr));
  results.emplace_back(make_strings_column(delim_indices, stream, mr));
  results.emplace_back(make_strings_column(right_indices, stream, mr));
  return std::make_unique<table>(std::move(results));
}

}  // namespace detail

// external APIs

std::unique_ptr<table> partition(strings_column_view const& input,
                                 string_scalar const& delimiter,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::partition(input, delimiter, stream, mr);
}

std::unique_ptr<table> rpartition(strings_column_view const& input,
                                  string_scalar const& delimiter,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rpartition(input, delimiter, stream, mr);
}

}  // namespace strings
}  // namespace cudf
