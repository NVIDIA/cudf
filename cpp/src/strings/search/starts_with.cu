/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Utility to return a bool column indicating whether each string satisfies
 * `predicate` with the given target string.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam Predicate Returns a bool value given a string and the target.
 *
 * @param strings Column of strings to check against the target.
 * @param target UTF-8 encoded string to check against each string.
 * @param predicate Returns true if the given string matches the target.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New BOOL column.
 */
template <typename Predicate>
std::unique_ptr<column> starts_ends_with_fn(strings_column_view const& strings,
                                            string_scalar const& target,
                                            Predicate predicate,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  if (target.size() == 0)  // empty target string returns true
  {
    auto const true_scalar =
      make_fixed_width_scalar<bool>(true, stream, cudf::get_current_device_resource_ref());
    auto results = make_column_from_scalar(*true_scalar, strings.size(), stream, mr);
    results->set_null_mask(cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                           strings.null_count());
    return results;
  }

  auto d_target       = string_view(target.data(), target.size());
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{strings_count},
                    d_results,
                    [d_strings, predicate, d_target] __device__(size_type idx) {
                      return !d_strings.is_null(idx) &&
                             bool{predicate(d_strings.element<string_view>(idx), d_target)};
                    });
  results->set_null_count(strings.null_count());
  return results;
}

/**
 * @brief Utility to return a bool column indicating whether each `strings[i]` satisfies
 * `predicate` with `targets[i]`.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam Predicate Returns a bool value given a string and its target.
 *
 * @param strings Column of strings to check against `targets[i]`.
 * @param targets Column of strings to check against `strings[i]`.
 * @param predicate Returns true if the given string matches its target.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New BOOL column.
 */
template <typename Predicate>
std::unique_ptr<column> starts_ends_with_fn(strings_column_view const& strings,
                                            strings_column_view const& targets,
                                            Predicate predicate,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(targets.size() == strings.size(),
               "strings and targets column must be the same size");

  auto targets_column = column_device_view::create(targets.parent(), stream);
  auto d_targets      = *targets_column;
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{strings.size()},
    d_results,
    [d_strings, predicate, d_targets] __device__(size_type idx) {
      // empty target string returns true
      if (d_targets.is_valid(idx) && d_targets.element<string_view>(idx).length() == 0) {
        return true;
      } else if (!d_strings.is_null(idx) && !d_targets.is_null(idx)) {
        return bool{
          predicate(d_strings.element<string_view>(idx), d_targets.element<string_view>(idx))};
      } else {
        return false;
      }
    });
  results->set_null_count(strings.null_count());
  return results;
}
}  // namespace

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto predicate = [] __device__(string_view d_string, string_view d_target) {
    return (d_target.size_bytes() <= d_string.size_bytes()) &&
           (d_target.compare(d_string.data(), d_target.size_bytes()) == 0);
  };
  return starts_ends_with_fn(strings, target, predicate, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto predicate = [] __device__(string_view d_string, string_view d_target) {
    return (d_target.size_bytes() <= d_string.size_bytes()) &&
           (d_target.compare(d_string.data(), d_target.size_bytes()) == 0);
  };
  return starts_ends_with_fn(strings, targets, predicate, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  auto predicate = [] __device__(string_view d_string, string_view d_target) {
    auto const str_size = d_string.size_bytes();
    auto const tgt_size = d_target.size_bytes();
    return (tgt_size <= str_size) &&
           (d_target.compare(d_string.data() + str_size - tgt_size, tgt_size) == 0);
  };

  return starts_ends_with_fn(strings, target, predicate, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  strings_column_view const& targets,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  auto predicate = [] __device__(string_view d_string, string_view d_target) {
    auto const str_size = d_string.size_bytes();
    auto const tgt_size = d_target.size_bytes();
    return (tgt_size <= str_size) &&
           (d_target.compare(d_string.data() + str_size - tgt_size, tgt_size) == 0);
  };

  return starts_ends_with_fn(strings, targets, predicate, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, target, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, targets, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, target, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  strings_column_view const& targets,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, targets, stream, mr);
}

}  // namespace strings
}  // namespace cudf
