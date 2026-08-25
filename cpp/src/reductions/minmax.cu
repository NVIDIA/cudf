/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/transform_reduce.h>

#include <type_traits>

namespace cudf {
namespace reduction {
namespace detail {

namespace {

/**
 * @brief Basic element for the minmax reduce operation.
 *
 * Stores the minimum and maximum values that have been encountered so far
 */
template <typename T>
struct minmax_pair {
  T min_val;
  T max_val;

  __host__ __device__ minmax_pair()
    : min_val(cudf::DeviceMin::identity<T>()), max_val(cudf::DeviceMax::identity<T>()) {};
  __host__ __device__ minmax_pair(T val) : min_val(val), max_val(val) {};
  __host__ __device__ minmax_pair(T min_val_, T max_val_) : min_val(min_val_), max_val(max_val_) {};
};

/**
 * @brief Reduce for the minmax operation and return a device scalar.
 *
 * @tparam Op Binary operator functor
 * @tparam InputIterator Input iterator Type
 * @tparam OutputType Output scalar type
 * @param d_in input iterator
 * @param num_items number of items to reduce
 * @param binary_op binary operator used to reduce
 * @param stream CUDA stream to run kernels on.
 * @return cudf::detail::device_scalar<OutputType>
 */
template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
auto reduce_device(InputIterator d_in, size_type num_items, Op binary_op, cuda::stream_ref stream)
{
  OutputType identity{};
  cudf::detail::device_scalar<OutputType> result{
    identity, stream, cudf::get_current_device_resource_ref()};

  // Allocate temporary storage
  size_t storage_bytes = 0;
  cub::DeviceReduce::Reduce(
    nullptr, storage_bytes, d_in, result.data(), num_items, binary_op, identity, stream.get());
  auto temp_storage =
    rmm::device_buffer{storage_bytes, stream, cudf::get_current_device_resource_ref()};

  // Run reduction
  cub::DeviceReduce::Reduce(temp_storage.data(),
                            storage_bytes,
                            d_in,
                            result.data(),
                            num_items,
                            binary_op,
                            identity,
                            stream.get());

  return result;
}

/**
 * @brief Functor that accepts two minmax_pairs and returns a
 * minmax_pair whose minimum and maximum values are the min() and max()
 * respectively of the minimums and maximums of the input pairs.
 */
template <typename T>
struct minmax_binary_op {
  __device__ minmax_pair<T> operator()(minmax_pair<T> const& lhs, minmax_pair<T> const& rhs) const
  {
    return minmax_pair<T>{cuda::std::min(lhs.min_val, rhs.min_val),
                          cuda::std::max(lhs.max_val, rhs.max_val)};
  }
};

/**
 * @brief Creates a minmax_pair<T> from a T
 */
template <typename T>
struct create_minmax {
  __device__ minmax_pair<T> operator()(T e) { return minmax_pair<T>{e}; }
};

/**
 * @brief Functor that takes a cuda::std::pair<T, bool> and produces a minmax_pair
 * that is <T, T> for minimum and maximum or <cudf::DeviceMin::identity<T>(),
 * cudf::DeviceMax::identity<T>()>
 */
template <typename T>
struct create_minmax_with_nulls {
  __device__ minmax_pair<T> operator()(cuda::std::pair<T, bool> i)
  {
    return i.second ? minmax_pair<T>{i.first} : minmax_pair<T>{};
  }
};

/**
 * @brief Functor to copy a minmax_pair result to individual scalar instances.
 *
 * @tparam T type of the data
 * @tparam ResultType result type to assign min, max to minmax_pair<T>
 */
template <typename T, typename ResultType = minmax_pair<T>>
struct assign_min_max {
  __device__ void operator()()
  {
    *min_data = result->min_val;
    *max_data = result->max_val;
  }

  ResultType* result;
  T* min_data;
  T* max_data;
};

template <typename T>
std::unique_ptr<cudf::scalar_type_t<T>> make_minmax_scalar(cudf::data_type type,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  if constexpr (cudf::is_fixed_point<T>()) {
    return std::make_unique<cudf::scalar_type_t<T>>(
      device_storage_type_t<T>{}, numeric::scale_type{type.scale()}, true, stream, mr);
  }
  return std::make_unique<cudf::scalar_type_t<T>>(T{}, true, stream, mr);
}

/**
 * @brief Computes a minmax_pair<T> reduction directly over a dictionary column's decoded key
 * values, i.e. `keys[indices[i]]` for each row `i`.
 *
 * @tparam T The dictionary's key type
 */
template <typename T>
auto reduce_dictionary(column_view const& col, cuda::stream_ref stream)
{
  auto d_dictionary = column_device_view::create(col, stream);
  if (col.has_nulls()) {
    auto pair_to_minmax = cuda::make_transform_iterator(
      cudf::dictionary::detail::make_dictionary_pair_iterator<T>(*d_dictionary, true),
      create_minmax_with_nulls<T>{});
    return reduce_device(pair_to_minmax, col.size(), minmax_binary_op<T>{}, stream);
  } else {
    auto col_to_minmax = cuda::make_transform_iterator(
      cudf::dictionary::detail::make_dictionary_iterator<T>(*d_dictionary), create_minmax<T>{});
    return reduce_device(col_to_minmax, col.size(), minmax_binary_op<T>{}, stream);
  }
}

/**
 * @brief Dispatch functor for minmax operation on a dictionary column, dispatched on the
 * dictionary's key type.
 */
struct minmax_dictionary_functor {
  template <typename T>
  static constexpr bool is_supported()
  {
    return !cudf::is_dictionary<T>() && !std::is_same_v<T, cudf::list_view> &&
           !std::is_same_v<T, cudf::struct_view>;
  }

  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    column_view const& col, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    requires(is_supported<T>() and !std::is_same_v<T, cudf::string_view>)
  {
    using storage_type  = device_storage_type_t<T>;
    auto dev_result     = reduce_dictionary<storage_type>(col, stream);
    auto const key_type = dictionary_column_view(col).keys().type();
    auto minimum        = make_minmax_scalar<T>(key_type, stream, mr);
    auto maximum        = make_minmax_scalar<T>(key_type, stream, mr);
    cudf::detail::device_single_thread(
      assign_min_max<storage_type>{dev_result.data(), minimum->data(), maximum->data()}, stream);
    return {std::move(minimum), std::move(maximum)};
  }

  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    column_view const& col, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    requires(std::is_same_v<T, cudf::string_view>)
  {
    auto dev_result        = reduce_dictionary<cudf::string_view>(col, stream);
    auto const host_result = dev_result.value(stream);
    return {std::make_unique<string_scalar>(host_result.min_val, true, stream, mr),
            std::make_unique<string_scalar>(host_result.max_val, true, stream, mr)};
  }

  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    column_view const&, cuda::stream_ref, rmm::device_async_resource_ref)
    requires(!is_supported<T>())
  {
    CUDF_FAIL("dictionary key type not supported for minmax() operation");
  }
};

/**
 * @brief Dispatch functor for minmax operation.
 *
 * This uses the reduce function to compute the min and max values
 * simultaneously for a column of data.
 *
 * @tparam T The input column's type
 */
struct minmax_functor {
  template <typename T>
  static constexpr bool is_supported()
  {
    return !(std::is_same_v<T, cudf::list_view> || std::is_same_v<T, cudf::struct_view>);
  }

  template <typename T>
  auto reduce(column_view const& col, cuda::stream_ref stream)
  {
    auto device_col = column_device_view::create(col, stream);
    // compute minimum and maximum values
    if (col.has_nulls()) {
      auto pair_to_minmax = cuda::make_transform_iterator(
        cudf::detail::make_pair_iterator<T, true>(*device_col), create_minmax_with_nulls<T>{});
      return reduce_device(pair_to_minmax, col.size(), minmax_binary_op<T>{}, stream);
    } else {
      auto col_to_minmax =
        cuda::make_transform_iterator(device_col->begin<T>(), create_minmax<T>{});
      return reduce_device(col_to_minmax, col.size(), minmax_binary_op<T>{}, stream);
    }
  }

  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const& col, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    requires(is_supported<T>() and !std::is_same_v<T, cudf::string_view> and
             !cudf::is_dictionary<T>())
  {
    using storage_type = device_storage_type_t<T>;
    // compute minimum and maximum values
    auto dev_result = reduce<storage_type>(col, stream);
    // create output scalars
    auto minimum = make_minmax_scalar<T>(col.type(), stream, mr);
    auto maximum = make_minmax_scalar<T>(col.type(), stream, mr);
    // copy dev_result to the output scalars
    cudf::detail::device_single_thread(
      assign_min_max<storage_type>{dev_result.data(), minimum->data(), maximum->data()}, stream);
    return {std::move(minimum), std::move(maximum)};
  }

  /**
   * @brief Specialization for strings column.
   */
  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const& col, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    requires(std::is_same_v<T, cudf::string_view>)
  {
    // compute minimum and maximum values
    auto dev_result = reduce<cudf::string_view>(col, stream);
    // copy the minmax_pair to the host; does not copy the strings
    auto const host_result = dev_result.value(stream);
    // strings are copied to create the scalars here
    return {std::make_unique<string_scalar>(host_result.min_val, true, stream, mr),
            std::make_unique<string_scalar>(host_result.max_val, true, stream, mr)};
  }

  /**
   * @brief Specialization for dictionary column.
   */
  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const& col, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    requires(cudf::is_dictionary<T>())
  {
    auto const keys_type = dictionary_column_view(col).keys().type();
    return type_dispatcher(keys_type, minmax_dictionary_functor{}, col, stream, mr);
  }

  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const&, cuda::stream_ref, rmm::device_async_resource_ref)
    requires(!is_supported<T>())
  {
    CUDF_FAIL("type not supported for minmax() operation");
  }
};

}  // namespace

/**
 * @copydoc cudf::minmax
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  cudf::column_view const& col, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (col.null_count() == col.size()) {
    // this handles empty and all-null columns; return scalars with valid==false.
    // For dictionary columns use the keys type — DICTIONARY32 has no scalar representation.
    auto const scalar_type = col.type().id() == type_id::DICTIONARY32
                               ? dictionary_column_view(col).keys().type()
                               : col.type();
    return {make_default_constructed_scalar(scalar_type, stream, mr),
            make_default_constructed_scalar(scalar_type, stream, mr)};
  }

  return type_dispatcher(col.type(), minmax_functor{}, col, stream, mr);
}
}  // namespace detail
}  // namespace reduction

std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  column_view const& col, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return reduction::detail::minmax(col, stream, mr);
}

}  // namespace cudf
