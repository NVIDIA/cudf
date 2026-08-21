/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/normalizing_iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/std/utility>

namespace cudf {
namespace detail {

/**
 * @brief The index normalizing input iterator.
 *
 * This is an iterator that can be used for index types (integers) without
 * requiring a type-specific instance. It can be used for any iterator
 * interface for reading an array of integer values of type
 * int8, int16, int32, int64, uint8, uint16, uint32, or uint64.
 * Reading specific elements always return a `size_type` integer.
 *
 * Use the indexalator_factory to create an appropriate input iterator
 * from a column_view.
 *
 * Example input iterator usage.
 * @code
 *  auto begin = indexalator_factory::create_input_iterator(gather_map);
 *  auto end   = begin + gather_map.size();
 *  auto result = detail::gather( source, begin, end, IGNORE, stream, mr );
 * @endcode
 *
 * @code
 *  auto begin = indexalator_factory::create_input_iterator(indices);
 *  auto end   = begin + indices.size();
 *  auto result = thrust::find(thrust::device, begin, end, size_type{12} );
 * @endcode
 */
struct input_indexalator : base_normalator<input_indexalator, cudf::size_type> {
  friend struct base_normalator<input_indexalator, cudf::size_type>;  // for CRTP

  using reference = cudf::size_type const;  // this keeps STL and thrust happy

  input_indexalator()                                    = default;
  input_indexalator(input_indexalator const&)            = default;
  input_indexalator(input_indexalator&&)                 = default;
  input_indexalator& operator=(input_indexalator const&) = default;
  input_indexalator& operator=(input_indexalator&&)      = default;

  /**
   * @brief Indirection operator returns the value at the current iterator position
   */
  __device__ inline cudf::size_type operator*() const { return operator[](0); }

  /**
   * @brief Dispatch functor for resolving a Integer value from any integer type
   */
  struct normalize_type {
    template <typename T, CUDF_ENABLE_IF(cudf::is_index_type<T>())>
    __device__ cudf::size_type operator()(void const* tp)
    {
      return static_cast<cudf::size_type>(*static_cast<T const*>(tp));
    }
    template <typename T, CUDF_ENABLE_IF(not cudf::is_index_type<T>())>
    __device__ cudf::size_type operator()(void const*)
    {
      CUDF_UNREACHABLE("only integral types are supported");
    }
  };

  /**
   * @brief Array subscript operator returns a value at the input
   * `idx` position as a `Integer` value.
   */
  __device__ inline cudf::size_type operator[](size_type idx) const
  {
    void const* tp = p_ + (static_cast<std::ptrdiff_t>(idx) * this->width_);
    return type_dispatcher(this->dtype_, normalize_type{}, tp);
  }

  /**
   * @brief Create an input index normalizing iterator
   *
   * Use the indexalator_factory to create an iterator instance.
   *
   * @param data   Pointer to an integer array in device memory.
   * @param dtype  Type of data in data
   * @param offset Applied to the data pointer per size of the type
   */
  CUDF_HOST_DEVICE input_indexalator(void const* data, data_type dtype, cudf::size_type offset = 0)
    : base_normalator<input_indexalator, cudf::size_type>(dtype), p_{static_cast<char const*>(data)}
  {
    p_ += static_cast<std::ptrdiff_t>(offset) * this->width_;
  }

 protected:
  char const* p_;  /// pointer to the integer data in device memory
};

/**
 * @brief The index normalizing output iterator
 *
 * This is an iterator that can be used for index types (integers) without
 * requiring a type-specific instance. It can be used for any iterator
 * interface for writing an array of integer values of type
 * int8, int16, int32, int64, uint8, uint16, uint32, or uint64.
 * Setting specific elements always accept `size_type` integer values.
 *
 * Use the indexalator_factory to create an appropriate output iterator
 * from a mutable_column_view.
 *
 * Example output iterator usage.
 * @code
 *  auto result_itr = indexalator_factory::create_output_iterator(indices->mutable_view());
 *  thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
 *                      input->begin<Element>(),
 *                      input->end<Element>(),
 *                      values->begin<Element>(),
 *                      values->end<Element>(),
 *                      result_itr,
 *                      cuda::std::less<Element>());
 * @endcode
 */
struct output_indexalator : base_normalator<output_indexalator, cudf::size_type> {
  friend struct base_normalator<output_indexalator, cudf::size_type>;  // for CRTP

  using reference = output_indexalator const&;  // required for output iterators

  output_indexalator()                                     = default;
  output_indexalator(output_indexalator const&)            = default;
  output_indexalator(output_indexalator&&)                 = default;
  output_indexalator& operator=(output_indexalator const&) = default;
  output_indexalator& operator=(output_indexalator&&)      = default;

  /**
   * @brief Indirection operator returns this iterator instance in order
   * to capture the `operator=(Integer)` calls.
   */
  __device__ inline reference operator*() const { return *this; }

  /**
   * @brief Array subscript operator returns an iterator instance at the specified `idx` position.
   *
   * This allows capturing the subsequent `operator=(Integer)` call in this class.
   */
  __device__ inline output_indexalator const operator[](size_type idx) const
  {
    output_indexalator tmp{*this};
    tmp.p_ += static_cast<std::ptrdiff_t>(idx) * this->width_;
    return tmp;
  }

  /**
   * @brief Dispatch functor for setting the index value from a size_type value.
   */
  struct normalize_type {
    template <typename T, CUDF_ENABLE_IF(cudf::is_index_type<T>())>
    __device__ void operator()(void* tp, cudf::size_type const value)
    {
      (*static_cast<T*>(tp)) = static_cast<T>(value);
    }
    template <typename T, CUDF_ENABLE_IF(not cudf::is_index_type<T>())>
    __device__ void operator()(void*, cudf::size_type const)
    {
      CUDF_UNREACHABLE("only index types are supported");
    }
  };

  /**
   * @brief Assign an Integer value to the current iterator position
   */
  __device__ inline reference operator=(cudf::size_type const value) const
  {
    void* tp = p_;
    type_dispatcher(this->dtype_, normalize_type{}, tp, value);
    return *this;
  }

  /**
   * @brief Create an output normalizing iterator
   *
   * @param data      Pointer to an integer array in device memory.
   * @param dtype Type of data in data
   */
  CUDF_HOST_DEVICE output_indexalator(void* data, data_type dtype)
    : base_normalator<output_indexalator, cudf::size_type>(dtype), p_{static_cast<char*>(data)}
  {
  }

 protected:
  char* p_;  /// pointer to the integer data in device memory
};

/**
 * @brief Use this class to create an indexalator instance.
 */
struct indexalator_factory {
  /**
   * @brief A type_dispatcher functor to create an input iterator from an indices column.
   */
  struct input_indexalator_fn {
    template <typename IndexType, CUDF_ENABLE_IF(is_index_type<IndexType>())>
    input_indexalator operator()(column_view const& indices)
    {
      return input_indexalator(indices.data<IndexType>(), indices.type());
    }
    template <typename IndexType, typename... Args, CUDF_ENABLE_IF(not is_index_type<IndexType>())>
    input_indexalator operator()(Args&&... args)
    {
      CUDF_FAIL("indices must be an index type");
    }
  };

  /**
   * @brief Use this class to create an indexalator to a scalar index.
   */
  struct input_indexalator_scalar_fn {
    template <typename IndexType, CUDF_ENABLE_IF(is_index_type<IndexType>())>
    input_indexalator operator()(scalar const& index)
    {
      // note: using static_cast<scalar_type_t<IndexType> const&>(index) creates a copy
      auto const scalar_impl = static_cast<scalar_type_t<IndexType> const*>(&index);
      return input_indexalator(scalar_impl->data(), index.type());
    }
    template <typename IndexType, typename... Args, CUDF_ENABLE_IF(not is_index_type<IndexType>())>
    input_indexalator operator()(Args&&... args)
    {
      CUDF_FAIL("scalar must be an index type");
    }
  };

  /**
   * @brief A type_dispatcher functor to create an output iterator from an indices column.
   */
  struct output_indexalator_fn {
    template <typename IndexType, CUDF_ENABLE_IF(is_index_type<IndexType>())>
    output_indexalator operator()(mutable_column_view const& indices)
    {
      return output_indexalator(indices.data<IndexType>(), indices.type());
    }
    template <typename IndexType, typename... Args, CUDF_ENABLE_IF(not is_index_type<IndexType>())>
    output_indexalator operator()(Args&&... args)
    {
      CUDF_FAIL("indices must be an index type");
    }
  };

  /**
   * @brief Create an input indexalator instance from an indices column.
   */
  static input_indexalator make_input_iterator(column_view const& indices)
  {
    return type_dispatcher(indices.type(), input_indexalator_fn{}, indices);
  }

  /**
   * @brief Create an input indexalator instance from an index scalar.
   */
  static input_indexalator make_input_iterator(cudf::scalar const& index)
  {
    return type_dispatcher(index.type(), input_indexalator_scalar_fn{}, index);
  }

  /**
   * @brief Create an output indexalator instance from an indices column.
   */
  static output_indexalator make_output_iterator(mutable_column_view const& indices)
  {
    return type_dispatcher(indices.type(), output_indexalator_fn{}, indices);
  }

  /**
   * @brief An index accessor that returns an index value if corresponding validity flag is true.
   *
   * This is suitable as an `optional_iterator`.
   */
  struct optional_index_accessor {
    input_indexalator iter;
    bitmask_type const* null_mask{};
    size_type const offset{};
    bool const has_nulls{};

    /**
     * @brief Create an accessor from a column_view.
     */
    optional_index_accessor(column_view const& col, bool has_nulls = false)
      : null_mask{col.null_mask()}, offset{col.offset()}, has_nulls{has_nulls}
    {
      if (has_nulls) { CUDF_EXPECTS(col.nullable(), "Unexpected non-nullable column."); }
      iter = make_input_iterator(col);
    }

    __device__ cuda::std::optional<size_type> operator()(size_type i) const
    {
      return has_nulls && !bit_is_set(null_mask, i + offset) ? cuda::std::nullopt
                                                             : cuda::std::make_optional(iter[i]);
    }
  };

  /**
   * @brief An index accessor that returns an index value if the scalar's validity flag is true.
   *
   * This is suitable as an `optional_iterator`.
   */
  struct scalar_optional_index_accessor {
    input_indexalator iter;
    bool const is_null;

    /**
     * @brief Create an accessor from a scalar.
     */
    scalar_optional_index_accessor(scalar const& input, rmm::cuda_stream_view stream)
      : is_null{!input.is_valid(stream)}
    {
      iter = indexalator_factory::make_input_iterator(input);
    }

    __device__ cuda::std::optional<size_type> operator()(size_type) const
    {
      return is_null ? cuda::std::nullopt : cuda::std::make_optional(*iter);
    }
  };

  /**
   * @brief Create an index iterator with an optional index accessor.
   */
  static auto make_input_optional_iterator(column_view const& col)
  {
    return make_counting_transform_iterator(0, optional_index_accessor{col, col.has_nulls()});
  }

  /**
   * @brief Create an index iterator with an optional index accessor for a scalar.
   */
  static auto make_input_optional_iterator(scalar const& input, rmm::cuda_stream_view stream)
  {
    return cuda::transform_iterator(cuda::make_constant_iterator<size_type>(0),
                                    scalar_optional_index_accessor{input, stream});
  }
};

}  // namespace detail
}  // namespace cudf
