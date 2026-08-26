/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace test {
namespace detail {
/**
 * @brief Base class for a wrapper around a `cudf::column`.
 *
 * Classes that derive from `column_wrapper` may be passed directly into any
 * API expecting a `column_view` or `mutable_column_view`.
 *
 * `column_wrapper` should not be instantiated directly.
 */
class column_wrapper {
 public:
  /**
   * @brief Implicit conversion operator to `column_view`.
   *
   * Allows passing in a `column_wrapper` (or any class deriving from
   * `column_wrapper`) to be passed into any API expecting a `column_view`
   * parameter.
   */
  operator column_view() const { return wrapped->view(); }

  /**
   * @brief Implicit conversion operator to `mutable_column_view`.
   *
   * Allows passing in a `column_wrapper` (or any class deriving from
   * `column_wrapper`) to be passed into any API expecting a
   * `mutable_column_view` parameter.
   */
  operator mutable_column_view() { return wrapped->mutable_view(); }

  /**
   * @brief Releases internal unique_ptr to wrapped column
   *
   * @return unique_ptr to wrapped column
   */
  std::unique_ptr<cudf::column> release() { return std::move(wrapped); }

 protected:
  std::unique_ptr<cudf::column> wrapped{};  ///< The wrapped column
};

/**
 * @brief Convert between source and target types when they differ and where possible.
 */
template <typename From, typename To>
struct fixed_width_type_converter {
  /**
   * @brief No conversion necessary: Same type, simply copy element to output.
   *
   * @tparam FromT Source type
   * @tparam ToT Target type
   * @param element Source value
   * @return The converted target value, same as source value
   */
  template <typename FromT                                      = From,
            typename ToT                                        = To,
            std::enable_if_t<std::is_same_v<FromT, ToT>, void>* = nullptr>
  constexpr ToT operator()(FromT element) const
  {
    return element;
  }

  /**
   * @brief Convert types if possible, otherwise construct target from source.
   *
   * @tparam FromT Source type
   * @tparam ToT Target type
   * @param element Source value
   * @return The converted target value
   */
  template <
    typename FromT          = From,
    typename ToT            = To,
    std::enable_if_t<!std::is_same_v<FromT, ToT> && (cudf::is_convertible<FromT, ToT>::value ||
                                                     std::is_constructible_v<ToT, FromT>),
                     void>* = nullptr>
  constexpr ToT operator()(FromT element) const
  {
    return static_cast<ToT>(element);
  }

  /**
   * @brief Convert integral values to timestamps
   *
   * @tparam FromT Source type
   * @tparam ToT Target type
   * @param element Source value
   * @return The converted target `timestamp` value
   */
  template <
    typename FromT                                                                  = From,
    typename ToT                                                                    = To,
    std::enable_if_t<std::is_integral_v<FromT> && cudf::is_timestamp<ToT>(), void>* = nullptr>
  constexpr ToT operator()(FromT element) const
  {
    return ToT{typename ToT::duration{element}};
  }
};

/**
 * @brief Creates a `device_buffer` containing the elements in the range `[begin,end)`.
 *
 * @tparam ElementTo The element type that is being created (non-`fixed_point`)
 * @tparam ElementFrom The element type used to create elements of type `ElementTo`
 * @tparam InputIterator Iterator type for `begin` and `end`
 * @param begin Beginning of the sequence of elements
 * @param end End of the sequence of elements
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used to allocate the returned buffer
 * @return rmm::device_buffer Buffer containing all elements in the range `[begin,end)`
 */
template <typename ElementTo,
          typename ElementFrom,
          typename InputIterator,
          std::enable_if_t<not cudf::is_fixed_point<ElementTo>()>* = nullptr>
rmm::device_buffer make_elements(InputIterator begin,
                                 InputIterator end,
                                 rmm::cuda_stream_view stream,
                                 cudf::memory_resources mr)
{
  static_assert(cudf::is_fixed_width<ElementTo>(), "Unexpected non-fixed width type.");
  auto const size = cudf::distance(begin, end);
  auto elements   = thrust::host_vector<ElementTo>(size);
  std::transform(
    begin, end, elements.begin(), fixed_width_type_converter<ElementFrom, ElementTo>{});
  rmm::device_buffer buffer{elements.data(), size * sizeof(ElementTo), stream, mr.get_output_mr()};
  stream.synchronize();  // wait for async H2D before host source is destroyed
  return buffer;
}

// The two signatures below are identical to the above overload apart from
// SFINAE so doxygen sees it as a duplicate.
//! @cond Doxygen_Suppress
/**
 * @brief Creates a `device_buffer` containing the elements in the range `[begin,end)`.
 *
 * @tparam ElementTo The element type that is being created (`fixed_point` specialization)
 * @tparam ElementFrom The element type used to create elements of type `ElementTo`
 * (non-`fixed-point`)
 * @tparam InputIterator Iterator type for `begin` and `end`
 * @param begin Beginning of the sequence of elements
 * @param end End of the sequence of elements
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used to allocate the returned buffer
 * @return rmm::device_buffer Buffer containing all elements in the range `[begin,end)`
 */
template <typename ElementTo,
          typename ElementFrom,
          typename InputIterator,
          std::enable_if_t<not cudf::is_fixed_point<ElementFrom>() and
                           cudf::is_fixed_point<ElementTo>()>* = nullptr>
rmm::device_buffer make_elements(InputIterator begin,
                                 InputIterator end,
                                 rmm::cuda_stream_view stream,
                                 cudf::memory_resources mr)
{
  using RepType   = typename ElementTo::rep;
  auto const size = cudf::distance(begin, end);
  auto elements   = thrust::host_vector<RepType>(size);
  std::transform(begin, end, elements.begin(), fixed_width_type_converter<ElementFrom, RepType>{});
  rmm::device_buffer buffer{elements.data(), size * sizeof(RepType), stream, mr.get_output_mr()};
  stream.synchronize();  // wait for async H2D before host source is destroyed
  return buffer;
}

/**
 * @brief Creates a `device_buffer` containing the elements in the range `[begin,end)`.
 *
 * @tparam ElementTo The element type that is being created (`fixed_point` specialization)
 * @tparam ElementFrom The element type used to create elements of type `ElementTo` (`fixed_point`)
 * @tparam InputIterator Iterator type for `begin` and `end`
 * @param begin Beginning of the sequence of elements
 * @param end End of the sequence of elements
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used to allocate the returned buffer
 * @return rmm::device_buffer Buffer containing all elements in the range `[begin,end)`
 */
template <typename ElementTo,
          typename ElementFrom,
          typename InputIterator,
          std::enable_if_t<cudf::is_fixed_point<ElementFrom>() and
                           cudf::is_fixed_point<ElementTo>()>* = nullptr>
rmm::device_buffer make_elements(InputIterator begin,
                                 InputIterator end,
                                 rmm::cuda_stream_view stream,
                                 cudf::memory_resources mr)
{
  using namespace numeric;
  using RepType = typename ElementTo::rep;

  CUDF_EXPECTS(std::all_of(begin, end, [](ElementFrom v) { return v.scale() == 0; }),
               "Only zero-scale fixed-point values are supported");

  auto const size = cudf::distance(begin, end);
  auto elements   = thrust::host_vector<RepType>(size);
  std::transform(begin, end, elements.begin(), [](ElementTo const fp) { return fp.value(); });
  rmm::device_buffer buffer{elements.data(), size * sizeof(RepType), stream, mr.get_output_mr()};
  stream.synchronize();  // wait for async H2D before host source is destroyed
  return buffer;
}
//! @endcond

/**
 * @brief Create a `std::vector` containing a validity indicator bitmask using
 * the range `[begin,end)` interpreted as booleans to indicate the state of
 * each bit.
 *
 * If `*(begin + i) == true`, then bit `i` is set to 1, else it is zero.
 *
 * @tparam ValidityIterator
 * @param begin The beginning of the validity indicator sequence
 * @param end The end of the validity indicator sequence
 * @return std::vector Contains a bitmask where bits are set for every
 * element in `[begin,end)` that evaluated to `true`.
 */
template <typename ValidityIterator>
std::pair<std::vector<bitmask_type>, cudf::size_type> make_null_mask_vector(ValidityIterator begin,
                                                                            ValidityIterator end)
{
  auto const size      = cudf::distance(begin, end);
  auto const num_words = cudf::bitmask_allocation_size_bytes(size) / sizeof(bitmask_type);

  auto null_mask  = std::vector<bitmask_type>(num_words, 0);
  auto null_count = cudf::size_type{0};
  for (auto i = 0; i < size; ++i) {
    if (*(begin + i)) {
      set_bit_unsafe(null_mask.data(), i);
    } else {
      ++null_count;
    }
  }

  return {std::move(null_mask), null_count};
}

/**
 * @brief Create a `device_buffer` containing a validity indicator bitmask using
 * the range `[begin,end)` interpreted as booleans to indicate the state of
 *each bit.
 *
 * If `*(begin + i) == true`, then bit `i` is set to 1, else it is zero.
 *
 * @tparam ValidityIterator
 * @param begin The beginning of the validity indicator sequence
 * @param end The end of the validity indicator sequence
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used to allocate the returned buffer
 * @return rmm::device_buffer Contains a bitmask where bits are set for every
 * element in `[begin,end)` that evaluated to `true`.
 */
template <typename ValidityIterator>
std::pair<rmm::device_buffer, cudf::size_type> make_null_mask(
  ValidityIterator begin,
  ValidityIterator end,
  rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
  cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
{
  auto [null_mask, null_count] = make_null_mask_vector(begin, end);
  rmm::device_buffer d_mask{null_mask.data(),
                            cudf::bitmask_allocation_size_bytes(cudf::distance(begin, end)),
                            stream,
                            mr.get_output_mr()};
  stream.synchronize();  // wait for async H2D before host source is destroyed
  return {std::move(d_mask), null_count};
}

/**
 * @brief Given the range `[begin,end)`, converts each value to a string and
 * then creates a packed vector of characters for each string and a vector of
 * offsets indicating the starting position of each string.
 *
 * @tparam StringsIterator A `std::string` must be constructible from
 * dereferencing a `StringsIterator`.
 * @tparam ValidityIterator Dereferencing a ValidityIterator must be
 * convertible to `bool`
 * @param begin The beginning of the sequence of values to convert to strings
 * @param end The end of the sequence of values to convert to strings
 * @param v The beginning of the validity indicator sequence
 * @return std::pair containing the vector of chars and offsets
 */
template <typename StringsIterator, typename ValidityIterator>
auto make_chars_and_offsets(StringsIterator begin, StringsIterator end, ValidityIterator v)
{
  std::vector<char> chars{};
  std::vector<int32_t> offsets(1, 0);
  for (auto str = begin; str < end; ++str) {
    std::string tmp = (*v++) ? std::string(*str) : std::string{};
    chars.insert(chars.end(), std::cbegin(tmp), std::cend(tmp));
    auto const last_offset = static_cast<std::size_t>(offsets.back());
    auto const next_offset = last_offset + tmp.length();
    CUDF_EXPECTS(next_offset < static_cast<std::size_t>(std::numeric_limits<int32_t>::max()),
                 "Cannot use strings_column_wrapper to build a large strings column");
    offsets.push_back(static_cast<int32_t>(next_offset));
  }
  return std::pair(std::move(chars), std::move(offsets));
};

}  // namespace detail

/**
 * @brief `column_wrapper` derived class for wrapping columns of fixed-width
 * elements.
 *
 * @tparam ElementTo The fixed-width element type that is created
 * @tparam SourceElementT The fixed-width element type that is used to create elements of type
 * `ElementTo`
 */
template <typename ElementTo, typename SourceElementT = ElementTo>
class fixed_width_column_wrapper : public detail::column_wrapper {
 public:
  /**
   * @brief Default constructor initializes an empty column with proper dtype
   */
  fixed_width_column_wrapper() : column_wrapper{}
  {
    wrapped = cudf::make_empty_column(cudf::type_to_id<ElementTo>());
  }

  /**
   * @brief Construct a non-nullable column of the fixed-width elements in the
   * range `[begin,end)`.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable column of INT32 elements with 5 elements: {0, 2, 4, 6, 8}
   * auto elements = make_counting_transform_iterator(0, [](auto i){return i*2;});
   * fixed_width_column_wrapper<int32_t> w(elements, elements + 5);
   * @endcode
   *
   * Note: similar to `std::vector`, this "range" constructor should be used
   *       with parentheses `()` and not braces `{}`. The latter should only
   *       be used for the `initializer_list` constructors
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename InputIterator>
  fixed_width_column_wrapper(InputIterator begin,
                             InputIterator end,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    auto const size = cudf::distance(begin, end);
    wrapped.reset(
      new cudf::column{cudf::data_type{cudf::type_to_id<ElementTo>()},
                       size,
                       detail::make_elements<ElementTo, SourceElementT>(begin, end, stream, mr),
                       rmm::device_buffer{},
                       0});
  }

  /**
   * @brief Construct a nullable column of the fixed-width elements in the range
   * `[begin,end)` using the range `[v, v + distance(begin,end))` interpreted
   * as booleans to indicate the validity of each element.
   *
   * If `v[i] == true`, element `i` is valid, else it is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable column of INT32 elements with 5 elements: {null, 1, null, 3, null}
   * auto elements = make_counting_transform_iterator(0, [](auto i){return i;});
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;})
   * fixed_width_column_wrapper<int32_t> w(elements, elements + 5, validity);
   * @endcode
   *
   * Note: similar to `std::vector`, this "range" constructor should be used
   *       with parentheses `()` and not braces `{}`. The latter should only
   *       be used for the `initializer_list` constructors
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   * @param v The beginning of the sequence of validity indicators
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename InputIterator, typename ValidityIterator>
  fixed_width_column_wrapper(InputIterator begin,
                             InputIterator end,
                             ValidityIterator v,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    auto const size              = cudf::distance(begin, end);
    auto [null_mask, null_count] = detail::make_null_mask(v, v + size, stream, mr);
    wrapped.reset(
      new cudf::column{cudf::data_type{cudf::type_to_id<ElementTo>()},
                       size,
                       detail::make_elements<ElementTo, SourceElementT>(begin, end, stream, mr),
                       std::move(null_mask),
                       null_count});
  }

  /**
   * @brief Construct a non-nullable column of fixed-width elements from an
   * initializer list.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable INT32 column with 4 elements: {1, 2, 3, 4}
   * fixed_width_column_wrapper<int32_t> w{{1, 2, 3, 4}};
   * @endcode
   *
   * @param elements The list of elements
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ElementFrom>
  fixed_width_column_wrapper(std::initializer_list<ElementFrom> elements,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : fixed_width_column_wrapper(std::cbegin(elements), std::cend(elements), stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column from a list of fixed-width elements
   * using another list to indicate the validity of each element.
   *
   * The validity of each element is determined by an `initializer_list` of
   * booleans where `true` indicates the element is valid, and `false` indicates
   * the element is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable INT32 column with 4 elements: {1, NULL, 3, NULL}
   * fixed_width_column_wrapper<int32_t> w{ {1,2,3,4}, {1, 0, 1, 0}};
   * @endcode
   *
   * @param elements The list of elements
   * @param validity The list of validity indicator booleans
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ElementFrom>
  fixed_width_column_wrapper(std::initializer_list<ElementFrom> elements,
                             std::initializer_list<bool> validity,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : fixed_width_column_wrapper(
        std::cbegin(elements), std::cend(elements), std::cbegin(validity), stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column from a list of fixed-width elements and
   * the range `[v, v + element_list.size())` interpreted as booleans to
   * indicate the validity of each element.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable INT32 column with 4 elements: {NULL, 1, NULL, 3}
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;})
   * fixed_width_column_wrapper<int32_t> w{ {1,2,3,4}, validity}
   * @endcode
   *
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be
   * convertible to `bool`
   * @param element_list The list of elements
   * @param v The beginning of the sequence of validity indicators
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ValidityIterator, typename ElementFrom>
  fixed_width_column_wrapper(std::initializer_list<ElementFrom> element_list,
                             ValidityIterator v,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : fixed_width_column_wrapper(std::cbegin(element_list), std::cend(element_list), v, stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column of the fixed-width elements in the range
   * `[begin,end)` using a validity initializer list to indicate the validity of each element.
   *
   * The validity of each element is determined by an `initializer_list` of
   * booleans where `true` indicates the element is valid, and `false` indicates
   * the element is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable column of INT32 elements with 5 elements: {null, 1, null, 3, null}
   * fixed_width_column_wrapper<int32_t> w(elements, elements + 5, {0, 1, 0, 1, 0});
   * @endcode
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   * @param validity The list of validity indicator booleans
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename InputIterator>
  fixed_width_column_wrapper(InputIterator begin,
                             InputIterator end,
                             std::initializer_list<bool> const& validity,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : fixed_width_column_wrapper(begin, end, std::cbegin(validity), stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column from a list of pairs of fixed-width
   * elements and validity booleans of each element.
   *
   * The validity of each element is determined by the boolean element in the pair
   * where `true` indicates the element is valid, and `false` indicates the
   * element is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable INT32 column with 4 elements: {1, NULL, 3, NULL}
   * using p = std::pair<int32_t, bool>;
   * fixed_width_column_wrapper<int32_t> w( p{1, true}, p{2, false}, p{3, true}, p{4, false} );
   * @endcode
   *
   * @param elements The list of pairs of element and validity booleans
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ElementFrom>
  fixed_width_column_wrapper(std::initializer_list<std::pair<ElementFrom, bool>> elements,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
  {
    auto values   = std::vector<ElementFrom>{};
    auto validity = std::vector<bool>{};
    values.reserve(elements.size());
    validity.reserve(elements.size());
    for (auto const& [value, valid] : elements) {
      values.push_back(value);
      validity.push_back(valid);
    }
    wrapped = fixed_width_column_wrapper<ElementTo, ElementFrom>(
                values.begin(), values.end(), validity.begin(), stream, mr)
                .release();
  }
};

/**
 * @brief A wrapper for a column of fixed-width elements.
 *
 * @tparam Rep The type of the column
 */
template <typename Rep>
class fixed_point_column_wrapper : public detail::column_wrapper {
 public:
  /**
   * @brief Construct a non-nullable column of the decimal elements in the range `[begin,end)`.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable column of DECIMAL32 elements with 5 elements: {0, 2, 4, 6, 8}
   * auto elements = make_counting_transform_iterator(0, [](auto i) { return i * 2;});
   * auto w = fixed_point_column_wrapper<int32_t>(elements, elements + 5, scale_type{0});
   * @endcode
   *
   * @tparam FixedPointRepIterator Iterator for fixed_point::rep
   *
   * @param begin The beginning of the sequence of elements
   * @param end   The end of the sequence of elements
   * @param scale The scale of the elements in the column
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename FixedPointRepIterator>
  fixed_point_column_wrapper(FixedPointRepIterator begin,
                             FixedPointRepIterator end,
                             numeric::scale_type scale,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    CUDF_EXPECTS(numeric::is_supported_representation_type<Rep>(), "not valid representation type");

    auto const size      = cudf::distance(begin, end);
    auto const elements  = thrust::host_vector<Rep>(begin, end);
    auto const id        = type_to_id<numeric::fixed_point<Rep, numeric::Radix::BASE_10>>();
    auto const data_type = cudf::data_type{id, static_cast<int32_t>(scale)};
    rmm::device_buffer data{elements.data(), size * sizeof(Rep), stream, mr.get_output_mr()};
    wrapped.reset(new cudf::column{data_type, size, std::move(data), rmm::device_buffer{}, 0});
    stream.synchronize();  // wait for async H2D before host source is destroyed
  }

  /**
   * @brief Construct a non-nullable column of decimal elements from an initializer list.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable `decimal32` column with 4 elements: {42.0, 4.2, 0.4}
   * auto const col = fixed_point_column_wrapper<int32_t>{{420, 42, 4}, scale_type{-1}};
   * @endcode
   *
   * @param values The initializer list of already shifted values
   * @param scale  The scale of the elements in the column
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  fixed_point_column_wrapper(std::initializer_list<Rep> values,
                             numeric::scale_type scale,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : fixed_point_column_wrapper(std::cbegin(values), std::cend(values), scale, stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column of the fixed-point elements from a range.
   *
   * Constructs a nullable column of the fixed-point elements in the range `[begin,end)` using the
   * range `[v, v + distance(begin,end))` interpreted as Booleans to indicate the validity of each
   * element.
   *
   * If `v[i] == true`, element `i` is valid, else it is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable column of DECIMAL32 elements with 5 elements: {null, 100, null, 300,
   * null}
   * auto elements = make_counting_transform_iterator(0, [](auto i){ return i; });
   * auto validity = make_counting_transform_iterator(0, [](auto i){ return i % 2; });
   * fixed_point_column_wrapper<int32_t> w(elements, elements + 5, validity, scale_type{2});
   * @endcode
   *
   * Note: similar to `std::vector`, this "range" constructor should be used
   *       with parentheses `()` and not braces `{}`. The latter should only
   *       be used for the `initializer_list` constructors
   *
   * @param begin The beginning of the sequence of elements
   * @param end   The end of the sequence of elements
   * @param v     The beginning of the sequence of validity indicators
   * @param scale The scale of the elements in the column
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename FixedPointRepIterator, typename ValidityIterator>
  fixed_point_column_wrapper(FixedPointRepIterator begin,
                             FixedPointRepIterator end,
                             ValidityIterator v,
                             numeric::scale_type scale,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    CUDF_EXPECTS(numeric::is_supported_representation_type<Rep>(), "not valid representation type");

    auto const size              = cudf::distance(begin, end);
    auto const elements          = thrust::host_vector<Rep>(begin, end);
    auto const id                = type_to_id<numeric::fixed_point<Rep, numeric::Radix::BASE_10>>();
    auto const data_type         = cudf::data_type{id, static_cast<int32_t>(scale)};
    auto [null_mask, null_count] = detail::make_null_mask(v, v + size, stream, mr);
    rmm::device_buffer data{elements.data(), size * sizeof(Rep), stream, mr.get_output_mr()};
    wrapped.reset(
      new cudf::column{data_type, size, std::move(data), std::move(null_mask), null_count});
    stream.synchronize();  // wait for async H2D before host source is destroyed
  }

  /**
   * @brief Construct a nullable column from an initializer list of decimal elements using another
   * list to indicate the validity of each element.
   *
   * The validity of each element is determined by an `initializer_list` of booleans where `true`
   * indicates the element is valid, and `false` indicates the element is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable INT32 column with 4 elements: {1, null, 3, null}
   * fixed_width_column_wrapper<int32_t> w{ {1,2,3,4}, {1, 0, 1, 0}, scale_type{0}};
   * @endcode
   *
   * @param elements The initializer list of elements
   * @param validity The initializer list of validity indicator booleans
   * @param scale    The scale of the elements in the column
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  fixed_point_column_wrapper(std::initializer_list<Rep> elements,
                             std::initializer_list<bool> validity,
                             numeric::scale_type scale,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : fixed_point_column_wrapper(
        std::cbegin(elements), std::cend(elements), std::cbegin(validity), scale, stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column from an initializer list of decimal elements and the
   * range `[v, v + element_list.size())` interpreted as booleans to indicate the validity of each
   * element.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable INT32 column with 4 elements: {null, 1, null, 3}
   * auto validity = make_counting_transform_iterator(0, [](auto i) { return i % 2; });
   * auto w        = fixed_width_column_wrapper<int32_t>{ {1,2,3,4}, validity, scale_type{0}};
   * @endcode
   *
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be convertible to `bool`
   *
   * @param element_list The initializer list of elements
   * @param v            The beginning of the sequence of validity indicators
   * @param scale        The scale of the elements in the column
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ValidityIterator>
  fixed_point_column_wrapper(std::initializer_list<Rep> element_list,
                             ValidityIterator v,
                             numeric::scale_type scale,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : fixed_point_column_wrapper(
        std::cbegin(element_list), std::cend(element_list), v, scale, stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column of the decimal elements in the range `[begin,end)` using a
   * validity initializer list to indicate the validity of each element.
   *
   * The validity of each element is determined by an `initializer_list` of booleans where `true`
   * indicates the element is valid, and `false` indicates the element is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable column of DECIMAL32 elements with 5 elements: {null, 1, null, 3, null}
   * fixed_point_column_wrapper<int32_t> w(elements, elements + 5, {0, 1, 0, 1, 0}, scale_type{0});
   * @endcode
   *
   * @tparam FixedPointRepIterator Iterator for fixed_point::rep
   *
   * @param begin    The beginning of the sequence of elements
   * @param end      The end of the sequence of elements
   * @param validity The initializer list of validity indicator booleans
   * @param scale    The scale of the elements in the column
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename FixedPointRepIterator>
  fixed_point_column_wrapper(FixedPointRepIterator begin,
                             FixedPointRepIterator end,
                             std::initializer_list<bool> const& validity,
                             numeric::scale_type scale,
                             rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                             cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : fixed_point_column_wrapper(begin, end, std::cbegin(validity), scale, stream, mr)
  {
  }
};

/**
 * @brief `column_wrapper` derived class for wrapping columns of strings.
 */
class strings_column_wrapper : public detail::column_wrapper {
 public:
  /**
   * @brief Default constructor initializes an empty column of strings
   */
  strings_column_wrapper() : column_wrapper{}
  {
    wrapped = cudf::make_empty_column(cudf::type_id::STRING);
  }

  /**
   * @brief Construct a non-nullable column of strings from the range
   * `[begin,end)`.
   *
   * Values in the sequence `[begin,end)` will each be converted to
   *`std::string` and a column will be created containing all of the strings.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable STRING column with 7 string elements:
   * // {"", "this", "is", "a", "column", "of", "strings"}
   * std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
   * strings_column_wrapper s(strings.begin(), strings.end());
   * @endcode
   *
   * @tparam StringsIterator A `std::string` must be constructible from
   * dereferencing a `StringsIterator`.
   * @param begin The beginning of the sequence
   * @param end The end of the sequence
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename StringsIterator>
  strings_column_wrapper(StringsIterator begin,
                         StringsIterator end,
                         rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                         cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    size_type num_strings = std::distance(begin, end);
    if (num_strings == 0) {
      wrapped = cudf::make_empty_column(cudf::type_id::STRING);
      return;
    }
    auto all_valid        = cuda::make_constant_iterator(true);
    auto [chars, offsets] = detail::make_chars_and_offsets(begin, end, all_valid);
    auto d_chars   = cudf::detail::make_device_uvector_async(chars, stream, mr.get_output_mr());
    auto d_offsets = std::make_unique<cudf::column>(
      cudf::detail::make_device_uvector(offsets, stream, mr.get_output_mr()),
      rmm::device_buffer{},
      0);
    wrapped =
      cudf::make_strings_column(num_strings, std::move(d_offsets), d_chars.release(), 0, {});
  }

  /**
   * @brief Construct a nullable column of strings from the range
   * `[begin,end)` using the range `[v, v + distance(begin,end))` interpreted
   * as booleans to indicate the validity of each string.
   *
   * Values in the sequence `[begin,end)` will each be converted to
   *`std::string` and a column will be created containing all of the strings.
   *
   * If `v[i] == true`, string `i` is valid, else it is null. If a string
   * `*(begin+i)` is null, it's value is ignored and treated as an empty string.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable STRING column with 7 string elements:
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * strings_column_wrapper s(strings.begin(), strings.end(), validity);
   * @endcode
   *
   * @tparam StringsIterator A `std::string` must be constructible from
   * dereferencing a `StringsIterator`.
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be convertible to `bool`
   *
   * @param begin The beginning of the sequence
   * @param end The end of the sequence
   * @param v The beginning of the sequence of validity indicators
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename StringsIterator, typename ValidityIterator>
  strings_column_wrapper(StringsIterator begin,
                         StringsIterator end,
                         ValidityIterator v,
                         rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                         cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    size_type num_strings = std::distance(begin, end);
    if (num_strings == 0) {
      wrapped = cudf::make_empty_column(cudf::type_id::STRING);
      return;
    }
    auto [chars, offsets]        = detail::make_chars_and_offsets(begin, end, v);
    auto [null_mask, null_count] = detail::make_null_mask_vector(v, v + num_strings);
    auto d_chars   = cudf::detail::make_device_uvector_async(chars, stream, mr.get_output_mr());
    auto d_offsets = std::make_unique<cudf::column>(
      cudf::detail::make_device_uvector_async(offsets, stream, mr.get_output_mr()),
      rmm::device_buffer{},
      0);
    auto d_bitmask = cudf::detail::make_device_uvector(null_mask, stream, mr.get_output_mr());
    wrapped        = cudf::make_strings_column(
      num_strings, std::move(d_offsets), d_chars.release(), null_count, d_bitmask.release());
  }

  /**
   * @brief Construct a non-nullable column of strings from a list of strings.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable STRING column with 7 string elements:
   * // {"", "this", "is", "a", "column", "of", "strings"}
   * strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"});
   * @endcode
   *
   * @param strings The list of strings
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  strings_column_wrapper(std::initializer_list<std::string> strings,
                         rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                         cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : strings_column_wrapper(std::cbegin(strings), std::cend(strings), stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column of strings from a list of strings and
   * the range `[v, v + strings.size())` interpreted as booleans to indicate the
   * validity of each string.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable STRING column with 7 string elements:
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"}, validity);
   * @endcode
   *
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be
   * convertible to `bool`
   * @param strings The list of strings
   * @param v The beginning of the sequence of validity indicators
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ValidityIterator>
  strings_column_wrapper(std::initializer_list<std::string> strings,
                         ValidityIterator v,
                         rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                         cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : strings_column_wrapper(std::cbegin(strings), std::cend(strings), v, stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column of strings from a list of strings and
   * a list of booleans to indicate the validity of each string.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable STRING column with 7 string elements:
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"},
   *                          {0,1,0,1,0,1,0});
   * @endcode
   *
   * @param strings The list of strings
   * @param validity The list of validity indicator booleans
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  strings_column_wrapper(std::initializer_list<std::string> strings,
                         std::initializer_list<bool> validity,
                         rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                         cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : strings_column_wrapper(
        std::cbegin(strings), std::cend(strings), std::cbegin(validity), stream, mr)
  {
  }

  /**
   * @brief Construct a nullable column from a list of pairs of strings
   * and validity booleans of each string.
   *
   * The validity of each string is determined by the boolean element in the pair
   * where `true` indicates the string is valid, and `false` indicates the
   * string is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable STRING column with 7 string elements:
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * using p = std::pair<std::string, bool>;
   * strings_column_wrapper s( p{"", false}, p{"this", true}, p{"is", false},
   *                           p{"a", true}, p{"column", false}, p{"of", true},
   *                           p{"strings", false} );
   * @endcode
   *
   * @param strings The list of pairs of strings and validity booleans
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  strings_column_wrapper(std::initializer_list<std::pair<std::string, bool>> strings,
                         rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                         cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
  {
    auto values   = std::vector<std::string>{};
    auto validity = std::vector<bool>{};
    values.reserve(strings.size());
    validity.reserve(strings.size());
    for (auto const& [value, valid] : strings) {
      values.push_back(value);
      validity.push_back(valid);
    }
    wrapped =
      strings_column_wrapper(values.begin(), values.end(), validity.begin(), stream, mr).release();
  }
};

/**
 * @brief `column_wrapper` derived class for wrapping dictionary columns.
 *
 * This class handles fixed-width type keys.
 *
 * @tparam KeyElementTo Specify a fixed-width type for the key values of the dictionary
 * @tparam SourceElementT The fixed-width element type that is used to create elements of type
 * `KeyElementTo`
 */
template <typename KeyElementTo, typename SourceElementT = KeyElementTo>
class dictionary_column_wrapper : public detail::column_wrapper {
 public:
  /**
   * @brief Cast to dictionary_column_view
   */
  operator dictionary_column_view() const { return cudf::dictionary_column_view{wrapped->view()}; }

  /**
   * @brief Default constructor initializes an empty column with dictionary type.
   */
  dictionary_column_wrapper() : column_wrapper{}
  {
    wrapped = cudf::make_empty_column(cudf::type_id::DICTIONARY32);
  }

  /**
   * @brief Construct a non-nullable dictionary column of the fixed-width elements in the
   * range `[begin,end)`.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable dictionary column of INT32 elements with 5 elements
   * std::vector<int32_t> elements{0, 2, 2, 6, 6};
   * dictionary_column_wrapper<int32_t> w(element.begin(), elements.end());
   * // keys = {0, 2, 6}, indices = {0, 1, 1, 2, 2}
   * @endcode
   *
   * @note Similar to `std::vector`, this "range" constructor should be used
   *       with parentheses `()` and not braces `{}`. The latter should only
   *       be used for the `initializer_list` constructors.
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename InputIterator>
  dictionary_column_wrapper(InputIterator begin,
                            InputIterator end,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    wrapped = cudf::dictionary::encode(fixed_width_column_wrapper<KeyElementTo, SourceElementT>(
                                         begin, end, stream, mr.get_temporary_mr()),
                                       cudf::data_type{type_id::INT32},
                                       stream,
                                       mr.get_output_mr());
  }

  /**
   * @brief Construct a nullable dictionary column of the fixed-width elements in the range
   * `[begin,end)` using the range `[v, v + distance(begin,end))` interpreted
   * as booleans to indicate the validity of each element.
   *
   * If `v[i] == true`, element `i` is valid, else it is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable dictionary column with 5 elements and a validity iterator.
   * std::vector<int32_t> elements{0, 2, 0, 6, 0};
   * // Validity iterator here sets even rows to null.
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;})
   * dictionary_column_wrapper<int32_t> w(elements, elements + 5, validity);
   * // keys = {2, 6}, indices = {NULL, 0, NULL, 1, NULL}
   * @endcode
   *
   * @note Similar to `std::vector`, this "range" constructor should be used
   *       with parentheses `()` and not braces `{}`. The latter should only
   *       be used for the `initializer_list` constructors.
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   * @param v The beginning of the sequence of validity indicators
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename InputIterator, typename ValidityIterator>
  dictionary_column_wrapper(InputIterator begin,
                            InputIterator end,
                            ValidityIterator v,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    wrapped = cudf::dictionary::encode(fixed_width_column_wrapper<KeyElementTo, SourceElementT>(
                                         begin, end, v, stream, mr.get_temporary_mr()),
                                       cudf::data_type{type_id::INT32},
                                       stream,
                                       mr.get_output_mr());
  }

  /**
   * @brief Construct a non-nullable dictionary column of fixed-width elements from an
   * initializer list.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable dictionary column with 4 elements.
   * dictionary_column_wrapper<int32_t> w{{1, 2, 3, 1}};
   * // keys = {1, 2, 3}, indices = {0, 1, 2, 0}
   * @endcode
   *
   * @param elements The list of elements
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ElementFrom>
  dictionary_column_wrapper(std::initializer_list<ElementFrom> elements,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : dictionary_column_wrapper(std::cbegin(elements), std::cend(elements), stream, mr)
  {
  }

  /**
   * @brief Construct a nullable dictionary column from a list of fixed-width elements
   * using another list to indicate the validity of each element.
   *
   * The validity of each element is determined by an `initializer_list` of
   * booleans where `true` indicates the element is valid, and `false` indicates
   * the element is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable dictionary column with 4 elements and validity initializer.
   * dictionary_column_wrapper<int32_t> w{ {1, 0, 3, 0}, {1, 0, 1, 0}};
   * // keys = {1, 3}, indices = {0, NULL, 1, NULL}
   * @endcode
   *
   * @param elements The list of elements
   * @param validity The list of validity indicator booleans
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ElementFrom>
  dictionary_column_wrapper(std::initializer_list<ElementFrom> elements,
                            std::initializer_list<bool> validity,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : dictionary_column_wrapper(
        std::cbegin(elements), std::cend(elements), std::cbegin(validity), stream, mr)
  {
  }

  /**
   * @brief Construct a nullable dictionary column from a list of fixed-width elements and
   * the range `[v, v + element_list.size())` interpreted as booleans to
   * indicate the validity of each element.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable dictionary column with 6 elements and a validity iterator.
   * // This validity iterator sets even rows to null.
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;})
   * dictionary_column_wrapper<int32_t> w{ {0, 4, 0, 4, 0, 5}, validity}
   * // keys = {4, 5}, indices = {NULL, 0, NULL, 0, NULL, 1}
   * @endcode
   *
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be convertible to `bool`
   * @param element_list The list of elements
   * @param v The beginning of the sequence of validity indicators
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ValidityIterator, typename ElementFrom>
  dictionary_column_wrapper(std::initializer_list<ElementFrom> element_list,
                            ValidityIterator v,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : dictionary_column_wrapper(std::cbegin(element_list), std::cend(element_list), v, stream, mr)
  {
  }

  /**
   * @brief Construct a nullable dictionary column of the fixed-width elements in the range
   * `[begin,end)` using a validity initializer list to indicate the validity of each element.
   *
   * The validity of each element is determined by an `initializer_list` of
   * booleans where `true` indicates the element is valid, and `false` indicates
   * the element is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable column of dictionary elements with 5 elements and validity initializer.
   * std::vector<int32_t> elements{0, 2, 2, 6, 6};
   * dictionary_width_column_wrapper<int32_t> w(elements, elements + 5, {0, 1, 0, 1, 0});
   * // keys = {2, 6}, indices = {NULL, 0, NULL, 1, NULL}
   * @endcode
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   * @param validity The list of validity indicator booleans
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename InputIterator>
  dictionary_column_wrapper(InputIterator begin,
                            InputIterator end,
                            std::initializer_list<bool> const& validity,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : dictionary_column_wrapper(begin, end, std::cbegin(validity), stream, mr)
  {
  }
};

/**
 * @brief `column_wrapper` derived class for wrapping a dictionary column with string keys.
 *
 * This is a specialization of the `dictionary_column_wrapper` class for strings.
 */
template <>
class dictionary_column_wrapper<std::string> : public detail::column_wrapper {
 public:
  /**
   * @brief Cast to dictionary_column_view
   *
   */
  operator dictionary_column_view() const { return cudf::dictionary_column_view{wrapped->view()}; }

  /**
   * @brief Access keys column view
   *
   * @return column_view to keys column
   */
  [[nodiscard]] column_view keys() const
  {
    return cudf::dictionary_column_view{wrapped->view()}.keys();
  }

  /**
   * @brief Access indices column view
   *
   * @return column_view to indices column
   */
  [[nodiscard]] column_view indices() const
  {
    return cudf::dictionary_column_view{wrapped->view()}.indices();
  }

  /**
   * @brief Default constructor initializes an empty dictionary column of strings
   */
  dictionary_column_wrapper() : dictionary_column_wrapper(std::initializer_list<std::string>{}) {}

  /**
   * @brief Construct a non-nullable dictionary column of strings from the range
   * `[begin,end)`.
   *
   * Values in the sequence `[begin,end)` will each be converted to
   *`std::string` and a dictionary column will be created by encoding the strings.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable dictionary column with 7 string elements
   * std::vector<std::string> strings{"", "aaa", "bbb", "aaa", "bbb, "ccc", "bbb"};
   * dictionary_column_wrapper<std::string> d(strings.begin(), strings.end());
   * // keys = {"","aaa","bbb","ccc"}, indices = {0, 1, 2, 1, 2, 3, 2}
   * @endcode
   *
   * @tparam StringsIterator A `std::string` must be constructible from
   *                         dereferencing a `StringsIterator`.
   * @param begin The beginning of the sequence
   * @param end The end of the sequence
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename StringsIterator>
  dictionary_column_wrapper(StringsIterator begin,
                            StringsIterator end,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    wrapped =
      cudf::dictionary::encode(strings_column_wrapper(begin, end, stream, mr.get_temporary_mr()),
                               cudf::data_type{type_id::INT32},
                               stream,
                               mr.get_output_mr());
  }

  /**
   * @brief Construct a nullable dictionary column of strings from the range
   * `[begin,end)` using the range `[v, v + distance(begin,end))` interpreted
   * as booleans to indicate the validity of each string.
   *
   * Values in the sequence `[begin,end)` will each be converted to
   * `std::string` and a dictionary column will be created by encoding the strings.
   *
   * If `v[i] == true`, string `i` is valid, else it is treated as null row.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable dictionary column with 7 strings elements and validity iterator.
   * std::vector<std::string> strings{"", "aaa", "", "aaa", "", "bbb", ""};
   * // Validity iterator sets even rows to null.
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * dictionary_column_wrapper<std::string> d(strings.begin(), strings.end(), validity);
   * // keys = {"aaa", "bbb"}, indices = {NULL, 0, NULL, 0, NULL, 1, NULL}
   * @endcode
   *
   * @tparam StringsIterator A `std::string` must be constructible from
   *                         dereferencing a `StringsIterator`.
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be
   *                          convertible to `bool`
   * @param begin The beginning of the sequence
   * @param end The end of the sequence
   * @param v The beginning of the sequence of validity indicators
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename StringsIterator, typename ValidityIterator>
  dictionary_column_wrapper(StringsIterator begin,
                            StringsIterator end,
                            ValidityIterator v,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    wrapped =
      cudf::dictionary::encode(strings_column_wrapper(begin, end, v, stream, mr.get_temporary_mr()),
                               cudf::data_type{type_id::INT32},
                               stream,
                               mr.get_output_mr());
  }

  /**
   * @brief Construct a non-nullable dictionary column of strings from a list of strings.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable dictionary column with 7 string elements.
   * dictionary_column_wrapper<std::string> d({"", "bb", "a", "bb", "a", "ccc", "a"});
   * // keys = {"","a","bb","ccc"}, indices = {0, 2, 1, 2, 1, 3, 1}
   * @endcode
   *
   * @param strings The list of strings
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  dictionary_column_wrapper(std::initializer_list<std::string> strings,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : dictionary_column_wrapper(std::cbegin(strings), std::cend(strings), stream, mr)
  {
  }

  /**
   * @brief Construct a nullable dictionary column of strings from a list of strings and
   * the range `[v, v + strings.size())` interpreted as booleans to indicate the
   * validity of each string.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable dictionary column with 7 string elements and a validity iterator.
   * // Validity iterator here sets even rows to null.
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * dictionary_column_wrapper<std::string> d({"", "bb", "", "bb", "", "a", ""}, validity);
   * // keys = {"a", "bb"}, indices = {NULL, 1, NULL, 1, NULL, 0, NULL}
   * @endcode
   *
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be convertible to `bool`
   * @param strings The list of strings
   * @param v The beginning of the sequence of validity indicators
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ValidityIterator>
  dictionary_column_wrapper(std::initializer_list<std::string> strings,
                            ValidityIterator v,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : dictionary_column_wrapper(std::cbegin(strings), std::cend(strings), v, stream, mr)
  {
  }

  /**
   * @brief Construct a nullable dictionary column of strings from a list of strings and
   * a list of booleans to indicate the validity of each string.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable STRING column with 7 string elements and validity initializer.
   * dictionary_column_wrapper<std::string> ({"", "a", "", "bb", "", "ccc", ""},
   *                                         {0,  1,   0,  1,    0,  1,     0});
   * // keys = {"a", "bb", "ccc"}, indices = {NULL, 0, NULL, 1, NULL, 2, NULL}
   * @endcode
   *
   * @param strings The list of strings
   * @param validity The list of validity indicator booleans
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  dictionary_column_wrapper(std::initializer_list<std::string> strings,
                            std::initializer_list<bool> validity,
                            rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                            cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    : dictionary_column_wrapper(
        std::cbegin(strings), std::cend(strings), std::cbegin(validity), stream, mr)
  {
  }
};

/**
 * @brief True when `T` is convertible to `rmm::cuda_stream_view`.
 */
template <typename T>
concept convertible_to_cuda_stream_view = std::is_convertible_v<T&, rmm::cuda_stream_view>;

/**
 * @brief `column_wrapper` derived class for wrapping columns of structs.
 */
class structs_column_wrapper : public detail::column_wrapper {
 public:
  /**
   * @brief Constructs a struct column from the specified list of pre-constructed child columns.
   *
   * The child columns are "adopted" by the struct column constructed here.
   *
   * Example usage:
   * @code{.cpp}
   * // The following constructs a column for struct< int, string >.
   * auto child_int_col = fixed_width_column_wrapper<int32_t>{ 1, 2, 3, 4, 5 }.release();
   * auto child_string_col = string_column_wrapper {"All", "the", "leaves", "are",
   * "brown"}.release();
   *
   * std::vector<std::unique_ptr<column>> child_columns;
   * child_columns.push_back(std::move(child_int_col));
   * child_columns.push_back(std::move(child_string_col));
   *
   * structs_column_wrapper structs_col{
   *  child_cols,
   *  {1,0,1,0,1} // Validity.
   * };
   *
   * auto struct_col {structs_col.release()};
   * @endcode
   *
   * The existing allocations in adopted child columns retain their original memory-resource
   * provenance. The supplied output resource controls the struct null mask and any child
   * allocations created while sanitizing null struct rows.
   *
   * To pass an explicit stream/mr with no parent nulls, pass an empty validity:
   * `structs_column_wrapper(std::move(children), {}, stream, mr)`.
   *
   * @param child_columns The vector of pre-constructed child columns
   * @param validity The vector of bools representing the column validity values
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used for new allocations owned by the returned column
   */
  structs_column_wrapper(std::vector<std::unique_ptr<cudf::column>>&& child_columns,
                         std::vector<bool> const& validity = {},
                         rmm::cuda_stream_view stream      = cudf::test::get_default_stream(),
                         cudf::memory_resources mr = cudf::get_current_device_resource_ref())
  {
    init(std::move(child_columns), validity, stream, mr);
  }

  /**
   * @brief Constructs a struct column from the list of column wrappers for child columns.
   *
   * Example usage:
   * @code{.cpp}
   * // The following constructs a column for struct< int, string >.
   * fixed_width_column_wrapper<int32_t> child_int_col_wrapper{ 1, 2, 3, 4, 5 };
   * string_column_wrapper child_string_col_wrapper {"All", "the", "leaves", "are", "brown"};
   *
   * structs_column_wrapper structs_col{
   *  {child_int_col_wrapper, child_string_col_wrapper}
   *  {1,0,1,0,1} // Validity.
   * };
   *
   * auto struct_col {structs_col.release()};
   * @endcode
   *
   * Child wrappers are deep-copied, so all allocations in the returned children use the supplied
   * output resource. The source wrappers retain their original allocations.
   *
   * To pass an explicit stream/mr with no parent nulls, pass an empty validity:
   * `structs_column_wrapper({wrappers}, {}, stream, mr)`.
   *
   * @param child_column_wrappers The list of child column wrappers
   * @param validity The vector of bools representing the column validity values
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  structs_column_wrapper(
    std::initializer_list<std::reference_wrapper<detail::column_wrapper>> child_column_wrappers,
    std::vector<bool> const& validity = {},
    rmm::cuda_stream_view stream      = cudf::test::get_default_stream(),
    cudf::memory_resources mr         = cudf::get_current_device_resource_ref())
  {
    std::vector<std::unique_ptr<cudf::column>> child_columns;
    child_columns.reserve(child_column_wrappers.size());
    std::transform(child_column_wrappers.begin(),
                   child_column_wrappers.end(),
                   std::back_inserter(child_columns),
                   [&](auto const& column_wrapper) {
                     return std::make_unique<cudf::column>(
                       column_wrapper.get(), stream, mr.get_output_mr());
                   });
    init(std::move(child_columns), validity, stream, mr);
  }

  /**
   * @brief Constructs a struct column from the list of column wrappers for child columns.
   *
   * Example usage:
   * @code{.cpp}
   * // The following constructs a column for struct< int, string >.
   * fixed_width_column_wrapper<int32_t> child_int_col_wrapper{ 1, 2, 3, 4, 5 };
   * string_column_wrapper child_string_col_wrapper {"All", "the", "leaves", "are", "brown"};
   *
   * structs_column_wrapper structs_col{
   *  {child_int_col_wrapper, child_string_col_wrapper}
   *  cudf::detail::make_counting_transform_iterator(0, [](auto i){ return i%2; }) // Validity.
   * };
   *
   * auto struct_col {structs_col.release()};
   * @endcode
   *
   * @param child_column_wrappers The list of child column wrappers
   * @param validity_iter Iterator returning the per-row validity bool
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename V>
  structs_column_wrapper(
    std::initializer_list<std::reference_wrapper<detail::column_wrapper>> child_column_wrappers,
    V validity_iter,
    rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
    cudf::memory_resources mr    = cudf::get_current_device_resource_ref())
    requires(!convertible_to_cuda_stream_view<V>)
  {
    std::vector<std::unique_ptr<cudf::column>> child_columns;
    child_columns.reserve(child_column_wrappers.size());
    std::transform(child_column_wrappers.begin(),
                   child_column_wrappers.end(),
                   std::back_inserter(child_columns),
                   [&](auto const& column_wrapper) {
                     return std::make_unique<cudf::column>(
                       column_wrapper.get(), stream, mr.get_output_mr());
                   });
    init(std::move(child_columns), validity_iter, stream, mr);
  }

 private:
  void init(std::vector<std::unique_ptr<cudf::column>>&& child_columns,
            std::vector<bool> const& validity,
            rmm::cuda_stream_view stream,
            cudf::memory_resources mr)
  {
    size_type num_rows = child_columns.empty() ? 0 : child_columns[0]->size();

    CUDF_EXPECTS(std::all_of(child_columns.begin(),
                             child_columns.end(),
                             [&](auto const& p_column) { return p_column->size() == num_rows; }),
                 "All struct member columns must have the same row count.");

    CUDF_EXPECTS(validity.size() <= 0 || static_cast<size_type>(validity.size()) == num_rows,
                 "Validity buffer must have as many elements as rows in the struct column.");

    auto [null_mask, null_count] = [&] {
      if (validity.size() <= 0) return std::make_pair(rmm::device_buffer{}, cudf::size_type{0});
      return cudf::test::detail::make_null_mask(validity.begin(), validity.end(), stream, mr);
    }();

    wrapped = cudf::make_structs_column(num_rows,
                                        std::move(child_columns),
                                        null_count,
                                        std::move(null_mask),
                                        stream,
                                        mr.get_output_mr());
  }

  template <typename V>
  void init(std::vector<std::unique_ptr<cudf::column>>&& child_columns,
            V validity_iterator,
            rmm::cuda_stream_view stream,
            cudf::memory_resources mr)
  {
    size_type const num_rows = child_columns.empty() ? 0 : child_columns[0]->size();

    CUDF_EXPECTS(std::all_of(child_columns.begin(),
                             child_columns.end(),
                             [&](auto const& p_column) { return p_column->size() == num_rows; }),
                 "All struct member columns must have the same row count.");

    std::vector<bool> validity(num_rows);
    std::copy(validity_iterator, validity_iterator + num_rows, validity.begin());

    init(std::move(child_columns), validity, stream, mr);
  }
};

}  // namespace test
}  // namespace CUDF_EXPORT cudf
