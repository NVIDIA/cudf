/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/scan.h>

#include <stdexcept>

namespace cudf {
namespace detail {

/**
 * @brief Iterator that can be used with a scan algorithm and also return the last element
 *
 * Use cudf::detail::make_sizes_to_offsets_iterator to create an instance of this class.
 *
 * @tparam ScanIterator Output iterator type for use in a scan operation
 * @tparam LastType Type used for final scan element
 */
template <typename ScanIterator, typename LastType>
struct sizes_to_offsets_iterator {
  using difference_type   = ptrdiff_t;
  using value_type        = LastType;
  using pointer           = LastType*;
  using reference         = sizes_to_offsets_iterator const&;
  using iterator_category = std::random_access_iterator_tag;

  using ScanType = cuda::std::iter_value_t<ScanIterator>;

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator& operator++()
  {
    ++itr_;
    return *this;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator operator++(int)
  {
    sizes_to_offsets_iterator tmp(*this);
    operator++();
    return tmp;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator& operator--()
  {
    --itr_;
    return *this;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator operator--(int)
  {
    sizes_to_offsets_iterator tmp(*this);
    operator--();
    return tmp;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator& operator+=(difference_type offset)
  {
    itr_ += offset;
    return *this;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator operator+(difference_type offset) const
  {
    sizes_to_offsets_iterator tmp(*this);
    tmp.itr_ += offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline friend sizes_to_offsets_iterator operator+(
    difference_type offset, sizes_to_offsets_iterator const& rhs)
  {
    sizes_to_offsets_iterator tmp{rhs};
    tmp.itr_ += offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator& operator-=(difference_type offset)
  {
    itr_ -= offset;
    return *this;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator operator-(difference_type offset) const
  {
    sizes_to_offsets_iterator tmp(*this);
    tmp.itr_ -= offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline friend sizes_to_offsets_iterator operator-(
    difference_type offset, sizes_to_offsets_iterator const& rhs)
  {
    sizes_to_offsets_iterator tmp{rhs};
    tmp.itr_ -= offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline difference_type operator-(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ - rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator==(sizes_to_offsets_iterator const& rhs) const
  {
    return rhs.itr_ == itr_;
  }
  CUDF_HOST_DEVICE inline bool operator!=(sizes_to_offsets_iterator const& rhs) const
  {
    return rhs.itr_ != itr_;
  }
  CUDF_HOST_DEVICE inline bool operator<(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ < rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator>(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ > rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator<=(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ <= rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator>=(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ >= rhs.itr_;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator const& operator*() const { return *this; }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator const operator[](int idx) const
  {
    sizes_to_offsets_iterator tmp{*this};
    tmp.itr_ += idx;
    return tmp;
  }

  /**
   * @brief Called to set the output of the scan operation to the current iterator position
   *
   * @param value Value to set to the current output
   * @return This iterator instance
   */
  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator const& operator=(LastType const value) const
  {
    *itr_ = static_cast<ScanType>(value);  // place into the output
    if (itr_ == end_) { *last_ = value; }  // also save the last value
    return *this;
  }

  sizes_to_offsets_iterator()                                            = default;
  sizes_to_offsets_iterator(sizes_to_offsets_iterator const&)            = default;
  sizes_to_offsets_iterator(sizes_to_offsets_iterator&&)                 = default;
  sizes_to_offsets_iterator& operator=(sizes_to_offsets_iterator const&) = default;
  sizes_to_offsets_iterator& operator=(sizes_to_offsets_iterator&&)      = default;

 protected:
  template <typename S, typename R>
  friend sizes_to_offsets_iterator<S, R> make_sizes_to_offsets_iterator(S, S, R*);

  /**
   * @brief Iterator constructor
   *
   * Use the make_sizes_to_offsets_iterator() to create an instance of this class
   */
  sizes_to_offsets_iterator(ScanIterator begin, ScanIterator end, LastType* last)
    : itr_{begin}, end_{cuda::std::prev(end)}, last_{last}
  {
  }

  ScanIterator itr_{};
  ScanIterator end_{};
  LastType* last_{};
};

/**
 * @brief Create an instance of a sizes_to_offsets_iterator
 *
 * @code{.pseudo}
 *  auto begin = // begin input iterator
 *  auto end = // end input iterator
 *  auto result = rmm::device_uvector(std::distance(begin, end), stream);
 *  auto last = cudf::detail::device_scalar<int64_t>(0, stream);
 *  auto itr = make_sizes_to_offsets_iterator(result.begin(),
 *                                            result.end(),
 *                                            last.data());
 *  thrust::exclusive_scan(
 *    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
 *    begin, end, itr, int64_t{0}
 *  );
 *  // last contains the value of the final element in the scan result
 * @endcode
 *
 * @tparam ScanIterator Output iterator type for use in a scan operation
 * @tparam LastType Type used for holding the final element value
 *
 * @param begin Output iterator for scan
 * @param end End of the output iterator for scan
 * @param last Last element in the scan is stored here
 * @return Instance of iterator
 */
template <typename ScanIterator, typename LastType>
static sizes_to_offsets_iterator<ScanIterator, LastType> make_sizes_to_offsets_iterator(
  ScanIterator begin, ScanIterator end, LastType* last)
{
  return sizes_to_offsets_iterator<ScanIterator, LastType>(begin, end, last);
}

/**
 * @brief Perform an exclusive-scan and capture the final element value
 *
 * This performs an exclusive-scan (addition only) on the given input `[begin, end)`.
 * The output of the scan is placed in `result` and the value of the last element is returned.
 *
 * This implementation will return the last element in `int64_t` or `uint64_t` precision
 * as appropriate regardless of the input or result types.
 * This can be used to check if the scan operation overflowed when the input and result are
 * declared as smaller types.
 *
 * Only integral types for input and result types are supported.
 *
 * Note that `begin == result` is allowed but `result` may not overlap `[begin,end)` otherwise the
 * behavior is undefined.
 *
 * @code{.pseudo}
 *   auto const bytes = cudf::detail::sizes_to_offsets(
 *     d_offsets, d_offsets + strings_count + 1, d_offsets, stream);
 *   CUDF_EXPECTS(bytes <= static_cast<int64_t>(std::numeric_limits<size_type>::max()),
 *               "Size of output exceeds the column size limit", std::overflow_error);
 * @endcode
 *
 * @tparam SizesIterator Iterator type for input of the scan using addition operation
 * @tparam OffsetsIterator Iterator type for the output of the scan
 *
 * @param begin Input iterator for scan
 * @param end End of the input iterator
 * @param result Output iterator for scan result
 * @param initial_offset Initial offset to add to scan
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The last element of the scan
 */
template <typename SizesIterator, typename OffsetsIterator>
auto sizes_to_offsets(SizesIterator begin,
                      SizesIterator end,
                      OffsetsIterator result,
                      int64_t initial_offset,
                      cuda::stream_ref stream)
{
  using SizeType = cuda::std::iter_value_t<SizesIterator>;
  static_assert(std::is_integral_v<SizeType>,
                "Only numeric types are supported by sizes_to_offsets");

  using LastType = std::conditional_t<std::is_signed_v<SizeType>, int64_t, uint64_t>;
  auto last_element =
    cudf::detail::device_scalar<LastType>(0, stream, cudf::get_current_device_resource_ref());
  auto output_itr =
    make_sizes_to_offsets_iterator(result, result + std::distance(begin, end), last_element.data());
  // This function uses the type of the initialization parameter as the accumulator type
  // when computing the individual scan output elements.
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         begin,
                         end,
                         output_itr,
                         static_cast<LastType>(initial_offset));
  return last_element.value(stream);
}

/**
 * @brief Create an offsets column to be a child of a compound column
 *
 * This function sets the offsets values by executing scan over the sizes in the provided
 * Iterator.
 *
 * The return also includes the total number of elements -- the last element value from the
 * scan.
 *
 * The returned column is always `type_id::INT32` since offsets children are 32-bit.
 *
 * @throw std::overflow_error if the total size of the scan (last element) greater than maximum
 * value of `int32_t`
 *
 * @tparam InputIterator Used as input to scan to set the offset values
 * @param begin The beginning of the input sequence
 * @param end The end of the input sequence
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Offsets column and total elements
 */
template <typename InputIterator>
std::pair<std::unique_ptr<column>, size_type> make_offsets_child_column(
  InputIterator begin,
  InputIterator end,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto count = static_cast<size_type>(std::distance(begin, end));
  auto offsets_column =
    make_numeric_column(data_type{type_id::INT32}, count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view = offsets_column->mutable_view();
  auto d_offsets    = offsets_view.template data<int32_t>();

  // The number of offsets is count+1 so to build the offsets from the sizes
  // using exclusive-scan technically requires count+1 input values even though
  // the final input value is never used.
  // The input iterator is wrapped here to allow the last value to be safely read.
  // The input sizes are deliberately not narrowed to the 32-bit offsets type here -- doing so
  // would corrupt individual sizes larger than `int32_t` before they reach the scan, so the
  // overflow check below could no longer detect the overflow. Narrowing happens only on write
  // to `d_offsets`, after the accumulated total has been validated.
  using SizeType = cuda::std::iter_value_t<InputIterator>;
  auto map_fn =
    cuda::proclaim_return_type<SizeType>([begin, count] __device__(size_type idx) -> SizeType {
      return idx < count ? begin[idx] : SizeType{0};
    });
  auto input_itr = cudf::detail::make_counting_transform_iterator(0, map_fn);
  // Use the sizes-to-offsets iterator to compute the total number of elements
  auto const total_elements =
    sizes_to_offsets(input_itr, input_itr + count + 1, d_offsets, 0, stream);
  // the offsets are 32-bit so the total must fit in an int32_t
  CUDF_EXPECTS(
    total_elements <= static_cast<decltype(total_elements)>(std::numeric_limits<int32_t>::max()),
    "Size of output exceeds the column size limit",
    std::overflow_error);

  offsets_column->set_null_count(0);
  return std::pair(std::move(offsets_column), static_cast<size_type>(total_elements));
}

}  // namespace detail
}  // namespace cudf
