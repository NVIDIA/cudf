/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <thrust/transform.h>

namespace cudf {

namespace {

using roaring_bitmap_32_type =
  cuco::experimental::roaring_bitmap<cuda::std::uint32_t, rmm::mr::polymorphic_allocator<char>>;
using roaring_bitmap_64_type =
  cuco::experimental::roaring_bitmap<cuda::std::uint64_t, rmm::mr::polymorphic_allocator<char>>;

/**
 * @brief Dispatches a roaring_bitmap::impl function based on the roaring bitmap type
 *
 * @tparam Func Type of function to dispatch
 * @param type Roaring bitmap type
 * @param func Function to dispatch
 * @return Result of the function dispatch
 */
template <typename Func>
decltype(auto) dispatch_roaring_bitmap_type(roaring_bitmap_type type, Func&& func)
{
  if (type == roaring_bitmap_type::BITS_32) {
    return std::forward<Func>(func).template operator()<roaring_bitmap_type::BITS_32>();
  }
  return std::forward<Func>(func).template operator()<roaring_bitmap_type::BITS_64>();
}

}  // namespace

class roaring_bitmap::impl {
 public:
  explicit impl(std::span<cuda::std::byte const> serialized_bitmap_data)
    : _serialized_bitmap_data{serialized_bitmap_data}
  {
  }

  impl(impl&&)            = default;
  impl& operator=(impl&&) = default;

  template <roaring_bitmap_type Type>
  void materialize(rmm::cuda_stream_view stream)
  {
    auto const bytes = _serialized_bitmap_data.data();

    if constexpr (Type == roaring_bitmap_type::BITS_32) {
      if (_bitmap32) { return; }
      _bitmap32 = std::make_unique<roaring_bitmap_32_type>(
        bytes, rmm::mr::polymorphic_allocator<char>{}, stream);
    } else {
      if (_bitmap64) { return; }
      _bitmap64 = std::make_unique<roaring_bitmap_64_type>(
        bytes, rmm::mr::polymorphic_allocator<char>{}, stream);
    }

    _serialized_bitmap_data = {};
  }

  template <roaring_bitmap_type Type>
  [[nodiscard]] bool empty() const
  {
    return bitmap<Type>().empty();
  }

  template <roaring_bitmap_type Type>
  [[nodiscard]] cuda::std::size_t size() const
  {
    return bitmap<Type>().size();
  }

  template <roaring_bitmap_type Type>
  [[nodiscard]] cuda::std::size_t size_bytes() const
  {
    return bitmap<Type>().size_bytes();
  }

  template <roaring_bitmap_type Type, typename InputIt, typename OutputIt>
  void contains_async(InputIt first, InputIt last, OutputIt output, rmm::cuda_stream_view stream)
  {
    if (first == last) { return; }

    materialize<Type>(stream);

    bitmap<Type>().contains_async(first, last, output, stream);
  }

 private:
  /**
   * @brief Returns the materialized roaring bitmap
   *
   * @tparam Type Roaring bitmap type
   * @return Materialized roaring bitmap
   */
  template <roaring_bitmap_type Type>
  [[nodiscard]] auto& bitmap() const
  {
    if constexpr (Type == roaring_bitmap_type::BITS_32) {
      CUDF_EXPECTS(_bitmap32,
                   "Roaring bitmap has not been materialized. Call materialize() first.");
      return *_bitmap32;
    } else {
      CUDF_EXPECTS(_bitmap64,
                   "Roaring bitmap has not been materialized. Call materialize() first.");
      return *_bitmap64;
    }
  }

  std::span<cuda::std::byte const> _serialized_bitmap_data;
  std::unique_ptr<roaring_bitmap_32_type> _bitmap32;
  std::unique_ptr<roaring_bitmap_64_type> _bitmap64;
};

roaring_bitmap::roaring_bitmap(roaring_bitmap_type type,
                               std::span<cuda::std::byte const> serialized_bitmap_data)
  : _type{type}
{
  CUDF_EXPECTS(not serialized_bitmap_data.empty(),
               "Encountered empty serialized roaring bitmap data",
               std::invalid_argument);
  _impl = std::make_unique<impl>(serialized_bitmap_data);
}

roaring_bitmap::~roaring_bitmap() = default;

roaring_bitmap::roaring_bitmap(roaring_bitmap&&) noexcept = default;

roaring_bitmap& roaring_bitmap::operator=(roaring_bitmap&&) noexcept = default;

void roaring_bitmap::materialize(rmm::cuda_stream_view stream) const
{
  dispatch_roaring_bitmap_type(
    _type, [&]<roaring_bitmap_type Type>() { _impl->materialize<Type>(stream); });
}

roaring_bitmap_type roaring_bitmap::type() const { return _type; }

bool roaring_bitmap::empty() const
{
  return dispatch_roaring_bitmap_type(
    _type, [&]<roaring_bitmap_type Type>() { return _impl->empty<Type>(); });
}

cuda::std::size_t roaring_bitmap::size() const
{
  return dispatch_roaring_bitmap_type(
    _type, [&]<roaring_bitmap_type Type>() { return _impl->size<Type>(); });
}

cuda::std::size_t roaring_bitmap::size_bytes() const
{
  return dispatch_roaring_bitmap_type(
    _type, [&]<roaring_bitmap_type Type>() { return _impl->size_bytes<Type>(); });
}

std::unique_ptr<cudf::column> roaring_bitmap::contains_async(
  cudf::column_view const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  auto result = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, keys.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  contains_async(keys, result->mutable_view(), stream);
  return result;
}

void roaring_bitmap::contains_async(cudf::column_view const& keys,
                                    cudf::mutable_column_view const& output,
                                    rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(output.type().id() == cudf::type_id::BOOL8,
               "Output column must be BOOL8",
               std::invalid_argument);
  CUDF_EXPECTS(output.size() >= keys.size(), "Output column size must be >= keys column size");

  if (_type == roaring_bitmap_type::BITS_32) {
    CUDF_EXPECTS(keys.type().id() == cudf::type_id::UINT32,
                 "Key column must be UINT32 for a 32-bit roaring bitmap",
                 std::invalid_argument);
    _impl->contains_async<roaring_bitmap_type::BITS_32>(keys.begin<cuda::std::uint32_t>(),
                                                        keys.end<cuda::std::uint32_t>(),
                                                        output.begin<bool>(),
                                                        stream);
  } else {
    CUDF_EXPECTS(keys.type().id() == cudf::type_id::UINT64,
                 "Key column must be UINT64 for a 64-bit roaring bitmap",
                 std::invalid_argument);
    _impl->contains_async<roaring_bitmap_type::BITS_64>(keys.begin<cuda::std::uint64_t>(),
                                                        keys.end<cuda::std::uint64_t>(),
                                                        output.begin<bool>(),
                                                        stream);
  }
}

}  // namespace cudf
