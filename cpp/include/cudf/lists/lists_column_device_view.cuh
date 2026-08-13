/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime.h>

namespace cudf {

/**
 * @brief Given a column_device_view, an instance of this class provides a
 * wrapper on this compound column for list operations.
 * Analogous to list_column_view.
 *
 * @ingroup column_classes
 */
class lists_column_device_view : private column_device_view {
 public:
  lists_column_device_view()                                = delete;
  ~lists_column_device_view()                               = default;
  lists_column_device_view(lists_column_device_view const&) = default;  ///< Copy constructor
  lists_column_device_view(lists_column_device_view&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return The reference to this lists column device view
   */
  lists_column_device_view& operator=(lists_column_device_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return The reference to this lists column device view
   */
  lists_column_device_view& operator=(lists_column_device_view&&) = default;

  /**
   * @brief Construct a new lists column device view object from a column device view.
   *
   * @param underlying_ The column device view to wrap
   */
  CUDF_HOST_DEVICE lists_column_device_view(column_device_view const& underlying_)
    : column_device_view(underlying_)
  {
#ifdef __CUDA_ARCH__
    cudf_assert(underlying_.type().id() == type_id::LIST and
                "lists_column_device_view only supports lists");
#else
    CUDF_EXPECTS(underlying_.type().id() == type_id::LIST,
                 "lists_column_device_view only supports lists");
#endif
  }

  using column_device_view::is_null;
  using column_device_view::nullable;
  using column_device_view::offset;
  using column_device_view::size;

  /**
   * @brief Fetches the offsets column of the underlying list column.
   *
   * @return The offsets column of the underlying list column
   */
  [[nodiscard]] __device__ inline column_device_view offsets() const
  {
    return column_device_view::child(lists_column_view::offsets_column_index);
  }

  /**
   * @brief Fetches the list offset value at a given row index while taking column offset into
   * account.
   *
   * @param idx The row index to fetch the list offset value at
   * @return The list offset value at a given row index while taking column offset into account
   */
  [[nodiscard]] __device__ inline size_type offset_at(size_type idx) const
  {
    return offsets().size() > 0 ? offsets().element<size_type>(offset() + idx) : 0;
  }

  /**
   * @brief Fetches the child column of the underlying list column.
   *
   * @return The child column of the underlying list column
   */
  [[nodiscard]] __device__ inline column_device_view child() const
  {
    return column_device_view::child(lists_column_view::child_column_index);
  }

  /**
   * @brief Fetches the child column of the underlying list column with offset and size applied
   *
   * @return The child column sliced relative to the parent's offset and size
   */
  [[nodiscard]] __device__ inline column_device_view get_sliced_child() const
  {
    auto start = offset_at(0);
    auto end   = offset_at(size());
    return child().slice(start, end - start);
  }
};

}  // namespace cudf
