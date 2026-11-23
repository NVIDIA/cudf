/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/errc.hpp>
#include <cudf/types.hpp>

/**
 * @file
 * @brief Internal encoding of evaluation statuses and their execution context.
 */

namespace cudf::detail {

/**
 * @brief Packed 64-bit status used for device-side error selection and host-side decoding.
 */
using packed_evaluation_status = uint64_t;

/**
 * @brief An evaluation status with logical row and row-program operator context.
 *
 * Row IR places an operator index in the upper 16 bits of its 32-bit `errc` return value.
 * The kernel then adds the logical row index to form a packed status:
 *
 * | Bits | Contents |
 * |------|----------|
 * | 63-32 | Logical row index, or all ones for an unspecified row |
 * | 31-16 | Index of the emitted fallible operator in the row program |
 * | 15-0 | Raw error code; zero means success |
 *
 * Among failures, `atomicMin` selects the lowest row index, then operator index, then error code.
 * An unspecified row sorts after every known row. Packed success is all ones, so it never wins
 * against a failure. The raw error code determines success even when operator context is present.
 */
struct evaluation_status {
  /**
   * @brief Maximum number of operator entries in a row program's error map.
   */
  static constexpr uint32_t max_error_map_entries = 0x10000;

  /**
   * @brief Packed success sentinel and initial value for an `atomicMin` error accumulator.
   */
  static constexpr packed_evaluation_status success = ~packed_evaluation_status{0};

  static_assert(sizeof(size_type) == sizeof(uint32_t),
                "Evaluation status packing requires a 32-bit size_type");

  errc error_code     = errc::SUCCESS;  ///< Raw error code without operator or row context.
  size_type row_index = -1;             ///< Logical row index, or -1 if unspecified.
  uint16_t op_index   = 0;              ///< Row-program operator index; 0 if none was attached.

  /**
   * @brief Attaches a row-program operator index while retaining the 32-bit `errc` return ABI.
   *
   * The raw error code occupies the low 16 bits and the operator index occupies the upper 16 bits.
   *
   * @pre `code` is an ABI-valid raw status without operator context.
   * @param code Raw evaluation status
   * @param index Index of the emitted fallible operator in the row program
   * @return Status carrying the raw error code and operator index
   */
  [[nodiscard]] CUDF_HOST_DEVICE static constexpr errc make_error(errc code, uint16_t index)
  {
    auto const packed = (uint32_t{index} << 16) | static_cast<uint16_t>(code);
    return static_cast<errc>(static_cast<error_int>(packed));
  }

  /**
   * @brief Attaches a logical row index to a row operation's status for `atomicMin` selection.
   *
   * A zero raw error code produces `success`, discarding any operator context.
   *
   * @param status ABI-valid raw status, optionally carrying row-IR operator context
   * @param row Logical row index, or -1 if unspecified; joins use the left row index
   * @return Packed failure status, or `success` when the raw error code is zero
   */
  [[nodiscard]] CUDF_HOST_DEVICE static constexpr packed_evaluation_status make(errc status,
                                                                                size_type row)
  {
    auto const bits = static_cast<uint32_t>(static_cast<error_int>(status));
    return static_cast<uint16_t>(bits) == 0
             ? success
             : (packed_evaluation_status{static_cast<uint32_t>(row)} << 32) | bits;
  }

  /**
   * @brief Decodes a packed evaluation status.
   *
   * @param packed Packed status produced by `make`, or the `success` sentinel
   * @return Decoded fields; `success` produces SUCCESS, an unspecified row, and operator index 0
   */
  [[nodiscard]] CUDF_HOST_DEVICE static constexpr evaluation_status unpack(
    packed_evaluation_status packed)
  {
    if (packed == success) { return {}; }
    return {.error_code = static_cast<errc>(static_cast<uint16_t>(packed)),
            .row_index  = static_cast<size_type>(static_cast<uint32_t>(packed >> 32)),
            .op_index   = static_cast<uint16_t>(packed >> 16)};
  }
};

}  // namespace cudf::detail
