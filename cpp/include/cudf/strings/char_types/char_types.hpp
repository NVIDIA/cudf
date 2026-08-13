/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/char_types/char_types_enum.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for identifying and filtering strings by character type
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_types
 * @{
 */

/**
 * @brief Returns a boolean column identifying string entries where all
 * characters are of the type specified.
 *
 * The output row entry will be set to false if the corresponding string element
 * is empty or has at least one character not of the specified type. If all
 * characters fit the type then true is set in that output row entry.
 *
 * To ignore all but specific types, set the `verify_types` to those types
 * which should be checked. Otherwise, the default `ALL_TYPES` will verify all
 * characters match `types`.
 *
 * @code{.pseudo}
 * Example:
 * s = ['ab', 'a b', 'a7', 'a B']
 * b1 = s.all_characters_of_type(s,LOWER)
 * b1 is [true, false, false, false]
 * b2 = s.all_characters_of_type(s,LOWER,LOWER|UPPER)
 * b2 is [true, true, true, false]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param input Strings instance for this operation
 * @param types The character types to check in each string
 * @param verify_types Only verify against these character types.
 *                     Default `ALL_TYPES` means return `true`
 *                     iff all characters match `types`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of boolean results for each string
 */
std::unique_ptr<column> all_characters_of_type(
  strings_column_view const& input,
  string_character_types types,
  string_character_types verify_types = string_character_types::ALL_TYPES,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());

/**
 * @brief Filter specific character types from a column of strings.
 *
 * To remove all characters of a specific type, set that type in
 * `types_to_remove` and set `types_to_keep` to `ALL_TYPES`.
 *
 * To filter out characters NOT of a select type, specify `ALL_TYPES` for
 * `types_to_remove` and which types to not remove in `types_to_keep`.
 *
 * @code{.pseudo}
 * Example:
 * s = ['ab', 'a b', 'a7bb', 'A7B234']
 * s1 = s.filter_characters_of_type(s,NUMERIC,"",ALL_TYPES)
 * s1 is ['ab', 'a b', 'abb', 'AB']
 * s2 = s.filter_characters_of_type(s,ALL_TYPES,"-",LOWER)
 * s2 is ['ab', 'a-b', 'a-bb', '------']
 * @endcode
 *
 * In `s1` all NUMERIC types have been removed.
 * In `s2` all non-LOWER types have been replaced.
 *
 * One but not both parameters `types_to_remove` and `types_to_keep` must
 * be set to `ALL_TYPES`.
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @throw cudf::logic_error if neither or both `types_to_remove` and
 *        `types_to_keep` are set to `ALL_TYPES`.
 *
 * @param input Strings instance for this operation
 * @param types_to_remove The character types to check in each string.
 *        Use `ALL_TYPES` here to specify `types_to_keep` instead.
 * @param replacement The replacement character to use when removing characters
 * @param types_to_keep Default `ALL_TYPES` means all characters of
 *        `types_to_remove` will be filtered.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return New strings column with the characters of specified types filtered out and replaced by
 * the specified replacement string
 */
std::unique_ptr<column> filter_characters_of_type(
  strings_column_view const& input,
  string_character_types types_to_remove,
  string_scalar const& replacement     = string_scalar(""),
  string_character_types types_to_keep = string_character_types::ALL_TYPES,
  rmm::cuda_stream_view stream         = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr    = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
