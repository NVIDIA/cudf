/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/dictionary.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include <cstddef>

/**
 * @file
 * @brief Concepts for classifying cudf column and scalar types.
 *
 * The concepts live in `cudf::concepts` so that generic names such as `duration` do not collide
 * with names brought into `cudf` code by using-directives, e.g. `using namespace
 * cuda::std::chrono`. Constraints that the standard library already expresses exactly, such as
 * `cuda::std::floating_point` or `cuda::std::totally_ordered_with`, are used directly instead.
 *
 * \code{cpp}
 * template <cudf::concepts::fixed_width T>
 * void foo(T value);
 *
 * template <typename T>
 *   requires(cudf::concepts::chrono<T> || cuda::std::floating_point<T>)
 * void bar(T value);
 * \endcode
 */

namespace CUDF_EXPORT cudf {
namespace concepts {

/**
 * @addtogroup utility_types
 * @{
 */

/**
 * @brief Satisfied if `T` is a Boolean type.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept boolean = cuda::std::same_as<cuda::std::remove_cv_t<T>, bool>;

/**
 * @brief Satisfied if `T` is an arithmetic type, including `bool`.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept arithmetic = cuda::std::is_arithmetic_v<T>;

/**
 * @brief Satisfied if `T` is an arithmetic type other than `bool`.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept arithmetic_not_bool = arithmetic<T> && !boolean<T>;

/**
 * @brief Satisfied if `T` is an integral type other than `bool`.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept integral_not_bool = cuda::std::integral<T> && !boolean<T>;

/**
 * @brief Satisfied if `T` is a cudf timestamp type.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept timestamp = cuda::std::same_as<cuda::std::remove_cv_t<T>, timestamp_D> ||
                    cuda::std::same_as<cuda::std::remove_cv_t<T>, timestamp_h> ||
                    cuda::std::same_as<cuda::std::remove_cv_t<T>, timestamp_m> ||
                    cuda::std::same_as<cuda::std::remove_cv_t<T>, timestamp_s> ||
                    cuda::std::same_as<cuda::std::remove_cv_t<T>, timestamp_ms> ||
                    cuda::std::same_as<cuda::std::remove_cv_t<T>, timestamp_us> ||
                    cuda::std::same_as<cuda::std::remove_cv_t<T>, timestamp_ns>;

/**
 * @brief Satisfied if `T` is a cudf duration type.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept duration = cuda::std::same_as<cuda::std::remove_cv_t<T>, duration_D> ||
                   cuda::std::same_as<cuda::std::remove_cv_t<T>, duration_h> ||
                   cuda::std::same_as<cuda::std::remove_cv_t<T>, duration_m> ||
                   cuda::std::same_as<cuda::std::remove_cv_t<T>, duration_s> ||
                   cuda::std::same_as<cuda::std::remove_cv_t<T>, duration_ms> ||
                   cuda::std::same_as<cuda::std::remove_cv_t<T>, duration_us> ||
                   cuda::std::same_as<cuda::std::remove_cv_t<T>, duration_ns>;

/**
 * @brief Satisfied if `T` is a cudf timestamp or duration type.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept chrono = timestamp<T> || duration<T>;

/**
 * @brief Satisfied if `T` is a supported `numeric::fixed_point` type.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept fixed_point = cuda::std::same_as<cuda::std::remove_cv_t<T>, numeric::decimal32> ||
                      cuda::std::same_as<cuda::std::remove_cv_t<T>, numeric::decimal64> ||
                      cuda::std::same_as<cuda::std::remove_cv_t<T>, numeric::decimal128> ||
                      cuda::std::same_as<cuda::std::remove_cv_t<T>,
                                         numeric::fixed_point<int32_t, numeric::Radix::BASE_2>> ||
                      cuda::std::same_as<cuda::std::remove_cv_t<T>,
                                         numeric::fixed_point<int64_t, numeric::Radix::BASE_2>> ||
                      cuda::std::same_as<cuda::std::remove_cv_t<T>,
                                         numeric::fixed_point<__int128_t, numeric::Radix::BASE_2>>;

/**
 * @brief Satisfied if elements of type `T` are fixed-width.
 *
 * Elements of a fixed-width type all have the same size in bytes.
 *
 * @note Defined in terms of `fixed_point<T>` so that a `fixed_point` constrained overload is more
 * constrained and preferred over a `fixed_width` constrained one.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept fixed_width = arithmetic<T> || chrono<T> || fixed_point<T>;

/**
 * @brief Satisfied if `T` is layout compatible with its "representation" type.
 *
 * For example, `duration_ns` is distinct from its concrete `int64_t` representation type, but
 * they are layout compatible. A `decimal32` is not, because it also carries a scale.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept rep_layout_compatible =
  arithmetic<T> || chrono<T> || cuda::std::same_as<cuda::std::remove_cv_t<T>, std::byte>;

/**
 * @brief Satisfied if `T` is the cudf dictionary type.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept dictionary = cuda::std::same_as<cuda::std::remove_cv_t<T>, dictionary32>;

/**
 * @brief Satisfied if `T` can be a dictionary key type.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept dictionary_key = !dictionary<T> && cuda::std::totally_ordered<T>;

/**
 * @brief Satisfied if `T` is a nested type.
 *
 * "Nested" types can have an arbitrarily deep list of descendants of the same type. Strings are
 * not a nested type, but lists are.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept nested = cuda::std::same_as<cuda::std::remove_cv_t<T>, list_view> ||
                 cuda::std::same_as<cuda::std::remove_cv_t<T>, struct_view>;

/**
 * @brief Satisfied if `T` is a compound type.
 *
 * Columns with "compound" elements are logically a single column of elements, but may be
 * concretely implemented with two or more columns. For example, a `STRING` column could contain
 * a column of offsets and a child column of characters.
 *
 * @tparam T The type to verify
 */
template <typename T>
concept compound =
  cuda::std::same_as<cuda::std::remove_cv_t<T>, string_view> || dictionary<T> || nested<T>;

/** @} */

}  // namespace concepts
}  // namespace CUDF_EXPORT cudf
