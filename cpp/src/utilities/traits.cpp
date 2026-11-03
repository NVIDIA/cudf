/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/dictionary.hpp>

namespace cudf {

namespace {
/**
 * @brief Helper functor to check if a specified type `T` supports relational comparisons.
 *
 */
struct unary_relationally_comparable_functor {
  /**
   * @brief Returns true if `T` supports relational comparisons.
   *
   * @tparam T Type to check
   * @return true if `T` supports relational comparisons
   */
  template <typename T>
  inline bool operator()() const
  {
    return cudf::is_relationally_comparable<T, T>();
  }
};
}  // namespace

/**
 * @brief Checks whether `data_type` `type` supports relational comparisons.
 *
 * @param type Data_type for comparison.
 * @return true If `type` supports relational comparisons.
 * @return false If `type` does not support relational comparisons.
 */
bool is_relationally_comparable(data_type type)
{
  return type_dispatcher(type, unary_relationally_comparable_functor{});
}

namespace {
/**
 * @brief Helper functor to check if a specified type `T` supports equality comparisons.
 *
 */
struct unary_equality_comparable_functor {
  /**
   * @brief Checks whether `T` supports equality comparisons.
   *
   * @tparam T Type to check
   * @return true if `T` supports equality comparisons
   */
  template <typename T>
  bool operator()() const
  {
    return cudf::is_equality_comparable<T, T>();
  }
};
}  // namespace

/**
 * @brief Checks whether `data_type` `type` supports equality comparisons.
 *
 * @param type Data_type for comparison.
 * @return true If `type` supports equality comparisons.
 * @return false If `type` does not support equality comparisons.
 */
bool is_equality_comparable(data_type type)
{
  return cudf::type_dispatcher(type, unary_equality_comparable_functor{});
}

struct is_numeric_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_numeric<T>();
  }
};

/**
 * @brief Indicates whether `type` is a numeric `data_type`.
 *
 * "Numeric" types are fundamental integral/floating point types such as `INT*`
 * or `FLOAT*`. Types that wrap a numeric type are not considered numeric, e.g.,
 *`TIMESTAMP`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is numeric
 * @return false `type` is not numeric
 */
bool is_numeric(data_type type) { return cudf::type_dispatcher(type, is_numeric_impl{}); }

struct is_index_type_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_index_type<T>();
  }
};

/**
 * @brief Indicates whether the type `type` is a index type.
 *
 * A type `T` is considered an index type if it is valid to use
 * elements of type `T` to index into a column. I.e.,
 * index types are integral types such as 'INT*' apart from 'bool'.
 *
 * @param type The `data_type` to verify
 * @return true `type` is index type
 * @return false `type` is not index type
 */
bool is_index_type(data_type type) { return cudf::type_dispatcher(type, is_index_type_impl{}); }

struct is_signed_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_signed<T>();
  }
};

/**
 * @brief Indicates whether `type` is a signed numeric `data_type`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is signed numeric
 */
bool is_signed(data_type type) { return cudf::type_dispatcher(type, is_signed_impl{}); }

struct is_unsigned_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_unsigned<T>();
  }
};

/**
 * @brief Indicates whether `type` is a unsigned numeric `data_type`.
 *
 * "Unsigned Numeric" types are fundamental integral types such as `UINT*`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is unsigned numeric
 * @return false `type` is signed numeric
 */
bool is_unsigned(data_type type) { return cudf::type_dispatcher(type, is_unsigned_impl{}); }

struct is_integral_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_integral<T>();
  }
};

bool is_integral(data_type type) { return cudf::type_dispatcher(type, is_integral_impl{}); }

struct is_integral_not_bool_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_integral_not_bool<T>();
  }
};

bool is_integral_not_bool(data_type type)
{
  return cudf::type_dispatcher(type, is_integral_not_bool_impl{});
}

struct is_numeric_not_bool_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_numeric_not_bool<T>();
  }
};

bool is_numeric_not_bool(data_type type)
{
  return cudf::type_dispatcher(type, is_numeric_not_bool_impl{});
}

struct is_floating_point_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_floating_point<T>();
  }
};

/**
 * @brief Indicates whether `type` is a floating point `data_type`.
 *
 * "Floating point" types are fundamental floating point types such as `FLOAT*`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is floating point
 * @return false `type` is not floating point
 */
bool is_floating_point(data_type type)
{
  return cudf::type_dispatcher(type, is_floating_point_impl{});
}

struct is_boolean_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_boolean<T>();
  }
};

/**
 * @brief Indicates whether `type` is a Boolean `data_type`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a Boolean
 * @return false `type` is not a Boolean
 */
bool is_boolean(data_type type) { return cudf::type_dispatcher(type, is_boolean_impl{}); }

struct is_fixed_point_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_fixed_point<T>();
  }
};

/**
 * @brief Indicates whether `type` is a fixed point `data_type`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a fixed point type
 * @return false `type` is not a fixed point type
 */
bool is_fixed_point(data_type type) { return cudf::type_dispatcher(type, is_fixed_point_impl{}); }

struct is_timestamp_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_timestamp<T>();
  }
};

/**
 * @brief Indicates whether `type` is a timestamp `data_type`.
 *
 * "Timestamp" types are int32_t or int64_t durations since the unix epoch.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a timestamp
 * @return false `type` is not a timestamp
 */
bool is_timestamp(data_type type) { return cudf::type_dispatcher(type, is_timestamp_impl{}); }

struct is_duration_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_duration<T>();
  }
};

/**
 * @brief Indicates whether `type` is a duration `data_type`.
 *
 * "Duration" types are int32_t or int64_t tick counts representing a time interval.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a duration
 * @return false `type` is not a duration
 */
bool is_duration(data_type type) { return cudf::type_dispatcher(type, is_duration_impl{}); }

struct is_chrono_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_chrono<T>();
  }
};

/**
 * @brief Indicates whether `type` is a chrono `data_type`.
 *
 * Chrono types include cudf timestamp types, which represent a point in time, and cudf
 * duration types that represent a time interval.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a chrono type
 * @return false `type` is not a chrono type
 */
bool is_chrono(data_type type) { return cudf::type_dispatcher(type, is_chrono_impl{}); }

struct is_dictionary_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_dictionary<T>();
  }
};

/**
 * @brief Indicates whether `type` is a dictionary `data_type`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a dictionary type
 * @return false `type` is not a dictionary type
 */
bool is_dictionary(data_type type) { return cudf::type_dispatcher(type, is_dictionary_impl{}); }

/**
 * @brief Indicates whether `type` is a valid dictionary key type
 *
 * @param type The `data_type` to verify
 * @return true `type` can be a dictionary type
 * @return false `type` cannot be a dictionary type
 */
bool is_dictionary_key(data_type type)
{
  return not is_dictionary(type) && is_relationally_comparable(type);
}

struct is_fixed_width_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_fixed_width<T>();
  }
};

/**
 * @brief Indicates whether elements of `type` are fixed-width.
 *
 * Elements of a fixed-width type all have the same size in bytes.
 *
 * @param type The `data_type` to verify
 * @return true `type` is fixed-width
 * @return false  `type` is variable-width
 */
bool is_fixed_width(data_type type) { return cudf::type_dispatcher(type, is_fixed_width_impl{}); }

struct is_compound_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_compound<T>();
  }
};

/**
 * @brief Indicates whether elements of `type` are compound.
 *
 * `column`s with "compound" elements are logically a single column of elements,
 * but may be concretely implemented with two or more `column`s. For example, a
 * `STRING` column could contain a `column` of offsets and a child `column` of
 * characters.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a compound type
 * @return false `type` is a simple type
 */
bool is_compound(data_type type) { return cudf::type_dispatcher(type, is_compound_impl{}); }

struct is_nested_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_nested<T>();
  }
};

/**
 * @brief Indicates whether `type` is a nested type
 *
 * "Nested" types are distinct from compound types in that they
 * can have an arbitrarily deep list of descendants of the same
 * type. Strings are not a nested type, but lists are.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a nested type
 * @return false `type` is not a nested type
 */
bool is_nested(data_type type) { return cudf::type_dispatcher(type, is_nested_impl{}); }

namespace {
template <typename FromType>
struct is_bit_castable_to_impl {
  template <typename ToType>
  constexpr bool operator()()
    requires(is_compound<ToType>())
  {
    return false;
  }

  template <typename ToType>
  constexpr bool operator()()
    requires(not is_compound<ToType>())
  {
    if (not cuda::std::is_trivially_copyable_v<FromType> ||
        not cuda::std::is_trivially_copyable_v<ToType>) {
      return false;
    }
    constexpr auto from_size = sizeof(cudf::device_storage_type_t<FromType>);
    constexpr auto to_size   = sizeof(cudf::device_storage_type_t<ToType>);
    return from_size == to_size;
  }
};

struct is_bit_castable_from_impl {
  template <typename FromType>
  constexpr bool operator()(data_type)
    requires(is_compound<FromType>())
  {
    return false;
  }

  template <typename FromType>
  constexpr bool operator()(data_type to)
    requires(not is_compound<FromType>())
  {
    return cudf::type_dispatcher(to, is_bit_castable_to_impl<FromType>{});
  }
};
}  // namespace

/**
 * @brief Indicates whether `from` is bit-castable to `to`.
 *
 * This casting is based on std::bit_cast. Data types that have the same size and are trivially
 * copyable are eligible for this casting.
 *
 * See `cudf::bit_cast()` which returns a zero-copy `column_view` when casting between
 * bit-castable types.
 *
 * @param from The `data_type` to convert from
 * @param to The `data_type` to convert to
 * @return `true` if the types are castable
 */
bool is_bit_castable(data_type from, data_type to)
{
  return type_dispatcher(from, is_bit_castable_from_impl{}, to);
}

}  // namespace cudf
