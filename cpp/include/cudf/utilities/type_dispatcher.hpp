/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/dictionary.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <string>

/**
 * @file
 * @brief Defines the mapping between `cudf::type_id` runtime type information
 * and concrete C++ types.
 */
namespace CUDF_EXPORT cudf {
/**
 * @addtogroup utility_dispatcher
 * @{
 */

/**
 * @brief Maps a C++ type to its corresponding `cudf::type_id`
 *
 * When explicitly passed a template argument of a given type, returns the
 * appropriate `type_id` enum for the specified C++ type.
 *
 * For example:
 *
 * ```
 * return cudf::base_type_to_id<int32_t>();        // Returns INT32
 * ```
 *
 * @tparam T The non-cv type to map to a `cudf::type_id`
 * @return The `cudf::type_id` corresponding to the specified type
 */
template <typename T>
CUDF_HOST_DEVICE inline constexpr type_id base_type_to_id()
{
  return type_id::EMPTY;
};

/**
 * @brief Maps a `cudf::type_id` types to its corresponding C++ type name string
 *
 */
struct type_to_name_impl {
  /**
   * @brief Maps a `cudf::type_id` types to its corresponding C++ type name string
   *
   * @return The C++ type name as string
   */
  template <typename T>
  inline std::string operator()()
  {
    return "void";
  }
};

template <cudf::type_id t>
struct id_to_type_impl {
  using type = void;
};
/**
 * @brief Maps a `cudf::type_id` to its corresponding concrete C++ type
 *
 * Example:
 * ```
 * static_assert(std::is_same<int32_t, id_to_type<id_type::INT32>);
 * ```
 * @tparam Id The `cudf::type_id` to map
 */
template <cudf::type_id Id>
using id_to_type = typename id_to_type_impl<Id>::type;

/**
 * @brief "Returns" the corresponding type that is stored on the device when using `cudf::column`
 *
 * For `decimal32`,  the storage type is an `int32_t`.
 * For `decimal64`,  the storage type is an `int64_t`.
 * For `decimal128`, the storage type is an `__int128_t`.
 *
 * Use this "type function" with the `using` type alias:
 * @code
 * using Type = device_storage_type_t<Element>;
 * @endcode
 *
 * @tparam T The literal type that is stored on the host
 */
// clang-format off
template <typename T>
using device_storage_type_t =
  std::conditional_t<std::is_same_v<numeric::decimal32,  T>, int32_t,
  std::conditional_t<std::is_same_v<numeric::decimal64,  T>, int64_t,
  std::conditional_t<std::is_same_v<numeric::decimal128, T>, __int128_t, T>>>;
// clang-format on

/**
 * @brief Maps a C++ type to its corresponding `cudf::type_id`
 *
 * When explicitly passed a template argument of a given type, returns the
 * appropriate `type_id` enum for the specified C++ type.
 *
 * For example:
 *
 * ```
 * return cudf::type_to_id<int32_t>();        // Returns INT32
 * ```
 *
 * @tparam T The type to map to a `cudf::type_id`
 * @return The `cudf::type_id` corresponding to the specified type
 */
template <typename T>
constexpr inline type_id type_to_id()
{
  return base_type_to_id<std::remove_cv_t<T>>();
}

/**
 * @brief Macro used to define a mapping between a concrete C++ type and a
 *`cudf::type_id` enum.

 * @param Type The concrete C++ type
 * @param Id The `cudf::type_id` enum
 */
#ifndef CUDF_TYPE_MAPPING
#define CUDF_TYPE_MAPPING(Type, Id)                        \
  template <>                                              \
  constexpr inline type_id base_type_to_id<Type>()         \
  {                                                        \
    return Id;                                             \
  }                                                        \
  template <>                                              \
  inline std::string type_to_name_impl::operator()<Type>() \
  {                                                        \
    return CUDF_STRINGIFY(Type);                           \
  }                                                        \
  template <>                                              \
  struct id_to_type_impl<Id> {                             \
    using type = Type;                                     \
  };
#endif

// Defines all of the mappings between C++ types and their corresponding `cudf::type_id` values.
CUDF_TYPE_MAPPING(int8_t, type_id::INT8)
CUDF_TYPE_MAPPING(int16_t, type_id::INT16)
CUDF_TYPE_MAPPING(int32_t, type_id::INT32)
CUDF_TYPE_MAPPING(int64_t, type_id::INT64)
CUDF_TYPE_MAPPING(uint8_t, type_id::UINT8)
CUDF_TYPE_MAPPING(uint16_t, type_id::UINT16)
CUDF_TYPE_MAPPING(uint32_t, type_id::UINT32)
CUDF_TYPE_MAPPING(uint64_t, type_id::UINT64)
CUDF_TYPE_MAPPING(float, type_id::FLOAT32)
CUDF_TYPE_MAPPING(double, type_id::FLOAT64)
CUDF_TYPE_MAPPING(bool, type_id::BOOL8)
CUDF_TYPE_MAPPING(cudf::timestamp_D, type_id::TIMESTAMP_DAYS)
CUDF_TYPE_MAPPING(cudf::timestamp_s, type_id::TIMESTAMP_SECONDS)
CUDF_TYPE_MAPPING(cudf::timestamp_ms, type_id::TIMESTAMP_MILLISECONDS)
CUDF_TYPE_MAPPING(cudf::timestamp_us, type_id::TIMESTAMP_MICROSECONDS)
CUDF_TYPE_MAPPING(cudf::timestamp_ns, type_id::TIMESTAMP_NANOSECONDS)
CUDF_TYPE_MAPPING(cudf::duration_D, type_id::DURATION_DAYS)
CUDF_TYPE_MAPPING(cudf::duration_s, type_id::DURATION_SECONDS)
CUDF_TYPE_MAPPING(cudf::duration_ms, type_id::DURATION_MILLISECONDS)
CUDF_TYPE_MAPPING(cudf::duration_us, type_id::DURATION_MICROSECONDS)
CUDF_TYPE_MAPPING(cudf::duration_ns, type_id::DURATION_NANOSECONDS)
CUDF_TYPE_MAPPING(cudf::dictionary32, type_id::DICTIONARY32)
CUDF_TYPE_MAPPING(cudf::string_view, type_id::STRING)
CUDF_TYPE_MAPPING(cudf::list_view, type_id::LIST)
CUDF_TYPE_MAPPING(numeric::decimal32, type_id::DECIMAL32)
CUDF_TYPE_MAPPING(numeric::decimal64, type_id::DECIMAL64)
CUDF_TYPE_MAPPING(numeric::decimal128, type_id::DECIMAL128)
CUDF_TYPE_MAPPING(cudf::struct_view, type_id::STRUCT)

/**
 * @brief Specialization to map 'char' type to type_id::INT8
 *
 * Required when passing device_uvector<char> to column constructor.
 * Possibly can be removed when PR 14202 is merged.
 *
 * @return id for 'char' type
 */
template <>  // CUDF_TYPE_MAPPING(char,INT8) causes duplicate id_to_type_impl definition
constexpr inline type_id base_type_to_id<char>()
{
  return type_id::INT8;
}

/**
 * @brief Checks if `fixed_point`-like types have template type `T` matching the column's
 * stored type id
 *
 * @tparam     T The type that is stored on the device
 * @param id   The `data_type::id` of the column
 * @return     `true` If T matches the stored column `type_id`
 * @return     `false` If T does not match the stored column `type_id`
 */
template <typename T>
constexpr bool type_id_matches_device_storage_type(type_id id)
{
  return (id == type_id::DECIMAL32 && std::is_same_v<T, int32_t>) ||
         (id == type_id::DECIMAL64 && std::is_same_v<T, int64_t>) ||
         (id == type_id::DECIMAL128 && std::is_same_v<T, __int128_t>) || id == type_to_id<T>();
}

/**
 * @brief Use this specialization on `type_dispatcher` whenever you only need to operate on the
 * underlying stored type.
 *
 * For example, `cudf::sort` in sort.cu uses `cudf::type_dispatcher<dispatch_storage_type>(...)`.
 * `cudf::gather` in gather.cuh also uses `cudf::type_dispatcher<dispatch_storage_type>(...)`.
 * However, reductions needs both `data_type` and underlying type, so cannot use this.
 */
template <cudf::type_id Id>
struct dispatch_storage_type {
  using type = device_storage_type_t<id_to_type<Id>>;  ///< The underlying type
};

template <typename T>
struct type_to_scalar_type_impl {
  using ScalarType = cudf::scalar;
};

/**
 * @brief Macro used to define scalar type and scalar device type for
 * `cudf::numeric_scalar` template class for numeric C++ types.
 *
 * @param Type The numeric C++ type
 */
#ifndef MAP_NUMERIC_SCALAR
#define MAP_NUMERIC_SCALAR(Type)                                     \
  template <>                                                        \
  struct type_to_scalar_type_impl<Type> {                            \
    using ScalarType       = cudf::numeric_scalar<Type>;             \
    using ScalarDeviceType = cudf::numeric_scalar_device_view<Type>; \
  };
#endif

MAP_NUMERIC_SCALAR(int8_t)
MAP_NUMERIC_SCALAR(int16_t)
MAP_NUMERIC_SCALAR(int32_t)
MAP_NUMERIC_SCALAR(int64_t)
MAP_NUMERIC_SCALAR(__int128_t)
MAP_NUMERIC_SCALAR(uint8_t)
MAP_NUMERIC_SCALAR(uint16_t)
MAP_NUMERIC_SCALAR(uint32_t)
MAP_NUMERIC_SCALAR(uint64_t)
MAP_NUMERIC_SCALAR(float)
MAP_NUMERIC_SCALAR(double)
MAP_NUMERIC_SCALAR(bool)

template <>
struct type_to_scalar_type_impl<std::string> {
  using ScalarType       = cudf::string_scalar;
  using ScalarDeviceType = cudf::string_scalar_device_view;
};

template <>
struct type_to_scalar_type_impl<cudf::string_view> {
  using ScalarType       = cudf::string_scalar;
  using ScalarDeviceType = cudf::string_scalar_device_view;
};

template <>
struct type_to_scalar_type_impl<numeric::decimal32> {
  using ScalarType       = cudf::fixed_point_scalar<numeric::decimal32>;
  using ScalarDeviceType = cudf::fixed_point_scalar_device_view<numeric::decimal32>;
};

template <>
struct type_to_scalar_type_impl<numeric::decimal64> {
  using ScalarType       = cudf::fixed_point_scalar<numeric::decimal64>;
  using ScalarDeviceType = cudf::fixed_point_scalar_device_view<numeric::decimal64>;
};

template <>
struct type_to_scalar_type_impl<numeric::decimal128> {
  using ScalarType       = cudf::fixed_point_scalar<numeric::decimal128>;
  using ScalarDeviceType = cudf::fixed_point_scalar_device_view<numeric::decimal128>;
};

template <>  // TODO: this is a temporary solution for make_pair_iterator
struct type_to_scalar_type_impl<cudf::dictionary32> {
  using ScalarType       = cudf::numeric_scalar<int32_t>;
  using ScalarDeviceType = cudf::numeric_scalar_device_view<int32_t>;
};

template <>  // TODO: this is to get compilation working. list scalars will be implemented at a
             // later time.
struct type_to_scalar_type_impl<cudf::list_view> {
  using ScalarType = cudf::list_scalar;
  // using ScalarDeviceType = cudf::list_scalar_device_view;
};

template <>  // TODO: Ditto, likewise.
struct type_to_scalar_type_impl<cudf::struct_view> {
  using ScalarType = cudf::struct_scalar;
  // using ScalarDeviceType = cudf::struct_scalar_device_view; // CALEB: TODO!
};

/**
 * @brief Macro used to define scalar type and scalar device type for
 * `cudf::timestamp_scalar` template class for timestamp C++ types.
 *
 * @param Type The timestamp C++ type
 */
#ifndef MAP_TIMESTAMP_SCALAR
#define MAP_TIMESTAMP_SCALAR(Type)                                     \
  template <>                                                          \
  struct type_to_scalar_type_impl<Type> {                              \
    using ScalarType       = cudf::timestamp_scalar<Type>;             \
    using ScalarDeviceType = cudf::timestamp_scalar_device_view<Type>; \
  };
#endif

MAP_TIMESTAMP_SCALAR(timestamp_D)
MAP_TIMESTAMP_SCALAR(timestamp_s)
MAP_TIMESTAMP_SCALAR(timestamp_ms)
MAP_TIMESTAMP_SCALAR(timestamp_us)
MAP_TIMESTAMP_SCALAR(timestamp_ns)

/**
 * @brief Macro used to define scalar type and scalar device type for
 * `cudf::duration_scalar` template class for duration C++ types.
 *
 * @param Type The duration C++ type
 */
#ifndef MAP_DURATION_SCALAR
#define MAP_DURATION_SCALAR(Type)                                     \
  template <>                                                         \
  struct type_to_scalar_type_impl<Type> {                             \
    using ScalarType       = cudf::duration_scalar<Type>;             \
    using ScalarDeviceType = cudf::duration_scalar_device_view<Type>; \
  };
#endif

MAP_DURATION_SCALAR(duration_D)
MAP_DURATION_SCALAR(duration_s)
MAP_DURATION_SCALAR(duration_ms)
MAP_DURATION_SCALAR(duration_us)
MAP_DURATION_SCALAR(duration_ns)

/**
 * @brief Maps a C++ type to the scalar type required to hold its value
 *
 * @tparam T The concrete C++ type to map
 */
template <typename T>
using scalar_type_t = typename type_to_scalar_type_impl<T>::ScalarType;

/**
 * @brief Maps a C++ type to the scalar device type required to hold its value
 *
 * @tparam T The concrete C++ type to map
 */
template <typename T>
using scalar_device_type_t = typename type_to_scalar_type_impl<T>::ScalarDeviceType;

/**
 * @brief Invokes an `operator()` template with the type instantiation based on
 * the specified `cudf::data_type`'s `id()`.
 *
 * Example usage with a functor that returns the size of the dispatched type:
 *
 * @code
 * struct size_of_functor{
 *  template <typename T>
 *  int operator()(){
 *    return sizeof(T);
 *  }
 * };
 * cudf::data_type t{INT32};
 * cudf::type_dispatcher(t, size_of_functor{});  // returns 4
 * @endcode
 *
 * The `type_dispatcher` uses `cudf::type_to_id<t>` to provide a default mapping
 * of `cudf::type_id`s to dispatched C++ types. However, this mapping may be
 * customized by explicitly specifying a user-defined trait struct for the
 * `IdTypeMap`. For example, to always dispatch `int32_t`
 *
 * @code
 * template<cudf::type_id t> struct always_int{ using type = int32_t; }
 *
 * // This will always invoke operator()<int32_t>
 * cudf::type_dispatcher<always_int>(data_type, f);
 * @endcode
 *
 * It is sometimes necessary to customize the dispatched functor's
 * `operator()` for different types.  This can be done in several ways.
 *
 * The first method is to use explicit template specialization. This is useful
 * for specializing behavior for single types. For example, a functor that
 * prints `int32_t` or `double` when invoked with either of those types, else it
 * prints `unhandled type`:
 *
 * @code
 * struct type_printer {
 *   template <typename ColumnType>
 *   void operator()() { std::cout << "unhandled type\n"; }
 * };
 *
 * // Due to a bug in g++, explicit member function specializations need to be
 * // defined outside of the class definition
 * template <>
 * void type_printer::operator()<int32_t>() { std::cout << "int32_t\n"; }
 *
 * template <>
 * void type_printer::operator()<double>() { std::cout << "double\n"; }
 * @endcode
 *
 * A second method is to use SFINAE with `std::enable_if_t`. This is useful for
 * specializing for a set of types that share some property. For example, a
 * functor that prints `integral` or `floating point` for integral or floating
 * point types:
 *
 * @code
 * struct integral_or_floating_point {
 *   template <typename ColumnType,
 *             std::enable_if_t<not std::is_integral_v<ColumnType>  and
 *                              not std::is_floating_point_v<ColumnType> >* = nullptr>
 *   void operator()() {
 *     std::cout << "neither integral nor floating point\n "; }
 *
 *   template <typename ColumnType,
 *             std::enable_if_t<std::is_integral_v<ColumnType> >* = nullptr>
 *   void operator()() { std::cout << "integral\n"; }
 *
 *   template <typename ColumnType,
 *             std::enable_if_t<std::is_floating_point_v<ColumnType> >* = nullptr>
 *   void operator()() { std::cout << "floating point\n"; }
 * };
 * @endcode
 *
 * For more info on SFINAE and `std::enable_if`, see
 * https://eli.thegreenplace.net/2014/sfinae-and-enable_if/
 *
 * The return type for all template instantiations of the functor's "operator()"
 * lambda must be the same, else there will be a compiler error as you would be
 * trying to return different types from the same function.
 *
 * @tparam IdTypeMap Maps a `cudf::type_id` to its dispatched C++ type
 * @tparam Functor The callable object's type
 * @tparam Ts Variadic parameter pack type
 * @param dtype The `cudf::data_type` whose `id()` determines which template
 * instantiation is invoked
 * @param f The callable whose `operator()` template is invoked
 * @param args Parameter pack of arguments forwarded to the `operator()`
 * invocation
 * @return Whatever is returned by the callable's `operator()`
 */
// This pragma disables a compiler warning that complains about the valid usage
// of calling a __host__ functor from this function which is __host__ __device__
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <template <cudf::type_id> typename IdTypeMap = id_to_type_impl,
          typename Functor,
          typename... Ts>
CUDF_HOST_DEVICE __forceinline__ constexpr decltype(auto) type_dispatcher(cudf::data_type dtype,
                                                                          Functor f,
                                                                          Ts&&... args)
{
  switch (dtype.id()) {
    case type_id::INT8:
      return f.template operator()<typename IdTypeMap<type_id::INT8>::type>(
        std::forward<Ts>(args)...);
    case type_id::INT16:
      return f.template operator()<typename IdTypeMap<type_id::INT16>::type>(
        std::forward<Ts>(args)...);
    case type_id::INT32:
      return f.template operator()<typename IdTypeMap<type_id::INT32>::type>(
        std::forward<Ts>(args)...);
    case type_id::INT64:
      return f.template operator()<typename IdTypeMap<type_id::INT64>::type>(
        std::forward<Ts>(args)...);
    case type_id::UINT8:
      return f.template operator()<typename IdTypeMap<type_id::UINT8>::type>(
        std::forward<Ts>(args)...);
    case type_id::UINT16:
      return f.template operator()<typename IdTypeMap<type_id::UINT16>::type>(
        std::forward<Ts>(args)...);
    case type_id::UINT32:
      return f.template operator()<typename IdTypeMap<type_id::UINT32>::type>(
        std::forward<Ts>(args)...);
    case type_id::UINT64:
      return f.template operator()<typename IdTypeMap<type_id::UINT64>::type>(
        std::forward<Ts>(args)...);
    case type_id::FLOAT32:
      return f.template operator()<typename IdTypeMap<type_id::FLOAT32>::type>(
        std::forward<Ts>(args)...);
    case type_id::FLOAT64:
      return f.template operator()<typename IdTypeMap<type_id::FLOAT64>::type>(
        std::forward<Ts>(args)...);
    case type_id::BOOL8:
      return f.template operator()<typename IdTypeMap<type_id::BOOL8>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_DAYS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_DAYS>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_SECONDS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_SECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_MILLISECONDS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_MILLISECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_MICROSECONDS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_MICROSECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_NANOSECONDS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_NANOSECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_DAYS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_DAYS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_SECONDS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_SECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_MILLISECONDS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_MILLISECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_MICROSECONDS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_MICROSECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_NANOSECONDS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_NANOSECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DICTIONARY32:
      return f.template operator()<typename IdTypeMap<type_id::DICTIONARY32>::type>(
        std::forward<Ts>(args)...);
    case type_id::STRING:
      return f.template operator()<typename IdTypeMap<type_id::STRING>::type>(
        std::forward<Ts>(args)...);
    case type_id::LIST:
      return f.template operator()<typename IdTypeMap<type_id::LIST>::type>(
        std::forward<Ts>(args)...);
    case type_id::DECIMAL32:
      return f.template operator()<typename IdTypeMap<type_id::DECIMAL32>::type>(
        std::forward<Ts>(args)...);
    case type_id::DECIMAL64:
      return f.template operator()<typename IdTypeMap<type_id::DECIMAL64>::type>(
        std::forward<Ts>(args)...);
    case type_id::DECIMAL128:
      return f.template operator()<typename IdTypeMap<type_id::DECIMAL128>::type>(
        std::forward<Ts>(args)...);
    case type_id::STRUCT:
      return f.template operator()<typename IdTypeMap<type_id::STRUCT>::type>(
        std::forward<Ts>(args)...);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Invalid type_id.");
#else
      CUDF_UNREACHABLE("Invalid type_id.");
#endif
    }
  }
}

// @cond
namespace detail {
template <typename T1>
struct double_type_dispatcher_second_type {
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  template <typename T2, typename F, typename... Ts>
  CUDF_HOST_DEVICE __forceinline__ decltype(auto) operator()(F&& f, Ts&&... args) const
  {
    return f.template operator()<T1, T2>(std::forward<Ts>(args)...);
  }
};

template <template <cudf::type_id> typename IdTypeMap>
struct double_type_dispatcher_first_type {
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  template <typename T1, typename F, typename... Ts>
  CUDF_HOST_DEVICE __forceinline__ decltype(auto) operator()(cudf::data_type type2,
                                                             F&& f,
                                                             Ts&&... args) const
  {
    return type_dispatcher<IdTypeMap>(type2,
                                      detail::double_type_dispatcher_second_type<T1>{},
                                      std::forward<F>(f),
                                      std::forward<Ts>(args)...);
  }
};
}  // namespace detail
// @endcond

/**
 * @brief Dispatches two type template parameters to a callable.
 *
 * This function expects a callable `f` with an `operator()` template accepting
 * two typename template parameters `T1` and `T2`.
 *
 * @param type1 The `data_type` used to dispatch a type for the first template
 * parameter of the callable `F`
 * @param type2 The `data_type` used to dispatch a type for the second template
 * parameter of the callable `F`
 * @param f The callable whose `operator()` template is invoked
 * @param args Parameter pack forwarded to the `operator()` invocation `F`.
 *
 * @return The result of invoking `f.template operator<T1, T2>(args)`
 */
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <template <cudf::type_id> typename IdTypeMap = id_to_type_impl, typename F, typename... Ts>
CUDF_HOST_DEVICE __forceinline__ constexpr decltype(auto) double_type_dispatcher(
  cudf::data_type type1, cudf::data_type type2, F&& f, Ts&&... args)
{
  return type_dispatcher<IdTypeMap>(type1,
                                    detail::double_type_dispatcher_first_type<IdTypeMap>{},
                                    type2,
                                    std::forward<F>(f),
                                    std::forward<Ts>(args)...);
}

/**
 * @brief Return a name for a given type.
 *
 * The returned type names are intended for error messages and are not
 * guaranteed to be stable.
 *
 * @param type The `data_type`
 * @return Name of the type
 */
std::string type_to_name(data_type type);

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
