/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <map>
#include <optional>

/**
 * @file generate_input.hpp
 * @brief Contains declarations of functions that generate columns filled with random data.
 *
 * Also includes the data profile descriptor classes.
 *
 * The create_random_table functions take a data profile, the information about table size and a
 * seed to deterministically generate a table with given parameters.
 *
 * Currently, the data generation is done on the CPU and the data is then copied to the device
 * memory.
 */

/**
 * @brief Identifies a group of related column's logical element types.
 */
enum class type_group_id : int32_t {
  INTEGRAL = static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS),
  INTEGRAL_SIGNED,
  FLOATING_POINT,
  NUMERIC,
  TIMESTAMP,
  DURATION,
  FIXED_POINT,
  COMPOUND,
  NESTED,
};

/**
 * @brief Identifies a probability distribution type.
 */
enum class distribution_id : int8_t {
  UNIFORM,    ///< Uniform sampling between the given bounds. Provides the best coverage of the
              ///< overall value range. Real data rarely has this distribution.
  NORMAL,     ///< Gaussian sampling - most samples are close to the middle of the range. Good for
              ///< simulating real-world numeric data.
  GEOMETRIC,  ///< Geometric sampling - highest chance to sample close to the lower bound. Good for
              ///< simulating real data with asymmetric distribution (unsigned values, timestamps).
};

// Default distribution types for each type
template <typename T, std::enable_if_t<cudf::is_chrono<T>()>* = nullptr>
distribution_id default_distribution_id()
{
  return distribution_id::GEOMETRIC;
}

template <typename T, std::enable_if_t<!std::is_unsigned_v<T> && cudf::is_numeric<T>()>* = nullptr>
distribution_id default_distribution_id()
{
  return distribution_id::NORMAL;
}

template <typename T,
          std::enable_if_t<!std::is_same_v<T, bool> && std::is_unsigned_v<T> &&
                           cudf::is_numeric<T>()>* = nullptr>
distribution_id default_distribution_id()
{
  return distribution_id::GEOMETRIC;
}

/**
 * @brief Default range for the timestamp types: 1970 - 2020.
 *
 * The 2020 timestamp is used as a lower bound to bias the geometric distribution to recent
 * timestamps.
 */
template <typename T, std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
std::pair<int64_t, int64_t> default_range()
{
  using cuda::std::chrono::duration_cast;
  auto const year = duration_cast<typename T::duration>(cudf::duration_D{365l});
  return {50 * year.count(), 0};
}

/**
 * @brief Default range for the duration types.
 *
 * If a geometric distribution is used, it will bias towards short duration values.
 */
template <typename T, std::enable_if_t<cudf::is_duration<T>()>* = nullptr>
std::pair<int64_t, int64_t> default_range()
{
  using cuda::std::chrono::duration_cast;
  auto const year = duration_cast<typename T::duration>(cudf::duration_D{365l});
  return {0, 2 * year.count()};
}

template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
std::pair<T, T> default_range()
{
  // Limits need to be such that `upper - lower` does not overflow
  return {std::numeric_limits<T>::lowest() / 2, std::numeric_limits<T>::max() / 2};
}

/**
 * @brief Enables partial specializations with SFINAE.
 */
template <typename T, typename Enable = void>
struct distribution_params;

/**
 * @brief Numeric values are parameterized with a distribution type and bounds of the same type.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<!std::is_same_v<T, bool> && cudf::is_numeric<T>()>> {
  distribution_id id;
  T lower_bound;
  T upper_bound;
};

/**
 * @brief Booleans are parameterized with the probability of getting `true` value.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<std::is_same_v<T, bool>>> {
  double probability_true;
};

/**
 * @brief Timestamps and durations are parameterized with a distribution type and int64_t bounds.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<cudf::is_chrono<T>()>> {
  distribution_id id;
  int64_t lower_bound;
  int64_t upper_bound;
};

/**
 * @brief Strings are parameterized by the distribution of their length and character range.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>> {
  distribution_params<uint32_t> length_params;
  unsigned char char_lower = 32;   ///< Lower bound of character range (inclusive)
  unsigned char char_upper = 137;  ///< Upper bound of character range (inclusive), >126 produces
                                   ///< UTF-8
};

/**
 * @brief Lists are parameterized by the distribution of their length, maximal nesting level, and
 * the element type.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<std::is_same_v<T, cudf::list_view>>> {
  cudf::type_id element_type;
  distribution_params<uint32_t> length_params;
  cudf::size_type max_depth;
};

/**
 * @brief Structs are parameterized by the maximal nesting level, and the leaf column types.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<std::is_same_v<T, cudf::struct_view>>> {
  std::vector<cudf::type_id> leaf_types;
  cudf::size_type max_depth;
};

/**
 * @brief Fixed-point values are parameterized with a distribution type, scale, and bounds of the
 * same type.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<cudf::is_fixed_point<T>()>> {
  distribution_id id;
  typename T::rep lower_bound;
  typename T::rep upper_bound;
  std::optional<numeric::scale_type> scale;
};

/**
 * @brief Returns a vector of types, corresponding to the input type or a type group.
 *
 * If the input is a `cudf::type_id` enumerator, function simply returns a vector containing this
 * type. If the input value corresponds to a `type_group_id` enumerator, function returns a vector
 * containing all types in the input group.
 *
 * @param id Integer equal to either a `cudf::type_id` enumerator or a `type_group_id` enumerator.
 */
std::vector<cudf::type_id> get_type_or_group(int32_t id);

/**
 * @brief Returns a vector of types, corresponding to the input types or type groups.
 *
 * If an element of the input vector is a `cudf::type_id` enumerator, function return value simply
 * includes this type. If an element of the input vector is a `type_group_id` enumerator, function
 * return value includes all types corresponding to the group enumerator.
 *
 * @param ids Vector of integers equal to either a `cudf::type_id` enumerator or a `type_group_id`
 * enumerator.
 */
std::vector<cudf::type_id> get_type_or_group(std::vector<int32_t> const& ids);

/**
 * @brief Contains data parameters for all types.
 *
 * This class exposes APIs to set and get distribution parameters for each supported type.
 * Parameters can be set for multiple types with a single call by passing a `type_group_id` instead
 * of `cudf::type_id`.
 *
 * All types have default parameters so it's not necessary to set the parameters before using them.
 */
class data_profile {
  std::map<cudf::type_id, distribution_params<uint64_t>> int_params;
  std::map<cudf::type_id, distribution_params<double>> float_params;
  distribution_params<cudf::string_view> string_dist_desc{{distribution_id::NORMAL, 0, 32}};
  distribution_params<cudf::list_view> list_dist_desc{
    cudf::type_id::INT32, {distribution_id::GEOMETRIC, 0, 64}, 2};
  distribution_params<cudf::struct_view> struct_dist_desc{
    {cudf::type_id::INT32, cudf::type_id::FLOAT32, cudf::type_id::STRING}, 2};
  std::map<cudf::type_id, distribution_params<numeric::decimal128>> decimal_params;

  double bool_probability_true           = 0.5;
  std::optional<double> null_probability = 0.01;
  cudf::size_type cardinality =
    2000;  /// Upper bound on the number of unique values generated if `0 <= cardinality < n`, where
           /// `n` is the total number of values to be generated. If `cardinality >= n`, n` unique
           /// values of the requested data type are generated.
  cudf::size_type avg_run_length = 4;

 public:
  template <typename T,
            std::enable_if_t<!std::is_same_v<T, bool> && cuda::std::is_integral_v<T>, T>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  template <typename T, std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  template <typename T, std::enable_if_t<std::is_same_v<T, bool>>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  template <typename T, std::enable_if_t<cudf::is_chrono<T>()>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::list_view>>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::struct_view>>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  [[nodiscard]] auto get_bool_probability_true() const { return bool_probability_true; }
  [[nodiscard]] auto get_null_probability() const { return null_probability; };
  [[nodiscard]] auto get_valid_probability() const { return 1. - null_probability.value_or(0.); };
  [[nodiscard]] auto get_cardinality() const { return cardinality; };
  [[nodiscard]] auto get_avg_run_length() const { return avg_run_length; };

  // Users should pass integral values for bounds when setting the parameters for types that have
  // discrete distributions (integers, strings, lists). Otherwise the call with have no effect.
  template <typename T,
            typename Type_enum,
            std::enable_if_t<cuda::std::is_integral_v<T>, T>* = nullptr>
  void set_distribution_params(Type_enum type_or_group,
                               distribution_id dist,
                               T lower_bound,
                               T upper_bound)
  {
    for (auto tid : get_type_or_group(static_cast<int32_t>(type_or_group))) {
      if (tid == cudf::type_id::STRING) {
        string_dist_desc.length_params = {
          dist, static_cast<uint32_t>(lower_bound), static_cast<uint32_t>(upper_bound)};
      } else if (tid == cudf::type_id::LIST) {
        list_dist_desc.length_params = {
          dist, static_cast<uint32_t>(lower_bound), static_cast<uint32_t>(upper_bound)};
      } else {
        int_params[tid] = {
          dist, static_cast<uint64_t>(lower_bound), static_cast<uint64_t>(upper_bound)};
      }
    }
  }

  // Users should pass floating point values for bounds when setting the parameters for types that
  // have continuous distributions (floating point types). Otherwise the call with have no effect.
  template <typename T,
            typename Type_enum,
            std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
  void set_distribution_params(Type_enum type_or_group,
                               distribution_id dist,
                               T lower_bound,
                               T upper_bound)
  {
    for (auto tid : get_type_or_group(static_cast<int32_t>(type_or_group))) {
      float_params[tid] = {
        dist, static_cast<double>(lower_bound), static_cast<double>(upper_bound)};
    }
  }

  // Users should pass integral values for bounds when setting the parameters for fixed-point.
  // Otherwise the call with have no effect.
  template <typename T,
            typename Type_enum,
            std::enable_if_t<cuda::std::is_integral_v<T>, T>* = nullptr>
  void set_distribution_params(Type_enum type_or_group,
                               distribution_id dist,
                               T lower_bound,
                               T upper_bound,
                               numeric::scale_type scale)
  {
    for (auto tid : get_type_or_group(static_cast<int32_t>(type_or_group))) {
      decimal_params[tid] = {
        dist, static_cast<__int128_t>(lower_bound), static_cast<__int128_t>(upper_bound), scale};
    }
  }

  template <typename T, typename Type_enum, std::enable_if_t<cudf::is_chrono<T>(), T>* = nullptr>
  void set_distribution_params(Type_enum type_or_group,
                               distribution_id dist,
                               typename T::rep lower_bound,
                               typename T::rep upper_bound)
  {
    for (auto tid : get_type_or_group(static_cast<int32_t>(type_or_group))) {
      int_params[tid] = {
        dist, static_cast<uint64_t>(lower_bound), static_cast<uint64_t>(upper_bound)};
    }
  }

  void set_bool_probability_true(double p);
  void set_null_probability(std::optional<double> p);
  void set_cardinality(cudf::size_type c) { cardinality = c; }
  void set_avg_run_length(cudf::size_type avg_rl) { avg_run_length = avg_rl; }
  void set_list_depth(cudf::size_type max_depth);
  void set_list_type(cudf::type_id type) { list_dist_desc.element_type = type; }
  void set_struct_depth(cudf::size_type max_depth);
  void set_struct_types(cudf::host_span<cudf::type_id const> types);
  void set_string_char_range(unsigned char lower, unsigned char upper);
};

/**
 * @brief Builder to construct data profiles for the random data generator.
 *
 * Setters can be chained to set multiple properties in a single expression.
 * For example, `data_profile` initialization
 * @code{.pseudo}
 * data_profile profile;
 * profile.set_null_probability(0.01);
 * profile.set_cardinality(0);
 * profile.set_distribution_params(cudf::type_id::INT32, distribution_id::UNIFORM, 0, 100);
 * @endcode
 * becomes
 * @code{.pseudo}
 * data_profile const profile =
 *   data_profile_builder().cardinality(0).null_probability(0.01).distribution(
 *     cudf::type_id::INT32, distribution_id::UNIFORM, 0, 100);
 * @endcode
 * The builder makes it easier to have immutable `data_profile` objects even with the complex
 * initialization. The `profile` object in the example above is initialized from
 * `data_profile_builder` using an implicit conversion operator.
 *
 * The builder API also includes a few additional convenience setters:
 * Overload of `distribution` that only takes the distribution type (not the range).
 * `no_validity`, which is a simpler equivalent of `null_probability(std::nullopt)`.
 */
class data_profile_builder {
  data_profile profile;

 public:
  /**
   * @brief Sets random distribution type for a given set of data types.
   *
   * Only the distribution type is set; the distribution will use the default range.
   *
   * @param type_or_group  Type or group ID, depending on whether the new distribution
   * applies to a single type or a subset of types
   * @param dist  Random distribution type
   * @tparam T Data type of the distribution range; does not need to match the data type
   * @return this for chaining
   */
  template <typename T, typename Type_enum>
  data_profile_builder& distribution(Type_enum type_or_group, distribution_id dist)
  {
    auto const range = default_range<T>();
    profile.set_distribution_params(type_or_group, dist, range.first, range.second);
    return *this;
  }

  /**
   * @brief Sets random distribution type and value range for a given set of data types.
   *
   * @tparam T Parameters that are forwarded to set_distribution_params
   * @return this for chaining
   */
  template <class... T>
  data_profile_builder& distribution(T&&... t)
  {
    profile.set_distribution_params(std::forward<T>(t)...);
    return *this;
  }

  /**
   * @brief Sets the probability that a randomly generated boolean element with be `true`.
   *
   * For example, passing `0.9` means that 90% of values in boolean columns with be `true`.
   *
   * @param p Probability of `true` values, in range [0..1]
   * @return this for chaining
   */
  data_profile_builder& bool_probability_true(double p)
  {
    profile.set_bool_probability_true(p);
    return *this;
  }

  /**
   * @brief Sets the probability that a randomly generated element will be `null`.
   *
   * @param p Probability of `null` values, in range [0..1]
   * @return this for chaining
   */
  data_profile_builder& null_probability(std::optional<double> p)
  {
    profile.set_null_probability(p);
    return *this;
  }

  /**
   * @brief Disables the creation of null mask in the output columns.
   *
   * @return this for chaining
   */
  data_profile_builder& no_validity()
  {
    profile.set_null_probability(std::nullopt);
    return *this;
  }

  /**
   * @brief Sets the maximum number of unique values in each output column.
   *
   * @param c Maximum number of unique values
   * @return this for chaining
   */
  data_profile_builder& cardinality(cudf::size_type c)
  {
    profile.set_cardinality(c);
    return *this;
  }

  /**
   * @brief Sets the average length of sequences of equal elements in output columns.
   *
   * @param avg_rl Average sequence length (run-length)
   * @return this for chaining
   */
  data_profile_builder& avg_run_length(cudf::size_type avg_rl)
  {
    profile.set_avg_run_length(avg_rl);
    return *this;
  }

  /**
   * @brief Sets the maximum nesting depth of generated list columns.
   *
   * @param max_depth maximum nesting depth
   * @return this for chaining
   */
  data_profile_builder& list_depth(cudf::size_type max_depth)
  {
    profile.set_list_depth(max_depth);
    return *this;
  }

  /**
   * @brief Sets the data type of list elements.
   *
   * @param type data type ID
   * @return this for chaining
   */
  data_profile_builder& list_type(cudf::type_id type)
  {
    profile.set_list_type(type);
    return *this;
  }

  /**
   * @brief Sets the maximum nesting depth of generated struct columns.
   *
   * @param max_depth maximum nesting depth
   * @return this for chaining
   */
  data_profile_builder& struct_depth(cudf::size_type max_depth)
  {
    profile.set_struct_depth(max_depth);
    return *this;
  }

  /**
   * @brief Sets the data types of struct fields.
   *
   * @param types data type IDs
   * @return this for chaining
   */
  data_profile_builder& struct_types(cudf::host_span<cudf::type_id const> types)
  {
    profile.set_struct_types(types);
    return *this;
  }

  /**
   * @brief Sets the character range for generated strings.
   *
   * Characters are uniformly distributed in the range [lower, upper].
   * Values > 126 will produce multi-byte UTF-8 characters.
   *
   * @param lower Lower bound of character range (inclusive, must be >= 32)
   * @param upper Upper bound of character range (inclusive)
   * @return this for chaining
   */
  data_profile_builder& string_char_range(unsigned char lower, unsigned char upper)
  {
    profile.set_string_char_range(lower, upper);
    return *this;
  }

  /**
   * @brief move data_profile member once it's built.
   */
  operator data_profile&&() { return std::move(profile); }
};

/**
 * @brief Strongly typed table size in bytes. Used to disambiguate overloads of
 * `create_random_table`.
 */
struct table_size_bytes {
  size_t size;
};

/**
 * @brief Strongly typed row count. Used to disambiguate overloads of `create_random_table`.
 */
struct row_count {
  cudf::size_type count;
};

/**
 * @brief Deterministically generates a table filled with data with the given parameters.
 *
 * @param dtype_ids Vector of requested column types
 * @param table_bytes Target size of the output table, in bytes. Some type may not produce columns
 * of exact size
 * @param data_params Optional, set of data parameters describing the data profile for each type
 * @param seed Optional, seed for the pseudo-random engine
 */
std::unique_ptr<cudf::table> create_random_table(std::vector<cudf::type_id> const& dtype_ids,
                                                 table_size_bytes table_bytes,
                                                 data_profile const& data_params = data_profile{},
                                                 unsigned seed                   = 1);

/**
 * @brief Deterministically generates a table filled with data with the given parameters.
 *
 * @param dtype_ids Vector of requested column types
 * @param num_rows Number of rows in the output table
 * @param data_params Optional, set of data parameters describing the data profile for each type
 * @param seed Optional, seed for the pseudo-random engine
 */
std::unique_ptr<cudf::table> create_random_table(std::vector<cudf::type_id> const& dtype_ids,
                                                 row_count num_rows,
                                                 data_profile const& data_params = data_profile{},
                                                 unsigned seed                   = 1);

/**
 * @brief Deterministically generates a column filled with data with the given parameters.
 *
 * @param dtype_id Requested column type
 * @param num_rows Number of rows in the output column
 * @param data_params Optional, set of data parameters describing the data profile
 * @param seed Optional, seed for the pseudo-random engine
 */
std::unique_ptr<cudf::column> create_random_column(cudf::type_id dtype_id,
                                                   row_count num_rows,
                                                   data_profile const& data_params = data_profile{},
                                                   unsigned seed                   = 1);

/**
 * @brief Deterministically generates a large string column filled with data with the given
 * parameters.
 *
 * @param num_rows Number of rows in the output column
 * @param row_width Width of each string in the column
 * @param hit_rate The hit rate percentage, ranging from 0 to 100
 */
std::unique_ptr<cudf::column> create_string_column(cudf::size_type num_rows,
                                                   cudf::size_type row_width,
                                                   int32_t hit_rate);

/**
 * @brief Generates an string column filled with ASCII characters only
 *
 * @param profile Data profile for the output column
 * @param num_rows Number of rows in the output column
 * @param seed Optional, seed for the pseudo-random engine
 */
std::unique_ptr<cudf::column> create_ascii_string_column(data_profile const& profile,
                                                         cudf::size_type num_rows,
                                                         unsigned seed = 1);

/**
 * @brief Generate sequence columns starting with value 0 in first row and increasing by 1 in
 * subsequent rows.
 *
 * @param dtype_ids Vector of requested column types
 * @param num_rows Number of rows in the output table
 * @param null_probability Optional, probability of a null value
 *  no value implies no null mask, =0 implies all valids, >=1 implies all nulls
 * @param seed Optional, seed for the pseudo-random engine
 * @return A table with the sequence columns.
 */
std::unique_ptr<cudf::table> create_sequence_table(
  std::vector<cudf::type_id> const& dtype_ids,
  row_count num_rows,
  std::optional<double> null_probability = std::nullopt,
  unsigned seed                          = 1);

/**
 * @brief Repeats the input data types cyclically to fill a vector of @ref num_cols
 * elements.
 *
 * @param dtype_ids Vector of requested column types
 * @param num_cols Number of types in the output vector
 * @return A vector of type_ids
 */
std::vector<cudf::type_id> cycle_dtypes(std::vector<cudf::type_id> const& dtype_ids,
                                        cudf::size_type num_cols);

/**
 * @brief Repeat the given two data types with a given ratio of a:b.
 *
 * The first dtype will have 'first_num' columns and the second will have 'num_cols - first_num'
 * columns.
 *
 * @param dtype_ids Pair of requested column types
 * @param num_cols Total number of columns in the output vector
 * @param first_num Total number of columns of type `dtype_ids.first`
 * @return A vector of type_ids
 */
std::vector<cudf::type_id> mix_dtypes(std::pair<cudf::type_id, cudf::type_id> const& dtype_ids,
                                      cudf::size_type num_cols,
                                      int first_num);
/**
 * @brief Create a random null mask object
 *
 * @param size number of rows
 * @param null_probability probability of a null value
 *  no value implies no null mask, =0 implies all valids, >=1 implies all nulls
 * @param seed Optional, seed for the pseudo-random engine
 * @return null mask device buffer with random null mask data and null count
 */
std::pair<rmm::device_buffer, cudf::size_type> create_random_null_mask(
  cudf::size_type size, std::optional<double> null_probability = std::nullopt, unsigned seed = 1);
