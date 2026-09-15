/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_input.hpp"
#include "random_distribution_factory.cuh"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/filling.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/random>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace {
/**
 * @brief Mersenne Twister pseudo-random engine.
 */
auto deterministic_engine(unsigned seed) { return cuda::std::philox4x32{seed}; }

/**
 *  Computes the mean value for a distribution of given type and value bounds.
 */
template <typename T>
double get_distribution_mean(distribution_params<T> const& dist)
{
  switch (dist.id) {
    case distribution_id::NORMAL:
    case distribution_id::UNIFORM: return (dist.lower_bound / 2.) + (dist.upper_bound / 2.);
    case distribution_id::GEOMETRIC: {
      // Geometric distribution is approximated by a half-normal distribution
      // Doubling the standard deviation because the dist range only includes half of the (unfolded)
      // normal distribution
      auto const gauss_std_dev   = std_dev_from_range(dist.lower_bound, dist.upper_bound) * 2;
      auto const half_gauss_mean = gauss_std_dev * sqrt(2. / M_PI);
      if (dist.lower_bound < dist.upper_bound)
        return dist.lower_bound + half_gauss_mean;
      else
        return dist.lower_bound - half_gauss_mean;
    }
    default: CUDF_FAIL("Unsupported distribution type.");
  }
}

/**
 * @brief Calculates the number of direct parents needed to generate a struct column hierarchy with
 * lowest maximum number of children in any nested column.
 *
 * Used to generate an "evenly distributed" struct column hierarchy with the given number of leaf
 * columns and nesting levels. The column tree is considered evenly distributed if all columns have
 * nearly the same number of child columns (difference not larger than one).
 */
int num_direct_parents(int num_lvls, int num_leaf_columns)
{
  // Estimated average number of children in the hierarchy;
  auto const num_children_avg = std::pow(num_leaf_columns, 1. / num_lvls);
  // Minimum number of children columns for any column in the hierarchy
  int const num_children_min = std::floor(num_children_avg);
  // Maximum number of children columns for any column in the hierarchy
  int const num_children_max = num_children_min + 1;

  // Minimum number of columns needed so that their number of children does not exceed the maximum
  int const min_for_current_nesting =
    std::ceil(static_cast<double>(num_leaf_columns) / num_children_max);
  // Minimum number of columns needed so that columns at the higher levels have at least the minimum
  // number of children
  int const min_for_upper_nesting = std::pow(num_children_min, num_lvls - 1);
  // Both conditions need to be satisfied
  return std::max(min_for_current_nesting, min_for_upper_nesting);
}

// Size of the null mask for each row, in bytes
[[nodiscard]] double row_null_mask_size(data_profile const& profile)
{
  return profile.get_null_probability().has_value() ? 1. / 8 : 0.;
}

/**
 * @brief Computes the average element size in a column, given the data profile.
 *
 * Random distribution parameters like average string length and maximum list nesting level affect
 * the element size of non-fixed-width columns. For lists and structs, `avg_element_size` is called
 * recursively to determine the size of nested columns.
 */
double avg_element_size(data_profile const& profile, cudf::data_type dtype);

// Utilities to determine the mean size of an element, given the data profile
template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
double non_fixed_width_size(data_profile const& profile)
{
  CUDF_FAIL("Should not be called, use `size_of` for this type instead");
}

template <typename T, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>())>
double non_fixed_width_size(data_profile const& profile)
{
  CUDF_FAIL("not implemented!");
}

template <>
double non_fixed_width_size<cudf::string_view>(data_profile const& profile)
{
  auto const dist = profile.get_distribution_params<cudf::string_view>().length_params;
  return get_distribution_mean(dist) * profile.get_valid_probability() + sizeof(cudf::size_type) +
         row_null_mask_size(profile);
}

double geometric_sum(size_t n, double p)
{
  if (p == 1) { return n; }
  return (1 - std::pow(p, n)) / (1 - p);
}

template <>
double non_fixed_width_size<cudf::list_view>(data_profile const& profile)
{
  auto const dist_params = profile.get_distribution_params<cudf::list_view>();
  auto const single_level_mean =
    get_distribution_mean(dist_params.length_params) * profile.get_valid_probability();

  // Leaf column size
  auto const element_size  = avg_element_size(profile, cudf::data_type{dist_params.element_type});
  auto const element_count = std::pow(single_level_mean, dist_params.max_depth);

  auto const offset_size = avg_element_size(profile, cudf::data_type{cudf::type_id::INT32});
  // Each nesting level includes offsets, this is the sum of all levels
  auto const total_offset_count = geometric_sum(dist_params.max_depth, single_level_mean);

  return element_size * element_count + offset_size * total_offset_count;
}

[[nodiscard]] cudf::size_type num_struct_columns(data_profile const& profile)
{
  auto const dist_params = profile.get_distribution_params<cudf::struct_view>();

  cudf::size_type children_count     = dist_params.leaf_types.size();
  cudf::size_type total_parent_count = 0;
  for (cudf::size_type lvl = dist_params.max_depth; lvl > 0; --lvl) {
    children_count = num_direct_parents(lvl, children_count);
    total_parent_count += children_count;
  }
  return total_parent_count;
}

template <>
double non_fixed_width_size<cudf::struct_view>(data_profile const& profile)
{
  auto const dist_params = profile.get_distribution_params<cudf::struct_view>();
  auto const total_children_size =
    std::accumulate(dist_params.leaf_types.cbegin(),
                    dist_params.leaf_types.cend(),
                    0ul,
                    [&](auto sum, auto type_id) {
                      return sum + avg_element_size(profile, cudf::data_type{type_id});
                    });

  // struct columns have a null mask for each row
  auto const structs_null_mask_size = num_struct_columns(profile) * row_null_mask_size(profile);

  return total_children_size + structs_null_mask_size;
}

struct non_fixed_width_size_fn {
  template <typename T>
  double operator()(data_profile const& profile)
  {
    return non_fixed_width_size<T>(profile);
  }
};

double avg_element_size(data_profile const& profile, cudf::data_type dtype)
{
  if (cudf::is_fixed_width(dtype)) { return cudf::size_of(dtype) + row_null_mask_size(profile); }
  return cudf::type_dispatcher(dtype, non_fixed_width_size_fn{}, profile);
}

/**
 * @brief bool generator with given probability [0.0 - 1.0] of returning true.
 */
struct bool_generator {
  cuda::std::philox4x32 engine;
  cuda::std::uniform_real_distribution<float> dist;
  double probability_true;
  bool_generator(cuda::std::philox4x32 engine, double probability_true)
    : engine(engine), dist{0, 1}, probability_true{probability_true}
  {
  }
  bool_generator(unsigned seed, double probability_true)
    : engine(seed), dist{0, 1}, probability_true{probability_true}
  {
  }

  __device__ bool operator()(size_t n)
  {
    engine.discard(n);
    // Short-circuit the degenerate endpoints so that probability_true == 1.0 (null_probability ==
    // 0) always yields all-valid and probability_true == 0.0 always yields all-null, independent of
    // the float distribution's endpoint behavior (which can return exactly 1.0f).
    if (probability_true >= 1.0) return true;
    if (probability_true <= 0.0) return false;
    return dist(engine) < probability_true;
  }
};

/**
 * @brief Functor that computes a random column element with the given data profile.
 *
 * The implementation is SFINAEd for different type groups. Currently only used for fixed-width
 * types.
 */
template <typename T, typename Enable = void>
struct random_value_fn;

/**
 * @brief Creates an random timestamp/duration value
 */
template <typename T>
struct random_value_fn<T, std::enable_if_t<cudf::is_chrono<T>()>> {
  distribution_fn<int64_t> seconds_gen;
  distribution_fn<int64_t> nanoseconds_gen;

  random_value_fn(distribution_params<T> params)
  {
    using cuda::std::chrono::duration_cast;

    std::pair<cudf::duration_s, cudf::duration_s> const range_s = {
      duration_cast<cuda::std::chrono::seconds>(typename T::duration{params.lower_bound}),
      duration_cast<cuda::std::chrono::seconds>(typename T::duration{params.upper_bound})};
    if (range_s.first != range_s.second) {
      seconds_gen =
        make_distribution<int64_t>(params.id, range_s.first.count(), range_s.second.count());

      nanoseconds_gen = make_distribution<int64_t>(distribution_id::UNIFORM, 0l, 1000000000l);
    } else {
      // Don't need a random seconds generator for sub-second intervals
      seconds_gen = [range_s](cuda::std::philox4x32&, size_t size) {
        rmm::device_uvector<int64_t> result(size, cudf::get_default_stream());
        thrust::uninitialized_fill(
          thrust::device, result.begin(), result.end(), range_s.second.count());
        return result;
      };

      std::pair<cudf::duration_ns, cudf::duration_ns> const range_ns = {
        duration_cast<cudf::duration_ns>(typename T::duration{params.lower_bound}),
        duration_cast<cudf::duration_ns>(typename T::duration{params.upper_bound})};
      nanoseconds_gen = make_distribution<int64_t>(distribution_id::UNIFORM,
                                                   std::min(range_ns.first.count(), 0l),
                                                   std::max(range_ns.second.count(), 0l));
    }
  }

  rmm::device_uvector<T> operator()(cuda::std::philox4x32& engine, unsigned size)
  {
    auto const sec = seconds_gen(engine, size);
    auto const ns  = nanoseconds_gen(engine, size);
    rmm::device_uvector<T> result(size, cudf::get_default_stream());
    thrust::transform(
      thrust::device,
      sec.begin(),
      sec.end(),
      ns.begin(),
      result.begin(),
      cuda::proclaim_return_type<T>([] __device__(int64_t sec_value, int64_t nanoseconds_value) {
        auto const timestamp_ns =
          cudf::duration_s{sec_value} + cudf::duration_ns{nanoseconds_value};
        // Return value in the type's precision
        return T(cuda::std::chrono::duration_cast<typename T::duration>(timestamp_ns));
      }));
    return result;
  }
};

/**
 * @brief Creates an random fixed_point value.
 */
template <typename T>
struct random_value_fn<T, std::enable_if_t<cudf::is_fixed_point<T>()>> {
  using DeviceType = cudf::device_storage_type_t<T>;
  DeviceType const lower_bound;
  DeviceType const upper_bound;
  distribution_fn<DeviceType> dist;
  std::optional<numeric::scale_type> scale;

  random_value_fn(distribution_params<T> const& desc)
    : lower_bound{desc.lower_bound},
      upper_bound{desc.upper_bound},
      dist{make_distribution<DeviceType>(desc.id, lower_bound, upper_bound)},
      scale{desc.scale}
  {
  }

  [[nodiscard]] numeric::scale_type get_scale(cuda::std::philox4x32& engine)
  {
    if (not scale.has_value()) {
      constexpr int max_scale = std::numeric_limits<DeviceType>::digits10;
      std::uniform_int_distribution<int> scale_dist{-max_scale, max_scale};
      std::mt19937 engine_scale(engine());
      scale = numeric::scale_type{scale_dist(engine_scale)};
    }
    return scale.value_or(numeric::scale_type{0});
  }

  rmm::device_uvector<DeviceType> operator()(cuda::std::philox4x32& engine, unsigned size)
  {
    return dist(engine, size);
  }
};

/**
 * @brief Creates an random numeric value with the given distribution.
 */
template <typename T>
struct random_value_fn<T, std::enable_if_t<!std::is_same_v<T, bool> && cudf::is_numeric<T>()>> {
  T const lower_bound;
  T const upper_bound;
  distribution_fn<T> dist;

  random_value_fn(distribution_params<T> const& desc)
    : lower_bound{desc.lower_bound},
      upper_bound{desc.upper_bound},
      dist{make_distribution<T>(desc.id, desc.lower_bound, desc.upper_bound)}
  {
  }

  auto operator()(cuda::std::philox4x32& engine, unsigned size) { return dist(engine, size); }
};

/**
 * @brief Creates an boolean value with given probability of returning `true`.
 */
template <typename T>
struct random_value_fn<T, typename std::enable_if_t<std::is_same_v<T, bool>>> {
  // Bernoulli distribution
  distribution_fn<bool> dist;

  random_value_fn(distribution_params<bool> const& desc)
    : dist{[valid_prob = desc.probability_true](cuda::std::philox4x32& engine,
                                                size_t size) -> rmm::device_uvector<bool> {
        rmm::device_uvector<bool> result(size, cudf::get_default_stream());
        thrust::tabulate(
          thrust::device, result.begin(), result.end(), bool_generator(engine, valid_prob));
        return result;
      }}
  {
  }
  auto operator()(cuda::std::philox4x32& engine, unsigned size) { return dist(engine, size); }
};

/**
 * @brief Generate indices within range [0 , cardinality) repeating with average run length
 * `avg_run_len`
 *
 * @param avg_run_len  Average run length of the generated indices
 * @param cardinality  Number of unique values in the output vector
 * @param num_rows     Number of indices to generate
 * @param engine       Random engine
 * @return Generated indices of type `cudf::size_type`
 */
rmm::device_uvector<cudf::size_type> sample_indices_with_run_length(cudf::size_type avg_run_len,
                                                                    cudf::size_type cardinality,
                                                                    cudf::size_type num_rows,
                                                                    cuda::std::philox4x32& engine)
{
  auto sample_dist = random_value_fn<cudf::size_type>{
    distribution_params<cudf::size_type>{distribution_id::UNIFORM, 0, cardinality - 1}};
  if (avg_run_len > 1) {
    auto avglen_dist =
      random_value_fn<int>{distribution_params<int>{distribution_id::UNIFORM, 1, 2 * avg_run_len}};
    auto const approx_run_len = num_rows / avg_run_len + 1;
    auto run_lens             = avglen_dist(engine, approx_run_len);
    thrust::inclusive_scan(
      thrust::device, run_lens.begin(), run_lens.end(), run_lens.begin(), cuda::std::plus<int>{});
    auto const samples_indices = sample_dist(engine, approx_run_len + 1);
    // This is gather.
    auto avg_repeated_sample_indices_iterator = cuda::transform_iterator(
      cuda::counting_iterator<cudf::size_type>{0},
      cuda::proclaim_return_type<cudf::size_type>(
        [rb              = run_lens.begin(),
         re              = run_lens.end(),
         samples_indices = samples_indices.begin()] __device__(cudf::size_type i) {
          auto sample_idx = thrust::upper_bound(thrust::seq, rb, re, i) - rb;
          return samples_indices[sample_idx];
        }));
    rmm::device_uvector<cudf::size_type> repeated_sample_indices(num_rows,
                                                                 cudf::get_default_stream());
    thrust::copy(thrust::device,
                 avg_repeated_sample_indices_iterator,
                 avg_repeated_sample_indices_iterator + num_rows,
                 repeated_sample_indices.begin());
    return repeated_sample_indices;
  } else {
    // generate n samples.
    return sample_dist(engine, num_rows);
  }
}

enum class string_encoding {
  ASCII,
  UTF8,
};

template <string_encoding Encoding = string_encoding::UTF8>
struct string_generator {
  char* chars;
  cuda::std::philox4x32 engine;
  cuda::std::uniform_int_distribution<unsigned char> char_dist;
  string_generator(char* c, cuda::std::philox4x32& engine)
    : chars(c), engine(engine), char_dist(32, Encoding == string_encoding::ASCII ? 126 : 137)
  // ~90% ASCII, ~10% UTF-8.
  // ~80% not-space, ~20% space.
  // range 32-127 is ASCII; 127-136 will be multi-byte UTF-8
  {
  }
  __device__ void operator()(cuda::std::tuple<int64_t, int64_t> str_begin_end)
  {
    auto begin = cuda::std::get<0>(str_begin_end);
    auto end   = cuda::std::get<1>(str_begin_end);
    engine.discard(begin);
    for (auto i = begin; i < end; ++i) {
      auto ch = char_dist(engine);
      if constexpr (Encoding == string_encoding::UTF8) {
        if (i == end - 1 && ch >= '\x7F') ch = ' ';  // last element ASCII only.
        if (ch >= '\x7F') {                          // x7F is at the top edge of ASCII
          chars[i++] = '\xC4';                       // these characters are assigned two bytes
          ch         = (ch >> 2) | 0x80;
        }
      }
      chars[i] = static_cast<char>(ch);
    }
  }
};

/**
 * @brief Create a UTF-8 string column with the average length.
 *
 */
template <string_encoding Encoding = string_encoding::UTF8>
std::unique_ptr<cudf::column> create_random_utf8_string_column(data_profile const& profile,
                                                               cuda::std::philox4x32& engine,
                                                               cudf::size_type num_rows)
{
  auto len_dist =
    random_value_fn<uint32_t>{profile.get_distribution_params<cudf::string_view>().length_params};
  auto valid_dist = random_value_fn<bool>(
    distribution_params<bool>{1. - profile.get_null_probability().value_or(0)});
  auto lengths   = len_dist(engine, num_rows + 1);
  auto null_mask = valid_dist(engine, num_rows + 1);
  auto stream    = cudf::get_default_stream();
  auto mr        = cudf::get_current_device_resource_ref();

  thrust::transform_if(
    thrust::device,
    lengths.begin(),
    lengths.end(),
    null_mask.begin(),
    lengths.begin(),
    cuda::proclaim_return_type<cudf::size_type>([] __device__(auto) { return 0; }),
    cuda::std::logical_not<bool>{});
  // offsets are created as INT32 or INT64 as appropriate
  auto [offsets, chars_length] = cudf::strings::detail::make_offsets_child_column(
    lengths.begin(), lengths.begin() + num_rows, stream, mr);
  // use the offsetalator to normalize the offset values for use by the string_generator
  auto offsets_itr = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());
  rmm::device_uvector<char> chars(chars_length, cudf::get_default_stream());
  thrust::for_each_n(thrust::device,
                     cuda::make_zip_iterator(cuda::std::make_tuple(offsets_itr, offsets_itr + 1)),
                     num_rows,
                     string_generator<Encoding>{chars.data(), engine});

  auto [result_bitmask, null_count] =
    profile.get_null_probability().has_value()
      ? cudf::bools_to_mask(cudf::device_span<bool const>(null_mask), stream)
      : std::pair{std::make_unique<rmm::device_buffer>(), 0};

  return cudf::make_strings_column(num_rows,
                                   std::move(offsets),
                                   chars.release(),
                                   null_count,
                                   std::move(*result_bitmask.release()));
}

// Forward declarations for create_rand_col_fn
template <typename T>
std::unique_ptr<cudf::column> create_random_column(data_profile const& profile,
                                                   cuda::std::philox4x32& engine,
                                                   cudf::size_type num_rows);

template <typename T>
  requires(cudf::is_numeric_not_bool<T>())
std::unique_ptr<cudf::column> create_distinct_rows_column(data_profile const& profile,
                                                          cuda::std::philox4x32& engine,
                                                          cudf::size_type num_rows);

template <typename T>
  requires(!cudf::is_numeric_not_bool<T>())
std::unique_ptr<cudf::column> create_distinct_rows_column(data_profile const& profile,
                                                          cuda::std::philox4x32& engine,
                                                          cudf::size_type num_rows);

/**
 * @brief Functor to dispatch create_random_column calls.
 */
struct create_rand_col_fn {
 public:
  template <typename T>
  std::unique_ptr<cudf::column> operator()(data_profile const& profile,
                                           cuda::std::philox4x32& engine,
                                           cudf::size_type num_rows)
  {
    if (profile.get_cardinality() >= num_rows) {
      return create_distinct_rows_column<T>(profile, engine, num_rows);
    }
    return create_random_column<T>(profile, engine, num_rows);
  }
};

/**
 * @brief Creates a column with random content of type @ref T.
 *
 * @param profile Parameters for the random generator
 * @param engine Pseudo-random engine
 * @param num_rows Size of the output column
 *
 * @tparam T Data type of the output column
 * @return Column filled with random data
 */
template <typename T>
std::unique_ptr<cudf::column> create_random_column(data_profile const& profile,
                                                   cuda::std::philox4x32& engine,
                                                   cudf::size_type num_rows)
{
  // Bernoulli distribution
  auto valid_dist = random_value_fn<bool>(
    distribution_params<bool>{1. - profile.get_null_probability().value_or(0)});
  auto value_dist = random_value_fn<T>{profile.get_distribution_params<T>()};

  using DeviceType            = cudf::device_storage_type_t<T>;
  cudf::data_type const dtype = [&]() {
    if constexpr (cudf::is_fixed_point<T>())
      return cudf::data_type{cudf::type_to_id<T>(), value_dist.get_scale(engine)};
    else
      return cudf::data_type{cudf::type_to_id<T>()};
  }();

  // Distribution for picking elements from the array of samples
  auto const avg_run_len = profile.get_avg_run_length();
  rmm::device_uvector<DeviceType> data(0, cudf::get_default_stream());
  rmm::device_uvector<bool> null_mask(0, cudf::get_default_stream());

  if (profile.get_cardinality() == 0 and avg_run_len == 1) {
    data      = value_dist(engine, num_rows);
    null_mask = valid_dist(engine, num_rows);
  } else {
    auto const cardinality = [profile_cardinality = profile.get_cardinality(), num_rows] {
      return (profile_cardinality == 0 or profile_cardinality > num_rows) ? num_rows
                                                                          : profile_cardinality;
    }();
    rmm::device_uvector<bool> samples_null_mask = valid_dist(engine, cardinality);
    rmm::device_uvector<DeviceType> samples     = value_dist(engine, cardinality);

    // generate n samples and gather.
    auto const sample_indices =
      sample_indices_with_run_length(avg_run_len, cardinality, num_rows, engine);
    data      = rmm::device_uvector<DeviceType>(num_rows, cudf::get_default_stream());
    null_mask = rmm::device_uvector<bool>(num_rows, cudf::get_default_stream());
    thrust::gather(
      thrust::device, sample_indices.begin(), sample_indices.end(), samples.begin(), data.begin());
    thrust::gather(thrust::device,
                   sample_indices.begin(),
                   sample_indices.end(),
                   samples_null_mask.begin(),
                   null_mask.begin());
  }

  auto [result_bitmask, null_count] =
    profile.get_null_probability().has_value()
      ? cudf::bools_to_mask(cudf::device_span<bool const>(null_mask))
      : std::pair{std::make_unique<rmm::device_buffer>(), 0};

  return std::make_unique<cudf::column>(
    dtype, num_rows, data.release(), std::move(*result_bitmask.release()), null_count);
}

/**
 * @brief Creates a string column with random content.
 *
 * @param profile Parameters for the random generator
 * @param engine Pseudo-random engine
 * @param num_rows Size of the output column
 *
 * @return Column filled with random strings
 */
template <>
std::unique_ptr<cudf::column> create_random_column<cudf::string_view>(data_profile const& profile,
                                                                      cuda::std::philox4x32& engine,
                                                                      cudf::size_type num_rows)
{
  auto const cardinality = std::min(profile.get_cardinality(), num_rows);
  auto const avg_run_len = profile.get_avg_run_length();

  auto sample_strings =
    create_random_utf8_string_column(profile, engine, cardinality == 0 ? num_rows : cardinality);
  if (cardinality == 0) { return sample_strings; }
  auto sample_indices = sample_indices_with_run_length(avg_run_len, cardinality, num_rows, engine);
  auto gather_map =
    cudf::device_span<cudf::size_type const>(sample_indices.data(), sample_indices.size());
  auto str_table = cudf::gather(cudf::table_view{{sample_strings->view()}},
                                gather_map,
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                cudf::negative_index_policy::NOT_ALLOWED);
  return std::move(str_table->release()[0]);
}

template <>
std::unique_ptr<cudf::column> create_random_column<cudf::dictionary32>(
  data_profile const& profile, cuda::std::philox4x32& engine, cudf::size_type num_rows)
{
  CUDF_FAIL("not implemented yet");
}

template <>
std::unique_ptr<cudf::column> create_random_column<cudf::struct_view>(data_profile const& profile,
                                                                      cuda::std::philox4x32& engine,
                                                                      cudf::size_type num_rows)
{
  auto const dist_params = profile.get_distribution_params<cudf::struct_view>();

  // Generate leaf columns
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(dist_params.leaf_types.size());
  std::transform(dist_params.leaf_types.cbegin(),
                 dist_params.leaf_types.cend(),
                 std::back_inserter(children),
                 [&](auto& type_id) {
                   return cudf::type_dispatcher(
                     cudf::data_type(type_id), create_rand_col_fn{}, profile, engine, num_rows);
                 });

  auto valid_dist = random_value_fn<bool>(
    distribution_params<bool>{1. - profile.get_null_probability().value_or(0)});

  // Generate the column bottom-up
  for (int lvl = dist_params.max_depth; lvl > 0; --lvl) {
    // Generating the next level
    std::vector<std::unique_ptr<cudf::column>> parents;
    parents.resize(num_direct_parents(lvl, children.size()));

    auto current_child = children.begin();
    for (auto current_parent = parents.begin(); current_parent != parents.end(); ++current_parent) {
      auto [null_mask, null_count] = [&]() {
        if (profile.get_null_probability().has_value()) {
          auto valids = valid_dist(engine, num_rows);
          return cudf::bools_to_mask(cudf::device_span<bool const>(valids),
                                     cudf::get_default_stream());
        }
        return std::pair{std::make_unique<rmm::device_buffer>(), 0};
      }();

      // Adopt remaining children as evenly as possible
      auto const num_to_adopt = cudf::util::div_rounding_up_unsafe(
        std::distance(current_child, children.end()), std::distance(current_parent, parents.end()));
      CUDF_EXPECTS(num_to_adopt > 0, "No children columns left to adopt");

      std::vector<std::unique_ptr<cudf::column>> children_to_adopt;
      children_to_adopt.insert(children_to_adopt.end(),
                               std::make_move_iterator(current_child),
                               std::make_move_iterator(current_child + num_to_adopt));
      current_child += children_to_adopt.size();

      *current_parent = cudf::make_structs_column(
        num_rows, std::move(children_to_adopt), null_count, std::move(*null_mask.release()));
    }

    if (lvl == 1) {
      CUDF_EXPECTS(parents.size() == 1, "There should be one top-level column");
      return std::move(parents.front());
    }
    children = std::move(parents);
  }
  CUDF_FAIL("Reached unreachable code in struct column creation");
}

template <typename T>
struct clamp_down {
  T max;
  clamp_down(T max) : max(max) {}
  __host__ __device__ T operator()(T x) const { return min(x, max); }
};

/**
 * @brief Creates a list column with random content.
 *
 * The data profile determines the list length distribution, number of nested level, and the data
 * type of the bottom level.
 *
 * @param profile Parameters for the random generator
 * @param engine Pseudo-random engine
 * @param num_rows Size of the output column
 *
 * @return Column filled with random lists
 */
template <>
std::unique_ptr<cudf::column> create_random_column<cudf::list_view>(data_profile const& profile,
                                                                    cuda::std::philox4x32& engine,
                                                                    cudf::size_type num_rows)
{
  auto const dist_params       = profile.get_distribution_params<cudf::list_view>();
  auto const single_level_mean = get_distribution_mean(dist_params.length_params);
  auto const raw_num_elements =
    static_cast<double>(num_rows) * std::pow(single_level_mean, dist_params.max_depth);
  cudf::size_type const num_elements = static_cast<cudf::size_type>(
    std::min(raw_num_elements, static_cast<double>(std::numeric_limits<cudf::size_type>::max())));

  auto leaf_column = cudf::type_dispatcher(
    cudf::data_type(dist_params.element_type), create_rand_col_fn{}, profile, engine, num_elements);
  auto len_dist =
    random_value_fn<uint32_t>{profile.get_distribution_params<cudf::list_view>().length_params};
  auto valid_dist = random_value_fn<bool>(
    distribution_params<bool>{1. - profile.get_null_probability().value_or(0)});

  // Generate the list column bottom-up
  auto list_column = std::move(leaf_column);
  for (int lvl = dist_params.max_depth; lvl > 0; --lvl) {
    // Generating the next level - offsets point into the current list column
    auto current_child_column = std::move(list_column);
    // Because single_level_mean is not a whole number, rounding errors can lead to slightly
    // different row count; top-level column needs to have exactly num_rows rows, so enforce it here
    cudf::size_type const current_num_rows =
      (lvl == 1) ? num_rows : std::lround(current_child_column->size() / single_level_mean);

    auto offsets = len_dist(engine, current_num_rows + 1);
    auto valids  = valid_dist(engine, current_num_rows);
    // to ensure these values <= current_child_column->size()
    auto output_offsets = cuda::make_transform_output_iterator(
      offsets.begin(), clamp_down{current_child_column->size()});

    thrust::exclusive_scan(thrust::device, offsets.begin(), offsets.end(), output_offsets);
    thrust::device_pointer_cast(offsets.end())[-1] =
      current_child_column->size();  // Always include all elements

    auto offsets_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                         current_num_rows + 1,
                                                         offsets.release(),
                                                         rmm::device_buffer{},
                                                         0);

    auto [null_mask, null_count] = profile.get_null_probability().has_value()
                                     ? cudf::bools_to_mask(cudf::device_span<bool const>(valids))
                                     : std::pair{std::make_unique<rmm::device_buffer>(), 0};

    list_column = cudf::make_lists_column(current_num_rows,
                                          std::move(offsets_column),
                                          std::move(current_child_column),
                                          null_count,
                                          std::move(*null_mask.release()));

    if (auto const cv = list_column->view();
        cudf::has_nonempty_nulls(cv, cudf::get_default_stream())) {
      list_column = cudf::purge_nonempty_nulls(
        cv, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    }
  }
  return list_column;  // return the top-level column
}

// Numeric types: create a sequence of unique values, then shuffle
template <typename T>
  requires(cudf::is_numeric_not_bool<T>())
std::unique_ptr<cudf::column> create_distinct_rows_column(data_profile const& profile,
                                                          cuda::std::philox4x32& engine,
                                                          cudf::size_type num_rows)
{
  auto init = cudf::make_fixed_width_scalar(T{});
  auto col  = cudf::sequence(num_rows, *init);

  // Shuffle to randomize order while preserving uniqueness
  thrust::shuffle(thrust::device,
                  col->mutable_view().template begin<T>(),
                  col->mutable_view().template end<T>(),
                  cuda::std::philox4x32(engine()));

  if (profile.get_null_probability().has_value()) {
    auto valid_dist =
      random_value_fn<bool>(distribution_params<bool>{1. - profile.get_null_probability().value()});
    auto null_mask = valid_dist(engine, num_rows);
    auto [result_bitmask, null_count] =
      cudf::bools_to_mask(cudf::device_span<bool const>(null_mask), cudf::get_default_stream());
    col->set_null_mask(std::move(*result_bitmask.release()), null_count);
  }

  return col;
}

// catch-all for all types not handled specifically above
template <typename T>
  requires(!cudf::is_numeric_not_bool<T>())
std::unique_ptr<cudf::column> create_distinct_rows_column(data_profile const& profile,
                                                          cuda::std::philox4x32& engine,
                                                          cudf::size_type num_rows)
{
  return create_random_column<T>(profile, engine, num_rows);
}

template <>
std::unique_ptr<cudf::column> create_distinct_rows_column<cudf::string_view>(
  data_profile const& profile, cuda::std::philox4x32& engine, cudf::size_type num_rows)
{
  auto col        = create_random_column<cudf::string_view>(profile, engine, num_rows);
  auto int_col    = cudf::sequence(num_rows, *cudf::make_fixed_width_scalar<int32_t>(0));
  auto int2strcol = cudf::strings::from_integers(int_col->view());
  auto concat_col = cudf::strings::concatenate(cudf::table_view({col->view(), int2strcol->view()}));
  return std::move(cudf::sample(cudf::table_view({concat_col->view()}), num_rows)->release()[0]);
}

template <>
std::unique_ptr<cudf::column> create_distinct_rows_column<cudf::list_view>(
  data_profile const& profile, cuda::std::philox4x32& engine, cudf::size_type num_rows)
{
  auto const dist_params = profile.get_distribution_params<cudf::list_view>();
  auto col               = create_random_column<cudf::list_view>(profile, engine, num_rows);
  auto zero              = cudf::make_fixed_width_scalar<cudf::size_type>(0);
  auto child_column      = cudf::sequence(num_rows, *zero);
  for (int lvl = dist_params.max_depth; lvl > 0; --lvl) {
    auto offsets_column = cudf::sequence(num_rows + 1, *zero);
    auto list_column    = cudf::make_lists_column(
      num_rows, std::move(offsets_column), std::move(child_column), 0, rmm::device_buffer{});
    if (auto const cv = list_column->view();
        cudf::has_nonempty_nulls(cv, cudf::get_default_stream())) {
      list_column = cudf::purge_nonempty_nulls(
        cv, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    }
    std::swap(child_column, list_column);
  }
  auto lists_col =
    cudf::lists::concatenate_rows(cudf::table_view({col->view(), child_column->view()}));
  return std::move(cudf::sample(cudf::table_view({lists_col->view()}), num_rows)->release()[0]);
}

template <>
std::unique_ptr<cudf::column> create_distinct_rows_column<cudf::dictionary32>(
  data_profile const& profile, cuda::std::philox4x32& engine, cudf::size_type num_rows)
{
  CUDF_FAIL("not implemented yet");
}

template <>
std::unique_ptr<cudf::column> create_distinct_rows_column<cudf::struct_view>(
  data_profile const& profile, cuda::std::philox4x32& engine, cudf::size_type num_rows)
{
  auto const dist_params = profile.get_distribution_params<cudf::struct_view>();
  auto col               = create_random_column<cudf::struct_view>(profile, engine, num_rows);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(cudf::sequence(num_rows, *cudf::make_fixed_width_scalar<int32_t>(0)));
  for (int lvl = dist_params.max_depth; lvl > 1; --lvl) {
    std::vector<std::unique_ptr<cudf::column>> parents;
    parents.push_back(
      cudf::create_structs_hierarchy(num_rows, std::move(children), 0, rmm::device_buffer{}));
    std::swap(parents, children);
  }
  auto const null_count = col->null_count();
  auto col_contents     = col->release();
  col_contents.children.push_back(std::move(children[0]));
  auto structs_col = cudf::create_structs_hierarchy(
    num_rows, std::move(col_contents.children), null_count, std::move(*col_contents.null_mask));
  return std::move(cudf::sample(cudf::table_view({structs_col->view()}), num_rows)->release()[0]);
}

}  // namespace

/**
 * @brief Repeats the input data types cyclically order to fill a vector of @ref num_cols
 * elements.
 */
std::vector<cudf::type_id> cycle_dtypes(std::vector<cudf::type_id> const& dtype_ids,
                                        cudf::size_type num_cols)
{
  if (dtype_ids.size() == static_cast<std::size_t>(num_cols)) { return dtype_ids; }
  std::vector<cudf::type_id> out_dtypes;
  out_dtypes.reserve(num_cols);
  for (cudf::size_type col = 0; col < num_cols; ++col)
    out_dtypes.push_back(dtype_ids[col % dtype_ids.size()]);
  return out_dtypes;
}

/**
 * @brief Repeat the given two data types with a given ratio of a:b.
 *
 * The first dtype will have 'first_num' columns and the second will have 'num_cols - first_num'
 * columns.
 */
std::vector<cudf::type_id> mix_dtypes(std::pair<cudf::type_id, cudf::type_id> const& dtype_ids,
                                      cudf::size_type num_cols,
                                      int first_num)
{
  std::vector<cudf::type_id> out_dtypes;
  out_dtypes.reserve(num_cols);
  for (cudf::size_type col = 0; col < first_num; ++col)
    out_dtypes.push_back(dtype_ids.first);
  for (cudf::size_type col = first_num; col < num_cols; ++col)
    out_dtypes.push_back(dtype_ids.second);
  return out_dtypes;
}

std::unique_ptr<cudf::table> create_random_table(std::vector<cudf::type_id> const& dtype_ids,
                                                 table_size_bytes table_bytes,
                                                 data_profile const& profile,
                                                 unsigned seed)
{
  auto const avg_row_bytes =
    std::accumulate(dtype_ids.begin(), dtype_ids.end(), double{0}, [&](auto sum, auto tid) {
      return sum + avg_element_size(profile, cudf::data_type(tid));
    });
  std::size_t const num_rows = std::lround(table_bytes.size / avg_row_bytes);
  CUDF_EXPECTS(num_rows > 0, "Table size is too small for the given data types");
  CUDF_EXPECTS(num_rows < std::numeric_limits<cudf::size_type>::max(),
               "Table size is too large for the given data types");

  return create_random_table(
    dtype_ids, row_count{static_cast<cudf::size_type>(num_rows)}, profile, seed);
}

std::unique_ptr<cudf::table> create_random_table(std::vector<cudf::type_id> const& dtype_ids,
                                                 row_count num_rows,
                                                 data_profile const& profile,
                                                 unsigned seed)
{
  auto seed_engine = deterministic_engine(seed);
  cuda::std::uniform_int_distribution<unsigned> seed_dist;

  std::vector<std::unique_ptr<cudf::column>> output_columns;
  std::transform(
    dtype_ids.begin(), dtype_ids.end(), std::back_inserter(output_columns), [&](auto tid) mutable {
      return create_random_column(tid, num_rows, profile, seed_dist(seed_engine));
    });
  return std::make_unique<cudf::table>(std::move(output_columns));
}

std::unique_ptr<cudf::column> create_random_column(cudf::type_id dtype_id,
                                                   row_count num_rows,
                                                   data_profile const& profile,
                                                   unsigned seed)
{
  auto engine = deterministic_engine(seed);
  return cudf::type_dispatcher(
    cudf::data_type(dtype_id), create_rand_col_fn{}, profile, engine, num_rows.count);
}

std::unique_ptr<cudf::table> create_sequence_table(std::vector<cudf::type_id> const& dtype_ids,
                                                   row_count num_rows,
                                                   std::optional<double> null_probability,
                                                   unsigned seed)
{
  auto seed_engine = deterministic_engine(seed);
  cuda::std::uniform_int_distribution<unsigned> seed_dist;

  auto columns = std::vector<std::unique_ptr<cudf::column>>(dtype_ids.size());
  std::transform(dtype_ids.begin(), dtype_ids.end(), columns.begin(), [&](auto dtype) mutable {
    auto init = cudf::make_default_constructed_scalar(cudf::data_type{dtype});
    init->set_valid_async(true);
    auto col = cudf::sequence(num_rows.count, *init);
    auto [mask, count] =
      create_random_null_mask(num_rows.count, null_probability, seed_dist(seed_engine));
    col->set_null_mask(std::move(mask), count);
    return col;
  });
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::column> create_string_column(cudf::size_type num_rows,
                                                   cudf::size_type row_width,
                                                   int32_t hit_rate)
{
  // build input table using the following data
  auto raw_data = cudf::test::strings_column_wrapper(
                    {
                      "123 abc 4567890 DEFGHI 0987 5W43",  // matches both patterns;
                      "012345 6789 01234 56789 0123 456",  // the rest do not match
                      "abc 4567890 DEFGHI 0987 Wxyz 123",
                      "abcdefghijklmnopqrstuvwxyz 01234",
                      "",
                      "AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01",
                      "9876543210,abcdefghijklmnopqrstU",
                      "9876543210,abcdefghijklmnopqrstU",
                      "123 édf 4567890 DéFG 0987 X5",
                      "1",
                    })
                    .release();

  if (row_width / 32 > 1) {
    std::vector<cudf::column_view> columns;
    for (int i = 0; i < row_width / 32; ++i) {
      columns.push_back(raw_data->view());
    }
    raw_data = cudf::strings::concatenate(cudf::table_view(columns));
  }
  auto data_view = raw_data->view();

  // compute number of rows in n_rows that should match
  auto const num_matches = (static_cast<int64_t>(num_rows) * hit_rate) / 100;

  // Create a randomized gather-map to build a column out of the strings in data.
  data_profile gather_profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 1, data_view.size() - 1);
  auto gather_table =
    create_random_table({cudf::type_id::INT32}, row_count{num_rows}, gather_profile);

  if (num_matches > 0) {  // guard against division by zero
    // Create scatter map by placing 0-index values throughout the gather-map
    auto zero_scalar  = cudf::numeric_scalar<int32_t>(0);
    auto scatter_data = cudf::sequence(
      num_matches, zero_scalar, cudf::numeric_scalar<int32_t>(num_rows / num_matches));
    gather_table = cudf::scatter({zero_scalar}, scatter_data->view(), gather_table->view());
  }

  auto gather_map = gather_table->view().column(0);
  auto table      = cudf::gather(cudf::table_view({data_view}), gather_map);

  return std::move(table->release().front());
}

std::pair<rmm::device_buffer, cudf::size_type> create_random_null_mask(
  cudf::size_type size, std::optional<double> null_probability, unsigned seed)
{
  if (not null_probability.has_value()) { return {rmm::device_buffer{}, 0}; }
  CUDF_EXPECTS(*null_probability >= 0.0 and *null_probability <= 1.0,
               "Null probability must be within the range [0.0, 1.0]");
  if (*null_probability == 0.0f) {
    return {cudf::create_null_mask(size, cudf::mask_state::ALL_VALID), 0};
  } else if (*null_probability == 1.0) {
    return {cudf::create_null_mask(size, cudf::mask_state::ALL_NULL), size};
  } else {
    return cudf::detail::valid_if(cuda::counting_iterator<cudf::size_type>{0},
                                  cuda::counting_iterator<cudf::size_type>{size},
                                  bool_generator{seed, 1.0 - *null_probability},
                                  cudf::get_default_stream(),
                                  cudf::get_current_device_resource_ref());
  }
}

std::unique_ptr<cudf::column> create_ascii_string_column(data_profile const& profile,
                                                         cudf::size_type num_rows,
                                                         unsigned seed)
{
  auto engine = deterministic_engine(seed);
  return create_random_utf8_string_column<string_encoding::ASCII>(profile, engine, num_rows);
}

std::vector<cudf::type_id> get_type_or_group(int32_t id)
{
  // identity transformation when passing a concrete type_id
  if (id < static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS))
    return {static_cast<cudf::type_id>(id)};

  // if the value is larger that type_id::NUM_TYPE_IDS, it's a group id
  type_group_id const group_id = static_cast<type_group_id>(id);

  using trait_fn       = bool (*)(cudf::data_type);
  trait_fn is_integral = [](cudf::data_type type) {
    return cudf::is_numeric(type) && !cudf::is_floating_point(type);
  };
  trait_fn is_integral_signed = [](cudf::data_type type) {
    return cudf::is_numeric(type) && !cudf::is_floating_point(type) && !cudf::is_unsigned(type);
  };
  auto fn = [&]() -> trait_fn {
    switch (group_id) {
      case type_group_id::FLOATING_POINT: return cudf::is_floating_point;
      case type_group_id::INTEGRAL: return is_integral;
      case type_group_id::INTEGRAL_SIGNED: return is_integral_signed;
      case type_group_id::NUMERIC: return cudf::is_numeric;
      case type_group_id::TIMESTAMP: return cudf::is_timestamp;
      case type_group_id::DURATION: return cudf::is_duration;
      case type_group_id::FIXED_POINT: return cudf::is_fixed_point;
      case type_group_id::COMPOUND: return cudf::is_compound;
      case type_group_id::NESTED: return cudf::is_nested;
      default: CUDF_FAIL("Invalid data type group");
    }
  }();
  std::vector<cudf::type_id> types;
  for (int type_int = 0; type_int < static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS); ++type_int) {
    auto const type = static_cast<cudf::type_id>(type_int);
    if (type != cudf::type_id::EMPTY && fn(cudf::data_type(type))) types.push_back(type);
  }
  return types;
}

std::vector<cudf::type_id> get_type_or_group(std::vector<int32_t> const& ids)
{
  std::vector<cudf::type_id> all_type_ids;
  for (auto& id : ids) {
    auto const type_ids = get_type_or_group(id);
    all_type_ids.insert(std::end(all_type_ids), std::cbegin(type_ids), std::cend(type_ids));
  }
  return all_type_ids;
}

void data_profile::set_bool_probability_true(double p)
{
  CUDF_EXPECTS(p >= 0. and p <= 1., "probability must be in range [0...1]");
  bool_probability_true = p;
}

void data_profile::set_null_probability(std::optional<double> p)
{
  CUDF_EXPECTS(p.value_or(0.) >= 0. and p.value_or(0.) <= 1.,
               "probability must be in range [0...1]");
  null_probability = p;
}

void data_profile::set_list_depth(cudf::size_type max_depth)
{
  CUDF_EXPECTS(max_depth > 0, "List depth must be positive");
  list_dist_desc.max_depth = max_depth;
}

void data_profile::set_struct_depth(cudf::size_type max_depth)
{
  CUDF_EXPECTS(max_depth > 0, "Struct depth must be positive");
  struct_dist_desc.max_depth = max_depth;
}

void data_profile::set_struct_types(cudf::host_span<cudf::type_id const> types)
{
  CUDF_EXPECTS(
    std::none_of(
      types.begin(), types.end(), [](auto& type) { return type == cudf::type_id::STRUCT; }),
    "Cannot include STRUCT as its own subtype");
  struct_dist_desc.leaf_types.assign(types.begin(), types.end());
}

void data_profile::set_string_char_range(unsigned char lower, unsigned char upper)
{
  CUDF_EXPECTS(lower <= upper, "Lower bound must be <= upper bound");
  string_dist_desc.char_lower = lower;
  string_dist_desc.char_upper = upper;
}

template <typename T, std::enable_if_t<!std::is_same_v<T, bool> && cuda::std::is_integral_v<T>, T>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  auto it = int_params.find(cudf::type_to_id<T>());
  if (it == int_params.end()) {
    auto const range = default_range<T>();
    return distribution_params<T>{default_distribution_id<T>(), range.first, range.second};
  } else {
    auto& desc = it->second;
    return {desc.id, static_cast<T>(desc.lower_bound), static_cast<T>(desc.upper_bound)};
  }
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, T>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  auto it = float_params.find(cudf::type_to_id<T>());
  if (it == float_params.end()) {
    auto const range = default_range<T>();
    return distribution_params<T>{default_distribution_id<T>(), range.first, range.second};
  } else {
    auto& desc = it->second;
    return {desc.id, static_cast<T>(desc.lower_bound), static_cast<T>(desc.upper_bound)};
  }
}

template <typename T, std::enable_if_t<std::is_same_v<T, bool>>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  return distribution_params<T>{bool_probability_true};
}

template <typename T, std::enable_if_t<cudf::is_chrono<T>()>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  auto it = int_params.find(cudf::type_to_id<T>());
  if (it == int_params.end()) {
    auto const range = default_range<T>();
    return distribution_params<T>{default_distribution_id<T>(), range.first, range.second};
  } else {
    auto& desc = it->second;
    return {
      desc.id, static_cast<int64_t>(desc.lower_bound), static_cast<int64_t>(desc.upper_bound)};
  }
}

template <typename T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  return string_dist_desc;
}

template <typename T, std::enable_if_t<std::is_same_v<T, cudf::list_view>>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  return list_dist_desc;
}

template <typename T, std::enable_if_t<std::is_same_v<T, cudf::struct_view>>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  return struct_dist_desc;
}

template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  using rep = typename T::rep;
  auto it   = decimal_params.find(cudf::type_to_id<T>());
  if (it == decimal_params.end()) {
    auto const range = default_range<rep>();
    auto const scale = std::optional<numeric::scale_type>{};
    return distribution_params<T>{default_distribution_id<rep>(), range.first, range.second, scale};
  } else {
    auto& desc = it->second;
    return {
      desc.id, static_cast<rep>(desc.lower_bound), static_cast<rep>(desc.upper_bound), desc.scale};
  }
}
