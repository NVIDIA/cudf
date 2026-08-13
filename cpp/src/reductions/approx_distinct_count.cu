/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/reduction/detail/approx_distinct_count.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <cmath>
#include <utility>

namespace cudf {
namespace detail {

namespace {

constexpr double hll_constant =
  1.04;  // ≈ β∞ = √(3ln2 − 1), Flajolet et al., "HyperLogLog: the analysis of a near-optimal
         // cardinality estimation algorithm"
constexpr std::int32_t min_precision = 4;
constexpr std::int32_t max_precision = 18;

/**
 * @brief Converts standard error to HLL precision
 *
 * Formula: precision = ceil(2 * log2(hll_constant / standard_error))
 *
 * @param standard_error The desired standard error
 * @return The calculated precision, clamped to valid range [4, 18]
 */
std::int32_t precision_from_standard_error(double standard_error)
{
  CUDF_EXPECTS(standard_error > 0, "Standard error must be positive", std::invalid_argument);

  auto const ratio     = hll_constant / standard_error;
  auto const precision = static_cast<std::int32_t>(std::ceil(2.0 * std::log2(ratio)));

  return std::clamp(precision, min_precision, max_precision);
}

/**
 * @brief Converts HLL precision to standard error
 *
 * Formula: standard_error = hll_constant / sqrt(2^precision)
 *
 * @param precision The HLL precision parameter
 * @return The standard error for the given precision
 */
constexpr double standard_error_from_precision(std::int32_t precision)
{
  return hll_constant / std::sqrt(static_cast<double>(1 << precision));
}

[[nodiscard]] std::int32_t check_precision(std::int32_t precision)
{
  CUDF_EXPECTS(precision >= min_precision && precision <= max_precision,
               "Precision must be in range [4, 18]",
               std::invalid_argument);
  return precision;
}

template <typename SpanT>
[[nodiscard]] SpanT check_sketch_span(SpanT sketch_span, std::int32_t precision)
{
  auto const expected_size =
    approx_distinct_count<cudf::hashing::detail::XXHash_64>::sketch_bytes(precision);
  CUDF_EXPECTS(sketch_span.size() == expected_size,
               "Sketch span size does not match expected size for precision",
               std::invalid_argument);
  CUDF_EXPECTS(reinterpret_cast<std::uintptr_t>(sketch_span.data()) %
                   approx_distinct_count<cudf::hashing::detail::XXHash_64>::sketch_alignment() ==
                 0,
               "Sketch span must be 4-byte aligned",
               std::invalid_argument);
  return sketch_span;
}

/**
 * @brief Device functor to check if a row is valid using a bitmask
 */
struct row_is_valid {
  bitmask_type const* row_bitmask;

  __device__ bool operator()(cudf::size_type row_idx) const noexcept
  {
    return cudf::bit_is_set(row_bitmask, row_idx);
  }
};

/**
 * @brief Device functor that returns NULL_HASH for rows containing null or NaN
 */
template <typename Hasher>
struct nan_to_null_hasher {
  using result_type = cuda::std::invoke_result_t<Hasher, cudf::size_type>;

  Hasher base_hasher;
  table_device_view d_table;

  __device__ result_type operator()(cudf::size_type row_idx) const noexcept
  {
    constexpr auto null_hash = cuda::std::numeric_limits<result_type>::max();

    for (cudf::size_type col_idx = 0; col_idx < d_table.num_columns(); ++col_idx) {
      auto const& col = d_table.column(col_idx);

      if (col.nullable() && col.is_null_nocheck(row_idx)) { return null_hash; }

      auto const type = col.type();
      if (type.id() == type_id::FLOAT32) {
        if (cuda::std::isnan(col.element<float>(row_idx))) { return null_hash; }
      } else if (type.id() == type_id::FLOAT64) {
        if (cuda::std::isnan(col.element<double>(row_idx))) { return null_hash; }
      }
    }

    return base_hasher(row_idx);
  }
};

/**
 * @brief Predicate to check if a row has any NaN values (for use with bitmask filtering)
 */
struct check_nans_predicate {
  table_device_view d_table;
  bitmask_type const* row_bitmask;

  __device__ bool operator()(cudf::size_type row_idx) const noexcept
  {
    if (row_bitmask != nullptr && !cudf::bit_is_set(row_bitmask, row_idx)) { return false; }

    for (cudf::size_type col_idx = 0; col_idx < d_table.num_columns(); ++col_idx) {
      auto const& col = d_table.column(col_idx);
      auto const type = col.type();
      if (type.id() == type_id::FLOAT32) {
        if (cuda::std::isnan(col.element<float>(row_idx))) { return false; }
      } else if (type.id() == type_id::FLOAT64) {
        if (cuda::std::isnan(col.element<double>(row_idx))) { return false; }
      }
    }
    return true;
  }
};

}  // namespace

template <template <typename> class Hasher>
approx_distinct_count<Hasher>::approx_distinct_count(
  table_view const& input,
  std::int32_t precision,
  null_policy null_handling,
  nan_policy nan_handling,
  rmm::cuda_stream_view stream,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _mr{std::move(mr)},
    _storage{rmm::device_uvector<register_type>{
      sketch_bytes(check_precision(precision)) / sizeof(register_type), stream, _mr}},
    _precision{precision},
    _null_handling{null_handling},
    _nan_handling{nan_handling}
{
  auto sketch_span = sketch();
  CUDF_CUDA_TRY(cudaMemsetAsync(sketch_span.data(), 0, sketch_span.size(), stream.value()));

  if (input.num_rows() > 0) { add(input, stream); }
}

template <template <typename> class Hasher>
approx_distinct_count<Hasher>::approx_distinct_count(
  table_view const& input,
  cudf::approx_distinct_count::desired_standard_error error,
  null_policy null_handling,
  nan_policy nan_handling,
  rmm::cuda_stream_view stream,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : approx_distinct_count{input,
                          precision_from_standard_error(error.value),
                          null_handling,
                          nan_handling,
                          stream,
                          std::move(mr)}
{
}

template <template <typename> class Hasher>
approx_distinct_count<Hasher>::approx_distinct_count(
  cuda::std::span<cuda::std::byte> sketch_span,
  std::int32_t precision,
  null_policy null_handling,
  nan_policy nan_handling,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _mr{std::move(mr)},
    _storage{check_sketch_span(sketch_span, check_precision(precision))},
    _precision{precision},
    _null_handling{null_handling},
    _nan_handling{nan_handling}
{
}

template <template <typename> class Hasher>
void approx_distinct_count<Hasher>::add(table_view const& input, rmm::cuda_stream_view stream)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) { return; }

  typename approx_distinct_count<Hasher>::hll_ref_type ref{sketch(), cuda::std::identity{}};

  auto const has_nulls = nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const preprocessed_input =
    cudf::detail::row::hash::preprocessed_table::create(input, stream);
  auto const row_hasher = cudf::detail::row::hash::row_hasher(preprocessed_input);
  auto const hash_key   = row_hasher.device_hasher<Hasher>(has_nulls);

  if (_null_handling == null_policy::INCLUDE) {
    if (_nan_handling == nan_policy::NAN_IS_NULL) {
      auto const d_table    = table_device_view::create(input, stream);
      auto const nan_hasher = nan_to_null_hasher{hash_key, *d_table};
      auto const hash_iter  = cudf::detail::make_counting_transform_iterator(0, nan_hasher);
      ref.add_async(hash_iter, hash_iter + num_rows, stream);
    } else {
      auto const hash_iter = cudf::detail::make_counting_transform_iterator(0, hash_key);
      ref.add_async(hash_iter, hash_iter + num_rows, stream);
    }
  } else {
    auto const hash_iter = cudf::detail::make_counting_transform_iterator(0, hash_key);
    auto const stencil   = cuda::counting_iterator<cudf::size_type>{0};

    if (_nan_handling == nan_policy::NAN_IS_VALID) {
      if (!has_nulls) {
        ref.add_async(hash_iter, hash_iter + num_rows, stream);
      } else {
        auto const row_bitmask =
          cudf::detail::bitmask_and(input, stream, cudf::get_current_device_resource_ref()).first;
        auto const pred = row_is_valid{static_cast<bitmask_type const*>(row_bitmask.data())};
        ref.add_if_async(hash_iter, hash_iter + num_rows, stencil, pred, stream);
      }
    } else {
      auto const d_table = table_device_view::create(input, stream);
      if (!has_nulls) {
        auto const pred = check_nans_predicate{*d_table, nullptr};
        ref.add_if_async(hash_iter, hash_iter + num_rows, stencil, pred, stream);
      } else {
        auto const row_bitmask =
          cudf::detail::bitmask_and(input, stream, cudf::get_current_device_resource_ref()).first;
        auto const bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
        auto const pred        = check_nans_predicate{*d_table, bitmask_ptr};
        ref.add_if_async(hash_iter, hash_iter + num_rows, stencil, pred, stream);
      }
    }
  }
}

template <template <typename> class Hasher>
void approx_distinct_count<Hasher>::merge(approx_distinct_count const& other,
                                          rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(_precision == other._precision,
               "Cannot merge sketches with different precisions",
               std::invalid_argument);
  CUDF_EXPECTS(_null_handling == other._null_handling,
               "Cannot merge sketches with different null handling policies",
               std::invalid_argument);
  CUDF_EXPECTS(_nan_handling == other._nan_handling,
               "Cannot merge sketches with different NaN handling policies",
               std::invalid_argument);

  typename approx_distinct_count<Hasher>::hll_ref_type ref{sketch(), cuda::std::identity{}};
  typename approx_distinct_count<Hasher>::hll_ref_type other_ref{
    const_cast<approx_distinct_count&>(other).sketch(), cuda::std::identity{}};
  ref.merge_async(other_ref, stream);
}

template <template <typename> class Hasher>
void approx_distinct_count<Hasher>::merge(cuda::std::span<cuda::std::byte const> sketch_span,
                                          rmm::cuda_stream_view stream)
{
  auto const checked = check_sketch_span(sketch_span, _precision);

  typename approx_distinct_count<Hasher>::hll_ref_type ref{sketch(), cuda::std::identity{}};
  typename approx_distinct_count<Hasher>::hll_ref_type other_ref{
    cuda::std::span<cuda::std::byte>{const_cast<cuda::std::byte*>(checked.data()), checked.size()},
    cuda::std::identity{}};
  ref.merge_async(other_ref, stream);
}

template <template <typename> class Hasher>
std::size_t approx_distinct_count<Hasher>::estimate(rmm::cuda_stream_view stream) const
{
  typename approx_distinct_count<Hasher>::hll_ref_type ref{
    const_cast<approx_distinct_count*>(this)->sketch(), cuda::std::identity{}};
  return ref.estimate(stream);
}

template <template <typename> class Hasher>
cuda::std::span<cuda::std::byte> approx_distinct_count<Hasher>::sketch() noexcept
{
  return std::visit(
    [](auto& storage) -> cuda::std::span<cuda::std::byte> {
      using T = std::decay_t<decltype(storage)>;
      if constexpr (std::is_same_v<T, rmm::device_uvector<register_type>>) {
        return {reinterpret_cast<cuda::std::byte*>(storage.data()),
                storage.size() * sizeof(register_type)};
      } else {
        return storage;  // already a byte span
      }
    },
    _storage);
}

template <template <typename> class Hasher>
cuda::std::span<cuda::std::byte const> approx_distinct_count<Hasher>::sketch() const noexcept
{
  return std::visit(
    [](auto const& storage) -> cuda::std::span<cuda::std::byte const> {
      using T = std::decay_t<decltype(storage)>;
      if constexpr (std::is_same_v<T, rmm::device_uvector<register_type>>) {
        return {reinterpret_cast<cuda::std::byte const*>(storage.data()),
                storage.size() * sizeof(register_type)};
      } else {
        return {storage.data(), storage.size()};  // convert span<byte> to span<byte const>
      }
    },
    _storage);
}

template <template <typename> class Hasher>
null_policy approx_distinct_count<Hasher>::null_handling() const noexcept
{
  return _null_handling;
}

template <template <typename> class Hasher>
nan_policy approx_distinct_count<Hasher>::nan_handling() const noexcept
{
  return _nan_handling;
}

template <template <typename> class Hasher>
std::int32_t approx_distinct_count<Hasher>::precision() const noexcept
{
  return _precision;
}

template <template <typename> class Hasher>
double approx_distinct_count<Hasher>::standard_error() const noexcept
{
  return standard_error_from_precision(_precision);
}

// Explicit instantiation for the default hasher
template class approx_distinct_count<cudf::hashing::detail::XXHash_64>;

}  // namespace detail

approx_distinct_count::~approx_distinct_count() = default;

approx_distinct_count::approx_distinct_count(table_view const& input,
                                             std::int32_t precision,
                                             null_policy null_handling,
                                             nan_policy nan_handling,
                                             rmm::cuda_stream_view stream,
                                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl(std::make_unique<impl_type>(
      input, precision, null_handling, nan_handling, stream, std::move(mr)))
{
}

approx_distinct_count::approx_distinct_count(table_view const& input,
                                             desired_standard_error error,
                                             null_policy null_handling,
                                             nan_policy nan_handling,
                                             rmm::cuda_stream_view stream,
                                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl(
      std::make_unique<impl_type>(input, error, null_handling, nan_handling, stream, std::move(mr)))
{
}

approx_distinct_count::approx_distinct_count(cuda::std::span<cuda::std::byte> sketch_span,
                                             std::int32_t precision,
                                             null_policy null_handling,
                                             nan_policy nan_handling)
  : _impl(std::make_unique<impl_type>(
      sketch_span, precision, null_handling, nan_handling, cudf::get_current_device_resource_ref()))
{
}

void approx_distinct_count::add(table_view const& input, rmm::cuda_stream_view stream)
{
  _impl->add(input, stream);
}

void approx_distinct_count::merge(approx_distinct_count const& other, rmm::cuda_stream_view stream)
{
  _impl->merge(*other._impl, stream);
}

void approx_distinct_count::merge(cuda::std::span<cuda::std::byte const> sketch_span,
                                  rmm::cuda_stream_view stream)
{
  _impl->merge(sketch_span, stream);
}

std::size_t approx_distinct_count::estimate(rmm::cuda_stream_view stream) const
{
  return _impl->estimate(stream);
}

cuda::std::span<cuda::std::byte> approx_distinct_count::sketch() noexcept
{
  return _impl->sketch();
}

cuda::std::span<cuda::std::byte const> approx_distinct_count::sketch() const noexcept
{
  return _impl->sketch();
}

null_policy approx_distinct_count::null_handling() const noexcept { return _impl->null_handling(); }

nan_policy approx_distinct_count::nan_handling() const noexcept { return _impl->nan_handling(); }

std::int32_t approx_distinct_count::precision() const noexcept { return _impl->precision(); }

double approx_distinct_count::standard_error() const noexcept { return _impl->standard_error(); }

std::size_t approx_distinct_count::sketch_bytes(std::int32_t precision)
{
  return impl_type::sketch_bytes(precision);
}

std::size_t approx_distinct_count::sketch_alignment() { return impl_type::sketch_alignment(); }

}  // namespace cudf
