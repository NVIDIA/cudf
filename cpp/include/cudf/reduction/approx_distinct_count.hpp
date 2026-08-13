/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <memory>

/**
 * @file
 * @brief Class for computing approximate distinct counts using a HyperLogLog sketch
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_reduction
 * @{
 */

// Forward declarations
namespace hashing::detail {
template <typename Key>
struct XXHash_64;
}

namespace detail {
template <template <typename> class Hasher>
class approx_distinct_count;
}

/**
 * @brief Object-oriented HyperLogLog sketch for approximate distinct counting.
 *
 * This class provides an object-oriented interface to HyperLogLog sketches, allowing
 * incremental addition of data and cardinality estimation.
 *
 * In the HLL literature, "sketch" refers to the register array that summarizes the input.
 * In this implementation, the sketch is stored as an `int32_t` device vector to enable
 * efficient device-side atomic operations.
 *
 * The implementation uses XXHash64 to hash table rows into 64-bit values, which are
 * then added to the HyperLogLog sketch without additional hashing (identity function).
 *
 * @par HyperLogLog Precision Parameter
 * The precision parameter (p) is the number of bits used to index into the register array.
 * It determines the number of registers (m = 2^p) in the HLL sketch:
 * - Memory usage: 2^p * 4 bytes (m registers of 4 bytes each for GPU atomics)
 * - Standard error: 1.04 / sqrt(m) = 1.04 / sqrt(2^p)
 *
 * Common precision values:
 * - p = 10: m = 1,024 registers, ~3.2% standard error, 4KB memory
 * - p = 12 (default): m = 4,096 registers, ~1.6% standard error, 16KB memory
 * - p = 14: m = 16,384 registers, ~0.8% standard error, 64KB memory
 * - p = 16: m = 65,536 registers, ~0.4% standard error, 256KB memory
 *
 * Valid range: p ∈ [4, 18]. This is not a hard theoretical limit but an empirically
 * recommended range:
 * - Below 4: Too few registers for HLL's statistical assumptions, resulting in high
 *   variance and unstable estimates.
 * - Above 18: Rapidly diminishing accuracy gains while incurring significant memory
 *   growth, making the structure no longer space-efficient for approximate counting.
 *
 * This range represents a practical engineering compromise from HLL++ and is widely
 * adopted by systems such as Apache Spark. The default of 12 aligns with Spark's
 * configuration and is the largest precision that fits efficiently in GPU shared memory,
 * enabling optimal performance for our implementation.
 *
 * Example usage:
 * @code{.cpp}
 *   auto adc = cudf::approx_distinct_count(table1);
 *   auto count1 = adc.estimate();
 *
 *   adc.add(table2);
 *   auto count2 = adc.estimate();
 * @endcode
 */
class approx_distinct_count {
 public:
  using impl_type =
    cudf::detail::approx_distinct_count<cudf::hashing::detail::XXHash_64>;  ///< Implementation type

  /**
   * @brief Strong type wrapper for the desired standard error constructor parameter
   *
   * Use this type to construct an `approx_distinct_count` with a desired error tolerance
   * instead of specifying precision directly.
   *
   * Example:
   * @code{.cpp}
   *   auto sketch = cudf::approx_distinct_count(
   *     table, cudf::approx_distinct_count::desired_standard_error{0.01});
   * @endcode
   */
  struct desired_standard_error {
    double value;  ///< The requested standard error value (must be positive)

    /**
     * @brief Constructs a desired_standard_error with the given value
     * @param v The requested standard error value (must be positive, e.g., 0.01 for ~1% error)
     */
    explicit constexpr desired_standard_error(double v) : value{v} {}
  };

  /**
   * @brief Constructs an approximate distinct count sketch from a table with specified precision
   *
   * @param input Table whose rows will be added to the sketch
   * @param precision The precision parameter for HyperLogLog (4-18). Higher precision gives
   *                  better accuracy but uses more memory. Default is 12.
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls (default: `EXCLUDE`)
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL` (default: `NAN_IS_NULL`)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the sketch storage
   */
  approx_distinct_count(table_view const& input,
                        std::int32_t precision       = 12,
                        null_policy null_handling    = null_policy::EXCLUDE,
                        nan_policy nan_handling      = nan_policy::NAN_IS_NULL,
                        rmm::cuda_stream_view stream = cudf::get_default_stream(),
                        cuda::mr::any_resource<cuda::mr::device_accessible> mr =
                          cudf::get_current_device_resource_ref());

  /**
   * @brief Constructs an approximate distinct count sketch from a table with specified standard
   * error
   *
   * This constructor allows specifying the desired standard error (error tolerance) directly,
   * which is more intuitive than specifying the precision parameter. The precision is calculated
   * as: `ceil(2 * log2(1.04 / standard_error))`.
   *
   * Since precision must be an integer, the actual standard error may be better (smaller)
   * than requested. Use the `standard_error()` getter to retrieve the actual value.
   *
   * @param input Table whose rows will be added to the sketch
   * @param error The desired standard error
   *              (e.g., `approx_distinct_count::desired_standard_error{0.01}` for ~1%)
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls (default: `EXCLUDE`)
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL` (default: `NAN_IS_NULL`)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the sketch storage
   *
   * @throws std::invalid_argument if standard_error value is not positive
   */
  approx_distinct_count(table_view const& input,
                        desired_standard_error error,
                        null_policy null_handling    = null_policy::EXCLUDE,
                        nan_policy nan_handling      = nan_policy::NAN_IS_NULL,
                        rmm::cuda_stream_view stream = cudf::get_default_stream(),
                        cuda::mr::any_resource<cuda::mr::device_accessible> mr =
                          cudf::get_current_device_resource_ref());

  /**
   * @brief Constructs a non-owning sketch that operates on user-allocated storage
   *
   * This constructor creates a sketch that operates directly on the provided storage
   * without copying. This enables zero-copy operations on pre-existing buffers, such as
   * sketch data stored in a column or received from another process.
   *
   * @warning The caller must ensure the storage remains valid for the lifetime of this
   * object. The sketch will read from and write to the provided storage directly.
   *
   * @param sketch_span The sketch bytes to operate on (must remain valid)
   * @param precision The precision parameter for the sketch (4-18)
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls (default: `EXCLUDE`)
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL` (default: `NAN_IS_NULL`)
   */
  approx_distinct_count(cuda::std::span<cuda::std::byte> sketch_span,
                        std::int32_t precision,
                        null_policy null_handling = null_policy::EXCLUDE,
                        nan_policy nan_handling   = nan_policy::NAN_IS_NULL);

  ~approx_distinct_count();

  approx_distinct_count(approx_distinct_count const&)            = delete;
  approx_distinct_count& operator=(approx_distinct_count const&) = delete;
  approx_distinct_count(approx_distinct_count&&) = default;  ///< Default move constructor
  /**
   * @brief Move assignment operator
   *
   * @return A reference to this object
   */
  approx_distinct_count& operator=(approx_distinct_count&&) = default;

  /**
   * @brief Adds rows from a table to the sketch
   *
   * @param input Table whose rows will be added
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void add(table_view const& input, rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Merges another sketch into this sketch
   *
   * After merging, this sketch will contain the combined distinct count estimate of both sketches.
   *
   * @throw std::invalid_argument if the sketches have different precision values
   * @throw std::invalid_argument if the sketches have different null handling policies
   * @throw std::invalid_argument if the sketches have different NaN handling policies
   *
   * @param other The sketch to merge into this sketch
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void merge(approx_distinct_count const& other,
             rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Merges a sketch from raw bytes into this sketch
   *
   * This allows merging sketches that have been serialized or created elsewhere, enabling
   * distributed distinct counting scenarios.
   *
   * @warning It is the caller's responsibility to ensure that the provided sketch span was created
   * with the same approx_distinct_count configuration (precision, null/NaN handling, etc.) as this
   * sketch. Merging incompatible sketches will produce incorrect results.
   *
   * @param sketch_span The sketch bytes to merge into this sketch
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void merge(cuda::std::span<cuda::std::byte const> sketch_span,
             rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Estimates the approximate number of distinct rows in the sketch
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Approximate number of distinct rows
   */
  [[nodiscard]] std::size_t estimate(
    rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

  /**
   * @brief Gets the raw sketch bytes for serialization or external merging
   *
   * The returned span provides access to the internal sketch storage.
   * This can be used to serialize the sketch, transfer it between processes,
   * or merge it with other sketches using the span-based merge API.
   *
   * @return A span view of the sketch bytes
   */
  [[nodiscard]] cuda::std::span<cuda::std::byte> sketch() noexcept;

  /**
   * @brief Gets the raw sketch bytes for serialization or external merging (const overload)
   *
   * The returned span provides access to the internal sketch storage.
   * This can be used to serialize the sketch, transfer it between processes,
   * or merge it with other sketches using the span-based merge API.
   *
   * @return A span view of the sketch bytes
   */
  [[nodiscard]] cuda::std::span<cuda::std::byte const> sketch() const noexcept;

  /**
   * @brief Gets the null handling policy for this sketch
   *
   * @return The null policy set at construction
   */
  [[nodiscard]] null_policy null_handling() const noexcept;

  /**
   * @brief Gets the NaN handling policy for this sketch
   *
   * @return The NaN policy set at construction
   */
  [[nodiscard]] nan_policy nan_handling() const noexcept;

  /**
   * @brief Gets the precision parameter for this sketch
   *
   * @return The precision value set at construction
   */
  [[nodiscard]] std::int32_t precision() const noexcept;

  /**
   * @brief Gets the standard error (error tolerance) for this sketch
   *
   * The standard error is calculated from precision as: `1.04 / sqrt(2^precision)`.
   * This represents the expected relative error of the cardinality estimate.
   *
   * @return The actual standard error based on the sketch's precision
   */
  [[nodiscard]] double standard_error() const noexcept;

  /**
   * @brief Gets the number of bytes required for sketch storage at a given precision
   *
   * @param precision The HLL precision parameter (4-18)
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] static std::size_t sketch_bytes(std::int32_t precision);

  /**
   * @brief Gets the alignment required for sketch storage
   *
   * @return The required alignment in bytes
   */
  [[nodiscard]] static std::size_t sketch_alignment();

 private:
  std::unique_ptr<impl_type> _impl;
};

/** @} */

}  // namespace CUDF_EXPORT cudf
