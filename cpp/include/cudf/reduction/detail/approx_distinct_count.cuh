/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/hyperloglog_ref.cuh>
#include <cuda/functional>
#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <variant>

namespace cudf {
namespace detail {

/**
 * @brief HyperLogLog-based approximate distinct count sketch for use by the public API
 *
 * This detail implementation provides the core HyperLogLog functionality used by the
 * public `cudf::approx_distinct_count` class. In the HLL literature, "sketch" refers to
 * the register array that summarizes the input. This class supports both owning and
 * non-owning sketch storage modes:
 *
 * - **Owning mode**: Allocates and manages its own sketch storage as an
 *   `rmm::device_uvector<int32_t>`. Used when constructing from a table or when copying
 *   from a span.
 *
 * - **Non-owning mode**: Operates on user-provided `cuda::std::span<int32_t>` sketch
 *   storage. Enables zero-copy operations on pre-existing buffers.
 *
 * Internally, the sketch uses `int32_t` registers for efficient 32-bit GPU atomics. The
 * public API exposes the sketch storage as `cuda::std::byte` spans for generic
 * serialization.
 *
 * @tparam Hasher The hash function template to use for hashing table rows. Must be compatible
 *                with cudf's row_hasher device_hasher interface (a template taking a Key type).
 */
template <template <typename> class Hasher>
class approx_distinct_count {
 private:
  /**
   * @brief HLL reference type used for sketch operations
   *
   * Uses cuco::hyperloglog_ref with uint64_t hash values and identity hash function
   * (since rows are pre-hashed by XXHash64).
   */
  using hll_ref_type =
    cuco::hyperloglog_ref<std::uint64_t, cuda::thread_scope_device, cuda::std::identity>;

  /**
   * @brief Register type for HLL sketch storage
   *
   * Uses cuco's register_type (int32_t) for efficient 32-bit GPU atomics.
   * Each register stores the maximum leading zero count for its bucket.
   */
  using register_type = typename hll_ref_type::register_type;

 public:
  /**
   * @brief Constructs an owning approximate distinct count sketch from a table with precision
   *
   * Allocates internal storage and adds all rows from the input table to the sketch.
   *
   * @param input Table whose rows will be added to the sketch
   * @param precision The precision parameter for HyperLogLog (4-18). Higher precision gives
   *                  better accuracy but uses more memory (2^precision * 4 bytes).
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL`
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the sketch storage
   */
  approx_distinct_count(table_view const& input,
                        std::int32_t precision,
                        null_policy null_handling,
                        nan_policy nan_handling,
                        rmm::cuda_stream_view stream,
                        cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  /**
   * @brief Constructs an owning approximate distinct count sketch from a table with standard
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
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL`
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the sketch storage
   */
  approx_distinct_count(table_view const& input,
                        cudf::approx_distinct_count::desired_standard_error error,
                        null_policy null_handling,
                        nan_policy nan_handling,
                        rmm::cuda_stream_view stream,
                        cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  /**
   * @brief Constructs a non-owning approximate distinct count sketch from user-allocated storage
   *
   * The sketch operates directly on the provided storage without copying. This enables
   * zero-copy operations on pre-existing buffers, such as sketch data stored in a column
   * or received from another process.
   *
   * @warning The caller must ensure the storage remains valid for the lifetime of this
   * object. The sketch will read from and write to the provided storage directly.
   *
   * @param sketch_span The sketch bytes to operate on (must remain valid, size = 2^precision * 4)
   * @param precision The precision parameter for the sketch (4-18)
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL`
   * @param mr Device memory resource (unused in non-owning mode, retained for member consistency)
   */
  approx_distinct_count(cuda::std::span<cuda::std::byte> sketch_span,
                        std::int32_t precision,
                        null_policy null_handling,
                        nan_policy nan_handling,
                        cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  approx_distinct_count()                                        = delete;
  ~approx_distinct_count()                                       = default;
  approx_distinct_count(approx_distinct_count const&)            = delete;
  approx_distinct_count& operator=(approx_distinct_count const&) = delete;
  approx_distinct_count(approx_distinct_count&&) noexcept        = default;
  approx_distinct_count& operator=(approx_distinct_count&&)      = default;

  /**
   * @brief Adds rows from a table to the sketch
   *
   * Each row in the input table is hashed and added to the HyperLogLog sketch.
   * Rows containing nulls are handled according to the null_handling policy
   * specified at construction. NaN values are handled according to the
   * nan_handling policy.
   *
   * @param input Table whose rows will be added
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void add(table_view const& input, rmm::cuda_stream_view stream);

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
  void merge(approx_distinct_count const& other, rmm::cuda_stream_view stream);

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
  void merge(cuda::std::span<cuda::std::byte const> sketch_span, rmm::cuda_stream_view stream);

  /**
   * @brief Estimates the approximate number of distinct rows in the sketch
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Approximate number of distinct rows
   */
  [[nodiscard]] std::size_t estimate(rmm::cuda_stream_view stream) const;

  /**
   * @brief Gets the raw sketch bytes
   *
   * @return A span view of the sketch bytes
   */
  [[nodiscard]] cuda::std::span<cuda::std::byte> sketch() noexcept;

  /**
   * @brief Gets the raw sketch bytes (const overload)
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
   * @return The precision value
   */
  [[nodiscard]] std::int32_t precision() const noexcept;

  /**
   * @brief Gets the standard error (error tolerance) for this sketch
   *
   * @return The actual standard error based on the sketch's precision
   */
  [[nodiscard]] double standard_error() const noexcept;

  /**
   * @brief Returns the number of bytes required to store a sketch at the given precision
   *
   * @param precision HyperLogLog precision parameter
   * @return Required sketch storage size in bytes
   */
  [[nodiscard]] static constexpr std::size_t sketch_bytes(std::int32_t precision)
  {
    return hll_ref_type::sketch_bytes(cuco::precision{precision});
  }

  /**
   * @brief Returns the required alignment for sketch storage
   *
   * @return Required sketch storage alignment in bytes
   */
  [[nodiscard]] static constexpr std::size_t sketch_alignment()
  {
    return hll_ref_type::sketch_alignment();
  }

 private:
  /**
   * @brief Storage type supporting both owning and non-owning modes
   *
   * - Owning: `device_uvector<register_type>` - allocates and manages storage
   * - Non-owning: `span<byte>` - operates on user-provided storage
   *
   * The public API always exposes storage as `span<byte>` via `sketch()`.
   */
  using storage_type =
    std::variant<rmm::device_uvector<register_type>,  ///< Owning storage (allocated internally)
                 cuda::std::span<cuda::std::byte>     ///< Non-owning storage (user-provided)
                 >;

  // Declared before `_storage` so it outlives the owning `device_uvector`, which only holds a
  // reference to this resource. Unused in non-owning (span) mode.
  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;  ///< Owns the sketch storage resource
  storage_type _storage;       ///< Sketch register storage (owning or non-owning)
  std::int32_t _precision;     ///< HLL precision parameter (determines 2^p registers)
  null_policy _null_handling;  ///< Null handling policy (immutable after construction)
  nan_policy _nan_handling;    ///< NaN handling policy (immutable after construction)
};

}  // namespace detail
}  // namespace cudf
