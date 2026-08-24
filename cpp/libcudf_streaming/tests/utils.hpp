/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief RAII temporary directory created under GTest's temp directory.
 *
 * The directory is created on construction and recursively removed on
 * destruction. Removal errors are ignored.
 */
class TempDir {
 public:
  TempDir() : path_(unique_path())
  {
    std::error_code ec;
    if (!std::filesystem::create_directories(path_, ec) || ec) {
      throw std::runtime_error("Failed to create temp directory: " + path_.string());
    }
  }

  ~TempDir() noexcept
  {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
    // Intentionally ignore errors in destructor.
  }

  TempDir(TempDir const&)            = delete;
  TempDir& operator=(TempDir const&) = delete;
  TempDir(TempDir&&)                 = delete;
  TempDir& operator=(TempDir&&)      = delete;

  /// @brief Returns the path to the temporary directory.
  [[nodiscard]] std::filesystem::path const& path() const noexcept { return path_; }

 private:
  static std::filesystem::path unique_path()
  {
    static std::atomic<std::uint64_t> counter{0};
    return std::filesystem::path(testing::TempDir()) /
           ("tmp_" + std::to_string(::getpid()) + "_" +
            std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
  }

  std::filesystem::path path_;
};

/// @brief User-defined literal for specifying memory sizes in KiB.
constexpr std::size_t operator"" _KiB(unsigned long long val) { return val * (1 << 10); }

/// @brief User-defined literal for specifying memory sizes in MiB.
constexpr std::size_t operator"" _MiB(unsigned long long val) { return val * (1ull << 20); }

/// @brief User-defined literal for specifying memory sizes in GiB.
constexpr std::size_t operator"" _GiB(unsigned long long val) { return val * (1 << 30); }

template <typename T>
[[nodiscard]] std::vector<T> iota_vector(std::size_t nelem, T start = 0)
{
  std::vector<T> ret(nelem);
  std::iota(ret.begin(), ret.end(), start);
  return ret;
}

template <typename T>
[[nodiscard]] inline std::unique_ptr<cudf::column> iota_column(std::size_t nrows, T start = 0)
{
  std::vector<T> vec = iota_vector(nrows, start);
  cudf::test::fixed_width_column_wrapper<T> ret(vec.begin(), vec.end());
  return ret.release();
}

template <std::integral T = std::int64_t>
[[nodiscard]] inline std::vector<T> random_vector(std::int64_t seed,
                                                  std::size_t nelem,
                                                  T min = std::numeric_limits<T>::min(),
                                                  T max = std::numeric_limits<T>::max())
{
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::uniform_int_distribution<T> dist(min, max);
  std::vector<T> ret(nelem);
  std::generate(ret.begin(), ret.end(), [&]() { return dist(rng); });
  return ret;
}

[[nodiscard]] inline std::unique_ptr<cudf::column> random_column(
  std::int64_t seed,
  std::size_t nrows,
  std::int64_t min = std::numeric_limits<std::int64_t>::min(),
  std::int64_t max = std::numeric_limits<std::int64_t>::max())
{
  std::vector<std::int64_t> vec = random_vector(seed, nrows, min, max);
  cudf::test::fixed_width_column_wrapper<std::int64_t> ret(vec.begin(), vec.end());
  return ret.release();
}

[[nodiscard]] inline cudf::table random_table_with_index(
  std::int64_t seed,
  std::size_t nrows,
  std::int64_t min = std::numeric_limits<std::int64_t>::min(),
  std::int64_t max = std::numeric_limits<std::int64_t>::max())
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(iota_column<std::int64_t>(nrows));
  cols.push_back(random_column(seed, nrows, min, max));
  return cudf::table(std::move(cols));
}

[[nodiscard]] inline cudf::table sort_table(
  cudf::table_view const& table, std::vector<cudf::size_type> const& /* column_indices */ = {0})
{
  if (table.num_columns() == 0) { return cudf::table(table); }
  return cudf::gather(table, cudf::sorted_order(table.select({0}))->view())->release();
}

[[nodiscard]] inline cudf::table sort_table(std::unique_ptr<cudf::table> const& table,
                                            std::vector<cudf::size_type> const& column_indices = {
                                              0})
{
  return sort_table(table->view(), column_indices);
}

/**
 * @brief Generate a packed data object with the given number of elements and offset.
 *
 * Both metadata and GPU data contain the same integer sequence.
 *
 * @param n_elements Number of elements in the sequence.
 * @param offset Starting value of the sequence.
 * @param stream CUDA stream for device allocation.
 * @param br Buffer resource used for allocations.
 * @return A packed data object containing metadata and GPU data.
 */
[[nodiscard]] inline rapidsmpf::PackedData generate_packed_data(int n_elements,
                                                                int offset,
                                                                rmm::cuda_stream_view stream,
                                                                rapidsmpf::BufferResource& br)
{
  auto values = iota_vector<int>(n_elements, offset);

  auto metadata = std::make_unique<std::vector<std::uint8_t>>(n_elements * sizeof(int));
  std::memcpy(metadata->data(), values.data(), n_elements * sizeof(int));

  auto data = std::make_unique<rmm::device_buffer>(
    values.data(), n_elements * sizeof(int), stream, br.device_mr());

  return {std::move(metadata), br.move(std::move(data), stream)};
}

/**
 * @brief Validate a packed data object by checking metadata and GPU data contents.
 *
 * @param packed_data Packed data object to validate.
 * @param n_elements Expected number of elements.
 * @param offset Expected starting value of the sequence.
 * @param stream CUDA stream used for device-host transfers.
 * @param br Buffer resource used for host allocation.
 */
inline void validate_packed_data(rapidsmpf::PackedData&& packed_data,
                                 int n_elements,
                                 int offset,
                                 rmm::cuda_stream_view stream,
                                 rapidsmpf::BufferResource& br)
{
  auto const& metadata = *packed_data.metadata;
  EXPECT_EQ(n_elements * sizeof(int), metadata.size());

  for (int i = 0; i < n_elements; i++) {
    int val;
    std::memcpy(&val, metadata.data() + i * sizeof(int), sizeof(int));
    EXPECT_EQ(offset + i, val);
  }

  EXPECT_EQ(n_elements * sizeof(int), packed_data.data->size);

  auto res          = br.reserve_or_fail(packed_data.data->size, rapidsmpf::MemoryType::HOST);
  auto data_on_host = br.move_to_host_buffer(std::move(packed_data.data), res);
  RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
  EXPECT_EQ(metadata, data_on_host->copy_to_uint8_vector());
}

/**
 * @brief Device memory resource that can inject stream-ordered delays.
 *
 * When enabled, each allocation enqueues a host callback on the allocation
 * stream that sleeps for a configurable duration. This blocks the CUDA stream
 * (making `cudaEventQuery` return not-ready) without blocking the host thread,
 * so the progress thread's event loop continues to run while data buffers
 * appear unready.
 */
class DelayedMemoryResource {
 public:
  DelayedMemoryResource(rmm::device_async_resource_ref upstream, std::chrono::milliseconds delay)
    : upstream_{upstream}, delay_{delay}
  {
  }

  void* allocate_sync(std::size_t, std::size_t)
  {
    RAPIDSMPF_FAIL("synchronous allocation not supported", std::invalid_argument);
  }

  void deallocate_sync(void*, std::size_t, std::size_t) noexcept
  {
    RAPIDSMPF_FATAL("synchronous deallocation not supported");
  }

  void* allocate(rmm::cuda_stream_view stream,
                 std::size_t size,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    void* ptr = upstream_.allocate(stream, size, alignment);
    if (size > 0) {
      RAPIDSMPF_CUDA_TRY(
        cudaLaunchHostFunc(stream.value(), sleep_on_stream, new std::chrono::milliseconds(delay_)));
    }
    return ptr;
  }

  void deallocate(rmm::cuda_stream_view stream,
                  void* ptr,
                  std::size_t size,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    upstream_.deallocate(stream, ptr, size, alignment);
  }

  bool operator==(DelayedMemoryResource const& other) const noexcept { return this == &other; }

  bool operator!=(DelayedMemoryResource const& other) const noexcept { return !(this == &other); }

  friend void get_property(DelayedMemoryResource const&, cuda::mr::device_accessible) noexcept {}

 private:
  static void CUDART_CB sleep_on_stream(void* user_data)
  {
    auto* delay = static_cast<std::chrono::milliseconds*>(user_data);
    std::this_thread::sleep_for(*delay);
    delete delay;
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::chrono::milliseconds delay_;
};

static_assert(cuda::mr::resource<DelayedMemoryResource>);
static_assert(cuda::mr::resource_with<DelayedMemoryResource, cuda::mr::device_accessible>);
