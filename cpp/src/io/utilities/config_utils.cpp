/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/utilities/error.hpp>

#include <kvikio/defaults.hpp>

#include <string>

namespace cudf::io {

namespace kvikio_integration {

void set_up_kvikio()
{
  static std::once_flag flag{};
  std::call_once(flag, [] {
    // Workaround for https://github.com/rapidsai/cudf/issues/14140, where cuFileDriverOpen errors
    // out if no CUDA calls have been made before it. This is a no-op if the CUDA context is already
    // initialized.
    CUDF_CUDA_TRY(cudaFree(nullptr));

    auto const compat_mode = kvikio::getenv_or("KVIKIO_COMPAT_MODE", kvikio::CompatMode::ON);
    kvikio::defaults::set_compat_mode(compat_mode);

    auto const nthreads = cudf::detail::getenv_or<unsigned int>("KVIKIO_NTHREADS", 4u);
    kvikio::defaults::set_thread_pool_nthreads(nthreads);
  });
}

}  // namespace kvikio_integration

namespace nvcomp_integration {

namespace {
/**
 * @brief Defines which nvCOMP usage to enable.
 */
enum class usage_policy : uint8_t { OFF, STABLE, ALWAYS };

/**
 * @brief Get the current usage policy.
 */
[[nodiscard]] usage_policy get_env_policy()
{
  auto const env_val = cudf::detail::getenv_or<std::string>("LIBCUDF_NVCOMP_POLICY", "STABLE");
  if (env_val == "OFF") return usage_policy::OFF;
  if (env_val == "STABLE") return usage_policy::STABLE;
  if (env_val == "ALWAYS") return usage_policy::ALWAYS;
  CUDF_FAIL("Invalid LIBCUDF_NVCOMP_POLICY value: " + env_val);
}
}  // namespace

[[nodiscard]] bool is_all_enabled() { return get_env_policy() == usage_policy::ALWAYS; }

[[nodiscard]] bool is_stable_enabled()
{
  return is_all_enabled() or get_env_policy() == usage_policy::STABLE;
}

}  // namespace nvcomp_integration

namespace integrated_memory_optimization {

[[nodiscard]] bool is_enabled()
{
  auto const policy = []() {
    auto const* env_val = std::getenv("LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION");
    if (env_val == nullptr) return std::string("AUTO");
    return std::string(env_val);
  }();

  if (policy == "OFF") return false;
  if (policy == "ON") return true;
  if (policy == "AUTO") return cudf::detail::has_integrated_memory();
  CUDF_FAIL("Invalid LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION value: " + policy);
}

}  // namespace integrated_memory_optimization
}  // namespace cudf::io
