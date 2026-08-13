/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/context.hpp"

#include "io/comp/nvcomp_adapter.hpp"
#include "jit/cache.hpp"

#include <cudf/context.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/utilities/error.hpp>

#include <unistd.h>

#include <filesystem>
#include <format>
#include <memory>

namespace cudf {

namespace {

int32_t get_driver_version()
{
  int32_t driver_version;
  CUDF_CUDA_TRY(cudaDriverGetVersion(&driver_version));
  return driver_version;
}

int32_t get_runtime_version()
{
  int32_t runtime_version;
  CUDF_CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

int32_t get_current_device_compute_capability()
{
  int32_t device;
  CUDF_CUDA_TRY(cudaGetDevice(&device));

  cudaDeviceProp props;
  CUDF_CUDA_TRY(cudaGetDeviceProperties(&props, device));

  return props.major * 10 + props.minor;
}

}  // namespace

context::context(context_config cfg, init_flags flags)
  : _config{std::move(cfg)},
    _jit_cache_init_flag{},
    _device_properties{
      get_driver_version(), get_runtime_version(), get_current_device_compute_capability()},
    _nvrtc_version{0},
    _nvjitlink_version{0}
{
  initialize_components(flags);
}

void context::ensure_nvcomp_loaded() { io::detail::nvcomp::load_nvcomp_library(); }

void context::ensure_jit_cache_initialized()
{
  std::call_once(_jit_cache_init_flag, [&]() {
    // make sure the required directories exist
    std::filesystem::create_directories(_config.rtcx_cache_dir);
    std::filesystem::create_directories(_config.jit_bundle_dir);
    std::filesystem::create_directories(_config.jit_pch_dir);
    std::filesystem::create_directories(_config.jit_tmp_dir);

    rtcx::initialize();

    _nvrtc_version     = rtcx::nvrtc_version();
    _nvjitlink_version = rtcx::nvjitlink_version();

    auto limits = rtcx::cache_limits{.num_mem_blobs     = _config.kernel_cache_limit_process,
                                     .num_mem_libraries = _config.kernel_cache_limit_process};

    _rtcx_cache = std::make_unique<rtcx::cache_t>(_config.rtcx_cache_dir,
                                                  _config.jit_tmp_dir,
                                                  limits,
                                                  bool{_config.preload_jit_cache},
                                                  bool{_config.disable_jit_cache});

    if (_config.clear_jit_cache) {
      _rtcx_cache->clear_memory_store();
      _rtcx_cache->clear_disk_store();
    }

    // note that jit_bundle depends on rtcx_cache, so we ensure rtcx_cache is initialized first.
    _jit_bundle = std::make_unique<jit_bundle_t>(_config.jit_bundle_dir, *_rtcx_cache);
  });
}

context::~context()
{
  _jit_bundle.reset();
  _rtcx_cache.reset();
  rtcx::teardown();
}

rtcx::cache_t& context::rtcx_cache()
{
  ensure_jit_cache_initialized();
  return *_rtcx_cache;
}

jit_bundle_t& context::jit_bundle()
{
  ensure_jit_cache_initialized();
  return *_jit_bundle;
}

bool context::dump_codegen() const { return _config.dump_codegen; }

bool context::use_jit() const { return _config.use_jit; }

context_config const& context::config() const { return _config; }

std::string const& context::get_jit_pch_dir() const { return _config.jit_pch_dir; }

context::device_properties const& context::get_device_properties() const
{
  return _device_properties;
}

std::optional<int32_t> context::nvrtc_version() const { return _nvrtc_version; }

std::optional<int32_t> context::nvjitlink_version() const { return _nvjitlink_version; }

void context::initialize_components(init_flags flags)
{
  if (has_flag(flags, init_flags::INIT_JIT_CACHE)) { ensure_jit_cache_initialized(); }

  if (has_flag(flags, init_flags::LOAD_NVCOMP)) { io::detail::nvcomp::load_nvcomp_library(); }
}

/**
 * @brief Returns the path to the CUDF kernel cache directory.
 */
std::filesystem::path get_cudf_kernel_cache_dir()
{
  static constexpr auto has_rwx = [](std::filesystem::path const& p) -> bool {
    // check if the process has read, write, and execute permissions on the directory
    return ::access(p.c_str(), R_OK | W_OK | X_OK) == 0;
  };

  static constexpr auto is_accessible_dir = [](std::filesystem::path const& p) -> bool {
    return std::filesystem::is_directory(p) && has_rwx(p);
  };

  static constexpr auto try_create_dirs = [](std::filesystem::path const& p) -> bool {
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return (!ec || ec == std::errc::file_exists) && is_accessible_dir(p);
  };

  if (auto path = detail::getenv_optional<std::string>("LIBCUDF_KERNEL_CACHE_PATH");
      path.has_value()) {
    // - if $LIBCUDF_KERNEL_CACHE_PATH exists, return it, otherwise create it
    // - if creation fails, warn and continue to next option
    // - check that we have read/write permissions to the directory, otherwise warn and continue
    // to next option
    if (try_create_dirs(*path)) {
      return *path;
    } else {
      CUDF_LOG_WARN(
        std::format("Environment variable {} is set to {}, but the process "
                    "could not create the directory. Ignoring.",
                    "LIBCUDF_KERNEL_CACHE_PATH",
                    *path));
    }
  }

  if (auto base = detail::getenv_optional<std::string>("XDG_CACHE_HOME"); base.has_value()) {
    auto path = std::filesystem::path(*base) / "libcudf";
    if (try_create_dirs(path)) { return path; }
  }

  if (auto base = detail::getenv_optional<std::string>("HOME"); base.has_value()) {
    if (is_accessible_dir(*base)) {
      // attempt to create the subdirectories if non-existent
      auto path = std::filesystem::path(*base) / ".cache" / "libcudf";
      if (try_create_dirs(path)) { return path; }
    }
  }

  if (auto base = detail::getenv_optional<std::string>("TMPDIR"); base.has_value()) {
    if (is_accessible_dir(*base)) {
      // attempt to create the subdirectory if non-existent
      auto path = std::filesystem::path(*base) / "libcudf";
      if (try_create_dirs(path)) { return path; }
    }
  }

  if (is_accessible_dir("/tmp")) {
    auto path = std::filesystem::path("/tmp") / "libcudf";
    if (try_create_dirs(path)) { return path; }
  }

  CUDF_FAIL(
    R"***(Unable to resolve cuDF kernel cache directory. Tried:
- ${LIBCUDF_KERNEL_CACHE_PATH}
- ${XDG_CACHE_HOME}/libcudf
- ${HOME}/.cache/libcudf
- ${TMPDIR}/libcudf
- /tmp/libcudf)***",
    std::runtime_error);
}

static std::optional<context> _context{std::nullopt};
static std::optional<std::once_flag> _context_init_flag{std::in_place};
static std::optional<std::once_flag> _context_deinit_flag{std::in_place};

}  // namespace cudf

namespace CUDF_EXPORT cudf {

void initialize(init_flags flags)
{
  std::call_once(*_context_init_flag, [&]() {
    auto const dump_codegen      = detail::get_bool_env_or("LIBCUDF_JIT_DUMP_CODEGEN", false);
    auto const use_jit           = detail::get_bool_env_or("LIBCUDF_JIT_ENABLED", false);
    auto const preload_jit_cache = detail::get_bool_env_or("LIBCUDF_KERNEL_CACHE_PRELOAD", false);
    auto const disable_jit_cache = detail::get_bool_env_or("LIBCUDF_KERNEL_CACHE_DISABLED", false);
    auto const clear_jit_cache   = detail::get_bool_env_or("LIBCUDF_KERNEL_CACHE_CLEAR", false);
    auto const disable_cuda_cache =
      detail::get_bool_env_or("LIBCUDF_JIT_DISABLE_CUDA_CACHE", false);
    auto const jit_verbose    = detail::get_bool_env_or("LIBCUDF_JIT_VERBOSE", false);
    auto const dump_jit_trace = detail::get_bool_env_or("LIBCUDF_JIT_DUMP_TRACE", false);
    auto const dump_jit_time_profile =
      detail::get_bool_env_or("LIBCUDF_JIT_DUMP_TIME_PROFILE", false);

    auto const kernel_cache_limit_process =
      detail::getenv_or("LIBCUDF_KERNEL_CACHE_LIMIT_PER_PROCESS", 16'384U);

    flags = flags | (use_jit ? init_flags::INIT_JIT_CACHE : init_flags::NONE);

    auto const cache_dir      = get_cudf_kernel_cache_dir();
    auto const jit_bundle_dir = cache_dir / "bundle";
    auto const rtcx_cache_dir = cache_dir / "rtcx_cache";
    auto const jit_pch_dir    = cache_dir / "pch";
    auto const jit_tmp_dir    = cache_dir / "tmp";

    context_config cfg{.dump_codegen               = dump_codegen,
                       .use_jit                    = use_jit,
                       .preload_jit_cache          = preload_jit_cache,
                       .disable_jit_cache          = disable_jit_cache,
                       .clear_jit_cache            = clear_jit_cache,
                       .disable_cuda_cache         = disable_cuda_cache,
                       .jit_verbose                = jit_verbose,
                       .dump_jit_trace             = dump_jit_trace,
                       .dump_jit_time_profile      = dump_jit_time_profile,
                       .rtcx_cache_dir             = rtcx_cache_dir,
                       .jit_bundle_dir             = jit_bundle_dir,
                       .jit_pch_dir                = jit_pch_dir,
                       .jit_tmp_dir                = jit_tmp_dir,
                       .kernel_cache_limit_process = kernel_cache_limit_process};

    _context.emplace(cfg, flags);
  });

  _context->initialize_components(flags);
}

void teardown()
{
  std::call_once(*_context_deinit_flag, [&]() {
    // reset the context to destroy all global objects and release resources, allowing for clean
    // re-initialization in the future if desired.
    _context.reset();
    _context_init_flag.emplace();
    _context_deinit_flag.emplace();
  });
}

void enable_jit_cache(bool enabled)
{
  auto& cache = get_context().rtcx_cache();
  cache.enable(enabled);
}

void clear_jit_cache()
{
  auto& cache = get_context().rtcx_cache();
  cache.clear_memory_store();
  cache.clear_disk_store();
}

context& get_context()
{
  cudf::initialize();
  return *_context;
}

}  // namespace CUDF_EXPORT cudf
