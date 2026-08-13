/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <cudf_cuda_embed.hpp>
#include <jit/cache.hpp>
#include <rtcx.hpp>
#include <runtime/context.hpp>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <future>

namespace CUDF_EXPORT cudf {

namespace {

void hash(XXH3_state_t* ctx, std::span<char const> input)
{
  XXH3_128bits_update(ctx, input.data(), input.size());
}

void hash(XXH3_state_t* ctx, std::span<char const* const> inputs)
{
  for (auto const* input : inputs) {
    XXH3_128bits_update(ctx, input, std::strlen(input));
    XXH3_128bits_update(ctx, "\0", 1);  // null terminator
  }
}

void hash(XXH3_state_t* ctx, std::span<rtcx::file_fragment const> file_fragments)
{
  for (auto const& fragment : file_fragments) {
    XXH3_128bits_update(ctx, fragment.path, std::strlen(fragment.path));
    XXH3_128bits_update(ctx, "\0", 1);  // null terminator
  }
}

void hash(XXH3_state_t* ctx, std::span<rtcx::memory_fragment const> memory_fragments)
{
  for (auto const& fragment : memory_fragments) {
    if (fragment.name != nullptr) {
      XXH3_128bits_update(ctx, fragment.name, std::strlen(fragment.name));
      XXH3_128bits_update(ctx, "\0", 1);  // null terminator
    } else {
      XXH3_128bits_update(ctx, fragment.data.data(), fragment.data.size());
      XXH3_128bits_update(ctx, "\0", 1);  // null terminator
    }
  }
}

void install_file_set(
  std::string_view target_dir,
  std::span<uint8_t const> compressed_binary,
  size_t uncompressed_size,
  std::span<std::size_t const[2]> file_ranges,  // NOLINT(modernize-avoid-c-arrays)
  std::span<char const* const> destinations,
  std::string_view compression)
{
  auto decompressed = rtcx::decompress_blob(compressed_binary, uncompressed_size, compression);
  for (size_t i = 0; i < file_ranges.size(); ++i) {
    auto file_data_range = file_ranges[i];
    auto file_data       = std::span{decompressed.data() + file_data_range[0], file_data_range[1]};
    auto dst_path        = destinations[i];
    auto target_path     = std::format("{}/{}", target_dir, dst_path);

    std::filesystem::create_directories(std::filesystem::path{target_path}.parent_path());

    std::ofstream file(target_path, std::ios::binary);
    if (!file) {
      throw std::runtime_error(
        std::format("Failed to open file for writing at path: {}", target_path));
    }

    file.write(reinterpret_cast<char const*>(file_data.data()), file_data.size());
    if (!file) {
      throw std::runtime_error(std::format("Failed to write file at path: {}", target_path));
    }
  }
}

std::string read_file_string(char const* path)
{
  std::ifstream file(std::string{path}, std::ios::binary | std::ios::ate);
  if (!file) { throw std::runtime_error(std::format("Failed to open file at path: {}", path)); }

  auto size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string contents(size, '\0');
  if (!file.read(contents.data(), size)) {
    throw std::runtime_error(std::format("Failed to read file at path: {}", path));
  }

  return contents;
}

void install_cudf_jit_files(std::string const& target_dir, std::string const& tmp_dir)
{
  // directory does not exist, so create it
  auto tmp_dir_path_str = std::format("{}/cudf-jit-tmpdir_XXXXXX", tmp_dir);
  char* tmp_dir_path    = ::mkdtemp(tmp_dir_path_str.data());
  CUDF_EXPECTS(
    tmp_dir_path != nullptr,
    std::format("Failed to create temporary directory for JIT file installation in tmp dir: {}",
                tmp_dir),
    std::runtime_error);

  install_file_set(tmp_dir_path,
                   cudf_cuda_embed::files,
                   cudf_cuda_embed::files_uncompressed_size,
                   cudf_cuda_embed::file_ranges,
                   cudf_cuda_embed::file_destinations,
                   cudf_cuda_embed::files_compression);

  // rename the temporary directory to the target install directory
  if (::rename(tmp_dir_path, target_dir.c_str()) == -1) {
    auto errc = errno;
    // another process created it
    if (errc == ENOTEMPTY || errc == EEXIST) {
      std::filesystem::remove_all(tmp_dir_path);
    } else {
      CUDF_FAIL(
        std::format("Failed to install JIT files to target directory: {} with error ({}): {}",
                    target_dir,
                    errc,
                    std::strerror(errc)),
        std::runtime_error);
    }
  }
}

}  // namespace

jit_bundle_t::jit_bundle_t(std::string install_dir, rtcx::cache_t& cache)
  : install_dir_{std::move(install_dir)}, cache_{&cache}
{
  ensure_installed();
}

void jit_bundle_t::ensure_installed() const
{
  CUDF_FUNC_RANGE();

  auto expected_hash = get_hash();
  auto expected_path = std::format("{}/{}", install_dir_, expected_hash);

  if (!std::filesystem::exists(expected_path)) {
    // ensure base install directory exists
    std::filesystem::create_directories(install_dir_);
    install_cudf_jit_files(expected_path.c_str(), cache_->get_tmp_dir());
  } else {
    // directory exists, perform minor sanity check
    CUDF_EXPECTS(std::filesystem::is_directory(expected_path),  // throws if path does not exist
                 std::format("JIT install path ({}) exists but is not a directory", expected_path),
                 std::runtime_error);
  }
}

std::string jit_bundle_t::get_hash() const
{
  auto str = rtcx::hash128_hex_string::make(cudf_cuda_embed::hash);
  return std::string{str.view()};
}

std::string jit_bundle_t::get_directory() const
{
  return std::format("{}/{}", install_dir_, get_hash());
}

std::vector<std::string> jit_bundle_t::get_include_directories() const
{
  std::vector<std::string> directories;
  auto base_dir = get_directory();

  for (auto dir : cudf_cuda_embed::include_directories) {
    directories.emplace_back(std::format("{}/{}", base_dir, dir));
  }

  return directories;
}

namespace {

constexpr int32_t make_cuda_version(int32_t major, int32_t minor, int32_t patch)
{
  return major * 1000 + minor * 10 + patch;
}

constexpr int32_t MIN_NVRTC_VERSION_PCH =
  make_cuda_version(12, 8, 0);  // minimum CUDA version for the "--pch" NVRTC flag
constexpr int32_t MIN_NVRTC_VERSION_MINIMAL =
  make_cuda_version(12, 8, 0);  // minimum CUDA version for the "--minimal" NVRTC flag

std::tuple<rtcx::library, rtcx::blob> compile_library(
  char const* name,
  char const* cuda_code,
  std::span<char const* const> extra_header_include_names,
  std::span<char const* const> extra_headers,
  std::span<char const* const> name_expressions)
{
  CUDF_FUNC_RANGE();

  auto& ctx               = cudf::get_context();
  auto& cfg               = ctx.config();
  auto& bundle            = ctx.jit_bundle();
  auto& device_properties = ctx.get_device_properties();
  auto sm                 = device_properties.compute_capability;
  auto nvrtc_version      = ctx.nvrtc_version().value();

  auto include_dirs = bundle.get_include_directories();
  auto use_pch      = nvrtc_version >= MIN_NVRTC_VERSION_PCH;
  auto use_minimal  = nvrtc_version >= MIN_NVRTC_VERSION_MINIMAL;

  std::vector<std::string> options;

  for (auto const& include_dir : include_dirs) {
    options.emplace_back(std::format("-I{}", include_dir));
  }

  options.emplace_back(std::format("--gpu-architecture=sm_{}", sm));

  options.emplace_back("--diag-suppress=47");
  options.emplace_back("--device-int128");

  if (sm >= 100) { options.emplace_back("--device-float128"); }

  options.emplace_back("-std=c++20");
  options.emplace_back("--device-as-default-execution-space");
  options.emplace_back("--generate-line-info");
  options.emplace_back("--dopt=on");

  if (use_minimal) { options.emplace_back("--minimal"); }

  if (use_pch) {
    options.emplace_back("--pch");

    if (cfg.jit_verbose) {
      options.emplace_back("--pch-verbose=true");
      options.emplace_back("--pch-messages=true");
    } else {
      options.emplace_back("--pch-verbose=false");
      options.emplace_back("--pch-messages=false");
    }
  } else {
    // suppress warning about nv_hdrstop directive on later CUDA versions
    options.emplace_back("--diag-suppress=161");
  }

  if (cfg.disable_cuda_cache) { options.emplace_back("--no-cache"); }

  if (cfg.dump_jit_trace) { options.emplace_back("--time=-"); }

  if (cfg.dump_jit_time_profile) {
    options.emplace_back(std::format("--fdevice-time-trace=cudf_kernel_{}_trace", name));
  }

  std::vector<char const*> options_cstr;
  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }

  auto params = rtcx::compile_params{.name                 = name,
                                     .source               = cuda_code,
                                     .header_include_names = extra_header_include_names,
                                     .headers              = extra_headers,
                                     .options              = options_cstr,
                                     .name_expressions     = name_expressions,
                                     .target_type          = rtcx::binary_type::CUBIN};

  auto cubin   = rtcx::compile(params);
  auto library = rtcx::load_library(cubin);
  auto blob    = rtcx::blob_t::from_buffer(std::move(cubin));

  return std::make_tuple(library, std::make_shared<rtcx::blob_t>(std::move(blob)));
}

rtcx::blob compile_fragment(char const* name,
                            char const* cuda_code,
                            std::span<char const* const> extra_header_include_names,
                            std::span<char const* const> extra_headers,
                            std::span<char const* const> name_expressions)
{
  CUDF_FUNC_RANGE();

  auto& ctx               = cudf::get_context();
  auto& cfg               = ctx.config();
  auto& bundle            = ctx.jit_bundle();
  auto& device_properties = ctx.get_device_properties();
  auto sm                 = device_properties.compute_capability;

  auto include_dirs = bundle.get_include_directories();
  auto pch_dir      = ctx.get_jit_pch_dir();

  auto nvrtc_version = ctx.nvrtc_version().value();

  auto use_pch     = nvrtc_version >= MIN_NVRTC_VERSION_PCH;
  auto use_minimal = nvrtc_version >= MIN_NVRTC_VERSION_MINIMAL;

  std::vector<std::string> options;

  for (auto const& include_dir : include_dirs) {
    options.emplace_back(std::format("-I{}", include_dir));
  }

  options.emplace_back(std::format("--gpu-architecture=sm_{}", sm));

  options.emplace_back("--diag-suppress=47");
  options.emplace_back("--device-int128");

  if (sm >= 100) { options.emplace_back("--device-float128"); }

  options.emplace_back("-std=c++20");
  options.emplace_back("--device-as-default-execution-space");
  options.emplace_back("--generate-line-info");
  options.emplace_back("--dopt=on");
  options.emplace_back("--dlink-time-opt");

  if (use_minimal) { options.emplace_back("--minimal"); }

  if (use_pch) {
    options.emplace_back("--pch");
    options.emplace_back(std::format("--pch-dir={}", pch_dir));

    if (cfg.jit_verbose) {
      options.emplace_back("--pch-verbose=true");
      options.emplace_back("--pch-messages=true");
    } else {
      options.emplace_back("--pch-verbose=false");
      options.emplace_back("--pch-messages=false");
    }
  } else {
    // suppress warning about nv_hdrstop directive on later CUDA versions
    options.emplace_back("--diag-suppress=161");
  }

  if (cfg.disable_cuda_cache) { options.emplace_back("--no-cache"); }

  if (cfg.dump_jit_trace) { options.emplace_back("--time=-"); }

  if (cfg.dump_jit_time_profile) {
    options.emplace_back(std::format("--fdevice-time-trace=cudf_kernel_{}_trace", name));
  }

  std::vector<char const*> options_cstr;
  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }

  auto params = rtcx::compile_params{.name                 = name,
                                     .source               = cuda_code,
                                     .header_include_names = extra_header_include_names,
                                     .headers              = extra_headers,
                                     .options              = options_cstr,
                                     .name_expressions     = name_expressions,
                                     .target_type          = rtcx::binary_type::LTO_IR};

  auto cubin = rtcx::compile(params);
  return std::make_shared<rtcx::blob_t>(rtcx::blob_t::from_buffer(std::move(cubin)));
}

}  // namespace

kernel get_kernel(std::string const& name,
                  std::string const& source_file_id,
                  std::span<char const* const> header_include_names,
                  std::span<char const* const> headers,
                  std::string const& kernel_instance)
{
  CUDF_FUNC_RANGE();

  auto& ctx               = cudf::get_context();
  auto& cache             = ctx.rtcx_cache();
  auto& bundle            = ctx.jit_bundle();
  auto& device_properties = ctx.get_device_properties();
  auto sm                 = device_properties.compute_capability;
  auto runtime            = device_properties.runtime_version;
  auto driver             = device_properties.driver_version;
  auto bundle_hash        = bundle.get_hash();

  auto source_file = std::format("{}/{}", bundle.get_directory(), source_file_id);

  auto spec = std::format(R"***(cuLibrary
name={}
binary_type=CUBIN
cuda_runtime={}
cuda_driver={}
arch={}
bundle={}
source_file={}
kernel_instance={}
)***",
                          name,
                          runtime,
                          driver,
                          sm,
                          bundle_hash,
                          source_file,
                          kernel_instance);

  XXH3_state_t state;
  XXH3_INITSTATE(&state);
  XXH3_128bits_reset(&state);
  hash(&state, spec);
  hash(&state, "header_include_names: ");
  hash(&state, header_include_names);
  hash(&state, "headers: ");
  hash(&state, headers);

  auto digest = XXH3_128bits_digest(&state);
  auto key    = rtcx::hash128{digest.high64, digest.low64};

  auto compile = [&] {
    auto bundle_dir = cudf::get_context().jit_bundle().get_directory();
    auto source     = read_file_string(source_file.c_str());
    return compile_library(name.c_str(), source.c_str(), header_include_names, headers, {});
  };

  auto fut = cache.get_or_add_library(key, rtcx::library_compile_func::from_functor(compile));

  auto lib = fut.get();
  return kernel{lib, lib->get_kernel("cudf_kernel_entry")};
}

rtcx::blob get_kernel_fragment(std::string const& name,
                               std::string const& source_file_id,
                               std::span<char const* const> header_include_names,
                               std::span<char const* const> headers,
                               std::string const& kernel_instance)
{
  CUDF_FUNC_RANGE();

  auto& ctx               = cudf::get_context();
  auto& cache             = ctx.rtcx_cache();
  auto& bundle            = ctx.jit_bundle();
  auto& device_properties = ctx.get_device_properties();
  auto runtime            = device_properties.runtime_version;
  auto driver             = device_properties.driver_version;
  auto sm                 = device_properties.compute_capability;
  auto bundle_hash        = bundle.get_hash();

  auto source_file = std::format("{}/{}", bundle.get_directory(), source_file_id);

  auto spec = std::format(R"***(objectFile
name={}
binary_type=LTO_IR
cuda_runtime={}
cuda_driver={}
arch={}
bundle={}
source_file={}
kernel_instance={}
)***",
                          name,
                          runtime,
                          driver,
                          sm,
                          bundle_hash,
                          source_file,
                          kernel_instance);

  XXH3_state_t state;
  XXH3_INITSTATE(&state);
  XXH3_128bits_reset(&state);
  hash(&state, spec);
  hash(&state, "header_include_names: ");
  hash(&state, header_include_names);
  hash(&state, "headers: ");
  hash(&state, headers);

  auto digest = XXH3_128bits_digest(&state);
  auto key    = rtcx::hash128{digest.high64, digest.low64};

  auto compile = [&] {
    auto bundle_dir = cudf::get_context().jit_bundle().get_directory();
    auto source     = read_file_string(source_file.c_str());
    return compile_fragment(name.c_str(), source.c_str(), header_include_names, headers, {});
  };

  auto fut = cache.get_or_add_blob(key, rtcx::blob_compile_func::from_functor(compile));

  return fut.get();
}

std::tuple<rtcx::library, rtcx::blob> link_library_uncached(
  char const* name,
  std::span<rtcx::file_fragment const> file_fragments,
  std::span<rtcx::memory_fragment const> memory_fragments)
{
  CUDF_FUNC_RANGE();

  auto& ctx               = cudf::get_context();
  auto& device_properties = ctx.get_device_properties();
  auto sm                 = device_properties.compute_capability;
  auto& cfg               = ctx.config();

  std::vector<std::string> options;

  options.emplace_back("-lto");
  options.emplace_back(std::format("-arch=sm_{}", sm));

  if (cfg.disable_cuda_cache) { options.emplace_back("--no-cache"); }

  if (cfg.jit_verbose) { options.emplace_back("-verbose"); }

  if (cfg.dump_jit_trace) { options.emplace_back("-time"); }

  std::vector<char const*> options_cstr;

  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }

  auto params = rtcx::link_params{.name             = name,
                                  .output_type      = rtcx::binary_type::CUBIN,
                                  .file_fragments   = file_fragments,
                                  .memory_fragments = memory_fragments,
                                  .link_options     = options_cstr};

  auto cubin   = rtcx::link_library(params);
  auto library = rtcx::load_library(cubin);
  auto blob    = rtcx::blob_t::from_buffer(std::move(cubin));

  return std::make_tuple(library, std::make_shared<rtcx::blob_t>(std::move(blob)));
}

kernel get_lto_linked_kernel(std::string const& name,
                             std::span<rtcx::file_fragment const> file_fragments,
                             std::span<rtcx::memory_fragment const> memory_fragments)
{
  CUDF_FUNC_RANGE();

  auto& ctx               = cudf::get_context();
  auto& cache             = ctx.rtcx_cache();
  auto& bundle            = ctx.jit_bundle();
  auto& device_properties = ctx.get_device_properties();
  auto runtime            = device_properties.runtime_version;
  auto driver             = device_properties.driver_version;
  auto sm                 = device_properties.compute_capability;
  auto bundle_hash        = bundle.get_hash();

  auto spec = std::format(R"***(cuLibrary
name={}
binary_type=CUBIN
cuda_runtime={}
cuda_driver={}
arch={}
bundle={}
)***",
                          name,
                          runtime,
                          driver,
                          sm,
                          bundle_hash);

  XXH3_state_t state;
  XXH3_INITSTATE(&state);
  XXH3_128bits_reset(&state);
  hash(&state, spec);
  hash(&state, "file_fragments: ");
  hash(&state, file_fragments);
  hash(&state, "memory_fragments: ");
  hash(&state, memory_fragments);

  auto digest = XXH3_128bits_digest(&state);
  auto key    = rtcx::hash128{digest.high64, digest.low64};

  auto compile = [&] {
    return link_library_uncached(name.c_str(), file_fragments, memory_fragments);
  };

  auto fut = cache.get_or_add_library(key, rtcx::library_compile_func::from_functor(compile));

  auto lib = fut.get();
  return kernel{lib, lib->get_kernel("cudf_kernel_entry")};
}

}  // namespace CUDF_EXPORT cudf
