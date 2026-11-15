/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/udf.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda_runtime.h>

#include <cudf_cuda_embed.hpp>
#include <jit/cache.hpp>
#include <jit/reflect.hpp>
#include <rtcx/rtcx.hpp>
#include <runtime/context.hpp>

#define XXH_INLINE_ALL
#include <unistd.h>
#include <xxhash.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

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
      // The bytes are linked as the given type, so both identify an unnamed fragment.
      XXH3_128bits_update(ctx, &fragment.type, sizeof(fragment.type));
      XXH3_128bits_update(ctx, fragment.data.data(), fragment.data.size());
      XXH3_128bits_update(ctx, "\0", 1);  // null terminator
    }
  }
}

void install_file_set(std::string_view target_dir,
                      std::span<uint8_t const> compressed_binary,
                      size_t uncompressed_size,
                      std::span<cudf_cuda_embed::range_t const> file_ranges,
                      std::span<char const* const> destinations,
                      std::string_view compression)
{
  auto decompressed = rtcx::decompress_blob(compressed_binary, uncompressed_size, compression);
  for (size_t i = 0; i < file_ranges.size(); ++i) {
    auto file_data   = file_ranges[i](std::span{decompressed});
    auto dst_path    = destinations[i];
    auto target_path = std::format("{}/{}", target_dir, dst_path);

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
constexpr int32_t MIN_NVRTC_VERSION_PROCESS_PCH_DIR =
  make_cuda_version(13, 2, 0);  // first NVRTC version that picks a per-process PCH directory
constexpr int32_t LTO_ARCHITECTURE =
  CUDF_LTO_ARCHITECTURE;  // target architecture for LTO IR compilation

/**
 * @brief Owns a PCH directory private to the current process.
 *
 * Without `--pch-dir`, NVRTC older than 13.2 writes automatic PCH files (e.g. `kernel.pch`) into
 * the current working directory. Processes sharing a working directory then overwrite and delete
 * each other's PCH files, which causes NVRTC compile errors and crashes. A PCH is only usable by
 * the NVRTC instance that created it, so a per-process directory loses no reuse.
 *
 * The directory is created on first use and removed at process exit. A forked child creates its
 * own directory and does not remove its parent's.
 */
class process_pch_dir {
 public:
  process_pch_dir()                                  = default;
  process_pch_dir(process_pch_dir const&)            = delete;
  process_pch_dir& operator=(process_pch_dir const&) = delete;
  process_pch_dir(process_pch_dir&&)                 = delete;
  process_pch_dir& operator=(process_pch_dir&&)      = delete;

  ~process_pch_dir()
  {
    if (!path_.empty() && owner_ == ::getpid()) {
      std::error_code ec;
      std::filesystem::remove_all(path_, ec);  // best effort; never throw at exit
    }
  }

  std::string get(std::string const& parent_dir)
  {
    std::lock_guard lock{mutex_};
    if (path_.empty() || owner_ != ::getpid()) {
      auto path_template = std::format("{}/nvrtc-pch-XXXXXX", parent_dir);
      std::filesystem::create_directories(parent_dir);
      auto const* created = ::mkdtemp(path_template.data());
      auto const error    = errno;
      CUDF_EXPECTS(created != nullptr,
                   std::format("Failed to create NVRTC PCH directory from template {}: {}",
                               path_template,
                               std::strerror(error)),
                   std::runtime_error);
      path_  = std::move(path_template);
      owner_ = ::getpid();
    }
    return path_;
  }

 private:
  std::mutex mutex_;
  std::string path_;
  pid_t owner_ = 0;
};

/**
 * @brief Appends the NVRTC automatic PCH options for this NVRTC version.
 */
void append_pch_options(std::vector<std::string>& options, int32_t nvrtc_version)
{
  if (nvrtc_version < MIN_NVRTC_VERSION_PCH) { return; }

  auto& cfg = cudf::get_context().config();

  options.emplace_back("--pch");

  // NVRTC 13.2+ already places automatic PCH files in a per-process temporary directory
  if (nvrtc_version < MIN_NVRTC_VERSION_PROCESS_PCH_DIR) {
    static process_pch_dir pch_dir;
    options.emplace_back(std::format("--pch-dir={}", pch_dir.get(cfg.jit_tmp_dir)));
  }

  if (cfg.jit_verbose) {
    options.emplace_back("--pch-verbose=true");
    options.emplace_back("--pch-messages=true");
  } else {
    options.emplace_back("--pch-verbose=false");
    options.emplace_back("--pch-messages=false");
  }
}

std::vector<std::string> make_lto_compile_options(char const* name,
                                                  std::span<char const* const> include_paths,
                                                  bool use_pch)
{
  auto& ctx = cudf::get_context();
  auto& cfg = ctx.config();

  auto nvrtc_version = ctx.nvrtc_version().value();

  auto use_minimal = nvrtc_version >= MIN_NVRTC_VERSION_MINIMAL;

  std::vector<std::string> options;
  // Let callers override bundled headers through their include directories.
  for (auto const* include_dir : include_paths) {
    options.emplace_back(std::format("-I{}", include_dir));
  }
  for (auto const& include_dir : ctx.jit_bundle().get_include_directories()) {
    options.emplace_back(std::format("-I{}", include_dir));
  }

  options.emplace_back(std::format("--gpu-architecture=sm_{}", LTO_ARCHITECTURE));

  options.emplace_back("--diag-suppress=47");
  options.emplace_back("--device-int128");

  if (LTO_ARCHITECTURE >= 100) { options.emplace_back("--device-float128"); }

  options.emplace_back("-std=c++20");
  options.emplace_back("--device-as-default-execution-space");
  options.emplace_back("--generate-line-info");
  options.emplace_back("--dopt=on");
  options.emplace_back("--dlink-time-opt");

  if (use_minimal) { options.emplace_back("--minimal"); }

  // suppress warnings if not in verbose mode
  if (!cfg.jit_verbose) { options.emplace_back("--disable-warnings"); }

  if (use_pch) { append_pch_options(options, nvrtc_version); }

  if (cfg.disable_cuda_cache) { options.emplace_back("--no-cache"); }

  if (cfg.dump_jit_trace) { options.emplace_back("--time=-"); }

  if (cfg.dump_jit_time_profile) {
    options.emplace_back(std::format("--fdevice-time-trace=cudf_kernel_{}_trace", name));
  }

  return options;
}

rtcx::blob compile_fragment(char const* name,
                            char const* cuda_code,
                            std::span<char const* const> header_include_names,
                            std::span<char const* const> headers,
                            std::span<std::string const> options)
{
  CUDF_FUNC_RANGE();

  std::vector<char const*> options_cstr;
  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }

  auto params = rtcx::compile_params{.name                 = name,
                                     .source               = cuda_code,
                                     .header_include_names = header_include_names,
                                     .headers              = headers,
                                     .options              = options_cstr,
                                     .target_type          = rtcx::binary_type::LTO_IR};

  auto fragment = rtcx::compile(params);

  return std::make_shared<rtcx::blob_t>(rtcx::blob_t::from_buffer(std::move(fragment)));
}

// Bundled kernels reuse their common PCH; specialization headers follow nv_hdrstop.
rtcx::blob compile_kernel_fragment(char const* name,
                                   char const* cuda_code,
                                   std::span<char const* const> header_include_names,
                                   std::span<char const* const> headers)
{
  CUDF_FUNC_RANGE();

  auto options = make_lto_compile_options(name, {}, true);

  return compile_fragment(name, cuda_code, header_include_names, headers, options);
}

rtcx::blob compile_user_fragment(char const* name,
                                 char const* cuda_code,
                                 std::span<char const* const> header_include_names,
                                 std::span<char const* const> headers,
                                 std::span<char const* const> include_paths,
                                 bool use_pch)
{
  CUDF_FUNC_RANGE();

  auto options = make_lto_compile_options(name, include_paths, use_pch);

  return compile_fragment(name, cuda_code, header_include_names, headers, options);
}

}  // namespace

lto_fragment compiled_fragment::view() const noexcept
{
  return lto_fragment{.type = lto_binary_type::LTO_IR, .data = blob_->view()};
}

rtcx::blob get_user_fragment(char const* name,
                             char const* source_code,
                             std::span<char const* const> header_include_names,
                             std::span<char const* const> headers,
                             std::span<char const* const> include_paths,
                             bool use_pch)
{
  CUDF_FUNC_RANGE();

  auto& ctx         = cudf::get_context();
  auto const& props = ctx.get_device_properties();
  auto spec         = std::format(
    "ltoFragment\n"
            "name={}\n"
            "nvrtc={}\n"
            "runtime={}\n"
            "driver={}\n"
            "arch={}\n"
            "bundle={}\n"
            "use_pch={}\n",
    name,
    ctx.nvrtc_version().value(),
    props.runtime_version,
    props.driver_version,
    LTO_ARCHITECTURE,
    ctx.jit_bundle().get_hash(),
    use_pch);

  XXH3_state_t state;
  XXH3_INITSTATE(&state);
  XXH3_128bits_reset(&state);
  hash(&state, spec);
  hash(&state, "include_names: ");
  hash(&state, header_include_names);
  hash(&state, "includes: ");
  hash(&state, headers);
  hash(&state, "include_paths: ");
  hash(&state, include_paths);

  hash(&state, "source: ");
  hash(&state, std::string_view{source_code});

  auto const digest = XXH3_128bits_digest(&state);
  auto const key    = rtcx::hash128{digest.high64, digest.low64};

  auto compile = [&] {
    return compile_user_fragment(
      name, source_code, header_include_names, headers, include_paths, use_pch);
  };

  return ctx.rtcx_cache()
    .get_or_add_blob(key, rtcx::blob_compile_func::from_functor(compile))
    .get();
}

compiled_fragment compile_cuda_to_lto_ir(char const* source_code,
                                         std::span<char const* const> include_names,
                                         std::span<char const* const> includes,
                                         std::span<char const* const> include_paths,
                                         bool use_pch,
                                         std::string_view pch_key)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(include_names.size() == includes.size(),
               "Header names and contents must have the same size",
               std::invalid_argument);

  auto const source_name = std::format("{}.cu", pch_key);
  return compiled_fragment(get_user_fragment(
    source_name.c_str(), source_code, include_names, includes, include_paths, use_pch));
}

std::optional<compiled_fragment> get_udf_adapter(std::string_view symbol,
                                                 std::string_view target_symbol,
                                                 bool is_null_aware,
                                                 std::span<transform_input_spec const> inputs,
                                                 std::span<transform_output_spec const> outputs,
                                                 bool use_physical_types,
                                                 int32_t index_count)
{
  // A UDF exporting the target symbol already provides the entry point; generating a forwarding
  // adapter would redefine it. This is also an optimization to avoid generating unnecessary code.
  if (symbol == target_symbol) { return std::nullopt; }

  auto parameters = jit::reflect_udf_parameter_types(
    is_null_aware, inputs, outputs, use_physical_types, index_count);
  auto append = [&](std::string& target, size_t i, bool with_type) {
    if (!target.empty()) { target += ", "; }
    if (with_type) { target += parameters[i] + " "; }
    target += std::format("arg_{}", i);
  };
  std::string declarations;
  std::string arguments;
  for (size_t i = 0; i < parameters.size(); ++i) {
    append(declarations, i, true);
    append(arguments, i, false);
  }
  auto source = std::format(R"CUDA(
#include <jit/udf_abi.cuh>
#pragma nv_hdrstop
namespace cudf_udf_abi {{
extern "C" __device__ cudf::errc {0}({1});
extern "C" __device__ cudf::errc {3}({1}) {{ return {0}({2}); }}
}}
)CUDA",
                            symbol,
                            declarations,
                            arguments,
                            target_symbol);
  return compile_cuda_to_lto_ir(source.c_str(), {}, {}, {}, true, "cudf_udf_adapter");
}

rtcx::binary_type as_rtcx_binary_type(lto_binary_type type)
{
  switch (type) {
    case lto_binary_type::LTO_IR: return rtcx::binary_type::LTO_IR;
    case lto_binary_type::FATBIN: return rtcx::binary_type::FATBIN;
    case lto_binary_type::PTX: return rtcx::binary_type::PTX;
    default:
      CUDF_FAIL(
        std::format("Unrecognized LTO binary type {} for LTO transform", static_cast<int>(type)),
        std::invalid_argument);
  }
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
                          LTO_ARCHITECTURE,
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
    return compile_kernel_fragment(name.c_str(), source.c_str(), header_include_names, headers);
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
  options.emplace_back("-kernels-used=cudf_kernel_entry");
  options.emplace_back("-optimize-unused-variables");
  options.emplace_back(std::format("-arch=sm_{}", sm));

  if (cfg.disable_cuda_cache) { options.emplace_back("--no-cache"); }

  if (cfg.jit_verbose) { options.emplace_back("-verbose"); }

  if (cfg.dump_jit_trace) { options.emplace_back("-time"); }

  std::vector<char const*> options_cstr;

  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }

  // Keep copies of unterminated PTX alive until linking finishes. Binary modules and
  // already-terminated PTX retain their original storage.
  std::vector<rtcx::memory_fragment> link_fragments(memory_fragments.begin(),
                                                    memory_fragments.end());
  std::vector<std::vector<uint8_t>> ptx_buffers;
  for (auto& fragment : link_fragments) {
    if (fragment.type == rtcx::binary_type::PTX &&
        (fragment.data.empty() || fragment.data.back() != 0)) {
      auto& ptx = ptx_buffers.emplace_back(fragment.data.begin(), fragment.data.end());
      ptx.push_back(0);
      fragment.data = ptx;
    }
  }

  auto params = rtcx::link_params{.name             = name,
                                  .output_type      = rtcx::binary_type::CUBIN,
                                  .file_fragments   = file_fragments,
                                  .memory_fragments = link_fragments,
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
