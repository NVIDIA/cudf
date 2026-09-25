/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

struct implementation_config {
  bool legacy;
  cudf::experimental::pack_compression compression;
  cudf::experimental::compressed_output_mode output_mode;
};

implementation_config get_implementation(std::string const& name)
{
  using compression = cudf::experimental::pack_compression;
  using output_mode = cudf::experimental::compressed_output_mode;
  if (name == "legacy") { return {true, compression::none, output_mode::compact}; }
  if (name == "prepared-uncompressed") { return {false, compression::none, output_mode::compact}; }
  if (name == "automatic-compact") { return {false, compression::automatic, output_mode::compact}; }
  if (name == "automatic-reserved") {
    return {false, compression::automatic, output_mode::reserved};
  }
  if (name == "cascaded-compact") { return {false, compression::cascaded, output_mode::compact}; }
  if (name == "cascaded-reserved") { return {false, compression::cascaded, output_mode::reserved}; }
  if (name == "zstd-compact") { return {false, compression::zstd, output_mode::compact}; }
  if (name == "zstd-reserved") { return {false, compression::zstd, output_mode::reserved}; }
  if (name == "snappy-compact") { return {false, compression::snappy, output_mode::compact}; }
  if (name == "snappy-reserved") { return {false, compression::snappy, output_mode::reserved}; }
  CUDF_FAIL("Unknown prepared-pack benchmark implementation");
}

std::unique_ptr<cudf::table> make_input(nvbench::state const& state)
{
  auto const size_bytes   = state.get_int64("size_mib") * 1024L * 1024L;
  auto const cardinality  = state.get_int64("cardinality");
  auto constexpr num_cols = 4;
  auto const num_rows     = static_cast<cudf::size_type>(size_bytes / (num_cols * sizeof(int32_t)));
  auto builder            = data_profile_builder()
                   .no_validity()
                   .cardinality(cardinality)
                   .distribution<int32_t>(cudf::type_id::INT32, distribution_id::UNIFORM);
  return create_random_table(
    cycle_dtypes({cudf::type_id::INT32}, num_cols), row_count{num_rows}, data_profile{builder});
}

void add_output_metrics(nvbench::state& state,
                        std::size_t uncompressed_bytes,
                        std::size_t retained_bytes)
{
  state.add_buffer_size(retained_bytes, "retained_payload_bytes", "retained_payload_bytes");
  auto& ratio = state.add_summary("uncompressed_to_retained_ratio");
  ratio.set_string("name", "Compression Ratio");
  ratio.set_string("description", "Uncompressed payload bytes divided by retained payload bytes");
  ratio.set_float64(
    "value", retained_bytes == 0 ? 1.0 : static_cast<double>(uncompressed_bytes) / retained_bytes);
}

void bench_pack_to_pinned_host(nvbench::state& state)
{
  auto const stream             = cudf::get_default_stream();
  auto const config             = get_implementation(state.get_string("implementation"));
  auto input                    = make_input(state);
  auto const uncompressed_bytes = cudf::packed_size(input->view(), stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<int8_t>(input->alloc_size());

  auto const mem_stats_logger = cudf::memory_stats_logger();
  std::size_t retained_bytes  = uncompressed_bytes;
  if (config.legacy) {
    auto host_output = cudf::detail::make_pinned_vector_async<uint8_t>(uncompressed_bytes, stream);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto packed = cudf::pack(input->view(), stream);
      CUDF_CUDA_TRY(cudaMemcpyAsync(host_output.data(),
                                    packed.gpu_data->data(),
                                    packed.gpu_data->size(),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      retained_bytes = packed.gpu_data->size();
    });
  } else {
    auto options        = cudf::experimental::pack_options{};
    options.compression = config.compression;
    options.output_mode = config.output_mode;
    auto plan           = cudf::experimental::prepare_pack(input->view(), options, stream);
    auto host_output =
      cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      retained_bytes = cudf::experimental::pack_into(
                         plan, cudf::device_span<uint8_t>{host_output.data(), host_output.size()})
                         .payload_bytes;
    });
  }

  add_output_metrics(state, uncompressed_bytes, retained_bytes);
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void bench_pack_to_device(nvbench::state& state)
{
  auto const stream             = cudf::get_default_stream();
  auto const config             = get_implementation(state.get_string("implementation"));
  auto input                    = make_input(state);
  auto const uncompressed_bytes = cudf::packed_size(input->view(), stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<int8_t>(input->alloc_size());

  auto const mem_stats_logger = cudf::memory_stats_logger();
  std::size_t retained_bytes  = uncompressed_bytes;
  if (config.legacy) {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto packed    = cudf::pack(input->view(), stream);
      retained_bytes = packed.gpu_data->size();
    });
  } else {
    auto options        = cudf::experimental::pack_options{};
    options.compression = config.compression;
    options.output_mode = config.output_mode;
    auto plan           = cudf::experimental::prepare_pack(input->view(), options, stream);
    rmm::device_buffer output(plan.sizes().payload_bytes, stream);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      retained_bytes =
        cudf::experimental::pack_into(
          plan, cudf::device_span<uint8_t>{static_cast<uint8_t*>(output.data()), output.size()})
          .payload_bytes;
    });
  }

  add_output_metrics(state, uncompressed_bytes, retained_bytes);
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void bench_encode_existing_pack_to_device(nvbench::state& state)
{
  auto const stream             = cudf::get_default_stream();
  auto const config             = get_implementation(state.get_string("implementation"));
  auto input                    = make_input(state);
  auto packed                   = cudf::pack(input->view(), stream);
  auto const uncompressed_bytes = packed.gpu_data->size();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<int8_t>(uncompressed_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  auto options                = cudf::experimental::pack_options{};
  options.compression         = config.compression;
  options.output_mode         = config.output_mode;
  auto plan                   = cudf::experimental::prepare_pack(packed, options, stream);
  rmm::device_buffer output(plan.sizes().payload_bytes, stream);
  std::size_t retained_bytes = uncompressed_bytes;
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    retained_bytes =
      cudf::experimental::pack_into(
        plan, cudf::device_span<uint8_t>{static_cast<uint8_t*>(output.data()), output.size()})
        .payload_bytes;
  });

  add_output_metrics(state, uncompressed_bytes, retained_bytes);
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void bench_restore_from_pinned_host(nvbench::state& state)
{
  auto const stream             = cudf::get_default_stream();
  auto const config             = get_implementation(state.get_string("implementation"));
  auto input                    = make_input(state);
  auto const uncompressed_bytes = cudf::packed_size(input->view(), stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  std::size_t retained_bytes = uncompressed_bytes;
  if (config.legacy) {
    auto packed = cudf::pack(input->view(), stream);
    auto host_output =
      cudf::detail::make_pinned_vector_async<uint8_t>(packed.gpu_data->size(), stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(host_output.data(),
                                  packed.gpu_data->data(),
                                  packed.gpu_data->size(),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    stream.sync();
    retained_bytes              = packed.gpu_data->size();
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      rmm::device_buffer device_payload(host_output.size(), stream);
      CUDF_CUDA_TRY(cudaMemcpyAsync(device_payload.data(),
                                    host_output.data(),
                                    host_output.size(),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      auto unpacked =
        cudf::unpack(packed.metadata->data(), static_cast<uint8_t const*>(device_payload.data()));
      [[maybe_unused]] auto restored = std::make_unique<cudf::table>(unpacked, stream);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  } else {
    auto options        = cudf::experimental::pack_options{};
    options.compression = config.compression;
    options.output_mode = config.output_mode;
    auto plan           = cudf::experimental::prepare_pack(input->view(), options, stream);
    auto host_output =
      cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);
    auto result = cudf::experimental::pack_into(
      plan, cudf::device_span<uint8_t>{host_output.data(), host_output.size()});
    stream.sync();
    retained_bytes    = result.payload_bytes;
    auto packed_input = cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{host_output.data(), result.payload_bytes},
      result.compression};
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      [[maybe_unused]] auto restored = cudf::experimental::materialize(packed_input, stream);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  }

  state.add_global_memory_reads<int8_t>(retained_bytes);
  state.add_global_memory_writes<int8_t>(input->alloc_size());
  add_output_metrics(state, uncompressed_bytes, retained_bytes);
}

void bench_restore_from_device(nvbench::state& state)
{
  auto const stream             = cudf::get_default_stream();
  auto const config             = get_implementation(state.get_string("implementation"));
  auto input                    = make_input(state);
  auto const uncompressed_bytes = cudf::packed_size(input->view(), stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  std::size_t retained_bytes = uncompressed_bytes;
  if (config.legacy) {
    auto packed    = cudf::pack(input->view(), stream);
    retained_bytes = packed.gpu_data->size();
    stream.sync();
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto unpacked =
        cudf::unpack(packed.metadata->data(), static_cast<uint8_t const*>(packed.gpu_data->data()));
      [[maybe_unused]] auto restored = std::make_unique<cudf::table>(unpacked, stream);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  } else {
    auto options        = cudf::experimental::pack_options{};
    options.compression = config.compression;
    options.output_mode = config.output_mode;
    auto plan           = cudf::experimental::prepare_pack(input->view(), options, stream);
    rmm::device_buffer payload(plan.sizes().payload_bytes, stream);
    auto result = cudf::experimental::pack_into(
      plan, cudf::device_span<uint8_t>{static_cast<uint8_t*>(payload.data()), payload.size()});
    stream.sync();
    retained_bytes    = result.payload_bytes;
    auto packed_input = cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(payload.data()),
                                       result.payload_bytes},
      result.compression};
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      [[maybe_unused]] auto restored = cudf::experimental::materialize(packed_input, stream);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  }

  state.add_global_memory_reads<int8_t>(retained_bytes);
  state.add_global_memory_writes<int8_t>(input->alloc_size());
  add_output_metrics(state, uncompressed_bytes, retained_bytes);
}

void bench_device_unpack_view(nvbench::state& state)
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_input(state);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  // Both APIs only reconstruct host-side metadata and return a borrowing table_view. CPU time is
  // therefore the meaningful NVBench measurement; the synchronized executor also records it.
  cudf::size_type volatile observed_columns = 0;
  if (state.get_string("implementation") == "legacy") {
    auto packed = cudf::pack(input->view(), stream);
    stream.sync();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto unpacked =
        cudf::unpack(packed.metadata->data(), static_cast<uint8_t const*>(packed.gpu_data->data()));
      observed_columns = unpacked.num_columns();
    });
  } else {
    auto plan = cudf::experimental::prepare_pack(input->view(), stream);
    rmm::device_buffer payload(plan.sizes().payload_bytes, stream);
    auto result = cudf::experimental::pack_into(
      plan, cudf::device_span<uint8_t>{static_cast<uint8_t*>(payload.data()), payload.size()});
    stream.sync();
    auto packed = cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(payload.data()),
                                       result.payload_bytes},
      result.compression};
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto unpacked    = cudf::experimental::unpack_view(packed);
      observed_columns = unpacked.num_columns();
    });
  }
  (void)observed_columns;
}

}  // namespace

auto const implementations = std::vector<std::string>{"legacy",
                                                      "prepared-uncompressed",
                                                      "automatic-compact",
                                                      "automatic-reserved",
                                                      "cascaded-compact",
                                                      "cascaded-reserved",
                                                      "zstd-compact",
                                                      "zstd-reserved",
                                                      "snappy-compact",
                                                      "snappy-reserved"};

auto const compact_implementations = std::vector<std::string>{"legacy",
                                                              "prepared-uncompressed",
                                                              "automatic-compact",
                                                              "cascaded-compact",
                                                              "zstd-compact",
                                                              "snappy-compact"};

NVBENCH_BENCH(bench_pack_to_pinned_host)
  .set_name("pack_to_pinned_host")
  .add_string_axis("implementation", implementations)
  .add_int64_axis("size_mib", {64})
  .add_int64_axis("cardinality", {16, 0});

NVBENCH_BENCH(bench_pack_to_device)
  .set_name("pack_to_device")
  .add_string_axis("implementation", compact_implementations)
  .add_int64_axis("size_mib", {64})
  .add_int64_axis("cardinality", {16, 0});

NVBENCH_BENCH(bench_encode_existing_pack_to_device)
  .set_name("encode_existing_pack_to_device")
  .add_string_axis("implementation",
                   {"automatic-compact", "cascaded-compact", "zstd-compact", "snappy-compact"})
  .add_int64_axis("size_mib", {64})
  .add_int64_axis("cardinality", {16, 0});

NVBENCH_BENCH(bench_restore_from_pinned_host)
  .set_name("restore_from_pinned_host")
  .add_string_axis("implementation", implementations)
  .add_int64_axis("size_mib", {64})
  .add_int64_axis("cardinality", {16, 0});

NVBENCH_BENCH(bench_restore_from_device)
  .set_name("restore_from_device")
  .add_string_axis("implementation", compact_implementations)
  .add_int64_axis("size_mib", {64})
  .add_int64_axis("cardinality", {16, 0});

NVBENCH_BENCH(bench_device_unpack_view)
  .set_name("device_unpack_view")
  .add_string_axis("implementation", {"legacy", "prepared-uncompressed"})
  .add_int64_axis("size_mib", {64})
  .add_int64_axis("cardinality", {16, 0});
