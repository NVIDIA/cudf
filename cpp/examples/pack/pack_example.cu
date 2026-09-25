/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace {

std::unique_ptr<cudf::table> make_input(cuda::stream_ref stream)
{
  constexpr cudf::size_type rows = 1 << 16;
  auto sequence                  = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED, stream);
  auto repeated = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, rows, cudf::mask_state::UNALLOCATED, stream);

  auto sequence_view = sequence->mutable_view();
  auto repeated_view = repeated->mutable_view();
  thrust::sequence(
    rmm::exec_policy(stream), sequence_view.begin<int32_t>(), sequence_view.end<int32_t>());
  thrust::fill(
    rmm::exec_policy(stream), repeated_view.begin<int64_t>(), repeated_view.end<int64_t>(), 42);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(sequence));
  columns.push_back(std::move(repeated));
  return std::make_unique<cudf::table>(std::move(columns));
}

void direct_uncompressed_spill(cudf::table_view input, cuda::stream_ref stream)
{
  auto plan = cudf::experimental::prepare_pack(input, stream);
  auto host_payload =
    cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);

  auto result = cudf::experimental::pack_into(
    plan, cudf::device_span<uint8_t>{host_payload.data(), host_payload.size()});
  stream.sync();  // A spill manager would normally wait on an event before consuming host bytes.

  auto restored = cudf::experimental::materialize(
    cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{host_payload.data(), result.payload_bytes},
      result.compression},
    stream);
  std::cout << "uncompressed spill: " << result.payload_bytes << " bytes, " << restored->num_rows()
            << " rows restored\n";
}

void asynchronous_reserved_spill(cudf::table_view input, cuda::stream_ref stream)
{
  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::cascaded;
  options.output_mode = cudf::experimental::compressed_output_mode::reserved;

  auto plan = cudf::experimental::prepare_pack(input, options, stream);
  auto host_payload =
    cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan, cudf::device_span<uint8_t>{host_payload.data(), host_payload.size()});

  // pack_into() did not query the final regional sizes. The spill allocation and input table must
  // remain alive until this stream completes; a real spill manager can record an event here.
  stream.sync();
  auto restored = cudf::experimental::materialize(
    cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{host_payload.data(), result.payload_bytes},
      result.compression},
    stream);
  std::cout << "reserved compressed spill: " << result.payload_bytes << " retained bytes, "
            << restored->num_rows() << " rows restored\n";
}

void compact_shuffle_block(cudf::table_view input, cuda::stream_ref stream)
{
  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::cascaded;
  options.output_mode = cudf::experimental::compressed_output_mode::compact;

  auto plan = cudf::experimental::prepare_pack(input, options, stream);
  auto host_payload =
    cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan, cudf::device_span<uint8_t>{host_payload.data(), host_payload.size()});

  // result.payload_bytes is the prefix a shuffle transport frames and sends. The receiver also
  // needs result.metadata and result.compression.
  auto received = cudf::experimental::packed_data_view{
    result.metadata,
    cudf::device_span<uint8_t const>{host_payload.data(), result.payload_bytes},
    result.compression};
  auto restored = cudf::experimental::materialize(received, stream);
  std::cout << "compact shuffle block: " << result.payload_bytes << " bytes, "
            << restored->num_rows() << " rows received\n";

  // Reusing the plan emits the same unchanged input batch without repeating table discovery,
  // region classification, layout, metadata, or codec setup.
  auto retry_payload =
    cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);
  [[maybe_unused]] auto retry = cudf::experimental::pack_into(
    plan, cudf::device_span<uint8_t>{retry_payload.data(), retry_payload.size()});
}

void automatic_compressed_spill(cudf::table_view input, cuda::stream_ref stream)
{
  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::automatic;

  auto plan = cudf::experimental::prepare_pack(input, options, stream);
  auto payload =
    cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);
  auto result =
    cudf::experimental::pack_into(plan, cudf::device_span<uint8_t>{payload.data(), payload.size()});

  // libcudf selects a codec for each physical region and retains a region uncompressed when the
  // selected codec does not save automatic_min_savings_bytes.
  auto restored = cudf::experimental::materialize(
    cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{payload.data(), result.payload_bytes},
      result.compression},
    stream);
  std::cout << "automatic compressed spill: " << result.payload_bytes << " bytes, "
            << restored->num_rows() << " rows restored\n";
}

void expert_region_selection(cudf::table_view input, cuda::stream_ref stream)
{
  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::automatic;
  auto builder        = cudf::experimental::make_pack_plan_builder(input, options, stream);

  for (auto& region : builder.regions()) {
    if (region.info.kind == cudf::experimental::pack_region_kind::validity) {
      region.options.codec = cudf::experimental::pack_compression::none;
    } else if (region.info.column_index == 0) {
      region.options.codec                   = cudf::experimental::pack_compression::cascaded;
      region.options.cascaded_num_RLEs       = 1;
      region.options.cascaded_num_deltas     = 2;
      region.options.cascaded_use_bitpacking = true;
    }
    // Other regions retain the builder's inherited automatic policy.
  }

  auto plan = std::move(builder).build();
  rmm::device_buffer payload(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan, cudf::device_span<uint8_t>{static_cast<uint8_t*>(payload.data()), payload.size()});
  auto restored = cudf::experimental::materialize(
    cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(payload.data()),
                                       result.payload_bytes},
      result.compression},
    stream);
  std::cout << "expert region selection: " << restored->num_rows() << " rows restored\n";
}

void compress_existing_shuffle_block(cudf::table_view input, cuda::stream_ref stream)
{
  // Some exchange pipelines receive ordinary packed columns before deciding whether compression
  // is worthwhile. The packed allocation remains the compression source and is not repacked.
  auto ordinary_pack = cudf::pack(input, stream);

  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::cascaded;
  options.output_mode = cudf::experimental::compressed_output_mode::compact;
  auto plan           = cudf::experimental::prepare_pack(ordinary_pack, options, stream);

  rmm::device_buffer encoded_payload(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan,
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(encoded_payload.data()),
                               encoded_payload.size()});
  auto restored = cudf::experimental::materialize(
    cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(encoded_payload.data()),
                                       result.payload_bytes},
      result.compression},
    stream);
  std::cout << "late-compressed shuffle block: " << result.payload_bytes << " bytes, "
            << restored->num_rows() << " rows restored\n";
}

void device_resident_zero_copy(cudf::table_view input, cuda::stream_ref stream)
{
  auto plan = cudf::experimental::prepare_pack(input, stream);
  rmm::device_buffer payload(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan, cudf::device_span<uint8_t>{static_cast<uint8_t*>(payload.data()), payload.size()});

  auto borrowed = cudf::experimental::unpack_view(cudf::experimental::packed_data_view{
    result.metadata,
    cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(payload.data()),
                                     result.payload_bytes},
    result.compression});
  std::cout << "device-resident zero-copy view: " << borrowed.num_rows() << " rows\n";
  // borrowed must not outlive either result.metadata or payload.
}

}  // namespace

int main()
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_input(stream);

  direct_uncompressed_spill(input->view(), stream);
  asynchronous_reserved_spill(input->view(), stream);
  compact_shuffle_block(input->view(), stream);
  automatic_compressed_spill(input->view(), stream);
  expert_region_selection(input->view(), stream);
  compress_existing_shuffle_block(input->view(), stream);
  device_resident_zero_copy(input->view(), stream);
}
