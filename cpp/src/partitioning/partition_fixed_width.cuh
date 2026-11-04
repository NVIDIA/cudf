// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>

#include <cuda/std/bit>
#include <cuda/std/cstring>
#include <cuda/std/memory>

namespace cudf::detail::partition_fixed {
template <int Width>
__device__ void copy_value(unsigned char* out, unsigned char const* in)
{
  cuda::std::memcpy(
    cuda::std::assume_aligned<Width>(out), cuda::std::assume_aligned<Width>(in), Width);
}
template <int Width>
struct single_output {
  unsigned char const* input;
  unsigned char* output;
  __device__ void operator()(size_type source, size_type destination) const
  {
    copy_value<Width>(output + std::size_t(destination) * Width,
                      input + std::size_t(source) * Width);
  }
};
struct data_column {
  unsigned char const* input;
  unsigned char* output;
  bitmask_type const* input_mask;
  bitmask_type* output_mask;
  size_type offset;
  size_type width;
};
template <bool Nullable>
struct table_output {
  data_column const* columns;
  size_type count;
  __device__ void operator()(size_type source, size_type destination) const
  {
    for (size_type c = 0; c < count; ++c) {
      auto const col = columns[c];
      auto const row = source + col.offset;
      bool valid     = true;
      if constexpr (Nullable)
        valid = !col.input_mask || (col.input_mask[row / 32] & (1u << (row % 32)));
      if (valid) {
        auto const* in = col.input + std::size_t(row) * col.width;
        auto* out      = col.output + std::size_t(destination) * col.width;
        switch (col.width) {
          case 1: copy_value<1>(out, in); break;
          case 2: copy_value<2>(out, in); break;
          case 4: copy_value<4>(out, in); break;
          case 8: copy_value<8>(out, in); break;
          case 16: copy_value<16>(out, in); break;
        }
        if constexpr (Nullable)
          if (col.output_mask) {
            atomicOr(col.output_mask + destination / 32, 1u << (destination % 32));
          }
      }
    }
  }
};

template <typename Launch>
std::pair<std::unique_ptr<table>, std::vector<size_type>> partition(
  table_view const& input,
  Launch&& launch,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const rows = input.num_rows();
  if (input.num_columns() == 1 && input.column(0).null_count() == 0) {
    auto const& col = input.column(0);
    auto output = make_fixed_width_column(col.type(), rows, mask_state::UNALLOCATED, stream, mr);
    auto run    = [&]<int Width>() {
      return launch(
        single_output<Width>{col.head<unsigned char>() + std::size_t(col.offset()) * Width,
                             output->mutable_view().head<unsigned char>()});
    };
    std::vector<size_type> offsets;
    switch (cudf::size_of(col.type())) {
      case 1: offsets = run.template operator()<1>(); break;
      case 2: offsets = run.template operator()<2>(); break;
      case 4: offsets = run.template operator()<4>(); break;
      case 8: offsets = run.template operator()<8>(); break;
      case 16: offsets = run.template operator()<16>(); break;
      default: CUDF_FAIL("Unsupported fixed-width storage size");
    }
    std::vector<std::unique_ptr<column>> columns;
    columns.push_back(std::move(output));
    return {std::make_unique<table>(std::move(columns), rows), std::move(offsets)};
  }
  std::vector<std::unique_ptr<column>> columns;
  columns.reserve(input.num_columns());
  std::vector<data_column> descriptors;
  bool nullable = false;
  for (auto const& col : input) {
    auto output =
      make_fixed_width_column(col.type(),
                              rows,
                              col.null_count() ? mask_state::ALL_NULL : mask_state::UNALLOCATED,
                              stream,
                              mr);
    output->set_null_count(col.null_count());
    if (col.null_count() != rows) {
      nullable |= col.has_nulls();
      descriptors.push_back({col.head<unsigned char>(),
                             output->mutable_view().head<unsigned char>(),
                             col.has_nulls() ? col.null_mask() : nullptr,
                             output->mutable_view().null_mask(),
                             col.offset(),
                             static_cast<size_type>(cudf::size_of(col.type()))});
    }
    columns.push_back(std::move(output));
  }
  auto device_columns =
    cudf::detail::make_device_buffer(descriptors, stream, cudf::get_current_device_resource_ref());
  std::vector<size_type> offsets;
  if (!nullable)
    offsets = launch(
      table_output<false>{device_columns.data(), static_cast<size_type>(descriptors.size())});
  else
    offsets =
      launch(table_output<true>{device_columns.data(), static_cast<size_type>(descriptors.size())});
  return {std::make_unique<table>(std::move(columns), rows), std::move(offsets)};
}
}  // namespace cudf::detail::partition_fixed
