/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "murmurhash3_x86_32_rtcx.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32_rtcx_tags.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rtcx/algorithm_planner.hpp>

#include <chrono>

namespace cudf::hashing::detail {
namespace {

rtcx::launcher_jit_cache& murmurhash3_x86_32_rtcx_cache()
{
  static rtcx::launcher_jit_cache cache;
  return cache;
}

enum class rtcx_murmurhash_schema { NONE, INT32, STRING, DICTIONARY_STRING };

rtcx_murmurhash_schema get_rtcx_murmurhash_schema(table_view const& input)
{
  if (input.num_rows() == 0 || input.num_columns() == 0) { return rtcx_murmurhash_schema::NONE; }

  auto all_int32             = true;
  auto all_strings           = true;
  auto all_dictionary_string = true;
  for (size_type column_index = 0; column_index < input.num_columns(); ++column_index) {
    auto const column = input.column(column_index);
    all_int32 = all_int32 && column.type().id() == type_id::INT32 && column.num_children() == 0;
    // STRING is a scalar type even though its offsets live in a child column.
    all_strings = all_strings && column.type().id() == type_id::STRING;
    all_dictionary_string =
      all_dictionary_string && column.type().id() == type_id::DICTIONARY32 &&
      column.num_children() == 2 &&
      column.child(dictionary_column_view::keys_column_index).type().id() == type_id::STRING;
  }
  if (all_int32) { return rtcx_murmurhash_schema::INT32; }
  if (all_strings) { return rtcx_murmurhash_schema::STRING; }
  if (all_dictionary_string) { return rtcx_murmurhash_schema::DICTIONARY_STRING; }
  return rtcx_murmurhash_schema::NONE;
}

}  // namespace

bool murmurhash3_x86_32_rtcx_enabled(table_view const& input)
{
  return get_rtcx_murmurhash_schema(input) != rtcx_murmurhash_schema::NONE;
}

std::unique_ptr<column> murmurhash3_x86_32_rtcx(table_view const& input,
                                                uint32_t seed,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  auto const schema = get_rtcx_murmurhash_schema(input);
  if (schema == rtcx_murmurhash_schema::NONE) {
    auto const preprocessed = cudf::detail::row::hash::preprocessed_table::create(
      input, stream, cudf::get_current_device_resource_ref());
    return murmurhash3_x86_32_rtcx_generic(
      preprocessed, input.num_rows(), seed, has_nulls(input), stream, mr);
  }
  auto output             = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);
  auto const preprocessed = cudf::detail::row::hash::preprocessed_table::create(
    input, stream, cudf::get_current_device_resource_ref());
  table_device_view const input_device_view{*preprocessed};
  auto output_device_view = mutable_column_device_view::create(output->mutable_view(), stream);

  auto const entrypoint = schema == rtcx_murmurhash_schema::INT32
                            ? "cudf_murmurhash3_x86_32_rtcx_entry"
                          : schema == rtcx_murmurhash_schema::STRING
                            ? "cudf_murmurhash3_x86_32_rtcx_string_entry"
                            : "cudf_murmurhash3_x86_32_rtcx_dictionary_string_entry";
  rtcx::algorithm_planner planner{entrypoint, murmurhash3_x86_32_rtcx_cache()};
  if (schema == rtcx_murmurhash_schema::INT32) {
    planner.add_static_fragment<rtcx_murmur::fragment_tag_entry_int32>();
    planner.add_static_fragment<rtcx_murmur::fragment_tag_hasher_int32>();
  } else if (schema == rtcx_murmurhash_schema::STRING) {
    planner.add_static_fragment<rtcx_murmur::fragment_tag_entry_string>();
    planner.add_static_fragment<rtcx_murmur::fragment_tag_hasher_string>();
  } else {
    planner
      .add_static_fragment<rtcx_murmur::fragment_tag_dictionary_string_entry_dictionary_string>();
    planner
      .add_static_fragment<rtcx_murmur::fragment_tag_dictionary_string_hasher_dictionary_string>();
  }
  auto const cache_size_before = murmurhash3_x86_32_rtcx_cache().size();
  auto const lookup_start      = std::chrono::steady_clock::now();
  auto const launcher          = planner.get_launcher();
  auto const lookup_elapsed    = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now() - lookup_start);
  CUDF_LOG_INFO(
    "MurmurHash3 RTCX planner lookup: cache %zu -> %zu, %lld us, linked cubin %zu bytes",
    cache_size_before,
    murmurhash3_x86_32_rtcx_cache().size(),
    static_cast<long long>(lookup_elapsed.count()),
    launcher->linked_cubin_size());

  constexpr auto threads_per_block = 256U;
  auto const blocks                = static_cast<unsigned int>(
    (input.num_rows() + static_cast<size_type>(threads_per_block) - 1) / threads_per_block);
  launcher
    ->dispatch<void(cudf::mutable_column_device_view, uint32_t, cudf::table_device_view, bool)>(
      stream.get(),
      dim3(blocks),
      dim3(threads_per_block),
      0,
      *output_device_view,
      seed,
      input_device_view,
      has_nulls(input));
  return output;
}

std::size_t murmurhash3_x86_32_rtcx_cache_size() { return murmurhash3_x86_32_rtcx_cache().size(); }

}  // namespace cudf::hashing::detail
