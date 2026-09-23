/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ndsh/utilities.hpp"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace ndsh::detail {
// Integer-derived counts and sums remain exact; floating metrics use a shared 1e-10 tolerance.
inline bool reference_equal(double value, double expected, bool exact = false)
{
  return std::isfinite(value) && std::isfinite(expected) &&
         (exact ? value == expected
                : std::abs(value - expected) <= 1e-10 * std::max(1.0, std::abs(expected)));
}

inline void reference_schema(cudf::table_view table, std::initializer_list<cudf::type_id> types)
{
  CUDF_EXPECTS(static_cast<std::size_t>(table.num_columns()) == types.size(),
               "Unexpected reference column count");
  int i = 0;
  for (auto type : types) {
    auto column = table.column(i++);
    CUDF_EXPECTS(column.type().id() == type && column.null_count() == 0,
                 "Unexpected reference type or nulls");
  }
}

struct reference_table_schema {
  char const* name;
  std::initializer_list<cudf::type_id> types;
};

inline void reference_input_schema(cudf::table_view const& table,
                                   std::string const& name,
                                   std::size_t index,
                                   std::initializer_list<reference_table_schema> schemas)
{
  CUDF_EXPECTS(index < schemas.size() && name == schemas.begin()[index].name,
               "Unexpected reference input table order: " + name);
  reference_schema(table, schemas.begin()[index].types);
}

inline cudf::table_view reference_output(table_with_names const& actual,
                                         std::vector<std::string> const& names,
                                         std::initializer_list<cudf::type_id> types,
                                         std::size_t rows,
                                         bool ordered = true)
{
  CUDF_EXPECTS(actual.column_names().size() == names.size() &&
                 static_cast<std::size_t>(actual.table().num_columns()) == names.size(),
               "Unexpected reference output column count");
  if (ordered) {
    CUDF_EXPECTS(actual.column_names() == names, "Unexpected reference output columns");
  } else {
    for (auto const& name : names) {
      CUDF_EXPECTS(
        std::count(actual.column_names().begin(), actual.column_names().end(), name) == 1,
        "Missing or duplicate reference output column: " + name);
    }
  }
  auto const table = ordered ? actual.table() : actual.select(names);
  reference_schema(table, types);
  CUDF_EXPECTS(static_cast<std::size_t>(table.num_rows()) == rows,
               "Reference output row count mismatch");
  return table;
}

template <typename T>
auto reference_host_copy(T const* data, std::size_t count, cuda::stream_ref stream)
{
  return cudf::detail::make_host_vector(cudf::device_span<T const>{data, count}, stream);
}

inline auto reference_host_strings(cudf::column_view column,
                                   cudf::size_type begin,
                                   cudf::size_type count,
                                   cuda::stream_ref stream)
{
  cudf::strings_column_view strings{column};
  auto const offsets = strings.offsets();
  CUDF_EXPECTS(
    (offsets.type().id() == cudf::type_id::INT32 || offsets.type().id() == cudf::type_id::INT64) &&
      offsets.null_count() == 0,
    "Unexpected reference string offsets");
  auto copy = [&](auto offset_type) {
    auto const host_offsets = reference_host_copy(
      offsets.data<decltype(offset_type)>() + column.offset() + begin, count + 1, stream);
    std::vector<std::string> result(count);
    if (host_offsets.back() == host_offsets.front()) return result;
    // Copy only this batch's character interval, not the entire backing column.
    auto const chars = reference_host_copy(strings.chars_begin(stream) + host_offsets.front(),
                                           host_offsets.back() - host_offsets.front(),
                                           stream);
    for (cudf::size_type i = 0; i < count; ++i) {
      result[i].assign(chars.data() + (host_offsets[i] - host_offsets.front()),
                       host_offsets[i + 1] - host_offsets[i]);
    }
    return result;
  };
  return offsets.type().id() == cudf::type_id::INT32 ? copy(int32_t{}) : copy(int64_t{});
}

// Borrow only the current batch; each copy retains the existing stream and sliced-column offset.
struct reference_batch {
  cudf::table_view const& table;
  cuda::stream_ref stream;
  cudf::size_type begin, count;

  template <typename T>
  auto values(int index) const
  {
    return reference_host_copy(table.column(index).data<T>() + begin, count, stream);
  }

  auto strings(int index) const
  {
    return reference_host_strings(table.column(index), begin, count, stream);
  }
};

template <typename Consume>
void for_reference_batches(cudf::table_view const& table,
                           cuda::stream_ref stream,
                           Consume&& consume)
{
  for (cudf::size_type begin = 0; begin < table.num_rows();) {
    auto const count = std::min<cudf::size_type>(1 << 20, table.num_rows() - begin);
    consume(reference_batch{table, stream, begin, count});
    begin += count;
  }
}
}  // namespace ndsh::detail
