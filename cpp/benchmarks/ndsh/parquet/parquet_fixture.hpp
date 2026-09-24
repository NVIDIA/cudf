/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Exercise Parquet predicate pushdown on synthetic NDS-H query inputs.
 * Uses an in-memory encoded fixture so correctness checks need no persistent dataset.
 */

#pragma once

#include "parquet_io.hpp"

#include <cudf/utilities/default_stream.hpp>

namespace ndsh {

/** Round-trip a synthetic input through the real Parquet reader, including predicate pushdown. */
inline std::unique_ptr<table_with_names> read_parquet_fixture(
  cudf::table_view input,
  std::vector<std::string> const& columns,
  std::unique_ptr<cudf::ast::operation> const& predicate)
{
  auto table = std::make_unique<cudf::table>(input);
  cuio_source_sink_pair source{io_type::HOST_BUFFER};
  write_to_parquet_device_buffer(table, columns, source);
  auto result = read_parquet(source.make_source_info(), columns, predicate);
  // The temporary encoded source must outlive any asynchronous reader work.
  cudf::get_default_stream().sync();
  return result;
}

}  // namespace ndsh
