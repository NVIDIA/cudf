/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/parquet/compact_protocol_reader.hpp"
#include "io/parquet/compact_protocol_writer.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/parquet_footer.hpp>
#include <cudf/utilities/span.hpp>

#include <cstdint>
#include <vector>

namespace cudf::io::parquet::experimental {

FileMetaData read_parquet_footer_bytes(std::span<uint8_t const> footer_bytes,
                                       thrift_mismatch_policy mode)
{
  CUDF_FUNC_RANGE();
  detail::CompactProtocolReader reader{footer_bytes.data(), footer_bytes.size(), mode};
  FileMetaData metadata;
  reader.read(&metadata);
  CUDF_EXPECTS(not reader.overread(), detail::CompactProtocolReader::overread_message);
  return metadata;
}

std::vector<uint8_t> write_parquet_footer_bytes(FileMetaData const& metadata)
{
  CUDF_FUNC_RANGE();
  std::vector<uint8_t> out;
  detail::CompactProtocolWriter writer{&out};
  writer.write(metadata);
  return out;
}

}  // namespace cudf::io::parquet::experimental
