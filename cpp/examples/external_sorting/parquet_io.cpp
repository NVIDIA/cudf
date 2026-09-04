/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "parquet_io.hpp"

#include "cudf/io/types.hpp"

#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <iostream>

namespace cudf {
namespace examples {

std::unique_ptr<cudf::table> read_parquet_file(std::string const& filepath,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  std::unique_ptr<cudf::table> table;
  return cudf::io::read_parquet(
           cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath)), stream)
    .tbl;
}

void write_parquet_file(std::string const& filepath,
                        cudf::table_view table_view,
                        rmm::cuda_stream_view stream)
{
  auto sink_info = cudf::io::sink_info(filepath);
  auto builder   = cudf::io::parquet_writer_options::builder(sink_info, table_view);

  // Create metadata for better compression
  auto table_metadata = cudf::io::table_input_metadata{table_view};
  for (cudf::size_type i = 0; i < table_view.num_columns(); ++i) {
    table_metadata.column_metadata[i].set_name("column_" + std::to_string(i));
  }

  auto options = builder.metadata(table_metadata).compression(cudf::io::compression_type::SNAPPY);
  cudf::io::write_parquet(options.build(), stream);
  stream.synchronize();
}

std::string construct_file_path(std::string const& input_dir, int filenum)
{
  std::string filepath = input_dir + "/part." + std::to_string(filenum) + ".parquet";
  return filepath;
}

}  // namespace examples
}  // namespace cudf
