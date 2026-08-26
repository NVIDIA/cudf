/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

/**
 * @file parquet_inspect_utils.hpp
 * @brief Utilities for `parquet_inspect` example
 */

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param pool Whether to use a pool memory resource.
 * @return Memory resource instance
 */
cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource(bool is_pool_used);

/**
 * @brief Reads parquet metadata (FileMetaData struct) from a file
 *
 * @param input_filepath Path to the input parquet file
 *
 * @return A tuple containing the parquet metadata and a boolean indicating if the file contains a
 * page index
 */
std::tuple<cudf::io::parquet::FileMetaData, bool> read_parquet_file_metadata(
  std::string_view input_filepath);

/**
 * @brief Writes row group metadata to a parquet file
 *
 * @param metadata Parquet file metadata
 * @param output_filepath Path to the output file
 * @param stream CUDA stream
 */
void write_rowgroup_metadata(cudf::io::parquet::FileMetaData const& metadata,
                             std::string const& output_filepath,
                             cuda::stream_ref stream);

/**
 * @brief Writes page metadata to a parquet file
 *
 * @param metadata Parquet file metadata
 * @param output_filepath Path to the output file
 * @param stream CUDA stream
 */
void write_page_metadata(cudf::io::parquet::FileMetaData const& metadata,
                         std::string const& output_filepath,
                         cuda::stream_ref stream);
