/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arrow_utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {

namespace {

std::unique_ptr<column> make_empty_column_from_schema(ArrowSchema const* schema,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  ArrowSchemaView schema_view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, nullptr));

  auto const type{arrow_to_cudf_type(&schema_view)};
  switch (type.id()) {
    case type_id::EMPTY: {
      return std::make_unique<column>(
        data_type(type_id::EMPTY), 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
    }
    case type_id::LIST: {
      return cudf::make_lists_column(0,
                                     cudf::make_empty_column(data_type{type_id::INT32}),
                                     make_empty_column_from_schema(schema->children[0], stream, mr),
                                     0,
                                     {});
    }
    case type_id::STRUCT: {
      std::vector<std::unique_ptr<column>> child_columns;
      child_columns.reserve(schema->n_children);
      std::transform(
        schema->children,
        schema->children + schema->n_children,
        std::back_inserter(child_columns),
        [&](auto const& child) { return make_empty_column_from_schema(child, stream, mr); });
      return cudf::make_structs_column(0, std::move(child_columns), 0, {}, stream, mr);
    }
    default: {
      return cudf::make_empty_column(type);
    }
  }
}

}  // namespace

std::unique_ptr<table> from_arrow_stream(ArrowArrayStream* input,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input != nullptr, "input ArrowArrayStream must not be NULL", std::invalid_argument);

  // Potential future optimization: Since the from_arrow API accepts an
  // ArrowSchema we're allocating one here instead of using a view, which we
  // could avoid with a different underlying implementation.
  ArrowSchema schema;
  NANOARROW_THROW_NOT_OK(ArrowArrayStreamGetSchema(input, &schema, nullptr));

  std::vector<std::unique_ptr<cudf::table>> chunks;
  // Keep each input chunk alive until the stream has executed the host-to-device copies enqueued
  // by `from_arrow`. Those copies use `cudaMemcpyBatchAsync` with `cudaMemcpySrcAccessOrderStream`,
  // which defers reading the host source until the stream reaches the copy.
  std::vector<nanoarrow::UniqueArray> sources;
  while (true) {
    nanoarrow::UniqueArray chunk;
    NANOARROW_THROW_NOT_OK(ArrowArrayStreamGetNext(input, chunk.get(), nullptr));
    if (chunk->release == nullptr) { break; }
    sources.push_back(std::move(chunk));
    chunks.push_back(from_arrow(&schema, sources.back().get(), stream, mr));
  }
  input->release(input);

  if (chunks.empty()) {
    if (schema.n_children == 0) {
      schema.release(&schema);
      return std::make_unique<cudf::table>();
    }

    // If there are no chunks but the schema has children, we need to construct a suitable empty
    // table.
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(chunks.size());
    std::transform(
      schema.children,
      schema.children + schema.n_children,
      std::back_inserter(columns),
      [&](auto const& child) { return make_empty_column_from_schema(child, stream, mr); });
    schema.release(&schema);
    return std::make_unique<cudf::table>(std::move(columns));
  }

  schema.release(&schema);

  // Ensure all host-to-device copies enqueued above have completed before `sources` releases the
  // host-side Arrow buffers.
  stream.synchronize();

  if (chunks.size() == 1) { return std::move(chunks[0]); }
  auto chunk_views = std::vector<table_view>{};
  chunk_views.reserve(chunks.size());
  std::transform(
    chunks.begin(), chunks.end(), std::back_inserter(chunk_views), [](auto const& chunk) {
      return chunk->view();
    });
  return cudf::detail::concatenate(chunk_views, stream, mr);
}

std::unique_ptr<column> from_arrow_stream_column(ArrowArrayStream* input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input != nullptr, "input ArrowArrayStream must not be NULL", std::invalid_argument);

  // Potential future optimization: Since the from_arrow API accepts an
  // ArrowSchema we're allocating one here instead of using a view, which we
  // could avoid with a different underlying implementation.
  ArrowSchema schema;
  NANOARROW_THROW_NOT_OK(ArrowArrayStreamGetSchema(input, &schema, nullptr));

  std::vector<std::unique_ptr<cudf::column>> chunks;
  // Keep each input chunk alive until the stream has executed the host-to-device copies enqueued
  // by `from_arrow_column`. Those copies use `cudaMemcpyBatchAsync` with
  // `cudaMemcpySrcAccessOrderStream`, which defers reading the host source until the stream reaches
  // the copy.
  std::vector<nanoarrow::UniqueArray> sources;
  while (true) {
    nanoarrow::UniqueArray chunk;
    NANOARROW_THROW_NOT_OK(ArrowArrayStreamGetNext(input, chunk.get(), nullptr));
    if (chunk->release == nullptr) { break; }
    sources.push_back(std::move(chunk));
    chunks.push_back(from_arrow_column(&schema, sources.back().get(), stream, mr));
  }
  input->release(input);

  if (chunks.empty()) {
    auto empty_column = make_empty_column_from_schema(&schema, stream, mr);
    schema.release(&schema);
    return empty_column;
  }

  schema.release(&schema);

  // Ensure all host-to-device copies enqueued above have completed before `sources` releases the
  // host-side Arrow buffers.
  stream.synchronize();

  if (chunks.size() == 1) { return std::move(chunks[0]); }
  auto chunk_views = std::vector<column_view>{};
  chunk_views.reserve(chunks.size());
  std::transform(
    chunks.begin(), chunks.end(), std::back_inserter(chunk_views), [](auto const& chunk) {
      return chunk->view();
    });
  return cudf::detail::concatenate(chunk_views, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> from_arrow_stream(ArrowArrayStream* input,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_arrow_stream(input, stream, mr);
}

std::unique_ptr<column> from_arrow_stream_column(ArrowArrayStream* input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_arrow_stream_column(input, stream, mr);
}

}  // namespace cudf
