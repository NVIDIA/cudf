/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <memory>
#include <utility>

namespace cudf::interop {

/**
 * @brief A wrapper around ArrowDeviceArray data used for flexible lifetime management.
 *
 * The arrow_array_container is the core object for storing and managing the
 * lifetime of arrow data in libcudf. Ultimately, data is always owned by this
 * container type regardless of the source. There are a few important cases to
 * consider:
 * 1. We are importing third-party Arrow device data: The data is moved
 *    directly into the container.
 * 2. We are converting a cudf arrow/table: We construct an ArrowDeviceArray
 *    that owns the data formerly owned by the cudf object and then fall back
 *    to case 1 with the new array.
 * 3. We are importing third-party Arrow host data: We construct a cudf
 *    column/table from the Arrow data and then fall back to case 2.
 *
 * Any export of arrow_column or arrow_table to an arrow array produces
 * an ArrowDeviceArray whose private_data is an instance of a cudf-internal
 * type (the ArrowArrayPrivateData struct) that also holds a shared pointer to
 * this container, ensuring shared ownership and of the data and compatible
 * management of its lifetime. All the release semantics boil down to a simple
 * deletion of a shared_ptr, so no actually freeing needs to be done manually.
 * The shared_ptr's reference counting is sufficient across all use cases. The
 * original array's release callback is called when the container is
 * destructed, which in practice means when all shared references to the
 * container are gone.
 */
struct arrow_array_container {
  arrow_array_container() = default;

  template <typename T>
  arrow_array_container(ArrowSchema&& schema_,
                        T input_,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
  {
    auto output = cudf::to_arrow_device(std::move(input_), stream, mr);
    ArrowSchemaMove(&schema_, &schema);
    ArrowDeviceArrayMove(output.get(), &owner);
  }

  arrow_array_container(ArrowSchema&& schema_,
                        ArrowDeviceArray&& input_,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
  {
    switch (input_.device_type) {
      case ARROW_DEVICE_CUDA:
      case ARROW_DEVICE_CUDA_HOST:
      case ARROW_DEVICE_CUDA_MANAGED: {
        ArrowSchemaMove(&schema_, &schema);
        ArrowDeviceArrayMove(&input_, &owner);
        break;
      }
      default: CUDF_FAIL("Unsupported ArrowDeviceArray type", std::runtime_error);
    }
  }
  ~arrow_array_container()
  {
    if (owner.array.release != nullptr) { ArrowArrayRelease(&owner.array); }
  }

  ArrowDeviceArray owner{};  //< ArrowDeviceArray that owns the data
  ArrowSchema schema{};      //< ArrowSchema that describes the data
};

cudf::column_metadata get_column_metadata(cudf::column_view const& input)
{
  cudf::column_metadata meta{};
  std::transform(
    input.child_begin(), input.child_end(), std::back_inserter(meta.children_meta), [](auto& cv) {
      return get_column_metadata(cv);
    });
  return meta;
}

std::vector<cudf::column_metadata> get_table_metadata(cudf::table_view const& input)
{
  auto meta = std::vector<cudf::column_metadata>{};
  std::transform(input.begin(), input.end(), std::back_inserter(meta), [](auto& cv) {
    return get_column_metadata(cv);
  });
  return meta;
}

namespace {

/**
 * @brief Private data for an ArrowArray that contains a struct array.
 *
 * This struct is used to manage the lifetimes of the children of a struct array.
 */
struct ArrowArrayPrivateData {
  std::shared_ptr<arrow_array_container> parent;
  std::vector<std::unique_ptr<ArrowArray>> children;
  std::vector<ArrowArray*> children_raw;
};

/**
 * @brief Release callback for an ArrowArray that contains a struct array.
 *
 * This function is called when the ArrowArray is released. It releases all of the children of the
 * struct array.
 *
 * @param array The ArrowArray to release
 */
void ArrayReleaseCallback(ArrowArray* array)
{
  auto private_data = reinterpret_cast<ArrowArrayPrivateData*>(array->private_data);
  for (auto& child : private_data->children) {
    child->release(child.get());
  }
  delete private_data;
  array->release = nullptr;
}

/**
 * @brief Copy an ArrowArray.
 *
 * This function shallow copies an ArrowArray and all of its children. It is
 * used to export cudf arrow objects to user-provided ArrowDeviceArrays.
 *
 * The @p input must be the ``owner`` member of the @p container OR a child of
 * the ``owner`` member of the @p container. If not, the behavior is undefined.
 *
 * @param output The ArrowArray to copy to
 * @param input The ArrowArray to copy from
 * @param container The container that owns the data
 */
void copy_array(ArrowArray* output,
                ArrowArray const* input,
                std::shared_ptr<arrow_array_container> container)
{
  auto private_data  = new ArrowArrayPrivateData{container};
  output->length     = input->length;
  output->null_count = input->null_count;
  output->offset     = input->offset;
  output->n_buffers  = input->n_buffers;
  output->n_children = input->n_children;
  output->buffers    = input->buffers;

  if (input->n_children > 0) {
    private_data->children_raw.resize(input->n_children);
    for (auto i = 0; i < input->n_children; ++i) {
      private_data->children.push_back(std::make_unique<ArrowArray>());
      private_data->children_raw[i] = private_data->children.back().get();
      copy_array(private_data->children_raw[i], input->children[i], container);
    }
  }
  output->children     = private_data->children_raw.data();
  output->dictionary   = input->dictionary;
  output->release      = ArrayReleaseCallback;
  output->private_data = private_data;
}

template <typename T>
void arrow_obj_to_arrow(T& obj,
                        std::shared_ptr<arrow_array_container> container,
                        ArrowDeviceArray* output,
                        ArrowDeviceType device_type,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  switch (device_type) {
    case ARROW_DEVICE_CUDA:
    case ARROW_DEVICE_CUDA_HOST:
    case ARROW_DEVICE_CUDA_MANAGED: {
      auto& device_arr = container->owner;
      copy_array(&output->array, &device_arr.array, container);
      output->device_id = device_arr.device_id;
      // We can reuse the sync event by reference from the input. The
      // destruction of that event is managed by the destruction of
      // the underlying ArrowDeviceArray of this table.
      output->sync_event  = device_arr.sync_event;
      output->device_type = device_type;
      break;
    }
    case ARROW_DEVICE_CPU: {
      auto out = cudf::to_arrow_host(obj.view(), stream, mr);
      ArrowArrayMove(&out->array, &output->array);
      output->device_id   = -1;
      output->sync_event  = nullptr;
      output->device_type = ARROW_DEVICE_CPU;
      break;
    }
    default: throw std::runtime_error("Unsupported ArrowDeviceArray type");
  }
}

}  // namespace

arrow_column::arrow_column(cudf::column&& input,
                           column_metadata const& metadata,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
  : container{[&] {
      auto table_meta = std::vector{metadata};
      auto tv         = cudf::table_view{{input.view()}};
      auto schema     = cudf::to_arrow_schema(tv, table_meta);
      return std::make_shared<arrow_array_container>(
        std::move(*schema->children[0]), std::move(input), stream, mr);
    }()}
{
  auto tmp     = from_arrow_device_column(&container->schema, &container->owner, stream, mr);
  view_columns = std::move(tmp.get_deleter().owned_mem_);
  cached_view  = *tmp;
}

arrow_column::arrow_column(ArrowSchema&& schema,
                           ArrowDeviceArray&& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  switch (input.device_type) {
    case ARROW_DEVICE_CPU: {
      auto col        = from_arrow_host_column(&schema, &input, stream, mr);
      auto tmp_column = arrow_column(std::move(*col), get_column_metadata(col->view()), stream, mr);
      container       = tmp_column.container;
      // Should always be non-null unless we're in some odd multithreaded
      // context but best to be safe.
      if (input.array.release != nullptr) { ArrowArrayRelease(&input.array); }
      break;
    }
    default:
      container =
        std::make_shared<arrow_array_container>(std::move(schema), std::move(input), stream, mr);
  }
  auto tmp     = from_arrow_device_column(&container->schema, &container->owner, stream, mr);
  view_columns = std::move(tmp.get_deleter().owned_mem_);
  cached_view  = *tmp;
}

arrow_column::arrow_column(ArrowSchema&& schema,
                           ArrowArray&& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  ArrowDeviceArray arr{.array = {}, .device_id = -1, .device_type = ARROW_DEVICE_CPU};
  ArrowArrayMove(&input, &arr.array);
  auto tmp     = arrow_column(std::move(schema), std::move(arr), stream, mr);
  container    = tmp.container;
  view_columns = std::move(tmp.view_columns);
  cached_view  = tmp.cached_view;
}

arrow_column::arrow_column(ArrowArrayStream&& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto col     = from_arrow_stream_column(&input, stream, mr);
  auto tmp     = arrow_column(std::move(*col), get_column_metadata(col->view()), stream, mr);
  container    = tmp.container;
  view_columns = std::move(tmp.view_columns);
  cached_view  = tmp.cached_view;
}

void arrow_column::to_arrow_schema(ArrowSchema* output,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr) const
{
  NANOARROW_THROW_NOT_OK(ArrowSchemaDeepCopy(&container->schema, output));
}

void arrow_column::to_arrow(ArrowDeviceArray* output,
                            ArrowDeviceType device_type,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const
{
  arrow_obj_to_arrow(*this, container, output, device_type, stream, mr);
}

column_view arrow_column::view() const { return cached_view; }

arrow_table::arrow_table(cudf::table&& input,
                         std::span<column_metadata const> metadata,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
  : container{[&]() {
      auto schema = cudf::to_arrow_schema(input.view(), metadata);
      return std::make_shared<arrow_array_container>(
        std::move(*schema), std::move(input), stream, mr);
    }()}
{
  auto tmp     = from_arrow_device(&container->schema, &container->owner, stream, mr);
  view_columns = std::move(tmp.get_deleter().owned_mem_);
  cached_view  = *tmp;
}

arrow_table::arrow_table(ArrowSchema&& schema,
                         ArrowDeviceArray&& input,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  switch (input.device_type) {
    case ARROW_DEVICE_CPU: {
      // I'm not sure if there is a more efficient approach than doing this
      // back-and-forth conversion without writing a lot of bespoke logic. I
      // suspect that the overhead of the memory copies will dwarf any extra
      // work here, but it's worth benchmarking to be sure.
      auto tbl       = from_arrow_host(&schema, &input, stream, mr);
      auto tmp_table = arrow_table(std::move(*tbl), get_table_metadata(tbl->view()), stream, mr);
      container      = tmp_table.container;
      if (input.array.release != nullptr) { ArrowArrayRelease(&input.array); }
      break;
    }
    default:
      container =
        std::make_shared<arrow_array_container>(std::move(schema), std::move(input), stream, mr);
  }
  auto tmp     = from_arrow_device(&container->schema, &container->owner, stream, mr);
  view_columns = std::move(tmp.get_deleter().owned_mem_);
  cached_view  = *tmp;
}

arrow_table::arrow_table(ArrowSchema&& schema,
                         ArrowArray&& input,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  ArrowDeviceArray arr{.array = {}, .device_id = -1, .device_type = ARROW_DEVICE_CPU};
  ArrowArrayMove(&input, &arr.array);
  auto tmp     = arrow_table(std::move(schema), std::move(arr), stream, mr);
  container    = tmp.container;
  view_columns = std::move(tmp.view_columns);
  cached_view  = tmp.cached_view;
}

arrow_table::arrow_table(ArrowArrayStream&& input,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  auto tbl     = from_arrow_stream(&input, stream, mr);
  auto tmp     = arrow_table(std::move(*tbl), get_table_metadata(tbl->view()), stream, mr);
  container    = tmp.container;
  view_columns = std::move(tmp.view_columns);
  cached_view  = tmp.cached_view;
}

void arrow_table::to_arrow_schema(ArrowSchema* output,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr) const
{
  NANOARROW_THROW_NOT_OK(ArrowSchemaDeepCopy(&container->schema, output));
}

void arrow_table::to_arrow(ArrowDeviceArray* output,
                           ArrowDeviceType device_type,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) const
{
  arrow_obj_to_arrow(*this, container, output, device_type, stream, mr);
}

table_view arrow_table::view() const { return cached_view; }

}  // namespace cudf::interop
