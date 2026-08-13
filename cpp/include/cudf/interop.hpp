/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/resource_ref.hpp>

#include <span>
#include <utility>

struct DLManagedTensor;

struct ArrowDeviceArray;

struct ArrowSchema;

struct ArrowArray;

struct ArrowArrayStream;

///@cond
// These are types from arrow that we are forward declaring for our API to
// avoid needing to include nanoarrow headers.
typedef int32_t ArrowDeviceType;  // NOLINT

#define ARROW_DEVICE_CUDA 2  // NOLINT
///@endcond

/**
 * @file
 * @brief APIs for converting between cudf tables/columns and DLPack tensors or Apache Arrow
 * arrays.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup interop_dlpack
 * @{
 */

/**
 * @brief Convert a DLPack DLTensor into a cudf table
 *
 * The `device_type` of the DLTensor must be `kDLCPU`, `kDLCuda`, or
 * `kDLCUDAHost`, and `device_id` must match the current device. The `ndim`
 * must be set to 1 or 2. The `dtype` must have 1 lane and the bitsize must
 * match a supported `cudf::data_type`.
 *
 * @note The managed tensor is not deleted by this function.
 *
 * @throw cudf::logic_error if the any of the DLTensor fields are unsupported
 *
 * @param managed_tensor a 1D or 2D column-major (Fortran order) tensor
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @return Table with a copy of the tensor data
 */
std::unique_ptr<table> from_dlpack(
  DLManagedTensor const* managed_tensor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convert a cudf table into a DLPack DLTensor
 *
 * All columns must have the same data type and this type must be numeric. The
 * columns may be nullable, but the null count must be zero. If the input table
 * is empty or has zero rows, the result will be nullptr.
 *
 * @note The `deleter` method of the returned `DLManagedTensor` must be used to
 * free the memory allocated for the tensor.
 *
 * @throw cudf::logic_error if the data types are not equal or not numeric,
 * or if any of columns have non-zero null count
 *
 * @param input Table to convert to DLPack
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned DLPack tensor's device memory
 *
 * @return 1D or 2D DLPack tensor with a copy of the table data, or nullptr
 */
DLManagedTensor* to_dlpack(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

/**
 * @addtogroup interop_arrow
 * @{
 */

/**
 * @brief Detailed metadata information for arrow array.
 *
 * This contains attributes of the column or type not natively supported by cudf.
 */
struct column_metadata {
  std::string name;                            ///< Name of the column
  std::string timezone;                        ///< Timezone of the column
  std::optional<int32_t> precision;            ///< Resulting decimal precision of the column
  std::vector<column_metadata> children_meta;  ///< Metadata of children of the column

  /**
   * @brief Construct a new column metadata object
   *
   * @param _name Name of the column
   */
  column_metadata(std::string _name) : name(std::move(_name)) {}
  column_metadata() = default;
};

/**
 * @brief typedef for a unique_ptr to an ArrowSchema with custom deleter
 *
 */
using unique_schema_t = std::unique_ptr<ArrowSchema, void (*)(ArrowSchema*)>;

/**
 * @brief typedef for a unique_ptr to an ArrowDeviceArray with a custom deleter
 *
 */
using unique_device_array_t = std::unique_ptr<ArrowDeviceArray, void (*)(ArrowDeviceArray*)>;

/**
 * @brief typedef for a vector of owning columns, used for conversion from ArrowDeviceArray
 *
 */
using owned_columns_t = std::vector<std::unique_ptr<cudf::column>>;

/**
 * @brief functor for a custom deleter to a unique_ptr of table_view
 *
 * When converting from an ArrowDeviceArray, there are cases where data can't
 * be zero-copy (i.e. bools or non-UINT32 dictionary indices). This custom deleter
 * is used to maintain ownership over the data allocated since a `cudf::table_view`
 * doesn't hold ownership.
 */
template <typename ViewType>
struct custom_view_deleter {
  /**
   * @brief Construct a new custom view deleter object
   *
   * @param owned Vector of owning columns
   */
  explicit custom_view_deleter(owned_columns_t&& owned) : owned_mem_{std::move(owned)} {}

  /**
   * @brief operator to delete the unique_ptr
   *
   * @param ptr Pointer to the object to be deleted
   */
  void operator()(ViewType* ptr) const { delete ptr; }

  owned_columns_t owned_mem_;  ///< Owned columns that must be deleted.
};

/**
 * @brief typedef for a unique_ptr to a `cudf::table_view` with custom deleter
 *
 */
using unique_table_view_t =
  std::unique_ptr<cudf::table_view, custom_view_deleter<cudf::table_view>>;

/**
 * @brief typedef for a unique_ptr to a `cudf::column_view` with custom deleter
 *
 */
using unique_column_view_t =
  std::unique_ptr<cudf::column_view, custom_view_deleter<cudf::column_view>>;

namespace interop {

struct arrow_array_container;

/**
 * @brief Helper function to generate empty column metadata (column with no
 * name) for arrow conversion.
 *
 * This function is helpful for internal conversions between host and device
 * data using existing arrow functions. It is also convenient for external
 * usage of the libcudf Arrow APIs to produce the canonical mapping from cudf
 * column names to Arrow column names (i.e. empty names with appropriate
 * nesting).
 *
 * @param input The column to generate metadata for
 * @return The metadata for the column
 */
cudf::column_metadata get_column_metadata(cudf::column_view const& input);

/**
 * @brief Helper function to generate empty table metadata (all columns with no
 * names) for arrow conversion.
 *
 * This function is helpful for internal conversions between host and device
 * data using existing arrow functions. It is also convenient for external
 * usage of the libcudf Arrow APIs to produce the canonical mapping from cudf
 * column names to Arrow column names (i.e. empty names with appropriate
 * nesting).
 *
 * @param input The table to generate metadata for
 * @return The metadata for the table
 */
std::vector<cudf::column_metadata> get_table_metadata(cudf::table_view const& input);

/**
 * @brief A standard interchange medium for ArrowDeviceArray data in cudf.
 *
 * This class provides a way to work with ArrowDeviceArray data in cudf without
 * sacrificing the APIs expected of a cudf column. On the other end, it
 * provides the shared lifetime management expected by arrow consumers rather
 * than the single-owner mechanism of cudf::column.
 */
class arrow_column {
 public:
  /**
   * @brief Construct a new arrow column object
   *
   * The input column will be moved into the arrow_column, so it is no longer
   * suitable for use afterwards.
   *
   * @param input cudf column to convert to arrow
   * @param metadata Column metadata for the column
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  arrow_column(cudf::column&& input,
               column_metadata const& metadata,
               rmm::cuda_stream_view stream      = cudf::get_default_stream(),
               rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct a new arrow column object
   *
   * The input array will be moved into the arrow_column, so it is no longer
   * suitable for use afterwards. For consistency, this is done even if the
   * source array points to host data.
   *
   * @param schema Arrow schema for the column
   * @param input ArrowDeviceArray data for the column
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  arrow_column(ArrowSchema&& schema,
               ArrowDeviceArray&& input,
               rmm::cuda_stream_view stream      = cudf::get_default_stream(),
               rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct a new arrow column object
   *
   * The input array will be released, so it is no longer suitable for use
   * afterwards. This is done for consistency with other constructors of arrow_table even though the
   * source data is always host data.
   *
   * @param schema Arrow schema for the column
   * @param input ArrowArray data for the column
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  arrow_column(ArrowSchema&& schema,
               ArrowArray&& input,
               rmm::cuda_stream_view stream      = cudf::get_default_stream(),
               rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct a new arrow column object
   *
   * The stream will be released after the column is created, so it is no longer
   * suitable for use afterwards. This is done for consistency with other constructors of
   * arrow_column even though the source data is always host data.
   *
   * @param input ArrowArrayStream data for the column
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  arrow_column(ArrowArrayStream&& input,
               rmm::cuda_stream_view stream      = cudf::get_default_stream(),
               rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Convert the column to an ArrowSchema
   *
   * The resulting schema is a deep copy of the arrow_column's schema and is
   * not tied to its lifetime.
   *
   * @param output ArrowSchema to populate with the column's schema
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  void to_arrow_schema(
    ArrowSchema* output,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Convert the column to an ArrowDeviceArray
   *
   * @param output ArrowDeviceArray to populate with the column's data
   * @param device_type ArrowDeviceType to set on the output
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  void to_arrow(ArrowDeviceArray* output,
                ArrowDeviceType device_type       = ARROW_DEVICE_CUDA,
                rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Get a view of the column data
   *
   * @return A view of the column data
   */
  [[nodiscard]] column_view view() const;

 private:
  std::shared_ptr<arrow_array_container>
    container;  ///< Shared pointer to container for the ArrowDeviceArray data; shared_ptr allows
                ///< re-export via to_arrow
  owned_columns_t view_columns;  ///< Cached view that manages ownership of non-view-only data.
  column_view cached_view;
};

/**
 * @brief A standard interchange medium for ArrowDeviceArray data in cudf.
 *
 * This class provides a way to work with ArrowDeviceArray data in cudf without
 * sacrificing the APIs expected of a cudf table. On the other end, it
 * provides the shared lifetime management expected by arrow consumers rather
 * than the single-owner mechanism of cudf::table.
 */
class arrow_table {
 public:
  /**
   * @brief Construct a new arrow table object
   *
   * The input table will be moved into the arrow_table, so it is no longer
   * suitable for use afterwards.
   *
   * @param input cudf table to convert to arrow
   * @param metadata The hierarchy of names of columns and children
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  arrow_table(cudf::table&& input,
              std::span<column_metadata const> metadata,
              rmm::cuda_stream_view stream      = cudf::get_default_stream(),
              rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct a new arrow table object
   *
   * The input array will be moved into the arrow_table, so it is no longer
   * suitable for use afterwards. For consistency, this is done even if the
   * source array points to host data.
   *
   * @param schema Arrow schema for the table
   * @param input ArrowDeviceArray data for the table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  arrow_table(ArrowSchema&& schema,
              ArrowDeviceArray&& input,
              rmm::cuda_stream_view stream      = cudf::get_default_stream(),
              rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct a new arrow table object
   *
   * The input array will be released, so it is no longer suitable for use
   * afterwards. This is done for consistency with other constructors of arrow_table even though the
   * source data is always host data.
   *
   * @param schema Arrow schema for the table
   * @param input ArrowArray data for the table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  arrow_table(ArrowSchema&& schema,
              ArrowArray&& input,
              rmm::cuda_stream_view stream      = cudf::get_default_stream(),
              rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct a new arrow table object
   *
   * The stream will be released after the table is created, so it is no longer
   * suitable for use afterwards. This is done for consistency with other constructors of
   * arrow_table even though the source data is always host data.
   *
   * @param input ArrowArrayStream data for the table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  arrow_table(ArrowArrayStream&& input,
              rmm::cuda_stream_view stream      = cudf::get_default_stream(),
              rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Convert the table to an ArrowSchema
   *
   * The resulting schema is a deep copy of the arrow_column's schema and is
   * not tied to its lifetime.
   *
   * @param output ArrowSchema to populate with the table's schema
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  void to_arrow_schema(
    ArrowSchema* output,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Convert the table to an ArrowDeviceArray
   *
   * @param output ArrowDeviceArray to populate with the table's data
   * @param device_type ArrowDeviceType to set on the output
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used for any allocations during conversion
   */
  void to_arrow(ArrowDeviceArray* output,
                ArrowDeviceType device_type       = ARROW_DEVICE_CUDA,
                rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Get a view of the table data
   *
   * @return A view of the table data
   */
  [[nodiscard]] table_view view() const;

 private:
  std::shared_ptr<arrow_array_container>
    container;  ///< Shared pointer to container for the ArrowDeviceArray data; shared_ptr allows
                ///< re-export via to_arrow
  owned_columns_t view_columns;  ///< Cached view that manages ownership of non-view-only data.
  table_view cached_view;
};

}  // namespace interop

/**
 * @brief Create ArrowSchema from cudf table and metadata
 *
 * Populates and returns an ArrowSchema C struct using a table and metadata.
 *
 * @note For decimals, since the precision is not stored for them in libcudf,
 * decimals will be converted to an Arrow decimal128 which has the widest precision that cudf
 * decimal type supports. For example, `numeric::decimal32` will be converted to Arrow decimal128
 * with the precision of 9 which is the maximum precision for 32-bit types. Similarly,
 * `numeric::decimal128` will be converted to Arrow decimal128 with the precision of 38.
 *
 * @param input Table to create a schema from
 * @param metadata Contains the hierarchy of names of columns and children
 * @return ArrowSchema generated from `input`
 */
unique_schema_t to_arrow_schema(cudf::table_view const& input,
                                std::span<column_metadata const> metadata);

/**
 * @brief Create `ArrowDeviceArray` from cudf table and metadata
 *
 * Populates the C struct ArrowDeviceArray without performing copies if possible.
 * This maintains the data on the GPU device and gives ownership of the table
 * and its buffers to the ArrowDeviceArray struct.
 *
 * After calling this function, the release callback on the returned ArrowDeviceArray
 * must be called to clean up the memory.
 *
 * @note For decimals, since the precision is not stored for them in libcudf
 * it will be converted to an Arrow decimal128 with the widest-precision the cudf decimal type
 * supports. For example, numeric::decimal32 will be converted to Arrow decimal128 of the precision
 * 9 which is the maximum precision for 32-bit types. Similarly, numeric::decimal128 will be
 * converted to Arrow decimal128 of the precision 38.
 *
 * @note Copies will be performed in the cases where cudf differs from Arrow
 * such as in the representation of bools (Arrow uses a bitmap, cudf uses 1-byte per value).
 *
 * @param table Input table, ownership of the data will be moved to the result
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for any allocations during conversion
 * @return ArrowDeviceArray which will have ownership of the GPU data, consumer must call release
 */
unique_device_array_t to_arrow_device(
  cudf::table&& table,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `ArrowDeviceArray` from cudf column and metadata
 *
 * Populates the C struct ArrowDeviceArray without performing copies if possible.
 * This maintains the data on the GPU device and gives ownership of the table
 * and its buffers to the ArrowDeviceArray struct.
 *
 * After calling this function, the release callback on the returned ArrowDeviceArray
 * must be called to clean up the memory.
 *
 * @note For decimals, since the precision is not stored for them in libcudf
 * it will be converted to an Arrow decimal128 with the widest-precision the cudf decimal type
 * supports. For example, numeric::decimal32 will be converted to Arrow decimal128 of the precision
 * 9 which is the maximum precision for 32-bit types. Similar, numeric::decimal128 will be
 * converted to Arrow decimal128 of the precision 38.
 *
 * @note Copies will be performed in the cases where cudf differs from Arrow such as
 * in the representation of bools (Arrow uses a bitmap, cudf uses 1 byte per value).
 *
 * @param col Input column, ownership of the data will be moved to the result
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for any allocations during conversion
 * @return ArrowDeviceArray which will have ownership of the GPU data
 */
unique_device_array_t to_arrow_device(
  cudf::column&& col,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `ArrowDeviceArray` from a table view
 *
 * Populates the C struct ArrowDeviceArray performing copies only if necessary.
 * This wraps the data on the GPU device and gives a view of the table data
 * to the ArrowDeviceArray struct. If the caller frees the data referenced by
 * the table_view, using the returned object results in undefined behavior.
 *
 * After calling this function, the release callback on the returned ArrowDeviceArray
 * must be called to clean up any memory created during conversion.
 *
 * @note For decimals, since the precision is not stored for them in libcudf
 * it will be converted to an Arrow decimal128 with the widest-precision the cudf decimal type
 * supports. For example, numeric::decimal32 will be converted to Arrow decimal128 of the precision
 * 9 which is the maximum precision for 32-bit types. Similarly, numeric::decimal128 will be
 * converted to Arrow decimal128 of the precision 38.
 *
 * Copies will be performed in the cases where cudf differs from Arrow:
 * - BOOL8: Arrow uses a bitmap and cudf uses 1 byte per value
 * - DECIMAL32 and DECIMAL64: Converted to Arrow decimal128
 * - STRING: Arrow expects a single value int32 offset child array for empty strings columns
 *
 * @param table Input table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for any allocations during conversion
 * @return ArrowDeviceArray which will have ownership of any copied data
 */
unique_device_array_t to_arrow_device(
  cudf::table_view const& table,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `ArrowDeviceArray` from a column view
 *
 * Populates the C struct ArrowDeviceArray performing copies only if necessary.
 * This wraps the data on the GPU device and gives a view of the column data
 * to the ArrowDeviceArray struct. If the caller frees the data referenced by
 * the column_view, using the returned object results in undefined behavior.
 *
 * After calling this function, the release callback on the returned ArrowDeviceArray
 * must be called to clean up any memory created during conversion.
 *
 * @note For decimals, since the precision is not stored for them in libcudf
 * it will be converted to an Arrow decimal128 with the widest-precision the cudf decimal type
 * supports. For example, numeric::decimal32 will be converted to Arrow decimal128 of the precision
 * 9 which is the maximum precision for 32-bit types. Similar, numeric::decimal128 will be
 * converted to Arrow decimal128 of the precision 38.
 *
 * Copies will be performed in the cases where cudf differs from Arrow:
 * - BOOL8: Arrow uses a bitmap and cudf uses 1 byte per value
 * - DECIMAL32 and DECIMAL64: Converted to Arrow decimal128
 * - STRING: Arrow expects a single value int32 offset child array for empty strings columns
 *
 * @param col Input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for any allocations during conversion
 * @return ArrowDeviceArray which will have ownership of any copied data
 */
unique_device_array_t to_arrow_device(
  cudf::column_view const& col,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Copy table view data to host and create `ArrowDeviceArray` for it
 *
 * Populates the C struct ArrowDeviceArray, copying the cudf data to the host. The
 * returned ArrowDeviceArray will have a device_type of CPU and will have no ties
 * to the memory referenced by the table view passed in. The deleter for the
 * returned unique_ptr will call the release callback on the ArrowDeviceArray
 * automatically.
 *
 * @note For decimals, since the precision is not stored for them in libcudf, it will
 * be converted to an Arrow decimal128 that has the widest-precision the cudf decimal type
 * supports. For example, numeric::decimal32 will be converted to Arrow decimal128 of the precision
 * 9 which is the maximum precision for 32-bit types. Similarly, numeric::decimal128 will be
 * converted to Arrow decimal128 of precision 38.
 *
 * @param table Input table
 * @param stream CUDA stream used for the device memory operations and kernel launches
 * @param mr Device memory resource used for any allocations during conversion
 * @return ArrowDeviceArray generated from input table
 */
unique_device_array_t to_arrow_host(
  cudf::table_view const& table,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Copy column view data to host and create `ArrowDeviceArray` for it
 *
 * Populates the C struct ArrowDeviceArray, copying the cudf data to the host. The
 * returned ArrowDeviceArray will have a device_type of CPU and will have no ties
 * to the memory referenced by the column view passed in. The deleter for the
 * returned unique_ptr will call the release callback on the ArrowDeviceArray
 * automatically.
 *
 * @note For decimals, since the precision is not stored for them in libcudf, it will
 * be converted to an Arrow decimal128 that has the widest-precision the cudf decimal type
 * supports. For example, numeric::decimal32 will be converted to Arrow decimal128 of the precision
 * 9 which is the maximum precision for 32-bit types. Similarly, numeric::decimal128 will be
 * converted to Arrow decimal128 of precision 38.
 *
 * @param col Input column
 * @param stream CUDA stream used for the device memory operations and kernel launches
 * @param mr Device memory resource used for any allocations during conversion
 * @return ArrowDeviceArray generated from input column
 */
unique_device_array_t to_arrow_host(
  cudf::column_view const& col,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Copy strings column data to host and create `ArrowDeviceArray` for it
 * using the ArrowBinaryView format
 *
 * Populates the ArrowDeviceArray, copying the cudf data to the host. The
 * returned ArrowDeviceArray will have a device_type of CPU and will have no ties
 * to the memory referenced by the column view passed in. The deleter for the
 * returned unique_ptr will call the release callback on the ArrowDeviceArray
 * automatically.
 *
 * @param col Input strings column view
 * @param stream CUDA stream used for the device memory operations and kernel launches
 * @param mr Device memory resource used for any allocations during conversion
 * @return ArrowDeviceArray generated from input column
 */
unique_device_array_t to_arrow_host_stringview(
  cudf::strings_column_view const& col,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `cudf::table` from given ArrowArray and ArrowSchema input
 *
 * @throws std::invalid_argument if either schema or input are NULL
 *
 * @throws cudf::data_type_error if the input array is not a struct array.
 *
 * @throws std::overflow_error if the input arrow object exceeds the column size limit.
 *
 * The conversion will not call release on the input Array.
 *
 * @param schema `ArrowSchema` pointer to describe the type of the data
 * @param input `ArrowArray` pointer that needs to be converted to cudf::table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate `cudf::table`
 * @return cudf table generated from given arrow data
 */
std::unique_ptr<cudf::table> from_arrow(
  ArrowSchema const* schema,
  ArrowArray const* input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `cudf::column` from a given ArrowArray and ArrowSchema input
 *
 * @throws std::invalid_argument if either schema or input are NULL
 *
 * The conversion will not call release on the input Array.
 *
 * @param schema `ArrowSchema` pointer to describe the type of the data
 * @param input `ArrowArray` pointer that needs to be converted to cudf::column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate `cudf::column`
 * @return cudf column generated from given arrow data
 */
std::unique_ptr<cudf::column> from_arrow_column(
  ArrowSchema const* schema,
  ArrowArray const* input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `cudf::table` from given ArrowDeviceArray input
 *
 * @throws std::invalid_argument if either schema or input are NULL
 *
 * @throws std::invalid_argument if the device_type is not `ARROW_DEVICE_CPU`
 *
 * @throws std::overflow_error if the input arrow object exceeds the column size limit.
 *
 * @throws cudf::data_type_error if the input array is not a struct array,
 * non-struct arrays should be passed to `from_arrow_host_column` instead.
 *
 * The conversion will not call release on the input Array.
 *
 * @param schema `ArrowSchema` pointer to describe the type of the data
 * @param input `ArrowDeviceArray` pointer to object owning the Arrow data
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to perform cuda allocation
 * @return cudf table generated from the given Arrow data
 */
std::unique_ptr<table> from_arrow_host(
  ArrowSchema const* schema,
  ArrowDeviceArray const* input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `cudf::table` from given ArrowArrayStream input
 *
 * @throws std::invalid_argument if input is NULL
 *
 * The conversion WILL release the input ArrayArrayStream and its constituent
 * arrays or schema since Arrow streams are not suitable for multiple reads.
 *
 * @param input `ArrowArrayStream` pointer to object that will produce ArrowArray data
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to perform cuda allocation
 * @return cudf table generated from the given Arrow data
 */
std::unique_ptr<table> from_arrow_stream(
  ArrowArrayStream* input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `cudf::column` from given ArrowArrayStream input
 *
 * @throws std::invalid_argument if input is NULL
 *
 * The conversion WILL release the input ArrayArrayStream and its constituent
 * arrays or schema since Arrow streams are not suitable for multiple reads.
 *
 * @param input `ArrowArrayStream` pointer to object that will produce ArrowArray data
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to perform cuda allocation
 * @return cudf column generated from the given Arrow data
 */
std::unique_ptr<column> from_arrow_stream_column(
  ArrowArrayStream* input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `cudf::column` from given ArrowDeviceArray input
 *
 * @throws std::invalid_argument if either schema or input are NULL
 *
 * @throws std::invalid_argument if the device_type is not `ARROW_DEVICE_CPU`
 *
 * @throws cudf::data_type_error if input arrow data type is not supported in cudf.
 *
 * @throws std::overflow_error if the input arrow object exceeds the column size limit.
 *
 * The conversion will not call release on the input Array.
 *
 * @param schema `ArrowSchema` pointer to describe the type of the data
 * @param input `ArrowDeviceArray` pointer to object owning the Arrow data
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to perform cuda allocation
 * @return cudf column generated from the given Arrow data
 */
std::unique_ptr<column> from_arrow_host_column(
  ArrowSchema const* schema,
  ArrowDeviceArray const* input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `cudf::table_view` from given `ArrowDeviceArray` and `ArrowSchema`
 *
 * Constructs a non-owning `cudf::table_view` using `ArrowDeviceArray` and `ArrowSchema`,
 * data must be accessible to the CUDA device. Because the resulting `cudf::table_view` will
 * not own the data, the `ArrowDeviceArray` must be kept alive for the lifetime of the result.
 * It is the responsibility of callers to ensure they call the release callback on the
 * `ArrowDeviceArray` after it is no longer needed, and that the `cudf::table_view` is not
 * accessed after this happens.
 *
 * @throws std::invalid_argument if device_type is not `ARROW_DEVICE_CUDA`, `ARROW_DEVICE_CUDA_HOST`
 * or `ARROW_DEVICE_CUDA_MANAGED`
 *
 * @throws cudf::data_type_error if the input array is not a struct array, non-struct
 * arrays should be passed to `from_arrow_device_column` instead.
 *
 * @throws cudf::data_type_error if the input arrow data type is not supported.
 *
 * @throws std::overflow_error if the input arrow object exceeds the column size limit.
 *
 * Each child of the input struct will be the columns of the resulting table_view.
 *
 * @note The custom deleter used for the unique_ptr to the table_view maintains ownership
 * over any memory which is allocated, such as converting boolean columns from the bitmap
 * used by Arrow to the 1-byte per value for cudf.
 *
 * @note If the input `ArrowDeviceArray` contained a non-null sync_event it is assumed
 * to be a `cudaEvent_t*` and the passed in stream will have `cudaStreamWaitEvent` called
 * on it with the event. This function, however, will not explicitly synchronize on the
 * stream.
 *
 * @param schema `ArrowSchema` pointer to object describing the type of the device array
 * @param input `ArrowDeviceArray` pointer to object owning the Arrow data
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to perform any allocations
 * @return `cudf::table_view` generated from given Arrow data
 */
unique_table_view_t from_arrow_device(
  ArrowSchema const* schema,
  ArrowDeviceArray const* input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create `cudf::column_view` from given `ArrowDeviceArray` and `ArrowSchema`
 *
 * Constructs a non-owning `cudf::column_view` using `ArrowDeviceArray` and `ArrowSchema`,
 * data must be accessible to the CUDA device. Because the resulting `cudf::column_view` will
 * not own the data, the `ArrowDeviceArray` must be kept alive for the lifetime of the result.
 * It is the responsibility of callers to ensure they call the release callback on the
 * `ArrowDeviceArray` after it is no longer needed, and that the `cudf::column_view` is not
 * accessed after this happens.
 *
 * @throws std::invalid_argument if device_type is not `ARROW_DEVICE_CUDA`, `ARROW_DEVICE_CUDA_HOST`
 * or `ARROW_DEVICE_CUDA_MANAGED`
 *
 * @throws cudf::data_type_error input arrow data type is not supported.
 *
 * @throws std::overflow_error if the input arrow object exceeds the column size limit.
 *
 * @note The custom deleter used for the unique_ptr to the table_view maintains ownership
 * over any memory which is allocated, such as converting boolean columns from the bitmap
 * used by Arrow to the 1-byte per value for cudf.
 *
 * @note If the input `ArrowDeviceArray` contained a non-null sync_event it is assumed
 * to be a `cudaEvent_t*` and the passed in stream will have `cudaStreamWaitEvent` called
 * on it with the event. This function, however, will not explicitly synchronize on the
 * stream.
 *
 * @param schema `ArrowSchema` pointer to object describing the type of the device array
 * @param input `ArrowDeviceArray` pointer to object owning the Arrow data
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to perform any allocations
 * @return `cudf::column_view` generated from given Arrow data
 */
unique_column_view_t from_arrow_device_column(
  ArrowSchema const* schema,
  ArrowDeviceArray const* input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
