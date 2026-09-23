/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream>

#include <dlpack/dlpack.h>

#include <algorithm>

namespace cudf {
namespace {
struct get_column_data_impl {
  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>())>
  void const* operator()(column_view const& col)
  {
    CUDF_FAIL("Unsupported type to convert to dlpack.");
  }

  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  void const* operator()(column_view const& col)
  {
    return col.data<T>();
  }
};

template <>
void const* get_column_data_impl::operator()<string_view>(column_view const& col)
{
  return nullptr;
}

void const* get_column_data(column_view const& col)
{
  return type_dispatcher(col.type(), get_column_data_impl{}, col);
}

data_type DLDataType_to_data_type(DLDataType type)
{
  CUDF_EXPECTS(type.lanes == 1, "Unsupported DLPack vector type");

  if (type.code == kDLInt) {
    switch (type.bits) {
      case 8: return data_type(type_id::INT8);
      case 16: return data_type(type_id::INT16);
      case 32: return data_type(type_id::INT32);
      case 64: return data_type(type_id::INT64);
      default: CUDF_FAIL("Unsupported bitsize for kDLInt");
    }
  } else if (type.code == kDLUInt) {
    switch (type.bits) {
      case 8: return data_type(type_id::UINT8);
      case 16: return data_type(type_id::UINT16);
      case 32: return data_type(type_id::UINT32);
      case 64: return data_type(type_id::UINT64);
      default: CUDF_FAIL("Unsupported bitsize for kDLUInt");
    }
  } else if (type.code == kDLFloat) {
    switch (type.bits) {
      case 32: return data_type(type_id::FLOAT32);
      case 64: return data_type(type_id::FLOAT64);
      default: CUDF_FAIL("Unsupported bitsize for kDLFloat");
    }
  } else {
    CUDF_FAIL("Invalid DLPack type code");
  }
}

struct data_type_to_DLDataType_impl {
  template <typename T>
  DLDataType operator()()
    requires(is_numeric<T>())
  {
    uint8_t const bits{sizeof(T) * 8};
    uint16_t const lanes{1};
    if (std::is_floating_point_v<T>) {
      return DLDataType{kDLFloat, bits, lanes};
    } else if (std::is_signed_v<T>) {
      return DLDataType{kDLInt, bits, lanes};
    } else {
      return DLDataType{kDLUInt, bits, lanes};
    }
  }

  template <typename T>
  DLDataType operator()()
    requires(not is_numeric<T>())
  {
    CUDF_FAIL("Conversion of non-numeric types to DLPack is unsupported");
  }
};

DLDataType data_type_to_DLDataType(data_type type)
{
  return type_dispatcher(type, data_type_to_DLDataType_impl{});
}

// Context object to own memory allocated for DLManagedTensor
struct dltensor_context {
  int64_t shape[2]{};    // NOLINT
  int64_t strides[2]{};  // NOLINT
  rmm::device_buffer buffer;

  static void deleter(DLManagedTensor* arg)
  {
    auto context = static_cast<dltensor_context*>(arg->manager_ctx);
    delete context;
    delete arg;
  }
};

}  // namespace

namespace detail {
std::unique_ptr<table> from_dlpack(DLManagedTensor const* managed_tensor,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(nullptr != managed_tensor, "managed_tensor is null");
  auto const& tensor = managed_tensor->dl_tensor;

  // We can copy from host or device pointers
  CUDF_EXPECTS(tensor.device.device_type == kDLCPU || tensor.device.device_type == kDLCUDA ||
                 tensor.device.device_type == kDLCUDAHost,
               "DLTensor device type must be CPU, CUDA or CUDAHost");

  // Make sure the current device ID matches the Tensor's device ID
  if (tensor.device.device_type != kDLCPU) {
    int device_id = 0;
    CUDF_CUDA_TRY(cudaGetDevice(&device_id));
    CUDF_EXPECTS(tensor.device.device_id == device_id, "DLTensor device ID must be current device");
  }

  // We only support 1D and 2D tensors with some restrictions on layout
  if (tensor.ndim == 1) {
    // 1D tensors must have dense layout (strides == nullptr <=> dense layout), or have shape (0,)
    CUDF_EXPECTS(nullptr == tensor.strides || tensor.strides[0] == 1 || tensor.shape[0] == 0,
                 "from_dlpack of 1D DLTensor only for unit-stride data");
  } else if (tensor.ndim == 2) {
    CUDF_EXPECTS(
      // Empty array is fine. If ncols == 0 then we get an empty dataframe
      // irrespective of nrows, which is slightly different behaviour from
      // cudf.DataFrame(np.empty((3, 0))) because there's no way to communicate
      // the index information out with a table view if no columns exist.
      (tensor.shape[0] == 0 || tensor.shape[1] == 0)
        // (N, 1) is fine as long as the 1D array has dense layout
        || (tensor.shape[1] == 1 && (nullptr == tensor.strides || tensor.strides[0] == 1))
        // Column major is fine as long as the fastest dimension has dense layout
        || (nullptr != tensor.strides && tensor.strides[0] == 1 &&
            tensor.strides[1] >= tensor.shape[0]),
      "from_dlpack of 2D DLTensor only for column-major unit-stride data");
  } else {
    CUDF_FAIL("DLTensor must be 1D or 2D");
  }
  CUDF_EXPECTS(tensor.shape[0] >= 0,
               "DLTensor first dim should be of shape greater than or equal to 0.");
  CUDF_EXPECTS(tensor.shape[0] <= std::numeric_limits<size_type>::max(),
               "DLTensor first dim exceeds the column size limit",
               std::overflow_error);
  if (tensor.ndim > 1) {
    CUDF_EXPECTS(tensor.shape[1] >= 0,
                 "DLTensor second dim should be of shape greater than or equal to 0.");
    CUDF_EXPECTS(tensor.shape[1] <= std::numeric_limits<size_type>::max(),
                 "DLTensor second dim exceeds the column size limit",
                 std::overflow_error);
  }
  size_t const num_columns = (tensor.ndim == 2) ? static_cast<size_t>(tensor.shape[1]) : 1;

  // Validate and convert data type to cudf
  data_type const dtype = DLDataType_to_data_type(tensor.dtype);

  size_t const byte_width = size_of(dtype);
  auto const num_rows     = static_cast<size_t>(tensor.shape[0]);
  size_t const bytes      = num_rows * byte_width;

  // For 2D tensors, if the strides pointer is not null, then strides[1] is the
  // number of elements (not bytes) between the start of each column
  size_t const col_stride = (tensor.ndim == 2 && nullptr != tensor.strides)
                              ? byte_width * tensor.strides[1]
                              : byte_width * num_rows;

  auto tensor_data = reinterpret_cast<uintptr_t>(tensor.data) + tensor.byte_offset;

  // Allocate columns and copy data from tensor
  std::vector<std::unique_ptr<column>> columns(num_columns);
  for (auto& col : columns) {
    col = make_numeric_column(dtype, num_rows, mask_state::UNALLOCATED, stream, mr);

    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      col->mutable_view().head<void>(), reinterpret_cast<void*>(tensor_data), bytes, stream));

    tensor_data += col_stride;
  }

  return std::make_unique<table>(std::move(columns));
}

DLManagedTensor* to_dlpack(table_view const& input,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.num_rows();
  auto const num_cols = input.num_columns();
  if (num_rows == 0 && num_cols == 0) { return nullptr; }

  // Ensure that type is convertible to DLDataType
  data_type const type    = input.column(0).type();
  DLDataType const dltype = data_type_to_DLDataType(type);

  // Ensure all columns are the same type
  CUDF_EXPECTS(cudf::all_have_same_types(input.begin(), input.end()),
               "All columns required to have same data type",
               cudf::data_type_error);

  // Ensure none of the columns have nulls
  CUDF_EXPECTS(
    std::none_of(input.begin(), input.end(), [](auto const& col) { return col.has_nulls(); }),
    "Input required to have null count zero");

  auto managed_tensor = std::make_unique<DLManagedTensor>();
  auto context        = std::make_unique<dltensor_context>();

  DLTensor& tensor = managed_tensor->dl_tensor;
  tensor.dtype     = dltype;

  tensor.ndim     = (num_cols > 1) ? 2 : 1;
  tensor.shape    = context->shape;
  tensor.shape[0] = num_rows;
  if (tensor.ndim > 1) {
    tensor.shape[1]   = num_cols;
    tensor.strides    = context->strides;
    tensor.strides[0] = num_rows > 1 ? 1 : 0;
    tensor.strides[1] = num_rows;
  }

  CUDF_CUDA_TRY(cudaGetDevice(&tensor.device.device_id));
  tensor.device.device_type = kDLCUDA;

  // If there is only one column, then a 1D tensor can just copy the pointer
  // to the data in the column, and the deleter should not delete the original
  // data. However, this is inconsistent with the 2D cases where we must do a
  // copy of each column's data into the dense tensor array. Also, if we don't
  // copy, then the original column data could be changed, which would change
  // the contents of the tensor, which might be surprising or cause issues.
  // Therefore, for now we ALWAYS do a copy of the data. If this becomes
  // a performance issue we can reevaluate in the future.

  size_t const stride_bytes = num_rows * size_of(type);
  size_t const total_bytes  = stride_bytes * num_cols;

  context->buffer = rmm::device_buffer(total_bytes, stream, mr);
  tensor.data     = context->buffer.data();

  auto tensor_data = reinterpret_cast<uintptr_t>(tensor.data);
  for (auto const& col : input) {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      reinterpret_cast<void*>(tensor_data), get_column_data(col), stride_bytes, stream));
    tensor_data += stride_bytes;
  }

  // Defer ownership of managed tensor to caller
  managed_tensor->deleter     = dltensor_context::deleter;
  managed_tensor->manager_ctx = context.release();

  // synchronize the stream because after the return the data may be accessed from the host before
  // the above async copies have completed (especially if pinned host memory is used).
  cudf::detail::sync_stream(stream);

  return managed_tensor.release();
}

}  // namespace detail

std::unique_ptr<table> from_dlpack(DLManagedTensor const* managed_tensor,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_dlpack(managed_tensor, stream, mr);
}

DLManagedTensor* to_dlpack(table_view const& input,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_dlpack(input, stream, mr);
}

}  // namespace cudf
