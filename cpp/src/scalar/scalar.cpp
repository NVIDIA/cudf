/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <algorithm>
#include <string>

namespace cudf {

namespace {

rmm::device_buffer make_null_mask(bool is_valid,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  return cudf::detail::create_null_mask(
    1, is_valid ? mask_state::ALL_VALID : mask_state::ALL_NULL, stream, mr);
}

size_type checked_string_size(std::size_t size)
{
  CUDF_EXPECTS(size <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Data exceeds the string size limit",
               std::overflow_error);
  return static_cast<size_type>(size);
}

template <typename T>
data_type fixed_width_storage_type()
{
  if constexpr (std::is_same_v<T, __int128_t>) {
    return data_type{type_id::DECIMAL128, 0};
  } else {
    return data_type{type_to_id<T>()};
  }
}

template <typename T>
column make_fixed_width_storage(T value,
                                data_type type,
                                bool is_valid,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  auto host_data = cudf::detail::make_pinned_vector<T>(1, stream);
  host_data[0]   = value;
  return column{type,
                1,
                rmm::device_buffer{host_data.data(), sizeof(T), stream, mr},
                make_null_mask(is_valid, stream, mr),
                is_valid ? 0 : 1};
}

template <typename T>
column make_fixed_width_storage(rmm::device_scalar<T> const& value,
                                data_type type,
                                bool is_valid,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  rmm::device_buffer data(sizeof(T), stream, mr);
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(data.data(), value.data(), sizeof(T), cudaMemcpyDeviceToDevice, stream.get()));
  return column{type, 1, std::move(data), make_null_mask(is_valid, stream, mr), is_valid ? 0 : 1};
}

std::unique_ptr<column> make_offsets_column(size_type size,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  auto offsets = cudf::detail::make_pinned_vector<size_type>(2, stream);
  offsets[0]   = 0;
  offsets[1]   = size;
  return std::make_unique<column>(
    data_type{type_id::INT32},
    2,
    rmm::device_buffer{offsets.data(), offsets.size() * sizeof(size_type), stream, mr},
    rmm::device_buffer{},
    0);
}

column assemble_string_storage(rmm::device_buffer&& chars,
                               std::unique_ptr<column>&& offsets,
                               bool is_valid,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(offsets));
  return column{data_type{type_id::STRING},
                1,
                std::move(chars),
                make_null_mask(is_valid, stream, mr),
                is_valid ? 0 : 1,
                std::move(children)};
}

column make_string_storage(rmm::device_buffer&& chars,
                           size_type size,
                           bool is_valid,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  return assemble_string_storage(
    std::move(chars), make_offsets_column(size, stream, mr), is_valid, stream, mr);
}

column make_list_storage(column&& elements,
                         bool is_valid,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
{
  auto const size = elements.size();
  std::vector<std::unique_ptr<column>> children;
  children.push_back(make_offsets_column(size, stream, mr));
  children.push_back(std::make_unique<column>(std::move(elements)));
  return column{data_type{type_id::LIST},
                1,
                rmm::device_buffer{},
                make_null_mask(is_valid, stream, mr),
                is_valid ? 0 : 1,
                std::move(children)};
}

column make_struct_storage(table&& children_table,
                           bool is_valid,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  return column{data_type{type_id::STRUCT},
                1,
                rmm::device_buffer{},
                make_null_mask(is_valid, stream, mr),
                is_valid ? 0 : 1,
                children_table.release()};
}

}  // namespace

struct string_scalar::string_storage {
  column storage;
  cudf::detail::host_vector<char> staging;
  size_type size;
};

string_scalar::string_storage string_scalar::make_storage(std::string_view string,
                                                          bool is_valid,
                                                          cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  auto const size = checked_string_size(string.size());
  auto offsets    = make_offsets_column(size, stream, mr);
  auto staging    = cudf::detail::make_pinned_vector<char>(string.size(), stream);
  std::copy(string.begin(), string.end(), staging.begin());
  auto chars = rmm::device_buffer(staging.data(), staging.size(), stream, mr);
  return string_storage{
    assemble_string_storage(std::move(chars), std::move(offsets), is_valid, stream, mr),
    std::move(staging),
    size};
}

string_scalar::string_scalar(string_storage&& storage)
  : scalar(std::move(storage.storage)), _size(storage.size), _host_data(std::move(storage.staging))
{
}

scalar::scalar(data_type type,
               bool is_valid,
               cuda::stream_ref stream,
               rmm::device_async_resource_ref mr)
  : _storage{type,
             1,
             rmm::device_buffer{cudf::size_of(type), stream, mr},
             make_null_mask(is_valid, stream, mr),
             is_valid ? 0 : 1}
{
}

scalar::scalar(column&& storage) : _storage(std::move(storage))
{
  CUDF_EXPECTS(_storage.size() == 1, "Scalar storage must contain exactly one row.");
}

scalar::scalar(scalar const& other, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
  : _storage(other._storage, stream, mr)
{
}

data_type scalar::type() const noexcept { return _storage.type(); }

void scalar::set_valid_async(bool is_valid, cuda::stream_ref stream)
{
  CUDF_CUDA_TRY(cudaMemsetAsync(_storage.mutable_view().null_mask(),
                                is_valid ? 0xff : 0,
                                bitmask_allocation_size_bytes(1),
                                stream.get()));
  _storage.set_null_count(is_valid ? 0 : 1);
}

bool scalar::is_valid(cuda::stream_ref) const { return _storage.null_count() == 0; }

bitmask_type const* scalar::validity_data() const { return _storage.view().null_mask(); }

scalar_column_view scalar::as_column_view() const { return scalar_column_view{_storage.view()}; }

mutable_scalar_column_view scalar::as_mutable_column_view()
{
  return mutable_scalar_column_view{_storage.mutable_view()};
}

string_scalar::string_scalar(std::string_view string,
                             bool is_valid,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : string_scalar(make_storage(string, is_valid, stream, mr))
{
}

string_scalar::string_scalar(string_scalar const& other,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : scalar(other, stream, mr), _size(other._size)
{
}

string_scalar::string_scalar(rmm::device_scalar<value_type>& data,
                             bool is_valid,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : string_scalar(data.value(stream), is_valid, stream, mr)
{
}

string_scalar::string_scalar(value_type const& source,
                             bool is_valid,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : scalar([&] {
      rmm::device_buffer chars(source.size_bytes(), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        chars.data(), source.data(), source.size_bytes(), cudaMemcpyDeviceToDevice, stream.get()));
      return make_string_storage(std::move(chars), source.size_bytes(), is_valid, stream, mr);
    }()),
    _size(source.size_bytes())
{
}

string_scalar::string_scalar(rmm::device_buffer&& data,
                             bool is_valid,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : string_scalar(std::move(data), checked_string_size(data.size()), is_valid, stream, mr)
{
}

string_scalar::string_scalar(rmm::device_buffer&& data,
                             size_type size,
                             bool is_valid,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : scalar(make_string_storage(std::move(data), size, is_valid, stream, mr)), _size(size)
{
}

string_scalar::value_type string_scalar::value(cuda::stream_ref stream) const
{
  return value_type{data(), size()};
}

size_type string_scalar::size() const { return _size; }

char const* string_scalar::data() const { return _storage.view().data<char>(); }

std::string string_scalar::to_string(cuda::stream_ref stream) const
{
  std::string result(size(), '\0');
  detail::cuda_memcpy(host_span<char>{result.data(), result.size()},
                      device_span<char const>{data(), static_cast<std::size_t>(size())},
                      stream);
  return result;
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(rep_type value,
                                          numeric::scale_type scale,
                                          bool is_valid,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{make_fixed_width_storage(
      value, data_type{type_to_id<T>(), static_cast<int32_t>(scale)}, is_valid, stream, mr)}
{
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(rep_type value,
                                          bool is_valid,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{make_fixed_width_storage(value, data_type{type_to_id<T>(), 0}, is_valid, stream, mr)}
{
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(T value,
                                          bool is_valid,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{make_fixed_width_storage(
      value.value(), data_type{type_to_id<T>(), value.scale()}, is_valid, stream, mr)}
{
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(rmm::device_scalar<rep_type>&& data,
                                          numeric::scale_type scale,
                                          bool is_valid,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{make_fixed_width_storage(data, data_type{type_to_id<T>(), scale}, is_valid, stream, mr)}
{
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(fixed_point_scalar<T> const& other,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{other, stream, mr}
{
}

template <typename T>
typename fixed_point_scalar<T>::rep_type fixed_point_scalar<T>::value(cuda::stream_ref stream) const
{
  rep_type result;
  detail::cuda_memcpy(
    host_span<rep_type>{&result, 1}, device_span<rep_type const>{data(), 1}, stream);
  return result;
}

template <typename T>
T fixed_point_scalar<T>::fixed_point_value(cuda::stream_ref stream) const
{
  return value_type{
    numeric::scaled_integer<rep_type>{value(stream), numeric::scale_type{type().scale()}}};
}

template <typename T>
typename fixed_point_scalar<T>::rep_type* fixed_point_scalar<T>::data()
{
  return _storage.mutable_view().data<rep_type>();
}

template <typename T>
typename fixed_point_scalar<T>::rep_type const* fixed_point_scalar<T>::data() const
{
  return _storage.view().data<rep_type>();
}

/**
 * @brief These define the valid fixed-point scalar types.
 *
 * See `is_fixed_point` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class fixed_point_scalar<numeric::decimal32>;
template class fixed_point_scalar<numeric::decimal64>;
template class fixed_point_scalar<numeric::decimal128>;

namespace CUDF_HIDDEN detail {

template <typename T>
fixed_width_scalar<T>::fixed_width_scalar(T value,
                                          bool is_valid,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
  : scalar(make_fixed_width_storage(value, fixed_width_storage_type<T>(), is_valid, stream, mr)),
    _bounce_buffer(cudf::detail::make_pinned_vector<T>(1, stream))
{
}

template <typename T>
fixed_width_scalar<T>::fixed_width_scalar(rmm::device_scalar<T>&& data,
                                          bool is_valid,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
  : scalar(make_fixed_width_storage(data, fixed_width_storage_type<T>(), is_valid, stream, mr)),
    _bounce_buffer(cudf::detail::make_pinned_vector<T>(1, stream))
{
}

template <typename T>
fixed_width_scalar<T>::fixed_width_scalar(fixed_width_scalar<T> const& other,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{other, stream, mr}, _bounce_buffer(cudf::detail::make_pinned_vector<T>(1, stream))
{
}

template <typename T>
void fixed_width_scalar<T>::set_value(T value, cuda::stream_ref stream)
{
  _bounce_buffer[0] = value;
  detail::cuda_memcpy_async<T>(device_span<T>{data(), 1}, _bounce_buffer, stream);
  this->set_valid_async(true, stream);
}

template <typename T>
T fixed_width_scalar<T>::value(cuda::stream_ref stream) const
{
  T result;
  detail::cuda_memcpy(host_span<T>{&result, 1}, device_span<T const>{data(), 1}, stream);
  return result;
}

template <typename T>
T* fixed_width_scalar<T>::data()
{
  return _storage.mutable_view().data<T>();
}

template <typename T>
T const* fixed_width_scalar<T>::data() const
{
  return _storage.view().data<T>();
}

/**
 * @brief These define the valid fixed-width scalar types.
 *
 * See `is_fixed_width` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class fixed_width_scalar<bool>;
template class fixed_width_scalar<int8_t>;
template class fixed_width_scalar<int16_t>;
template class fixed_width_scalar<int32_t>;
template class fixed_width_scalar<int64_t>;
template class fixed_width_scalar<__int128_t>;
template class fixed_width_scalar<uint8_t>;
template class fixed_width_scalar<uint16_t>;
template class fixed_width_scalar<uint32_t>;
template class fixed_width_scalar<uint64_t>;
template class fixed_width_scalar<float>;
template class fixed_width_scalar<double>;
template class fixed_width_scalar<timestamp_D>;
template class fixed_width_scalar<timestamp_s>;
template class fixed_width_scalar<timestamp_ms>;
template class fixed_width_scalar<timestamp_us>;
template class fixed_width_scalar<timestamp_ns>;
template class fixed_width_scalar<duration_D>;
template class fixed_width_scalar<duration_s>;
template class fixed_width_scalar<duration_ms>;
template class fixed_width_scalar<duration_us>;
template class fixed_width_scalar<duration_ns>;

}  // namespace CUDF_HIDDEN detail

template <typename T>
numeric_scalar<T>::numeric_scalar(T value,
                                  bool is_valid,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>(value, is_valid, stream, mr)
{
}

template <typename T>
numeric_scalar<T>::numeric_scalar(rmm::device_scalar<T>&& data,
                                  bool is_valid,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>(std::forward<rmm::device_scalar<T>>(data), is_valid, stream, mr)
{
}

template <typename T>
numeric_scalar<T>::numeric_scalar(numeric_scalar<T> const& other,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>{other, stream, mr}
{
}

/**
 * @brief These define the valid numeric scalar types.
 *
 * See `is_numeric` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class numeric_scalar<bool>;
template class numeric_scalar<int8_t>;
template class numeric_scalar<int16_t>;
template class numeric_scalar<int32_t>;
template class numeric_scalar<int64_t>;
template class numeric_scalar<__int128_t>;
template class numeric_scalar<uint8_t>;
template class numeric_scalar<uint16_t>;
template class numeric_scalar<uint32_t>;
template class numeric_scalar<uint64_t>;
template class numeric_scalar<float>;
template class numeric_scalar<double>;

template <typename T>
chrono_scalar<T>::chrono_scalar(T value,
                                bool is_valid,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>(value, is_valid, stream, mr)
{
}

template <typename T>
chrono_scalar<T>::chrono_scalar(rmm::device_scalar<T>&& data,
                                bool is_valid,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>(std::forward<rmm::device_scalar<T>>(data), is_valid, stream, mr)
{
}

template <typename T>
chrono_scalar<T>::chrono_scalar(chrono_scalar<T> const& other,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>{other, stream, mr}
{
}

/**
 * @brief These define the valid chrono scalar types.
 *
 * See `is_chrono` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class chrono_scalar<timestamp_D>;
template class chrono_scalar<timestamp_s>;
template class chrono_scalar<timestamp_ms>;
template class chrono_scalar<timestamp_us>;
template class chrono_scalar<timestamp_ns>;
template class chrono_scalar<duration_D>;
template class chrono_scalar<duration_s>;
template class chrono_scalar<duration_ms>;
template class chrono_scalar<duration_us>;
template class chrono_scalar<duration_ns>;

template <typename T>
duration_scalar<T>::duration_scalar(rep_type value,
                                    bool is_valid,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
  : chrono_scalar<T>(T{value}, is_valid, stream, mr)
{
}

template <typename T>
duration_scalar<T>::duration_scalar(duration_scalar<T> const& other,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
  : chrono_scalar<T>{other, stream, mr}
{
}

template <typename T>
typename duration_scalar<T>::rep_type duration_scalar<T>::count(cuda::stream_ref stream)
{
  return this->value(stream).count();
}

/**
 * @brief These define the valid duration scalar types.
 *
 * See `is_duration` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class duration_scalar<duration_D>;
template class duration_scalar<duration_s>;
template class duration_scalar<duration_ms>;
template class duration_scalar<duration_us>;
template class duration_scalar<duration_ns>;

template <typename T>
typename timestamp_scalar<T>::rep_type timestamp_scalar<T>::ticks_since_epoch(
  cuda::stream_ref stream)
{
  return this->value(stream).time_since_epoch().count();
}

/**
 * @brief These define the valid timestamp scalar types.
 *
 * See `is_timestamp` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class timestamp_scalar<timestamp_D>;
template class timestamp_scalar<timestamp_s>;
template class timestamp_scalar<timestamp_ms>;
template class timestamp_scalar<timestamp_us>;
template class timestamp_scalar<timestamp_ns>;

template <typename T>
template <typename D>
timestamp_scalar<T>::timestamp_scalar(D const& value,
                                      bool is_valid,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
  : chrono_scalar<T>(T{typename T::duration{value}}, is_valid, stream, mr)
{
}

template <typename T>
timestamp_scalar<T>::timestamp_scalar(timestamp_scalar<T> const& other,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
  : chrono_scalar<T>{other, stream, mr}
{
}

#define TS_CTOR(TimestampType, DurationType)                  \
  template timestamp_scalar<TimestampType>::timestamp_scalar( \
    DurationType const&, bool, cuda::stream_ref, rmm::device_async_resource_ref);

/**
 * @brief These are the valid combinations of duration types to timestamp types.
 */
TS_CTOR(timestamp_D, duration_D)
TS_CTOR(timestamp_D, int32_t)
TS_CTOR(timestamp_s, duration_D)
TS_CTOR(timestamp_s, duration_s)
TS_CTOR(timestamp_s, int64_t)
TS_CTOR(timestamp_ms, duration_D)
TS_CTOR(timestamp_ms, duration_s)
TS_CTOR(timestamp_ms, duration_ms)
TS_CTOR(timestamp_ms, int64_t)
TS_CTOR(timestamp_us, duration_D)
TS_CTOR(timestamp_us, duration_s)
TS_CTOR(timestamp_us, duration_ms)
TS_CTOR(timestamp_us, duration_us)
TS_CTOR(timestamp_us, int64_t)
TS_CTOR(timestamp_ns, duration_D)
TS_CTOR(timestamp_ns, duration_s)
TS_CTOR(timestamp_ns, duration_ms)
TS_CTOR(timestamp_ns, duration_us)
TS_CTOR(timestamp_ns, duration_ns)
TS_CTOR(timestamp_ns, int64_t)

list_scalar::list_scalar(cudf::column_view const& data,
                         bool is_valid,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
  : scalar(make_list_storage(column{data, stream, mr}, is_valid, stream, mr))
{
}

list_scalar::list_scalar(cudf::column&& data,
                         bool is_valid,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
  : scalar(make_list_storage(std::move(data), is_valid, stream, mr))
{
}

list_scalar::list_scalar(list_scalar const& other,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
  : scalar{other, stream, mr}
{
}

column_view list_scalar::view() const
{
  return _storage.num_children() == 0 ? column_view{} : _storage.view().child(1);
}

struct_scalar::struct_scalar(struct_scalar const& other,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : scalar{other, stream, mr}
{
}

struct_scalar::struct_scalar(table_view const& data,
                             bool is_valid,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : scalar(make_struct_storage(
      init_data(table{data, stream, mr}, is_valid, stream, mr), is_valid, stream, mr))
{
  assert_valid_size();
}

struct_scalar::struct_scalar(std::span<column_view const> data,
                             bool is_valid,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : scalar(make_struct_storage(
      init_data(table{table_view{std::vector<column_view>{data.begin(), data.end()}}, stream, mr},
                is_valid,
                stream,
                mr),
      is_valid,
      stream,
      mr))
{
  assert_valid_size();
}

struct_scalar::struct_scalar(table&& data,
                             bool is_valid,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
  : scalar(
      make_struct_storage(init_data(std::move(data), is_valid, stream, mr), is_valid, stream, mr))
{
  assert_valid_size();
}

table_view struct_scalar::view() const
{
  auto const storage = _storage.view();
  return table_view{std::vector<column_view>{storage.child_begin(), storage.child_end()}};
}

void struct_scalar::assert_valid_size()
{
  auto const tv = view();
  CUDF_EXPECTS(
    std::all_of(tv.begin(), tv.end(), [](column_view const& col) { return col.size() == 1; }),
    "Struct scalar inputs must have exactly 1 row");
}

table struct_scalar::init_data(table&& data,
                               bool is_valid,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  if (is_valid) { return std::move(data); }

  auto data_cols = data.release();

  // push validity mask down
  auto const validity = cudf::detail::create_null_mask(
    1, mask_state::ALL_NULL, stream, cudf::get_current_device_resource_ref());
  for (auto& col : data_cols) {
    col = cudf::structs::detail::superimpose_and_sanitize_nulls(
      static_cast<bitmask_type const*>(validity.data()), 1, std::move(col), stream, mr);
  }

  return table{std::move(data_cols)};
}

}  // namespace cudf
