/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/durations.hpp>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

struct generated_test_data {
  generated_test_data(cudf::size_type length)
    : int64_data(length),
      bool_data(length),
      string_data(length),
      validity(length),
      bool_validity(length),
      list_offsets(length + 1)
  {
    auto validity_generator = []() { return rand() % 7 != 0; };

    std::generate(int64_data.begin(), int64_data.end(), []() { return rand() % 500000; });
    std::generate(bool_data.begin(), bool_data.end(), validity_generator);
    std::generate(
      string_data.begin(), string_data.end(), []() { return rand() % 7 != 0 ? "CUDF" : "Rocks"; });
    std::generate(validity.begin(), validity.end(), validity_generator);
    std::generate(bool_validity.begin(), bool_validity.end(), validity_generator);

    std::transform(bool_validity.cbegin(),
                   bool_validity.cend(),
                   std::back_inserter(bool_data_validity),
                   [](auto val) { return static_cast<uint8_t>(val); });
    list_validity = bool_data_validity;

    constexpr cudf::size_type length_of_individual_list = 3;
    list_offsets[0]                                     = 0;
    for (cudf::size_type i = 0; i < length; ++i) {
      list_offsets[i + 1] = list_offsets[i] + (list_validity[i] ? length_of_individual_list : 0);
    }
    list_int64_data.resize(list_offsets.back());
    list_int64_data_validity.resize(list_int64_data.size());
    std::generate(list_int64_data.begin(), list_int64_data.end(), []() { return rand() % 500000; });
    std::generate(
      list_int64_data_validity.begin(), list_int64_data_validity.end(), validity_generator);
  }

  std::vector<int64_t> int64_data;
  std::vector<bool> bool_data;
  std::vector<std::string> string_data;
  std::vector<uint8_t> validity;
  std::vector<bool> bool_validity;
  std::vector<uint8_t> bool_data_validity;
  std::vector<int64_t> list_int64_data;
  std::vector<uint8_t> list_int64_data_validity;
  std::vector<int32_t> list_offsets;
  std::vector<uint8_t> list_validity;
};

// no-op allocator/deallocator to set into ArrowArray buffers that we don't
// want to own their buffers.
static ArrowBufferAllocator noop_alloc = (struct ArrowBufferAllocator){
  .reallocate = [](ArrowBufferAllocator*, uint8_t* ptr, int64_t, int64_t) -> uint8_t* {
    return ptr;
  },
  .free         = [](ArrowBufferAllocator*, uint8_t*, int64_t) {},
  .private_data = nullptr,
};

// populate an ArrowArray with pointers to the raw device buffers of a cudf::column_view
// and use the no-op alloc so that the ArrowArray doesn't presume ownership of the data
template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>() and !std::is_same_v<T, bool>, void> populate_from_col(
  ArrowArray* arr, cudf::column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();
  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc));
  ArrowArrayValidityBitmap(arr)->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(view.size());
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(view.null_mask()));
  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc));
  ArrowArrayBuffer(arr, 1)->size_bytes = sizeof(T) * view.size();
  ArrowArrayBuffer(arr, 1)->data       = const_cast<uint8_t*>(view.data<uint8_t>());
}

// populate an ArrowArray from a boolean cudf column. Since Arrow and cudf
// still represent boolean arrays differently, we have to use bools_to_mask
// and give the ArrowArray object ownership of the device data.
template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, void> populate_from_col(ArrowArray* arr,
                                                                  cudf::column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();

  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc));
  ArrowArrayValidityBitmap(arr)->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(view.size());
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(view.null_mask()));

  auto bitmask = cudf::bools_to_mask(view);
  auto ptr     = reinterpret_cast<uint8_t*>(bitmask.first->data());
  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(
    ArrowArrayBuffer(arr, 1),
    ArrowBufferDeallocator(
      [](ArrowBufferAllocator* alloc, uint8_t*, int64_t) {
        auto buf = reinterpret_cast<std::unique_ptr<rmm::device_buffer>*>(alloc->private_data);
        delete buf;
      },
      new std::unique_ptr<rmm::device_buffer>(std::move(bitmask.first)))));
  ArrowArrayBuffer(arr, 1)->size_bytes = cudf::bitmask_allocation_size_bytes(view.size());
  ArrowArrayBuffer(arr, 1)->data       = ptr;
}

// populate an ArrowArray with the string data buffers of a cudf column_view
// using no-op allocator so the ArrowArray knows it doesn't have ownership
// of the device buffers.
template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, void> populate_from_col(
  ArrowArray* arr, cudf::column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();

  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc));
  ArrowArrayValidityBitmap(arr)->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(view.size());
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(view.null_mask()));

  cudf::strings_column_view sview{view};
  if (view.size() > 0) {
    NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc));
    ArrowArrayBuffer(arr, 1)->size_bytes = sizeof(int32_t) * sview.offsets().size();
    ArrowArrayBuffer(arr, 1)->data       = const_cast<uint8_t*>(sview.offsets().data<uint8_t>());
    NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 2), noop_alloc));
    ArrowArrayBuffer(arr, 2)->size_bytes = sview.chars_size(cudf::get_default_stream());
    ArrowArrayBuffer(arr, 2)->data       = const_cast<uint8_t*>(view.data<uint8_t>());
  } else {
    auto zero          = cudf::detail::device_scalar<int32_t>(0, cudf::get_default_stream());
    uint8_t const* ptr = reinterpret_cast<uint8_t*>(zero.data());
    nanoarrow::BufferInitWrapped(ArrowArrayBuffer(arr, 1), std::move(zero), ptr, 4);
  }
}

template <typename KEY_TYPE, typename IND_TYPE>
void populate_dict_from_col(ArrowArray* arr, cudf::dictionary_column_view dview)
{
  arr->length     = dview.size();
  arr->null_count = dview.null_count();
  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc));
  ArrowArrayValidityBitmap(arr)->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(dview.size());
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(dview.null_mask()));

  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc));
  ArrowArrayBuffer(arr, 1)->size_bytes = sizeof(IND_TYPE) * dview.indices().size();
  ArrowArrayBuffer(arr, 1)->data       = const_cast<uint8_t*>(dview.indices().data<uint8_t>());

  populate_from_col<KEY_TYPE>(arr->dictionary, dview.keys());
}

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, nanoarrow::UniqueArray>
get_nanoarrow_tables(cudf::size_type length = 10000);

void populate_list_from_col(ArrowArray* arr, cudf::lists_column_view view);

std::unique_ptr<cudf::table> get_cudf_table();

template <typename T>
struct nanoarrow_storage_type {};

#define DEFINE_NANOARROW_STORAGE(T, NanoType)                    \
  template <>                                                    \
  struct nanoarrow_storage_type<T> {                             \
    static constexpr ArrowType type = NANOARROW_TYPE_##NanoType; \
  }

DEFINE_NANOARROW_STORAGE(int8_t, INT8);
DEFINE_NANOARROW_STORAGE(int16_t, INT16);
DEFINE_NANOARROW_STORAGE(int32_t, INT32);
DEFINE_NANOARROW_STORAGE(int64_t, INT64);
DEFINE_NANOARROW_STORAGE(uint8_t, UINT8);
DEFINE_NANOARROW_STORAGE(uint16_t, UINT16);
DEFINE_NANOARROW_STORAGE(uint64_t, UINT64);
DEFINE_NANOARROW_STORAGE(bool, BOOL);
DEFINE_NANOARROW_STORAGE(cudf::duration_D, INT32);
DEFINE_NANOARROW_STORAGE(cudf::duration_s, INT64);
DEFINE_NANOARROW_STORAGE(cudf::duration_ms, INT64);
DEFINE_NANOARROW_STORAGE(cudf::duration_us, INT64);
DEFINE_NANOARROW_STORAGE(cudf::duration_ns, INT64);
DEFINE_NANOARROW_STORAGE(__int128_t, DECIMAL128);

#undef DEFINE_NANOARROW_STORAGE

template <typename T>
struct nanoarrow_decimal_type {};

template <>
struct nanoarrow_decimal_type<int32_t> {
  static constexpr ArrowType type = NANOARROW_TYPE_DECIMAL32;
};

template <>
struct nanoarrow_decimal_type<int64_t> {
  static constexpr ArrowType type = NANOARROW_TYPE_DECIMAL64;
};

template <>
struct nanoarrow_decimal_type<__int128_t> {
  static constexpr ArrowType type = NANOARROW_TYPE_DECIMAL128;
};

template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>() and !std::is_same_v<T, bool>, nanoarrow::UniqueArray>
get_nanoarrow_array(std::vector<T> const& data, std::vector<uint8_t> const& mask = {})
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(tmp.get(), nanoarrow_storage_type<T>::type));

  if (!mask.empty()) {
    ArrowBitmap bitmap;
    ArrowBitmapInit(&bitmap);
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&bitmap, mask.size()));
    ArrowBitmapAppendInt8Unsafe(&bitmap, reinterpret_cast<int8_t const*>(mask.data()), mask.size());

    ArrowArraySetValidityBitmap(tmp.get(), &bitmap);
    tmp->null_count =
      data.size() -
      ArrowBitCountSet(ArrowArrayValidityBitmap(tmp.get())->buffer.data, 0, mask.size());
  }

  ArrowBuffer buf;
  ArrowBufferInit(&buf);
  NANOARROW_THROW_NOT_OK(
    ArrowBufferAppend(&buf, reinterpret_cast<void const*>(data.data()), sizeof(T) * data.size()));
  NANOARROW_THROW_NOT_OK(ArrowArraySetBuffer(tmp.get(), 1, &buf));

  tmp->length = data.size();

  return tmp;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, nanoarrow::UniqueArray> get_nanoarrow_array(
  std::vector<bool> const& data, std::vector<bool> const& mask = {})
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(tmp.get(), NANOARROW_TYPE_BOOL));

  auto to_arrow_bitmap = [](std::vector<bool> const& b) -> ArrowBitmap {
    ArrowBitmap out;
    ArrowBitmapInit(&out);
    NANOARROW_THROW_NOT_OK(ArrowBitmapResize(&out, b.size(), 1));
    // TODO: Investigate clang-tidy issue further after nanoarrow is made compliant
    std::memset(out.buffer.data, 0, out.buffer.size_bytes);  // NOLINT

    for (size_t i = 0; i < b.size(); ++i) {
      ArrowBitSetTo(out.buffer.data, i, static_cast<uint8_t>(b[i]));
    }

    return out;
  };

  if (!mask.empty()) {
    auto validity_bitmap = to_arrow_bitmap(mask);
    ArrowArraySetValidityBitmap(tmp.get(), &validity_bitmap);
    tmp->null_count =
      mask.size() -
      ArrowBitCountSet(ArrowArrayValidityBitmap(tmp.get())->buffer.data, 0, mask.size());
  }

  auto raw_buffer = to_arrow_bitmap(data);
  NANOARROW_THROW_NOT_OK(ArrowArraySetBuffer(tmp.get(), 1, &raw_buffer.buffer));
  tmp->length = data.size();

  return tmp;
}

template <typename T, typename B>
nanoarrow::UniqueArray get_nanoarrow_array(std::initializer_list<T> elements,
                                           std::initializer_list<B> validity = {})
{
  std::vector<B> mask(validity);
  std::vector<T> data(elements);

  return get_nanoarrow_array<T>(data, mask);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, nanoarrow::UniqueArray> get_nanoarrow_array(
  std::vector<std::string> const& data, std::vector<uint8_t> const& mask = {})
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(tmp.get(), NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(ArrowArrayValidityBitmap(tmp.get()), mask.size()));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(tmp.get()));
  NANOARROW_THROW_NOT_OK(ArrowArrayReserve(tmp.get(), data.size()));

  for (size_t i = 0; i < data.size(); ++i) {
    if (!mask.empty() && mask[i] == 0) {
      NANOARROW_THROW_NOT_OK(ArrowArrayAppendNull(tmp.get(), 1));
    } else {
      NANOARROW_THROW_NOT_OK(ArrowArrayAppendString(tmp.get(), ArrowCharView(data[i].c_str())));
    }
  }

  return tmp;
}

template <typename KEY_TYPE, typename IND_TYPE>
nanoarrow::UniqueArray get_nanoarrow_dict_array(std::vector<KEY_TYPE> const& keys,
                                                std::vector<IND_TYPE> const& ind,
                                                std::vector<uint8_t> const& validity = {})
{
  auto indices_array = get_nanoarrow_array<IND_TYPE>(ind, validity);
  NANOARROW_THROW_NOT_OK(ArrowArrayAllocateDictionary(indices_array.get()));

  auto keys_array = get_nanoarrow_array<KEY_TYPE>(keys);
  keys_array.move(indices_array->dictionary);

  return indices_array;
}

template <typename T>
nanoarrow::UniqueArray get_nanoarrow_list_array(std::vector<T> const& data,
                                                std::vector<int32_t> const& offsets,
                                                std::vector<uint8_t> const& data_validity = {},
                                                std::vector<uint8_t> const& list_validity = {})
{
  auto data_array = get_nanoarrow_array<T>(data, data_validity);

  nanoarrow::UniqueArray tmp;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(tmp.get(), NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), 1));
  data_array.move(tmp->children[0]);

  tmp->length = offsets.size() - 1;
  if (!list_validity.empty()) {
    ArrowBitmap bitmap;
    ArrowBitmapInit(&bitmap);
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&bitmap, list_validity.size()));
    ArrowBitmapAppendInt8Unsafe(
      &bitmap, reinterpret_cast<int8_t const*>(list_validity.data()), list_validity.size());

    ArrowArraySetValidityBitmap(tmp.get(), &bitmap);
    tmp->null_count =
      tmp->length -
      ArrowBitCountSet(ArrowArrayValidityBitmap(tmp.get())->buffer.data, 0, list_validity.size());
  }

  ArrowBuffer buf;
  ArrowBufferInit(&buf);
  NANOARROW_THROW_NOT_OK(ArrowBufferAppend(
    &buf, reinterpret_cast<void const*>(offsets.data()), sizeof(int32_t) * offsets.size()));
  NANOARROW_THROW_NOT_OK(ArrowArraySetBuffer(tmp.get(), 1, &buf));

  return tmp;
}

template <typename T>
nanoarrow::UniqueArray get_nanoarrow_list_array(std::initializer_list<T> data,
                                                std::initializer_list<int32_t> offsets,
                                                std::initializer_list<uint8_t> data_validity = {},
                                                std::initializer_list<uint8_t> list_validity = {})
{
  std::vector<T> data_vector(data);
  std::vector<int32_t> offset(offsets);
  std::vector<uint8_t> data_mask(data_validity);
  std::vector<uint8_t> list_mask(list_validity);
  return get_nanoarrow_list_array<T>(data_vector, offset, data_mask, list_mask);
}

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, generated_test_data>
get_nanoarrow_cudf_table(cudf::size_type length);

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, nanoarrow::UniqueArray>
get_nanoarrow_host_tables(cudf::size_type length);

void slice_host_nanoarrow(ArrowArray* arr, int64_t start, int64_t end);

template <typename T>
std::enable_if_t<std::disjunction_v<std::is_same<T, int32_t>,
                                    std::is_same<T, int64_t>,
                                    std::is_same<T, __int128_t>>,
                 std::size_t>
get_decimal_precision()
{
  return std::numeric_limits<T>::digits10;
}

struct VectorOfArrays {
  std::vector<nanoarrow::UniqueArray> arrays;
  nanoarrow::UniqueSchema schema;
  size_t index{0};

  static int get_schema(ArrowArrayStream* stream, ArrowSchema* out_schema)
  {
    auto private_data = static_cast<VectorOfArrays*>(stream->private_data);

    NANOARROW_THROW_NOT_OK(ArrowSchemaDeepCopy(private_data->schema.get(), out_schema));
    return 0;
  }

  static int get_next(ArrowArrayStream* stream, ArrowArray* out_array)
  {
    auto private_data = static_cast<VectorOfArrays*>(stream->private_data);
    if (private_data->index >= private_data->arrays.size()) {
      out_array->release = nullptr;
      return 0;
    }
    ArrowArrayMove(private_data->arrays[private_data->index++].get(), out_array);
    return 0;
  }

  static char const* get_last_error(ArrowArrayStream* stream) { return nullptr; }

  static void release(ArrowArrayStream* stream)
  {
    delete static_cast<VectorOfArrays*>(stream->private_data);
  }
};

void makeStreamFromArrays(std::vector<nanoarrow::UniqueArray> arrays,
                          nanoarrow::UniqueSchema schema,
                          ArrowArrayStream* out);

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, ArrowArrayStream>
get_nanoarrow_stream(int num_copies);
