/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

template <typename T>
using fwcw  = cudf::test::fixed_width_column_wrapper<T>;
using strcw = cudf::test::strings_column_wrapper;
template <typename T>
using lcw     = cudf::test::lists_column_wrapper<T>;
using structs = cudf::test::structs_column_wrapper;

/// Column names for `cudf::to_arrow_schema`, matching the nesting of `col`.
cudf::column_metadata make_metadata(cudf::column_view const& col, std::string name)
{
  auto meta = cudf::column_metadata{std::move(name)};
  switch (col.type().id()) {
    case cudf::type_id::LIST:
      meta.children_meta.emplace_back("offsets");
      meta.children_meta.push_back(
        make_metadata(col.child(cudf::lists_column_view::child_column_index), "element"));
      break;
    case cudf::type_id::STRUCT:
      for (cudf::size_type i = 0; i < col.num_children(); ++i) {
        meta.children_meta.push_back(make_metadata(col.child(i), "f" + std::to_string(i)));
      }
      break;
    case cudf::type_id::DICTIONARY32:
      if (col.num_children() == 2) {
        meta.children_meta.push_back(make_metadata(col.child(1), "keys"));
      }
      break;
    default: break;
  }
  return meta;
}

/// Column names for `cudf::to_arrow_schema` of `table`.
std::vector<cudf::column_metadata> make_columns_metadata(cudf::table_view const& table)
{
  std::vector<cudf::column_metadata> meta;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    meta.push_back(make_metadata(table.column(i), "c" + std::to_string(i)));
  }
  return meta;
}

/// A view of `array` interpreted with `schema`, released on destruction.
struct array_view {
  ArrowArrayView view{};
  array_view(ArrowSchema const* schema, ArrowArray const* array)
  {
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&view, schema, nullptr));
    ArrowError error;
    if (ArrowArrayViewSetArray(&view, array, &error) != NANOARROW_OK) {
      throw std::runtime_error(error.message);
    }
  }
  ~array_view() { ArrowArrayViewReset(&view); }
  array_view(array_view const&)            = delete;
  array_view& operator=(array_view const&) = delete;
};

/**
 * @brief Compare `n` logical rows of two Arrow arrays, starting at row `e` of `expected` and
 * row `a` of `actual`.
 *
 * Compares values and nulls, not the physical layout, since `cudf::to_arrow_host` keeps
 * slicing offsets that packed data does not have.
 */
void compare_rows(
  ArrowArrayView const* expected, int64_t e, ArrowArrayView const* actual, int64_t a, int64_t n)
{
  ASSERT_EQ(expected->storage_type, actual->storage_type);
  for (int64_t i = 0; i < n; ++i) {
    auto const ei      = e + i;
    auto const ai      = a + i;
    auto const is_null = ArrowArrayViewIsNull(expected, ei);
    ASSERT_EQ(is_null, ArrowArrayViewIsNull(actual, ai)) << "row " << i;
    if (is_null) { continue; }
    switch (expected->storage_type) {
      case NANOARROW_TYPE_BOOL:
        ASSERT_EQ(ArrowArrayViewGetIntUnsafe(expected, ei), ArrowArrayViewGetIntUnsafe(actual, ai));
        break;
      case NANOARROW_TYPE_STRING:
      case NANOARROW_TYPE_LARGE_STRING: {
        auto const ev = ArrowArrayViewGetBytesUnsafe(expected, ei);
        auto const av = ArrowArrayViewGetBytesUnsafe(actual, ai);
        ASSERT_EQ(std::string(ev.data.as_char, ev.size_bytes),
                  std::string(av.data.as_char, av.size_bytes));
        break;
      }
      case NANOARROW_TYPE_LIST: {
        auto const e_begin = ArrowArrayViewListChildOffset(expected, expected->offset + ei);
        auto const e_end   = ArrowArrayViewListChildOffset(expected, expected->offset + ei + 1);
        auto const a_begin = ArrowArrayViewListChildOffset(actual, actual->offset + ai);
        auto const a_end   = ArrowArrayViewListChildOffset(actual, actual->offset + ai + 1);
        ASSERT_EQ(e_end - e_begin, a_end - a_begin);
        compare_rows(expected->children[0], e_begin, actual->children[0], a_begin, e_end - e_begin);
        break;
      }
      case NANOARROW_TYPE_STRUCT:
        ASSERT_EQ(expected->n_children, actual->n_children);
        for (int64_t c = 0; c < expected->n_children; ++c) {
          compare_rows(expected->children[c],
                       expected->offset + ei,
                       actual->children[c],
                       actual->offset + ai,
                       1);
        }
        break;
      default: {
        if (expected->dictionary != nullptr) {
          ASSERT_NE(actual->dictionary, nullptr);
          compare_rows(expected->dictionary,
                       ArrowArrayViewGetIntUnsafe(expected, ei),
                       actual->dictionary,
                       ArrowArrayViewGetIntUnsafe(actual, ai),
                       1);
          break;
        }
        auto const size = expected->layout.element_size_bits[1] / 8;
        ASSERT_EQ(size, actual->layout.element_size_bits[1] / 8);
        auto const* ev = expected->buffer_views[1].data.as_uint8 + (expected->offset + ei) * size;
        auto const* av = actual->buffer_views[1].data.as_uint8 + (actual->offset + ai) * size;
        ASSERT_EQ(std::memcmp(ev, av, size), 0) << "row " << i;
      }
    }
  }
}

/// Recursively compare the formats, names and flags of two schemas.
void expect_same_schema(ArrowSchema const* expected, ArrowSchema const* actual)
{
  ASSERT_EQ(std::string(expected->format), std::string(actual->format));
  EXPECT_EQ(std::string(expected->name ? expected->name : ""),
            std::string(actual->name ? actual->name : ""));
  EXPECT_EQ(expected->flags, actual->flags);
  ASSERT_EQ(expected->n_children, actual->n_children);
  for (int64_t i = 0; i < expected->n_children; ++i) {
    expect_same_schema(expected->children[i], actual->children[i]);
  }
  ASSERT_EQ(expected->dictionary == nullptr, actual->dictionary == nullptr);
  if (expected->dictionary != nullptr) {
    expect_same_schema(expected->dictionary, actual->dictionary);
  }
}

/// The metadata and payload of a packed table, with the payload copied to host memory.
struct host_packed {
  std::vector<uint8_t> metadata;
  std::vector<uint8_t> data;
};

/// Pack `table` with `cudf::pack` and copy the payload to host memory.
host_packed pack_to_host(cudf::table_view const& table)
{
  auto const stream = cudf::test::get_default_stream();
  auto packed       = cudf::pack(table, stream);
  host_packed result{packed.metadata ? *packed.metadata : std::vector<uint8_t>{},
                     std::vector<uint8_t>(packed.gpu_data->size())};
  if (!result.data.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(result.data.data(),
                                  packed.gpu_data->data(),
                                  result.data.size(),
                                  cudaMemcpyDefault,
                                  stream.get()));
  }
  stream.sync();
  return result;
}

/// Pack `table` with `pack_into` and the given codec, and copy the payload to host memory.
host_packed pack_into_host(cudf::table_view const& table,
                           cudf::experimental::pack_compression compression)
{
  namespace cx      = cudf::experimental;
  auto const stream = cudf::test::get_default_stream();
  auto const plan   = cx::prepare_pack(table, cx::pack_options{compression}, stream);
  rmm::device_buffer buffer(plan.sizes().payload_bytes, stream);
  auto result =
    cx::pack_into(plan, std::span<uint8_t>{static_cast<uint8_t*>(buffer.data()), buffer.size()});
  host_packed out{std::move(result.metadata), std::vector<uint8_t>(result.payload_bytes)};
  if (!out.data.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      out.data.data(), buffer.data(), out.data.size(), cudaMemcpyDefault, stream.get()));
  }
  stream.sync();
  return out;
}

/// Pack `table` with `chunked_pack` and copy the payload to host memory, chunk by chunk.
host_packed chunked_pack_to_host(cudf::table_view const& table)
{
  auto const stream = cudf::test::get_default_stream();
  rmm::device_buffer bounce(1024 * 1024, stream);
  auto packer = cudf::chunked_pack::create(table, bounce.size(), stream);
  host_packed out{*packer->build_metadata(),
                  std::vector<uint8_t>(packer->get_total_contiguous_size())};
  std::size_t offset = 0;
  while (packer->has_next()) {
    auto const bytes =
      packer->next(cudf::device_span<uint8_t>{static_cast<uint8_t*>(bounce.data()), bounce.size()});
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      out.data.data() + offset, bounce.data(), bytes, cudaMemcpyDefault, stream.get()));
    stream.sync();
    offset += bytes;
  }
  return out;
}

/// The packers whose output `unpack_to_arrow_host` reads.
enum class packer { pack, pack_into, chunked_pack };

/// Tests of `unpack_to_arrow_host` and `unpack_to_arrow_schema`.
struct PackedArrowHostTest : public cudf::test::BaseFixture {
  /// Check that `unpack_to_arrow_host` on the output of `pack_with`, uncompressed, matches
  /// `cudf::to_arrow_host`.
  void expect_unpack_matches_to_arrow_host(cudf::table_view const& table,
                                           packer pack_with = packer::pack)
  {
    auto const packed = [&] {
      switch (pack_with) {
        case packer::pack: return pack_to_host(table);
        case packer::pack_into:
          return pack_into_host(table, cudf::experimental::pack_compression::none);
        case packer::chunked_pack: return chunked_pack_to_host(table);
      }
      CUDF_FAIL("Unknown packer");
    }();
    auto const meta = make_columns_metadata(table);

    auto harness = cudf::test::memory_resource_test_harness{};
    cudf::unique_device_array_t got{nullptr, nullptr};
    cudf::unique_schema_t got_schema{nullptr, nullptr};
    {
      auto const scope = harness.fail_on_current_device_resource_use();
      got              = cudf::experimental::unpack_to_arrow_host(packed.metadata, packed.data);
      got_schema       = cudf::experimental::unpack_to_arrow_schema(packed.metadata, meta);
    }
    ASSERT_EQ(got->device_type, ARROW_DEVICE_CPU);

    // `unpack_to_arrow_schema` matches `cudf::to_arrow_schema` of the unpacked table.
    // The reference schema comes from the payload unpacked on the device.
    auto const stream = cudf::test::get_default_stream();
    auto const d_data = rmm::device_buffer{packed.data.data(), packed.data.size(), stream};
    auto const unpacked =
      cudf::unpack(std::span<uint8_t const>{packed.metadata},
                   packed.data.empty() ? nullptr : static_cast<uint8_t const*>(d_data.data()));
    auto const expected_schema = cudf::to_arrow_schema(unpacked, meta);
    expect_same_schema(expected_schema.get(), got_schema.get());

    auto const expected = cudf::to_arrow_host(table, stream);
    auto const e        = array_view{expected_schema.get(), &expected->array};
    auto a              = array_view{got_schema.get(), &got->array};
    ASSERT_EQ(e.view.length, a.view.length);
    compare_rows(&e.view, 0, &a.view, 0, e.view.length);

    // The result is a valid Arrow array on its own.
    ArrowError error;
    EXPECT_EQ(ArrowArrayViewValidate(&a.view, NANOARROW_VALIDATION_LEVEL_FULL, &error),
              NANOARROW_OK)
      << error.message;
  }
};

/// A table with a column of every fixed-width type, some with nulls.
std::unique_ptr<cudf::table> make_fixed_width_table()
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(fwcw<int8_t>({1, -2, 3, 4, 5}, null_at(1)).release());
  cols.push_back(fwcw<int16_t>({1, 2, 3, 4, 5}).release());
  cols.push_back(fwcw<int32_t>({1, 2, 3, 4, 5}, nulls_at({0, 4})).release());
  cols.push_back(fwcw<int64_t>({1, 2, 3, 4, 5}).release());
  cols.push_back(fwcw<uint8_t>({1, 2, 3, 4, 5}).release());
  cols.push_back(fwcw<uint16_t>({1, 2, 3, 4, 5}, null_at(2)).release());
  cols.push_back(fwcw<uint32_t>({1, 2, 3, 4, 5}).release());
  cols.push_back(fwcw<uint64_t>({1, 2, 3, 4, 5}).release());
  cols.push_back(fwcw<float>({1.5f, 2.f, 3.f, 4.f, 5.f}, null_at(3)).release());
  cols.push_back(fwcw<double>({1.5, 2., 3., 4., 5.}).release());
  cols.push_back(fwcw<bool>({true, false, true, true, false}, null_at(4)).release());
  cols.push_back(fwcw<bool>({true, false, true, true, false}).release());
  cols.push_back(cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>(
                   {1, 2, 3, 4, 5})
                   .release());
  cols.push_back(cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>(
                   {1, 2, 3, 4, 5}, null_at(0))
                   .release());
  cols.push_back(
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
      {1, 2, 3, 4, 5})
      .release());
  cols.push_back(
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>(
      {1, 2, 3, 4, 5})
      .release());
  cols.push_back(
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep>(
      {1, 2, 3, 4, 5})
      .release());
  cols.push_back(
    cudf::test::fixed_width_column_wrapper<cudf::duration_s, cudf::duration_s::rep>({1, 2, 3, 4, 5})
      .release());
  cols.push_back(cudf::test::fixed_width_column_wrapper<cudf::duration_ms, cudf::duration_ms::rep>(
                   {1, 2, 3, 4, 5})
                   .release());
  cols.push_back(cudf::test::fixed_width_column_wrapper<cudf::duration_us, cudf::duration_us::rep>(
                   {1, 2, 3, 4, 5})
                   .release());
  cols.push_back(cudf::test::fixed_width_column_wrapper<cudf::duration_ns, cudf::duration_ns::rep>(
                   {1, 2, 3, 4, 5}, null_at(1))
                   .release());
  cols.push_back(cudf::test::fixed_point_column_wrapper<int32_t>(
                   {1, 2, 3, 4, 5}, null_at(2), numeric::scale_type{-2})
                   .release());
  cols.push_back(
    cudf::test::fixed_point_column_wrapper<int64_t>({1, 2, 3, 4, 5}, numeric::scale_type{3})
      .release());
  cols.push_back(
    cudf::test::fixed_point_column_wrapper<__int128_t>({1, 2, 3, 4, 5}, numeric::scale_type{-10})
      .release());
  return std::make_unique<cudf::table>(std::move(cols));
}

/// A table of strings, lists, a struct and dictionaries, some with nulls.
std::unique_ptr<cudf::table> make_nested_table()
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(strcw({"a", "", "bcd", "efghijklmnop", "q"}, null_at(1)).release());
  cols.push_back(strcw({"", "", "", "", ""}).release());
  cols.push_back(lcw<int32_t>{{{1, 2}, {}, {3}, {4, 5, 6}, {7}}, null_at(1)}.release());
  cols.push_back(lcw<cudf::string_view>{{"a", "b"}, {}, {"c"}, {"", "d"}, {"e"}}.release());
  cols.push_back(lcw<int64_t>{{{1}, {2, 3}}, {}, {{}}, {{4}, {}, {5}}, {{6}}}.release());
  {
    auto ints = fwcw<int32_t>({1, 2, 3, 4, 5}, null_at(3));
    auto strs = strcw({"x", "y", "z", "w", "v"}, null_at(0));
    auto list = lcw<int32_t>{{1}, {2, 3}, {}, {4}, {5}};
    cols.push_back(structs({ints, strs, list}, null_at(2)).release());
  }
  {
    auto keys = strcw({"b", "a", "b", "c", "a"}, null_at(2));
    cols.push_back(cudf::dictionary::encode(keys));
  }
  {
    auto values = fwcw<int64_t>({10, 20, 10, 30, 20});
    cols.push_back(cudf::dictionary::encode(values, cudf::data_type{cudf::type_id::INT8}));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

TEST_F(PackedArrowHostTest, UnpackFixedWidth)
{
  expect_unpack_matches_to_arrow_host(make_fixed_width_table()->view());
}

TEST_F(PackedArrowHostTest, UnpackNested)
{
  expect_unpack_matches_to_arrow_host(make_nested_table()->view());
}

TEST_F(PackedArrowHostTest, UnpackSliced)
{
  // Packing a slice rebases offsets to zero and shifts null masks to bit zero.
  auto const fixed  = make_fixed_width_table();
  auto const nested = make_nested_table();
  for (auto [begin, end] : {std::pair{1, 4}, std::pair{3, 5}, std::pair{2, 2}}) {
    expect_unpack_matches_to_arrow_host(cudf::slice(fixed->view(), {begin, end})[0]);
    expect_unpack_matches_to_arrow_host(cudf::slice(nested->view(), {begin, end})[0]);
  }
}

TEST_F(PackedArrowHostTest, UnpackManyRows)
{
  // More than one null mask word and a row count that is not a multiple of 32
  auto constexpr num_rows = 1000;
  auto const seq = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto const validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 7 != 0; });
  auto const ints = fwcw<int32_t>(seq, seq + num_rows, validity);
  auto const bool_iter =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 == 0; });
  auto const bools        = fwcw<bool>(bool_iter, bool_iter + num_rows, validity);
  auto const strings_iter = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return std::string(i % 17, 'a' + i % 26); });
  auto const strs  = strcw(strings_iter, strings_iter + num_rows, validity);
  auto const table = cudf::table_view{{ints, bools, strs}};
  expect_unpack_matches_to_arrow_host(table);
  expect_unpack_matches_to_arrow_host(cudf::slice(table, {37, 999})[0]);
}

TEST_F(PackedArrowHostTest, UnpackLargeStrings)
{
  auto offsets = fwcw<int64_t>({0, 1, 3, 3, 6}).release();
  std::string const chars{"abcdef"};
  auto const stream = cudf::test::get_default_stream();
  auto strings =
    cudf::make_strings_column(4,
                              std::move(offsets),
                              rmm::device_buffer{chars.data(), chars.size(), stream},
                              1,
                              cudf::test::detail::make_null_mask(null_at(2), null_at(2) + 4).first);
  expect_unpack_matches_to_arrow_host(cudf::table_view{{strings->view()}});
}

TEST_F(PackedArrowHostTest, UnpackEmpty)
{
  // Zero rows
  auto const zero_rows = make_nested_table();
  expect_unpack_matches_to_arrow_host(cudf::slice(zero_rows->view(), {0, 0})[0]);
  // Zero columns, with and without rows
  expect_unpack_matches_to_arrow_host(cudf::table_view{std::vector<cudf::column_view>{}, 7});
  expect_unpack_matches_to_arrow_host(cudf::table_view{});
}

TEST_F(PackedArrowHostTest, UnpackEmptyDictionary)
{
  // A dictionary without children, as `make_empty_column` creates
  auto const dictionary = cudf::make_empty_column(cudf::data_type{cudf::type_id::DICTIONARY32});
  expect_unpack_matches_to_arrow_host(cudf::table_view{{dictionary->view()}});
}

TEST_F(PackedArrowHostTest, UnpackPackInto)
{
  expect_unpack_matches_to_arrow_host(make_fixed_width_table()->view(), packer::pack_into);
  expect_unpack_matches_to_arrow_host(make_nested_table()->view(), packer::pack_into);
}

TEST_F(PackedArrowHostTest, UnpackChunkedPack)
{
  expect_unpack_matches_to_arrow_host(make_fixed_width_table()->view(), packer::chunked_pack);
  expect_unpack_matches_to_arrow_host(make_nested_table()->view(), packer::chunked_pack);
  // Larger than the 1 MB bounce buffer, so the payload is copied in several chunks
  auto const seq  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto const ints = fwcw<int32_t>(seq, seq + 300'000);
  expect_unpack_matches_to_arrow_host(cudf::table_view{{ints}}, packer::chunked_pack);
}

TEST_F(PackedArrowHostTest, UnpackCompressed)
{
  auto const seq    = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto const ints   = fwcw<int64_t>(seq, seq + 100'000);
  auto const table  = cudf::table_view{{ints}};
  auto const packed = pack_into_host(table, cudf::experimental::pack_compression::cascaded);
  EXPECT_THROW(cudf::experimental::unpack_to_arrow_host(packed.metadata, packed.data),
               std::invalid_argument);
}

TEST_F(PackedArrowHostTest, UnpackOutOfBounds)
{
  auto const table      = make_fixed_width_table();
  auto const packed     = pack_to_host(table->view());
  auto const short_data = std::span<uint8_t const>{packed.data.data(), packed.data.size() / 2};
  EXPECT_THROW(cudf::experimental::unpack_to_arrow_host(packed.metadata, short_data),
               std::out_of_range);
  auto const short_metadata =
    std::span<uint8_t const>{packed.metadata.data(), packed.metadata.size() - 1};
  EXPECT_THROW(cudf::experimental::unpack_to_arrow_host(short_metadata, packed.data),
               cudf::logic_error);
}

TEST_F(PackedArrowHostTest, SchemaMetadataErrors)
{
  auto const table  = make_nested_table();
  auto const packed = pack_to_host(table->view());
  auto const meta   = make_columns_metadata(table->view());

  auto const too_few = std::span<cudf::column_metadata const>{meta}.first(meta.size() - 1);
  EXPECT_THROW(cudf::experimental::unpack_to_arrow_schema(packed.metadata, too_few),
               cudf::logic_error);
  std::vector<cudf::column_metadata> too_many{meta};
  too_many.emplace_back("extra");
  EXPECT_THROW(cudf::experimental::unpack_to_arrow_schema(packed.metadata, too_many),
               cudf::logic_error);
  auto const short_metadata =
    std::span<uint8_t const>{packed.metadata.data(), packed.metadata.size() - 1};
  EXPECT_THROW(cudf::experimental::unpack_to_arrow_schema(short_metadata, meta), cudf::logic_error);
}
