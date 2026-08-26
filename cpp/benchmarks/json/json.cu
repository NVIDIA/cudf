/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/json/json.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

std::vector<std::string> const Books{
  R"json({
"category": "reference",
"author": "Nigel Rees",
"title": "Sayings of the Century",
"price": 8.95
})json",
  R"json({
"category": "fiction",
"author": "Evelyn Waugh",
"title": "Sword of Honour",
"price": 12.99
})json",
  R"json({
"category": "fiction",
"author": "Herman Melville",
"title": "Moby Dick",
"isbn": "0-553-21311-3",
"price": 8.99
})json",
  R"json({
"category": "fiction",
"author": "J. R. R. Tolkien",
"title": "The Lord of the Rings",
"isbn": "0-395-19395-8",
"price": 22.99
})json"};
constexpr int Approx_book_size = 110;
std::vector<std::string> const Bicycles{
  R"json({"color": "red", "price": 9.95})json",
  R"json({"color": "green", "price": 29.95})json",
  R"json({"color": "blue", "price": 399.95})json",
  R"json({"color": "yellow", "price": 99.95})json",
  R"json({"color": "mauve", "price": 199.95})json",
};
constexpr int Approx_bicycle_size = 33;
std::string Misc{"\n\"expensive\": 10\n"};

struct json_benchmark_row_builder {
  int const desired_bytes;
  cudf::size_type const num_rows;
  cudf::column_device_view const d_books_bicycles[2];  // Books, Bicycles strings
  cudf::column_device_view const d_book_pct;           // Book percentage
  cudf::column_device_view const d_misc_order;         // Misc-Store order
  cudf::column_device_view const d_store_order;        // Books-Bicycles order
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  // internal data structure for {bytes, out_ptr} with operator+=
  struct bytes_and_ptr {
    cudf::size_type bytes;
    char* ptr;
    __device__ bytes_and_ptr& operator+=(cudf::string_view const& str_append)
    {
      bytes += str_append.size_bytes();
      if (ptr) { ptr = cudf::strings::detail::copy_string(ptr, str_append); }
      return *this;
    }
  };

  __device__ inline void copy_items(int this_idx,
                                    cudf::size_type num_items,
                                    bytes_and_ptr& output_str)
  {
    cudf::string_view comma(",\n", 2);
    for (int i = 0; i < num_items; i++) {
      if (i > 0) { output_str += comma; }
      int idx   = threadIdx.x % d_books_bicycles[this_idx].size();
      auto item = d_books_bicycles[this_idx].element<cudf::string_view>(idx);
      output_str += item;
    }
  }

  __device__ void operator()(cudf::size_type idx)
  {
    int num_books       = 2;
    int num_bicycles    = 2;
    int remaining_bytes = max(
      0, desired_bytes - ((num_books * Approx_book_size) + (num_bicycles * Approx_bicycle_size)));

    // divide up the remainder between books and bikes
    auto book_pct = d_book_pct.element<float>(idx);
    // {Misc, store} OR {store, Misc}
    // store: {books, bicycles} OR store: {bicycles, books}
    float bicycle_pct = 1.0f - book_pct;
    num_books += (remaining_bytes * book_pct) / Approx_book_size;
    num_bicycles += (remaining_bytes * bicycle_pct) / Approx_bicycle_size;

    char* out_ptr = d_chars ? d_chars + d_offsets[idx] : nullptr;
    bytes_and_ptr output_str{0, out_ptr};
    //
    cudf::string_view comma(",\n", 2);
    cudf::string_view brace1("{\n", 2);
    cudf::string_view store_member_start[2]{{"\"book\": [\n", 10}, {"\"bicycle\": [\n", 13}};
    cudf::string_view store("\"store\": {\n", 11);
    cudf::string_view Misc{"\"expensive\": 10", 15};
    cudf::string_view brace2("\n}", 2);
    cudf::string_view square2{"\n]", 2};

    output_str += brace1;
    if (d_misc_order.element<bool>(idx)) {  // Misc. first.
      output_str += Misc;
      output_str += comma;
    }
    output_str += store;
    for (int store_order = 0; store_order < 2; store_order++) {
      if (store_order > 0) { output_str += comma; }
      int this_idx    = (d_store_order.element<bool>(idx) == store_order);
      auto& mem_start = store_member_start[this_idx];
      output_str += mem_start;
      copy_items(this_idx, this_idx == 0 ? num_books : num_bicycles, output_str);
      output_str += square2;
    }
    output_str += brace2;
    if (!d_misc_order.element<bool>(idx)) {  // Misc, if not first.
      output_str += comma;
      output_str += Misc;
    }
    output_str += brace2;
    if (!output_str.ptr) { d_sizes[idx] = output_str.bytes; }
  }
};

auto build_json_string_column(int desired_bytes, int num_rows)
{
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::FLOAT32, distribution_id::UNIFORM, 0.0, 1.0);
  auto float_2bool_columns =
    create_random_table({cudf::type_id::FLOAT32, cudf::type_id::BOOL8, cudf::type_id::BOOL8},
                        row_count{num_rows},
                        profile);

  cudf::test::strings_column_wrapper books(Books.begin(), Books.end());
  cudf::test::strings_column_wrapper bicycles(Bicycles.begin(), Bicycles.end());
  auto d_books       = cudf::column_device_view::create(books);
  auto d_bicycles    = cudf::column_device_view::create(bicycles);
  auto d_book_pct    = cudf::column_device_view::create(float_2bool_columns->get_column(0));
  auto d_misc_order  = cudf::column_device_view::create(float_2bool_columns->get_column(1));
  auto d_store_order = cudf::column_device_view::create(float_2bool_columns->get_column(2));
  json_benchmark_row_builder jb{
    desired_bytes, num_rows, {*d_books, *d_bicycles}, *d_book_pct, *d_misc_order, *d_store_order};
  auto [offsets, chars] = cudf::strings::detail::make_strings_children(
    jb, num_rows, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  return cudf::make_strings_column(num_rows, std::move(offsets), chars.release(), 0, {});
}

static std::string queries[] = {"$",
                                "$.store",
                                "$.store.book",
                                "$.store.*",
                                "$.store.book[*]",
                                "$.store.book[*].category",
                                "$.store['bicycle']",
                                "$.store.book[*]['isbn']",
                                "$.store.bicycle[1]"};

static void bench_query(nvbench::state& state)
{
  srand(5236);

  auto const num_rows              = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const desired_bytes         = static_cast<cudf::size_type>(state.get_int64("bytes"));
  auto const query                 = state.get_int64("query");
  std::string_view const json_path = queries[query];

  auto const stream = cudf::get_default_stream();
  auto input        = build_json_string_column(desired_bytes, num_rows);
  cudf::strings_column_view scv(input->view());
  size_t num_chars = scv.chars_size(stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  // This isn't strictly 100% accurate. a given query isn't necessarily
  // going to visit every single incoming character but in spirit it does.
  state.add_global_memory_reads<nvbench::int8_t>(num_chars);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    [[maybe_unused]] auto result = cudf::get_json_object(scv, json_path);
  });
}

NVBENCH_BENCH(bench_query)
  .set_name("json_path")
  .add_int64_axis("bytes", {300, 600, 4096})
  .add_int64_axis("num_rows", {100, 1000, 100000, 400000})
  .add_int64_axis("query", {0, 1, 2, 3, 4, 5, 6, 7, 8});
