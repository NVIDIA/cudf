/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/ndsh_data_generator/ndsh_data_generator.hpp>
#include <benchmarks/common/nvtx_ranges.hpp>
#include <benchmarks/common/table_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/join/join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <future>
#include <iterator>
#include <unordered_set>
#include <utility>

namespace {

std::vector<std::string> const ORDERS_SCHEMA   = {"o_orderkey",
                                                  "o_custkey",
                                                  "o_orderstatus",
                                                  "o_totalprice",
                                                  "o_orderdate",
                                                  "o_orderpriority",
                                                  "o_clerk",
                                                  "o_shippriority",
                                                  "o_comment"};
std::vector<std::string> const LINEITEM_SCHEMA = {"l_orderkey",
                                                  "l_partkey",
                                                  "l_suppkey",
                                                  "l_linenumber",
                                                  "l_quantity",
                                                  "l_extendedprice",
                                                  "l_discount",
                                                  "l_tax",
                                                  "l_returnflag",
                                                  "l_linestatus",
                                                  "l_shipdate",
                                                  "l_commitdate",
                                                  "l_receiptdate",
                                                  "l_shipinstruct",
                                                  "l_shipmode",
                                                  "l_comment"};
std::vector<std::string> const PART_SCHEMA     = {"p_partkey",
                                                  "p_name",
                                                  "p_mfgr",
                                                  "p_brand",
                                                  "p_type",
                                                  "p_size",
                                                  "p_container",
                                                  "p_retailprice",
                                                  "p_comment"};
std::vector<std::string> const PARTSUPP_SCHEMA = {
  "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"};
std::vector<std::string> const SUPPLIER_SCHEMA = {
  "s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"};
std::vector<std::string> const CUSTOMER_SCHEMA = {"c_custkey",
                                                  "c_name",
                                                  "c_address",
                                                  "c_nationkey",
                                                  "c_phone",
                                                  "c_acctbal",
                                                  "c_mktsegment",
                                                  "c_comment"};
std::vector<std::string> const NATION_SCHEMA   = {
  "n_nationkey", "n_name", "n_regionkey", "n_comment"};
std::vector<std::string> const REGION_SCHEMA = {"r_regionkey", "r_name", "r_comment"};

std::unordered_map<std::string, std::vector<std::string> const> const SCHEMAS = {
  {"orders", ORDERS_SCHEMA},
  {"lineitem", LINEITEM_SCHEMA},
  {"part", PART_SCHEMA},
  {"partsupp", PARTSUPP_SCHEMA},
  {"supplier", SUPPLIER_SCHEMA},
  {"customer", CUSTOMER_SCHEMA},
  {"nation", NATION_SCHEMA},
  {"region", REGION_SCHEMA}};

class device_buffer_sink final : public cudf::io::data_sink {
 public:
  device_buffer_sink(std::size_t capacity, rmm::cuda_stream_view stream)
    : buffer_{capacity, stream}, stream_{stream}
  {
  }

  void host_write(void const* data, std::size_t size) override
  {
    auto* destination = reserve(size);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(destination, data, size, cudaMemcpyHostToDevice, stream_.value()));
    stream_.synchronize();
  }

  [[nodiscard]] bool supports_device_write() const override { return true; }

  [[nodiscard]] bool is_device_write_preferred(std::size_t) const override { return true; }

  void device_write(void const* gpu_data, std::size_t size, cuda::stream_ref stream) override
  {
    device_write_async(gpu_data, size, stream).get();
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       std::size_t size,
                                       cuda::stream_ref stream) override
  {
    auto* destination = reserve(size);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(destination, gpu_data, size, cudaMemcpyDeviceToDevice, stream.get()));
    return std::async(std::launch::deferred, [stream] { stream.sync(); });
  }

  void flush() override {}

  [[nodiscard]] std::size_t bytes_written() override { return size_; }

  [[nodiscard]] void const* data() const { return buffer_.data(); }

 private:
  void* reserve(std::size_t size)
  {
    CUDF_EXPECTS(size <= buffer_.size() - size_, "Parquet device sink capacity exceeded");
    auto* destination = static_cast<std::byte*>(buffer_.data()) + size_;
    size_ += size;
    return destination;
  }

  rmm::device_buffer buffer_;
  rmm::cuda_stream_view stream_;
  std::size_t size_{};
};
}  // namespace

ndsh_parquet_source::~ndsh_parquet_source()
{
  for (auto const& buffer : buffers_) {
    (void)cudaFreeHost(buffer.data);
  }
}

ndsh_parquet_source::ndsh_parquet_source(ndsh_parquet_source&& other) noexcept
  : buffers_{std::move(other.buffers_)}
{
  other.buffers_.clear();
}

ndsh_parquet_source& ndsh_parquet_source::operator=(ndsh_parquet_source&& other) noexcept
{
  if (this != &other) {
    for (auto const& buffer : buffers_) {
      (void)cudaFreeHost(buffer.data);
    }
    buffers_ = std::move(other.buffers_);
    other.buffers_.clear();
  }
  return *this;
}

[[nodiscard]] cudf::io::source_info ndsh_parquet_source::make_source_info() const
{
  std::vector<cudf::host_span<std::byte const>> spans;
  spans.reserve(buffers_.size());
  std::transform(
    buffers_.begin(), buffers_.end(), std::back_inserter(spans), [](auto const& buffer) {
      return cudf::host_span<std::byte const>(reinterpret_cast<std::byte const*>(buffer.data),
                                              buffer.size);
    });
  return cudf::io::source_info(
    cudf::host_span<cudf::host_span<std::byte const>>(spans.data(), spans.size()));
}

void ndsh_parquet_source::append_from_device(void const* device_data,
                                             std::size_t size,
                                             rmm::cuda_stream_view stream)
{
  void* host_buffer{};
  CUDF_CUDA_TRY(cudaMallocHost(&host_buffer, size));
  buffers_.push_back({host_buffer, size});
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(host_buffer, device_data, size, cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
}

cudf::table_view table_with_names::table() const { return tbl->view(); }

cudf::column_view table_with_names::column(std::string const& col_name) const
{
  return tbl->view().column(column_id(col_name));
}

std::vector<std::string> const& table_with_names::column_names() const { return col_names; }

cudf::size_type table_with_names::column_id(std::string const& col_name) const
{
  auto it = std::find(col_names.begin(), col_names.end(), col_name);
  if (it == col_names.end()) {
    std::string err_msg = "Column `" + col_name + "` not found";
    throw std::runtime_error(err_msg);
  }
  return std::distance(col_names.begin(), it);
}

table_with_names& table_with_names::append(std::unique_ptr<cudf::column>& col,
                                           std::string const& col_name)
{
  auto cols = tbl->release();
  cols.push_back(std::move(col));
  tbl = std::make_unique<cudf::table>(std::move(cols));
  col_names.push_back(col_name);
  return (*this);
}

cudf::table_view table_with_names::select(std::vector<std::string> const& col_names) const
{
  CUDF_BENCHMARK_RANGE();
  std::vector<cudf::size_type> col_indices;
  for (auto const& col_name : col_names) {
    col_indices.push_back(column_id(col_name));
  }
  return tbl->select(col_indices);
}

void table_with_names::to_parquet(std::string const& filepath) const
{
  CUDF_BENCHMARK_RANGE();
  auto const sink_info = cudf::io::sink_info(filepath);
  cudf::io::table_metadata metadata;
  metadata.schema_info =
    std::vector<cudf::io::column_name_info>(col_names.begin(), col_names.end());
  auto const table_input_metadata = cudf::io::table_input_metadata{metadata};
  auto builder = cudf::io::parquet_writer_options::builder(sink_info, tbl->view());
  builder.metadata(table_input_metadata);
  auto const options = builder.build();
  cudf::io::write_parquet(options);
}

std::unique_ptr<cudf::table> join_and_gather(cudf::table_view const& left_input,
                                             cudf::table_view const& right_input,
                                             std::vector<cudf::size_type> const& left_on,
                                             std::vector<cudf::size_type> const& right_on,
                                             cudf::null_equality compare_nulls)
{
  CUDF_BENCHMARK_RANGE();
  constexpr auto oob_policy = cudf::out_of_bounds_policy::DONT_CHECK;
  auto [left_join_indices, right_join_indices] =
    cudf::inner_join(left_input.select(left_on),
                     right_input.select(right_on),
                     compare_nulls,
                     cudf::get_default_stream(),
                     cudf::get_current_device_resource_ref());

  auto const left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto const right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto const left_indices_col  = cudf::column_view{left_indices_span};
  auto const right_indices_col = cudf::column_view{right_indices_span};

  auto const left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
  auto const right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

std::unique_ptr<table_with_names> apply_inner_join(
  std::unique_ptr<table_with_names> const& left_input,
  std::unique_ptr<table_with_names> const& right_input,
  std::vector<std::string> const& left_on,
  std::vector<std::string> const& right_on,
  cudf::null_equality compare_nulls)
{
  CUDF_BENCHMARK_RANGE();
  std::vector<cudf::size_type> left_on_indices;
  std::vector<cudf::size_type> right_on_indices;
  std::transform(
    left_on.begin(), left_on.end(), std::back_inserter(left_on_indices), [&](auto const& col_name) {
      return left_input->column_id(col_name);
    });
  std::transform(right_on.begin(),
                 right_on.end(),
                 std::back_inserter(right_on_indices),
                 [&](auto const& col_name) { return right_input->column_id(col_name); });
  auto table = join_and_gather(
    left_input->table(), right_input->table(), left_on_indices, right_on_indices, compare_nulls);
  std::vector<std::string> merged_column_names;
  merged_column_names.reserve(left_input->column_names().size() +
                              right_input->column_names().size());
  std::copy(left_input->column_names().begin(),
            left_input->column_names().end(),
            std::back_inserter(merged_column_names));
  std::copy(right_input->column_names().begin(),
            right_input->column_names().end(),
            std::back_inserter(merged_column_names));
  return std::make_unique<table_with_names>(std::move(table), merged_column_names);
}

std::unique_ptr<table_with_names> apply_filter(std::unique_ptr<table_with_names> const& table,
                                               cudf::ast::operation const& predicate)
{
  CUDF_BENCHMARK_RANGE();
  auto const boolean_mask = cudf::compute_column(table->table(), predicate);
  auto result_table       = cudf::apply_boolean_mask(table->table(), boolean_mask->view());
  return std::make_unique<table_with_names>(std::move(result_table), table->column_names());
}

std::unique_ptr<table_with_names> apply_mask(std::unique_ptr<table_with_names> const& table,
                                             std::unique_ptr<cudf::column> const& mask)
{
  CUDF_BENCHMARK_RANGE();
  auto result_table = cudf::apply_boolean_mask(table->table(), mask->view());
  return std::make_unique<table_with_names>(std::move(result_table), table->column_names());
}

std::unique_ptr<table_with_names> apply_groupby(std::unique_ptr<table_with_names> const& table,
                                                groupby_context_t const& ctx)
{
  CUDF_BENCHMARK_RANGE();
  auto const keys = table->select(ctx.keys);
  cudf::groupby::groupby groupby_obj(keys);
  std::vector<std::string> result_column_names;
  result_column_names.insert(result_column_names.end(), ctx.keys.begin(), ctx.keys.end());
  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto& [value_col, aggregations] : ctx.values) {
    requests.emplace_back(cudf::groupby::aggregation_request());
    for (auto& agg : aggregations) {
      if (agg.first == cudf::aggregation::Kind::SUM) {
        requests.back().aggregations.push_back(
          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      } else if (agg.first == cudf::aggregation::Kind::MEAN) {
        requests.back().aggregations.push_back(
          cudf::make_mean_aggregation<cudf::groupby_aggregation>());
      } else if (agg.first == cudf::aggregation::Kind::COUNT_ALL) {
        requests.back().aggregations.push_back(
          cudf::make_count_aggregation<cudf::groupby_aggregation>());
      } else {
        throw std::runtime_error("Unsupported aggregation");
      }
      result_column_names.push_back(agg.second);
    }
    requests.back().values = table->column(value_col);
  }
  auto agg_results = groupby_obj.aggregate(requests);
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  for (auto i = 0; i < agg_results.first->num_columns(); i++) {
    auto col = std::make_unique<cudf::column>(agg_results.first->get_column(i));
    result_columns.push_back(std::move(col));
  }
  for (size_t i = 0; i < agg_results.second.size(); i++) {
    for (size_t j = 0; j < agg_results.second[i].results.size(); j++) {
      result_columns.push_back(std::move(agg_results.second[i].results[j]));
    }
  }
  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
  return std::make_unique<table_with_names>(std::move(result_table), result_column_names);
}

std::unique_ptr<table_with_names> apply_orderby(std::unique_ptr<table_with_names> const& table,
                                                std::vector<std::string> const& sort_keys,
                                                std::vector<cudf::order> const& sort_key_orders)
{
  CUDF_BENCHMARK_RANGE();
  std::vector<cudf::column_view> column_views;
  for (auto& key : sort_keys) {
    column_views.push_back(table->column(key));
  }
  auto result_table =
    cudf::sort_by_key(table->table(), cudf::table_view{column_views}, sort_key_orders);
  return std::make_unique<table_with_names>(std::move(result_table), table->column_names());
}

std::unique_ptr<table_with_names> apply_reduction(cudf::column_view const& column,
                                                  cudf::aggregation::Kind const& agg_kind,
                                                  std::string const& col_name)
{
  CUDF_BENCHMARK_RANGE();
  auto const agg            = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto const result         = cudf::reduce(column, *agg, column.type());
  cudf::size_type const len = 1;
  auto col                  = cudf::make_column_from_scalar(*result, len);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col));
  auto result_table                  = std::make_unique<cudf::table>(std::move(columns));
  std::vector<std::string> col_names = {col_name};
  return std::make_unique<table_with_names>(std::move(result_table), col_names);
}

std::unique_ptr<table_with_names> read_parquet(
  cudf::io::source_info const& source_info,
  std::vector<std::string> const& columns,
  std::unique_ptr<cudf::ast::operation> const& predicate)
{
  CUDF_BENCHMARK_RANGE();
  auto builder = cudf::io::parquet_reader_options_builder(source_info);
  if (!columns.empty()) { builder.column_names(columns); }
  if (predicate) { builder.filter(*predicate); }
  auto const options       = builder.build();
  auto table_with_metadata = cudf::io::read_parquet(options);
  std::vector<std::string> column_names;
  for (auto const& col_info : table_with_metadata.metadata.schema_info) {
    column_names.push_back(col_info.name);
  }
  return std::make_unique<table_with_names>(std::move(table_with_metadata.tbl), column_names);
}

std::tm make_tm(int year, int month, int day)
{
  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon  = month - 1;
  tm.tm_mday = day;
  return tm;
}

int32_t days_since_epoch(int year, int month, int day)
{
  std::tm tm             = make_tm(year, month, day);
  std::tm epoch          = make_tm(1970, 1, 1);
  std::time_t time       = std::mktime(&tm);
  std::time_t epoch_time = std::mktime(&epoch);
  double diff            = std::difftime(time, epoch_time) / (60 * 60 * 24);
  return static_cast<int32_t>(diff);
}

void write_to_parquet_device_buffer(std::unique_ptr<cudf::table> const& table,
                                    std::vector<std::string> const& col_names,
                                    ndsh_parquet_source& source,
                                    std::size_t max_parquet_file_bytes)
{
  CUDF_BENCHMARK_RANGE();
  CUDF_EXPECTS(max_parquet_file_bytes > 0, "Maximum Parquet file size must be positive");
  auto const stream = cudf::get_default_stream();

  // Prepare the table metadata
  cudf::io::table_metadata metadata;
  std::vector<cudf::io::column_name_info> col_name_infos;
  for (auto& col_name : col_names) {
    col_name_infos.push_back(cudf::io::column_name_info(col_name));
  }
  metadata.schema_info            = col_name_infos;
  auto const table_input_metadata = cudf::io::table_input_metadata{metadata};

  auto const est_size             = static_cast<std::size_t>(estimate_size(table->view()));
  constexpr auto SINK_SLACK_BYTES = 64ul << 20;  // Parquet metadata and compression overhead
  auto const num_partitions =
    std::max<std::size_t>(1, cudf::util::div_rounding_up_safe(est_size, max_parquet_file_bytes));
  auto const rows_per_partition = cudf::util::div_rounding_up_safe(
    table->num_rows(), static_cast<cudf::size_type>(num_partitions));
  std::vector<cudf::size_type> splits(num_partitions - 1);
  std::generate_n(splits.begin(), splits.size(), [rows_per_partition, i = 0]() mutable {
    return (i += rows_per_partition);
  });
  auto const partitions = cudf::split(table->view(), splits, stream);

  for (auto const& partition : partitions) {
    auto const partition_size = static_cast<std::size_t>(estimate_size(partition));
    auto const sink_capacity  = partition_size + (partition_size / 20) + SINK_SLACK_BYTES;
    device_buffer_sink sink{sink_capacity, stream};
    auto const sink_info = cudf::io::sink_info(&sink);

    auto const write_range = cudf::benchmark::scoped_range{"write_parquet_to_device_buffer"};
    {
      auto builder = cudf::io::parquet_writer_options::builder(sink_info, partition);
      builder.metadata(table_input_metadata);
      auto const options = builder.build();
      cudf::io::write_parquet(options, stream);
    }

    auto const copy_range = cudf::benchmark::scoped_range{"copy_parquet_device_buffer_to_host"};
    source.append_from_device(sink.data(), sink.bytes_written(), stream);
  }
}

void generate_parquet_data_sources(double scale_factor,
                                   std::vector<std::string> const& table_names,
                                   ndsh_data_sources& sources,
                                   ndsh_data_generation_options const& options)
{
  CUDF_BENCHMARK_RANGE();
  auto const mr = cudf::get_current_device_resource_ref();

  std::unordered_set<std::string> const requested_table_names = [&table_names]() {
    if (table_names.empty()) {
      return std::unordered_set<std::string>{
        "orders", "lineitem", "part", "partsupp", "supplier", "customer", "nation", "region"};
    }
    return std::unordered_set(table_names.begin(), table_names.end());
  }();
  std::for_each(requested_table_names.begin(),
                requested_table_names.end(),
                [&](auto const& table_name) { sources.try_emplace(table_name); });

  auto const stream = cudf::get_default_stream();

  if (sources.count("orders") or sources.count("lineitem")) {
    if (sources.count("orders")) {
      auto orders =
        cudf::datagen::generate_orders(scale_factor, options.orders_per_chunk, stream, mr);
      write_to_parquet_device_buffer(
        orders, SCHEMAS.at("orders"), sources.at("orders"), options.max_parquet_file_bytes);
    }
    if (sources.count("lineitem")) {
      auto lineitem_schema = LINEITEM_SCHEMA;
      if (!options.include_lineitem_comment) { lineitem_schema.pop_back(); }
      cudf::datagen::generate_lineitem_partitions(
        scale_factor,
        options.orders_per_chunk,
        [&](auto lineitem) {
          write_to_parquet_device_buffer(
            lineitem, lineitem_schema, sources.at("lineitem"), options.max_parquet_file_bytes);
        },
        options.include_lineitem_comment,
        stream,
        mr);
    }
  }

  if (sources.count("part")) {
    auto part = cudf::datagen::generate_part(scale_factor, stream, mr);
    write_to_parquet_device_buffer(
      part, SCHEMAS.at("part"), sources.at("part"), options.max_parquet_file_bytes);
  }

  if (sources.count("partsupp")) {
    auto partsupp = cudf::datagen::generate_partsupp(scale_factor, stream, mr);
    write_to_parquet_device_buffer(
      partsupp, SCHEMAS.at("partsupp"), sources.at("partsupp"), options.max_parquet_file_bytes);
  }

  if (sources.count("supplier")) {
    auto supplier = cudf::datagen::generate_supplier(scale_factor, stream, mr);
    write_to_parquet_device_buffer(
      supplier, SCHEMAS.at("supplier"), sources.at("supplier"), options.max_parquet_file_bytes);
  }

  if (sources.count("customer")) {
    auto customer = cudf::datagen::generate_customer(scale_factor, stream, mr);
    write_to_parquet_device_buffer(
      customer, SCHEMAS.at("customer"), sources.at("customer"), options.max_parquet_file_bytes);
  }

  if (sources.count("nation")) {
    auto nation = cudf::datagen::generate_nation(stream, mr);
    write_to_parquet_device_buffer(
      nation, SCHEMAS.at("nation"), sources.at("nation"), options.max_parquet_file_bytes);
  }

  if (sources.count("region")) {
    auto region = cudf::datagen::generate_region(stream, mr);
    write_to_parquet_device_buffer(
      region, SCHEMAS.at("region"), sources.at("region"), options.max_parquet_file_bytes);
  }
}
