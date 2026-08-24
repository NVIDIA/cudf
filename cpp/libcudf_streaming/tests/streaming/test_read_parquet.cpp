/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "base_streaming_fixture.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_streaming/parquet.hpp>
#include <cudf_streaming/partition.hpp>
#include <cudf_streaming/partition_utils.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/coll/allgather.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>

#include <any>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace cudf_streaming;

class StreamingReadParquet : public BaseStreamingFixture {
 protected:
  void SetUp() override
  {
    BaseStreamingFixture::SetUp();
    constexpr int nfiles = 10;
    constexpr int nrows  = 10;

    temp_dir = std::filesystem::temp_directory_path() / "rapidsmpf_read_parquet_test";

    for (int i = 0; i < nfiles; ++i) {
      std::ostringstream filename_stream;
      filename_stream << std::setw(3) << std::setfill('0') << i << ".pq";
      std::filesystem::path filepath = temp_dir / filename_stream.str();
      source_files.push_back(filepath.string());
    }

    if (GlobalEnvironment->comm_->rank() == 0) {
      std::filesystem::create_directories(temp_dir);

      int start = 0;
      for (auto& file : source_files) {
        auto values = std::ranges::iota_view(start, start + nrows);
        cudf::test::fixed_width_column_wrapper<std::int32_t> col(values.begin(), values.end());

        std::vector<std::unique_ptr<cudf::column>> columns;
        columns.push_back(col.release());
        auto table = std::make_unique<cudf::table>(std::move(columns));

        cudf::io::sink_info sink{file};
        auto options = cudf::io::parquet_writer_options::builder(sink, table->view()).build();
        cudf::io::write_parquet(options);
        start += nrows + nrows / 2;
      }
    }

    GlobalEnvironment->barrier();
  }

  void TearDown() override
  {
    GlobalEnvironment->barrier();

    if (GlobalEnvironment->comm_->rank() == 0 && std::filesystem::exists(temp_dir)) {
      std::filesystem::remove_all(temp_dir);
    }

    BaseStreamingFixture::TearDown();
  }

  [[nodiscard]] cudf::io::source_info get_source_info(bool truncate_file_list) const
  {
    if (truncate_file_list) {
      std::vector<std::string> files(source_files.begin(), source_files.begin() + 2);
      return cudf::io::source_info(files);
    } else {
      return cudf::io::source_info(source_files);
    }
  }

  std::filesystem::path temp_dir;
  std::vector<std::string> source_files;
};

using ReadParquetParams =
  std::tuple<std::optional<std::int64_t>, std::optional<std::int64_t>, bool, bool>;

class StreamingReadParquetParams : public StreamingReadParquet,
                                   public ::testing::WithParamInterface<ReadParquetParams> {};

INSTANTIATE_TEST_SUITE_P(ReadParquetCombinations,
                         StreamingReadParquetParams,
                         ::testing::Combine(
                           // skip_rows
                           ::testing::Values(std::nullopt,
                                             std::optional<std::int64_t>{7},
                                             std::optional<std::int64_t>{19},
                                             std::optional<std::int64_t>{113}),
                           // num_rows
                           ::testing::Values(std::nullopt,
                                             std::optional<std::int64_t>{0},
                                             std::optional<std::int64_t>{3},
                                             std::optional<std::int64_t>{31},
                                             std::optional<std::int64_t>{83}),
                           // use_filter
                           ::testing::Values(false, true),
                           // truncate file list
                           ::testing::Values(false, true)),
                         [](::testing::TestParamInfo<ReadParquetParams> const& info) {
                           auto const& skip_rows          = std::get<0>(info.param);
                           auto const& num_rows           = std::get<1>(info.param);
                           auto const& use_filter         = std::get<2>(info.param);
                           auto const& truncate_file_list = std::get<3>(info.param);
                           std::string result             = "skip_rows_";
                           result +=
                             skip_rows.has_value() ? std::to_string(skip_rows.value()) : "none";
                           result += "_num_rows_";
                           result +=
                             num_rows.has_value() ? std::to_string(num_rows.value()) : "all";
                           if (use_filter) {
                             result += "_with_filter";
                           } else {
                             result += "_no_filter";
                           }
                           if (truncate_file_list) {
                             result += "_one_file";
                           } else {
                             result += "_all_files";
                           }
                           return result;
                         });

TEST_P(StreamingReadParquetParams, ReadParquet)
{
  auto [skip_rows, num_rows, use_filter, truncate_file_list] = GetParam();
  auto source                                                = get_source_info(truncate_file_list);

  auto options = cudf::io::parquet_reader_options::builder(source).build();
  if (skip_rows.has_value()) { options.set_skip_rows(skip_rows.value()); }
  if (num_rows.has_value()) { options.set_num_rows(num_rows.value()); }
  auto filter_expr = [&]() -> std::unique_ptr<filter> {
    if (!use_filter) { return nullptr; }
    auto stream = ctx->br()->stream_pool()->get_stream();
    auto owner  = new std::vector<std::any>;
    owner->push_back(std::make_shared<cudf::numeric_scalar<std::int32_t>>(15, true, stream));
    owner->push_back(std::make_shared<cudf::ast::literal>(
      *std::any_cast<std::shared_ptr<cudf::numeric_scalar<std::int32_t>>>(owner->at(0))));
    owner->push_back(std::make_shared<cudf::ast::column_reference>(0));
    owner->push_back(std::make_shared<cudf::ast::operation>(
      cudf::ast::ast_operator::LESS,
      *std::any_cast<std::shared_ptr<cudf::ast::column_reference>>(owner->at(2)),
      *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(1))));
    return std::make_unique<filter>(
      stream,
      *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->back()),
      rapidsmpf::OwningWrapper(static_cast<void*>(owner),
                               [](void* p) { delete static_cast<std::vector<std::any>*>(p); }));
  }();
  auto expected = [&]() {
    if (filter_expr != nullptr) {
      auto expected_options = options;
      expected_options.set_filter(filter_expr->filter);
      filter_expr->stream.synchronize();
      auto expected = cudf::io::read_parquet(expected_options).tbl;
      filter_expr->stream.synchronize();
      return expected;
    } else {
      return cudf::io::read_parquet(options).tbl;
    }
  }();
  auto ch = ctx->create_channel();
  std::vector<rapidsmpf::streaming::Actor> actors;

  actors.push_back(cudf_streaming::actor::read_parquet(
    ctx, GlobalEnvironment->comm_, ch, 4, options, 3, std::move(filter_expr)));

  std::vector<rapidsmpf::streaming::Message> messages;
  actors.push_back(rapidsmpf::streaming::actor::pull_from_channel(ctx, ch, messages));

  if (GlobalEnvironment->comm_->nranks() > 1 &&
      (skip_rows.value_or(0) > 0 || num_rows.has_value())) {
    // We don't yet implement skip_rows/num_rows in multi-rank mode
    EXPECT_THROW(rapidsmpf::streaming::run_actor_network(std::move(actors)), std::logic_error);
    return;
  }
  rapidsmpf::streaming::run_actor_network(std::move(actors));

  rapidsmpf::coll::AllGather allgather(GlobalEnvironment->comm_,
                                       /* op_id = */ 0,
                                       br.get());

  for (auto& msg : messages) {
    auto chunk            = msg.release<table_chunk>();
    auto seq              = msg.sequence_number();
    auto [reservation, _] = br->reserve(
      rapidsmpf::MemoryType::DEVICE, chunk.make_available_cost(), rapidsmpf::AllowOverbooking::YES);
    chunk               = chunk.make_available(reservation);
    auto packed_columns = cudf::pack(chunk.table_view(), chunk.stream(), br->device_mr());
    auto packed_data =
      rapidsmpf::PackedData{std::move(packed_columns.metadata),
                            br->move(std::move(packed_columns.gpu_data), chunk.stream())};

    allgather.insert(seq, std::move(packed_data));
  }

  allgather.insert_finished();

  // May as well check on all ranks, so we also mildly exercise the allgather.
  auto gathered_packed_data = allgather.wait_and_extract(rapidsmpf::coll::AllGather::Ordered::YES);
  auto result               = cudf_streaming::unpack_and_concat(
    std::move(gathered_packed_data), rmm::cuda_stream_default, br.get());
  EXPECT_EQ(result->num_rows(), expected->num_rows());
  EXPECT_EQ(result->num_columns(), expected->num_columns());
  EXPECT_EQ(result->num_columns(), 1);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected->view());
}
