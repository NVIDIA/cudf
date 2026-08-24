/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cudf_streaming/parquet.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/memory_resource>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <any>
#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rapidsmpf::ndsh {
namespace detail {

/**
 * @brief List all parquet files in a given path.
 *
 * @param root_path The path to look in.
 *
 * @return If `root_path` names a regular file that ends with `.parquet` then a singleton
 * vector of just that file. If `root_path` is a directory, then a vector containing all
 * regular files in that directory whose name ends with `.parquet`, in the order they are
 * listed.
 *
 * @throws std::runtime_error if the `root_path` doesn't name a regular file or a
 * directory. Or if it does name a regular file, but that file doesn't end in `.parquet`.
 */
[[nodiscard]] std::vector<std::string> list_parquet_files(std::string const root_path);

/**
 * @brief Get the path to a given table
 *
 * @param input_directory Input directory
 * @param table_name Name of table to find.
 *
 * @return Path to given table.
 */
[[nodiscard]] std::string get_table_path(std::string const& input_directory,
                                         std::string const& table_name);

/**
 * @brief Get cudf data types for all columns from parquet metadata.
 *
 * Reads parquet metadata to determine the cudf data type for each column.
 * The data types are inferred from the first file found for the given table.
 *
 * @param input_directory Directory containing input parquet files
 * @param table_name Name of the table (e.g., "lineitem")
 * @return Map from column name to cudf data type
 */
[[nodiscard]] std::map<std::string, cudf::data_type> get_column_types(
  std::string const& input_directory, std::string const& table_name);

}  // namespace detail

/**
 * @brief Create a date comparison filter expression.
 *
 * Creates a filter that compares a date column against a literal date value.
 * The operation will be equivalent to
 * "<column_name> <op> DATE '<year>-<month>-<day>'".
 *
 * @tparam timestamp_type The timestamp type to use for the filter scalar
 * (e.g., cudf::timestamp_D or cudf::timestamp_ms)
 * @param stream CUDA stream to use
 * @param date The date to compare against
 * @param column_name The name of the column to compare
 * @param op The comparison operator (e.g., LESS, LESS_EQUAL, GREATER)
 * @return Filter expression with proper lifetime management
 */
template <typename timestamp_type>
std::unique_ptr<cudf_streaming::filter> make_date_filter(rmm::cuda_stream_view stream,
                                                         cuda::std::chrono::year_month_day date,
                                                         std::string const& column_name,
                                                         cudf::ast::ast_operator op)
{
  auto owner    = new std::vector<std::any>;
  auto sys_days = cuda::std::chrono::sys_days(date);
  owner->push_back(std::make_shared<cudf::timestamp_scalar<timestamp_type>>(
    sys_days.time_since_epoch(), true, stream));
  owner->push_back(std::make_shared<cudf::ast::literal>(
    *std::any_cast<std::shared_ptr<cudf::timestamp_scalar<timestamp_type>>>(owner->at(0))));
  owner->push_back(std::make_shared<cudf::ast::column_name_reference>(column_name));
  owner->push_back(std::make_shared<cudf::ast::operation>(
    op,
    *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(owner->at(2)),
    *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(1))));
  return std::make_unique<cudf_streaming::filter>(
    stream,
    *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->back()),
    OwningWrapper(static_cast<void*>(owner),
                  [](void* p) { delete static_cast<std::vector<std::any>*>(p); }));
}

/**
 * @brief Create a date range filter expression.
 *
 * Creates a filter that checks if a date column falls within a half-open range.
 * The operation will be equivalent to
 * "<column_name> >= DATE '<start>' AND <column_name> < DATE '<end>'".
 *
 * @tparam timestamp_type The timestamp type to use for the filter scalars
 * (e.g., cudf::timestamp_D or cudf::timestamp_ms)
 * @param stream CUDA stream to use
 * @param start_date The start date (inclusive) of the range
 * @param end_date The end date (exclusive) of the range
 * @param column_name The name of the column to compare
 * @return Filter expression with proper lifetime management
 */
template <typename timestamp_type>
std::unique_ptr<cudf_streaming::filter> make_date_range_filter(
  rmm::cuda_stream_view stream,
  cuda::std::chrono::year_month_day start_date,
  cuda::std::chrono::year_month_day end_date,
  std::string const& column_name)
{
  auto owner = new std::vector<std::any>;

  // 0: column_reference
  owner->push_back(std::make_shared<cudf::ast::column_name_reference>(column_name));

  // 1, 2: Scalars for start and end dates
  owner->push_back(std::make_shared<cudf::timestamp_scalar<timestamp_type>>(
    cuda::std::chrono::sys_days(start_date).time_since_epoch(), true, stream));
  owner->push_back(std::make_shared<cudf::timestamp_scalar<timestamp_type>>(
    cuda::std::chrono::sys_days(end_date).time_since_epoch(), true, stream));

  // 3, 4: Literals for start and end dates
  owner->push_back(std::make_shared<cudf::ast::literal>(
    *std::any_cast<std::shared_ptr<cudf::timestamp_scalar<timestamp_type>>>(owner->at(1))));
  owner->push_back(std::make_shared<cudf::ast::literal>(
    *std::any_cast<std::shared_ptr<cudf::timestamp_scalar<timestamp_type>>>(owner->at(2))));

  // 5: (GE, column, literal<start>)
  owner->push_back(std::make_shared<cudf::ast::operation>(
    cudf::ast::ast_operator::GREATER_EQUAL,
    *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(owner->at(0)),
    *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(3))));

  // 6: (LT, column, literal<end>)
  owner->push_back(std::make_shared<cudf::ast::operation>(
    cudf::ast::ast_operator::LESS,
    *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(owner->at(0)),
    *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(4))));

  // 7: (AND, GE, LT)
  owner->push_back(std::make_shared<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND,
    *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->at(5)),
    *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->at(6))));

  return std::make_unique<cudf_streaming::filter>(
    stream,
    *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->back()),
    OwningWrapper(static_cast<void*>(owner),
                  [](void* p) { delete static_cast<std::vector<std::any>*>(p); }));
}

/**
 * @brief Sink messages into a channel and discard them.
 *
 * @param ctx Streaming context
 * @param ch Channel to discard messages from.
 *
 * @return Coroutine representing the shutdown and discard of the channel.
 */
[[nodiscard]] streaming::Actor sink_channel(std::shared_ptr<streaming::Context> ctx,
                                            std::shared_ptr<streaming::Channel> ch);

/**
 * @brief Consume messages from a channel and discard them.
 *
 * @param ctx Streaming context
 * @param ch Channel to consume messages from.
 *
 * @note If the channel contains `table_chunk`s, moves them to device and prints small
 * amount of detail about them (row and column count).
 *
 * @return Coroutine representing consuming and discarding messages in channel.
 */
[[nodiscard]] streaming::Actor consume_channel(std::shared_ptr<streaming::Context> ctx,
                                               std::shared_ptr<streaming::Channel> ch_in);

///< @brief Communicator type to use
enum class CommType : std::uint8_t {
  SINGLE,  ///< Single process communicator
  MPI,     ///< MPI backed communicator
  UCXX,    ///< UCXX backed communicator
  MAX,     ///< Max value
};

[[nodiscard]] constexpr std::array<std::string_view, static_cast<std::size_t>(CommType::MAX)>
comm_type_names()
{
  return {"single", "mpi", "ucxx"};
}

[[nodiscard]] bool is_comm_type_available(CommType comm_type);

[[nodiscard]] std::string available_comm_types();

[[nodiscard]] std::optional<CommType> parse_comm_type(std::string_view name);

///< @brief Configuration options for the query
struct ProgramOptions {
  int num_streaming_threads{1};  ///< Number of streaming threads to use
  int num_iterations{2};         ///< Number of iterations of query to run
  int num_streams{16};           ///< Number of streams in stream pool
#ifdef CUDF_STREAMING_HAVE_UCXX
  CommType comm_type{CommType::UCXX};  ///< Type of communicator to create
#elif defined(CUDF_STREAMING_HAVE_MPI)
  CommType comm_type{CommType::MPI};  ///< Type of communicator to create
#else
  CommType comm_type{CommType::SINGLE};  ///< Type of communicator to create
#endif
  std::optional<std::chrono::milliseconds>
    periodic_spill;  ///< Duration between background periodic spilling checks
  cudf::size_type num_rows_per_chunk{100'000'000};  ///< Number of rows to produce per chunk read
  std::optional<double> spill_device_limit{std::nullopt};  ///< Optional fractional spill limit
  bool no_pinned_host_memory{false};                       ///< Disable pinned host memory?
  bool use_shuffle_join = false;                           ///< Use shuffle join for "big" joins?
  std::string output_file;                                 ///< File to write output to
  std::string input_directory;                             ///< Directory containing input files.
};

/**
 * @brief Parse commandline arguments
 *
 * @param argc Number of arguments
 * @param argv Arguments
 *
 * @return `ProgramOptions` struct with parsed arguments.
 */
ProgramOptions parse_arguments(int argc, char** argv);

/**
 * @brief Create a streaming execution context and communicator for a query.
 *
 * @param arguments Arguments to configure the context
 * @param mr The device memory resource to use for all allocations.
 *
 * @return Pair of shared pointer to new streaming context and communicator.
 */
std::pair<std::shared_ptr<streaming::Context>, std::shared_ptr<Communicator>> create_context(
  ProgramOptions& arguments, cuda::mr::any_resource<cuda::mr::device_accessible> mr);

/**
 * @brief Finalize MPI when going out of scope.
 */
struct FinalizeMPI {
  ~FinalizeMPI() noexcept;
};
}  // namespace rapidsmpf::ndsh
