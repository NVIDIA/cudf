/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>

#include <cudf_streaming/parquet.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/core/spillable_messages.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudf_streaming::actor {

namespace {

/**
 * @brief Per-context cache for file-backed messages.
 *
 * FileCache caches file read results by storing message copies in the associated
 * Context's SpillableMessages instance. By tying cached data to the Context, the
 * lifetime of cached entries matches the lifetime of the Context itself.
 *
 * Each cache instance is scoped to a single Context and is shared across callers
 * using that Context.
 *
 * The cache is thread-safe.
 */
class FileCache {
 public:
  struct Key {
    std::vector<std::string> filepaths;
    std::int64_t skip_rows;
    std::size_t skip_bytes;
    std::optional<std::int64_t> num_rows;
    std::optional<std::int64_t> num_bytes;
    std::optional<std::vector<std::string>> column_names;
    std::optional<std::vector<cudf::size_type>> column_indices;
    std::vector<std::vector<cudf::size_type>> row_groups;

    // Lexicographical comparison of all data members.
    auto operator<=>(Key const&) const = default;
  };

  /**
   * @brief Construct a FileCache.
   *
   * @param mem_type Memory type used for cache storage.
   */
  FileCache(rapidsmpf::MemoryType mem_type = rapidsmpf::MemoryType::HOST) : mem_type_{mem_type} {}

  /**
   * @brief Insert a message into the cache.
   *
   * The message is copied into the memory type configured for this cache
   * and stored in the associated Context's SpillableMessages instance.
   *
   * @param ctx Streaming context.
   * @param key Cache key identifying the message.
   * @param msg Message to cache.
   * @return True if the message was inserted, false if the key already existed.
   */
  bool insert(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
              Key key,
              rapidsmpf::streaming::Message const& msg)
  {
    auto reservation = ctx->br()->reserve_or_fail(msg.copy_cost(), mem_type_);
    auto msg_copy    = msg.copy(reservation);

    std::lock_guard lock(mutex_);
    if (cache_.contains(key)) { return false; }
    cache_.emplace(std::move(key), ctx->spillable_messages()->insert(std::move(msg_copy)));
    return true;
  }

  /**
   * @brief Retrieve a cached message.
   *
   * If the key exists, the cached message is copied out of spillable storage
   * using newly reserved memory, prioritizing memory types in `MEMORY_TYPES`
   * order.
   *
   * @param ctx Streaming context.
   * @param key Cache key to look up.
   * @return The cached message, or std::nullopt if the key is not present.
   */
  std::optional<rapidsmpf::streaming::Message> get(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx, Key const& key) const
  {
    auto& stats = *ctx->statistics();

    stats.add_report_entry("unbounded_file_read_cache hits",
                           {"unbounded_file_read_cache hits"},
                           rapidsmpf::Statistics::Formatter::HitRate);

    rapidsmpf::streaming::SpillableMessages::MessageId mid;
    {
      std::lock_guard lock(mutex_);
      auto it = cache_.find(key);
      if (it == cache_.end()) {
        stats.add_stat("unbounded_file_read_cache hits", 0);
        return std::nullopt;
      }
      mid = it->second;
    }
    auto const size = ctx->spillable_messages()->get_content_description(mid).content_size();

    stats.add_stat("unbounded_file_read_cache hits", 1);
    stats.add_bytes_stat("unbounded_file_read_cache saved", size);
    auto reservation = ctx->br()->reserve_or_fail(size, rapidsmpf::MEMORY_TYPES);
    return ctx->spillable_messages()->copy(mid, reservation);
  }

  /**
   * @brief Get the FileCache instance for a Context.
   *
   * Each Context has exactly one FileCache instance for the lifetime of the
   * process. If the `unbounded_file_read_cache` option is disabled, this
   * function returns nullptr.
   *
   * @param ctx Context used to identify the cache instance. The same Context must
   * be used for all subsequent insert and get operations.
   * @return Shared pointer to the per-context FileCache, or nullptr if the cache
   * is disabled.
   */
  static std::shared_ptr<FileCache> instance(std::shared_ptr<rapidsmpf::streaming::Context> ctx)
  {
    static std::mutex mutex;
    static std::unordered_map<std::size_t, std::shared_ptr<FileCache>> instances;

    std::lock_guard lock(mutex);
    auto const id = ctx->uid();
    auto it       = instances.find(id);
    if (it != instances.end()) { return it->second; }

    // Get the memory type of the file cache, if enabled.
    auto const mem_type = ctx->options().get<std::optional<rapidsmpf::MemoryType>>(
      "unbounded_file_read_cache", [](auto const& s) -> std::optional<rapidsmpf::MemoryType> {
        auto val = rapidsmpf::parse_optional(s);
        if (!val.has_value() || val->empty()) { return std::nullopt; }
        return rapidsmpf::parse_string<rapidsmpf::MemoryType>(s);
      });

    if (mem_type.has_value()) {
      auto ret = std::make_shared<FileCache>(*mem_type);
      instances.emplace(id, ret);
      return ret;
    }
    return nullptr;
  }

 private:
  mutable std::mutex mutex_;
  std::map<Key, rapidsmpf::streaming::SpillableMessages::MessageId> cache_;
  rapidsmpf::MemoryType mem_type_;
};

/**
 * @brief Read a single chunk from a parquet source.
 *
 * @param ctx The execution context to use.
 * @param stream The stream on which to read the chunk.
 * @param options The parquet reader options describing the data to read.
 * @param sequence_number The ordered chunk id to reconstruct original ordering of the
 * data.
 * @return Message representing the read chunk.
 */
rapidsmpf::streaming::Message read_parquet_chunk(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                                 rmm::cuda_stream_view stream,
                                                 cudf::io::parquet_reader_options options,
                                                 std::uint64_t sequence_number)
{
  auto do_read_parquet = [&]() -> rapidsmpf::streaming::Message {
    return to_message(
      sequence_number,
      std::make_unique<table_chunk>(
        cudf::io::read_parquet(options, stream, ctx->br()->device_mr()).tbl, stream));
  };

  auto file_cache = FileCache::instance(ctx);
  if (file_cache == nullptr) { return do_read_parquet(); }

  FileCache::Key key{.filepaths      = options.get_source().filepaths(),
                     .skip_rows      = options.get_skip_rows(),
                     .skip_bytes     = options.get_skip_bytes(),
                     .num_rows       = options.get_num_rows(),
                     .num_bytes      = options.get_num_bytes(),
                     .column_names   = options.get_column_names(),
                     .column_indices = options.get_column_indices(),
                     .row_groups     = options.get_row_groups()};

  auto msg = file_cache->get(ctx, key);
  if (msg.has_value()) { return std::move(*msg); }

  auto ret = do_read_parquet();
  file_cache->insert(ctx, key, ret);
  return ret;
}

struct ChunkDesc {
  std::uint64_t sequence_number;
  std::int64_t skip_rows;
  std::int64_t num_rows;
  cudf::io::source_info source;
};

/**
 * @brief Read chunks and send them to an output channel.
 *
 * @param ctx Execution context to use.
 * @param ch_out Channel to send output to.
 * @param options Template reader options.
 * @param chunks List of chunks from the input files to read. Processed in order.
 * @param idx Index of the next chunk to process.
 *
 * @return Coroutine representing the processing of all chunks.
 */
rapidsmpf::streaming::Actor produce_chunks(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::BoundedQueue> ch_out,
  std::vector<ChunkDesc>& chunks,
  cudf::io::parquet_reader_options options)
{
  // ShutdownAtExit c{ch_out};
  co_await ctx->executor()->schedule();
  for (auto& chunk : chunks) {
    cudf::io::parquet_reader_options chunk_options{options};
    chunk_options.set_skip_rows(chunk.skip_rows);
    chunk_options.set_num_rows(chunk.num_rows);
    chunk_options.set_source(chunk.source);
    auto stream = ctx->br()->stream_pool()->get_stream();
    auto ticket = co_await ch_out->acquire();
    if (!ticket.has_value()) {
      // Semaphore (and hence output channel) shutdown
      break;
    }
    // Having acquire a ticket, let's move to a new thread.
    co_await ctx->executor()->schedule();
    // TODO: This reads the metadata ntasks times.
    // See https://github.com/rapidsai/cudf/issues/20311
    auto [msg, exception] = [&]() -> std::pair<rapidsmpf::streaming::Message, std::exception_ptr> {
      try {
        return {read_parquet_chunk(ctx, stream, chunk_options, chunk.sequence_number), nullptr};
      } catch (...) {
        return {rapidsmpf::streaming::Message{}, std::current_exception()};
      }
    }();
    if (exception != nullptr) {
      co_await ch_out->shutdown();
      std::rethrow_exception(exception);
    }
    auto sent = co_await ticket->send(std::move(msg));
    if (!sent) {
      // Output channel is shutdown, no need for more reads.
      break;
    }
  }
  co_await ch_out->drain(ctx->executor());
}
}  // namespace

rapidsmpf::streaming::Actor read_parquet(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                         std::shared_ptr<rapidsmpf::Communicator> comm,
                                         std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                         std::size_t num_producers,
                                         cudf::io::parquet_reader_options options,
                                         cudf::size_type num_rows_per_chunk,
                                         std::unique_ptr<filter> filter)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_out};
  co_await ctx->executor()->schedule();
  auto const size = rapidsmpf::safe_cast<std::size_t>(comm->nranks());
  auto const rank = rapidsmpf::safe_cast<std::size_t>(comm->rank());
  auto source     = options.get_source();
  RAPIDSMPF_EXPECTS(source.type() == cudf::io::io_type::FILEPATH,
                    "Only implemented for file sources");
  // TODO: To handle this we need a prefix scan across all the ranks of the total
  // number of rows that would be read by previous ranks.
  RAPIDSMPF_EXPECTS(size == 1 || !options.get_num_rows().has_value(),
                    "Reading subset of rows not yet supported in multi-rank execution");
  // TODO: To handle this we need a prefix scan across all the ranks of the total
  // number of rows that would be read by previous ranks.
  RAPIDSMPF_EXPECTS(size == 1 || options.get_skip_rows() == 0,
                    "Skipping rows not yet supported in multi-rank execution");
  auto files = source.filepaths();
  RAPIDSMPF_EXPECTS(files.size() > 0, "Must have at least one file to read");
  RAPIDSMPF_EXPECTS(!options.get_filter().has_value(),
                    "Do not set filter on options, use the filter argument");
  if (filter != nullptr) {
    options.set_filter(filter->filter);
    // Let's just join all the possible streams here rather than inducing cross-stream
    // deps in the tasks
    rapidsmpf::cuda_stream_join(
      std::ranges::transform_view(
        std::ranges::iota_view(std::size_t{0}, ctx->br()->stream_pool()->get_pool_size()),
        [&](auto i) { return ctx->br()->stream_pool()->get_stream(i); }),
      std::ranges::single_view(filter->stream));
  }
  // TODO: Handle case where multiple ranks are reading from a single file.
  auto const files_per_rank =
    rapidsmpf::safe_cast<int>(files.size() / size + (rank < (files.size() % size)));
  auto const file_offset =
    rapidsmpf::safe_cast<int>(rank * (files.size() / size) + std::min(rank, files.size() % size));
  auto local_files =
    std::vector(files.begin() + file_offset, files.begin() + file_offset + files_per_rank);
  std::uint64_t sequence_number = 0;
  std::vector<std::vector<ChunkDesc>> chunks_per_producer(num_producers);
  auto const num_files = local_files.size();
  // Estimate number of rows per file
  std::size_t files_per_chunk = 1;
  if (num_files > 1) {
    auto nrows = cudf::io::read_parquet_metadata(cudf::io::source_info(local_files[0])).num_rows();
    files_per_chunk =
      nrows > 0
        ? rapidsmpf::safe_cast<std::size_t>(std::max<std::int64_t>(num_rows_per_chunk / nrows, 1))
        : 1;
  }
  auto to_skip = options.get_skip_rows();
  auto to_read = options.get_num_rows().value_or(std::numeric_limits<std::int64_t>::max());
  for (std::size_t file_offset = 0; file_offset < num_files; file_offset += files_per_chunk) {
    std::vector<std::string> chunk_files;
    auto const nchunk_files = std::min(num_files - file_offset, files_per_chunk);
    std::ranges::copy_n(local_files.begin() + rapidsmpf::safe_cast<std::int64_t>(file_offset),
                        rapidsmpf::safe_cast<std::int64_t>(nchunk_files),
                        std::back_inserter(chunk_files));
    auto source = cudf::io::source_info(chunk_files);
    // Must read [skip_rows, skip_rows + num_rows) from full fileset
    auto chunk_rows      = cudf::io::read_parquet_metadata(source).num_rows() - to_skip;
    auto chunk_skip_rows = to_skip;
    // If the chunk is larger than the number rows we need to skip, on the next
    // iteration we don't need to skip any more rows, otherwise we must skip the
    // remainder.
    to_skip = std::max(0l, -chunk_rows);
    while (chunk_rows > 0 && to_read > 0) {
      auto rows_read =
        std::min({rapidsmpf::safe_cast<std::int64_t>(num_rows_per_chunk), chunk_rows, to_read});
      chunks_per_producer[sequence_number % num_producers].emplace_back(
        sequence_number, chunk_skip_rows, rows_read, source);
      sequence_number++;
      to_read = std::max(0l, to_read - rows_read);
      chunk_skip_rows += rows_read;
      chunk_rows -= rows_read;
    }
  }
  if (std::ranges::all_of(chunks_per_producer, [](auto&& v) { return v.empty(); })) {
    if (local_files.size() > 0) {
      // If we're on the hook to read some files, but the skip_rows/num_rows setup
      // meant our slice was empty, send an empty table of correct shape.
      // Anyone with no files will just immediately close their output channel.
      auto empty_opts = options;
      empty_opts.set_source(cudf::io::source_info(local_files[0]));
      empty_opts.set_skip_rows(0);
      empty_opts.set_num_rows(0);
      co_await ctx->executor()->schedule(ch_out->send(
        read_parquet_chunk(ctx, ctx->br()->stream_pool()->get_stream(), std::move(empty_opts), 0)));
    }
  } else {
    std::vector<rapidsmpf::streaming::Actor> read_tasks;
    read_tasks.reserve(1 + num_producers);
    auto lineariser = rapidsmpf::streaming::Lineariser(ctx, ch_out, num_producers);
    auto queues     = lineariser.get_queues();
    for (std::size_t i = 0; i < num_producers; i++) {
      read_tasks.push_back(produce_chunks(ctx, queues[i], chunks_per_producer[i], options));
    }
    read_tasks.push_back(lineariser.drain());
    rapidsmpf::streaming::coro_results(co_await coro::when_all(std::move(read_tasks)));
  }
  co_await ch_out->drain(ctx->executor());
  if (filter != nullptr) {
    // Let's just join all the possible streams here rather than inducing cross-stream
    // deps in the tasks
    rapidsmpf::cuda_stream_join(
      std::ranges::single_view(filter->stream),
      std::ranges::transform_view(
        std::ranges::iota_view(std::size_t{0}, ctx->br()->stream_pool()->get_pool_size()),
        [&](auto i) { return ctx->br()->stream_pool()->get_stream(i); }));
  }
}
}  // namespace cudf_streaming::actor
