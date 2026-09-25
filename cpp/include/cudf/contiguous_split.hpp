/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/packed_types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

/**
 * @file
 * @brief Table APIs for contiguous_split, pack, unpack, and metadata
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup copy_split
 * @{
 */

/**
 * @brief Performs a deep-copy split of a `table_view` into a vector of `packed_table` where each
 * `packed_table` is using a single contiguous block of memory for all of the split's column data.
 *
 * The memory for the output views is allocated in a single contiguous `rmm::device_buffer` returned
 * in the `packed_table`. There is no top-level owning table.
 *
 * The returned views of `input` are constructed from a vector of indices, that indicate
 * where each split should occur. The `i`th returned `table_view` is sliced as
 * `[0, splits[i])` if `i`=0, else `[splits[i], input.size())` if `i` is the last view and
 * `[splits[i-1], splits[i]]` otherwise.
 *
 * For all `i` it is expected `splits[i] <= splits[i+1] <= input.size()`.
 * For a `splits` size N, there will always be N+1 splits in the output.
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory contained in the `all_data` field of the
 * returned packed_table.
 *
 * @note Every output partition of a dictionary column holds a copy of the complete keys child,
 * including keys that the partition's rows do not reference.
 *
 * @code{.pseudo}
 * Example:
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * splits:  {2, 5, 9}
 * output:  [{{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}},
 *           {{50, 52}, {54, 56, 58}, {60, 62, 64, 66}, {68}}]
 * @endcode
 *
 *
 * @throws std::out_of_range if `splits` has end index > size of `input`.
 * @throws std::out_of_range When the value in `splits` is not in the range [0, input.size()).
 * @throws std::invalid_argument When the values in the `splits` are 'strictly decreasing'.
 *
 * @param input View of a table to split
 * @param splits A vector of indices where the view will be split
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr An optional memory resource to use for all returned device allocations
 * @return The set of requested views of `input` indicated by the `splits` and the viewed memory
 * buffer
 */
std::vector<packed_table> contiguous_split(
  cudf::table_view const& input,
  std::vector<size_type> const& splits,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

namespace detail {

/**
 * @brief A helper struct containing the state of contiguous_split, whether the caller
 * is using the single-pass contiguous_split or chunked_pack.
 *
 */
struct contiguous_split_state;
}  // namespace detail

/**
 * @brief Perform a chunked "pack" operation of the input `table_view` using a user provided
 * buffer of size `user_buffer_size`.
 *
 * The intent of this operation is to be used in a streamed fashion at times of GPU
 * out-of-memory, where we want to minimize the number of small cudaMemcpy calls and
 * tracking of all the metadata associated with cudf tables. Because of the memory constraints,
 * all thrust and scratch memory allocations are using the passed-in memory resource exclusively,
 * not a per-device memory resource.
 *
 * This class defines two methods that must be used in concert to carry out the chunked_pack:
 * has_next and next. Here is an example:
 *
 * @code{.pseudo}
 * // Create a table_view
 * cudf::table_view tv = ...;
 *
 * // Choose a memory resource (optional). This memory resource is used for scratch/thrust temporary
 * // data. In memory constrained cases, this can be used to set aside scratch memory
 * // for `chunked_pack` at the beginning of a program.
 * auto mr = cudf::get_current_device_resource_ref();
 * cuda::stream_ref stream = cudf::get_default_stream();
 *
 * // Define a buffer size for each chunk: the larger the buffer is, the more SMs can be
 * // occupied by this algorithm.
 * //
 * // Internally, the GPU unit of work is a 1MB batch. When we instantiate `cudf::chunked_pack`,
 * // all the 1MB batches for the source table_view are computed up front. Additionally,
 * // chunked_pack calculates the number of iterations that are required to go through all those
 * // batches given a `user_buffer_size` buffer. The number of 1MB batches in each iteration (chunk)
 * // equals the number of CUDA blocks that will be used for the main kernel launch.
 * //
 * std::size_t user_buffer_size = 128*1024*1024;
 *
 * auto chunked_packer = cudf::chunked_pack::create(tv, user_buffer_size, stream, mr);
 *
 * std::size_t host_offset = 0;
 * auto host_buffer = ...; // obtain a host buffer you would like to copy to
 *
 * while (chunked_packer->has_next()) {
 *   // get a user buffer of size `user_buffer_size`
 *   cudf::device_span<uint8_t> user_buffer = ...;
 *   std::size_t bytes_copied = chunked_packer->next(user_buffer);
 *
 *   // buffer will hold the contents of at most `user_buffer_size` bytes
 *   // of the contiguously packed input `table_view`. You are now free to copy
 *   // this memory somewhere else, for example, to host.
 *   cudaMemcpyAsync(
 *     host_buffer.data() + host_offset,
 *     user_buffer.data(),
 *     bytes_copied,
 *     cudaMemcpyDefault,
 *     stream.get());
 *
 *   host_offset += bytes_copied;
 * }
 * @endcode
 */
class chunked_pack {
 public:
  /**
   * @brief Construct a `chunked_pack` class.
   *
   * @param input source `table_view` to pack
   * @param user_buffer_size buffer size (in bytes) that will be passed on `next`. Must be
   *                         at least 1MB
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param temp_mr An optional memory resource to be used for temporary and scratch allocations
   * only
   */
  explicit chunked_pack(
    cudf::table_view const& input,
    std::size_t user_buffer_size,
    cuda::stream_ref stream                = cudf::get_default_stream(),
    rmm::device_async_resource_ref temp_mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Destructor that will be implemented as default. Declared with definition here because
   * contiguous_split_state is incomplete at this stage.
   */
  ~chunked_pack();

  /**
   * @brief Obtain the total size of the contiguously packed `table_view`.
   *
   * @return total size (in bytes) of all the chunks
   */
  [[nodiscard]] std::size_t get_total_contiguous_size() const;

  /**
   * @brief Function to check if there are chunks left to be copied.
   *
   * @return true if there are chunks left to be copied, and false otherwise
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @brief Packs the next chunk into `user_buffer`. This should be called as long as
   * `has_next` returns true. If `next` is called when `has_next` is false, an exception
   * is thrown.
   *
   * @throws cudf::logic_error If the size of `user_buffer` is different than `user_buffer_size`
   * @throws cudf::logic_error If called after all chunks have been copied
   *
   * @param user_buffer device span target for the chunk. The size of this span must equal
   *                    the `user_buffer_size` parameter passed at construction
   * @return The number of bytes that were written to `user_buffer` (at most
   *          `user_buffer_size`)
   */
  [[nodiscard]] std::size_t next(cudf::device_span<uint8_t> const& user_buffer);

  /**
   * @brief Build the opaque metadata for all added columns.
   *
   * @return A vector containing the serialized column metadata
   */
  [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> build_metadata() const;

  /**
   * @brief Creates a `chunked_pack` instance to perform a "pack" of the `table_view`
   * "input", where a buffer of `user_buffer_size` is filled with chunks of the
   * overall operation. This operation can be used in cases where GPU memory is constrained.
   *
   * The memory resource (`temp_mr`) could be a special memory resource to be used in
   * situations when GPU memory is low and we want scratch and temporary allocations to
   * happen from a small reserved pool of memory. Note that it defaults to the regular cuDF
   * per-device resource.
   *
   * @throws cudf::logic_error When user_buffer_size is less than 1MB
   *
   * @param input source `table_view` to pack
   * @param user_buffer_size buffer size (in bytes) that will be passed on `next`. Must be
   *                         at least 1MB
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param temp_mr RMM memory resource to be used for temporary and scratch allocations only
   * @return a unique_ptr of chunked_pack
   */
  [[nodiscard]] static std::unique_ptr<chunked_pack> create(
    cudf::table_view const& input,
    std::size_t user_buffer_size,
    cuda::stream_ref stream                = cudf::get_default_stream(),
    rmm::device_async_resource_ref temp_mr = cudf::get_current_device_resource_ref());

 private:
  // internal state of contiguous split
  std::unique_ptr<detail::contiguous_split_state> state;
};

/**
 * @brief Deep-copy a `table_view` into a serialized contiguous memory format.
 *
 * The metadata from the `table_view` is copied into a host vector of bytes and the data from the
 * `table_view` is copied into a `device_buffer`. Pass the output of this function into
 * `cudf::unpack` to deserialize.
 *
 * @param input View of the table to pack
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr An optional memory resource to use for all returned device allocations
 * @return packed_columns A struct containing the serialized metadata and data in contiguous host
 *         and device memory respectively
 */
packed_columns pack(cudf::table_view const& input,
                    cuda::stream_ref stream           = cudf::get_default_stream(),
                    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Compute the size in bytes of the contiguous memory buffer needed to pack the input table.
 *
 * This function computes the total contiguous size that would be required to pack the input
 * table using `pack()` or `chunked_pack`, without actually performing the packing operation.
 * This is useful for pre-allocating memory or determining if a table will fit in available memory.
 *
 * @param input View of the table to compute the packed size for
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param temp_mr An optional memory resource to use for temporary allocations
 * @return The size in bytes required to store the packed table data (not including metadata)
 */
std::size_t packed_size(
  cudf::table_view const& input,
  cuda::stream_ref stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref temp_mr = cudf::get_current_device_resource_ref());

namespace experimental {

/**
 * @brief Compression algorithms supported by the prepared pack prototype.
 */
enum class pack_compression {
  none,       ///< Preserve the current uncompressed packed representation
  automatic,  ///< Select a codec independently for each physical region
  cascaded,   ///< nvCOMP Cascaded with an NVCOMP_NATIVE self-describing bitstream
  zstd,       ///< nvCOMP Zstd with an NVCOMP_NATIVE self-describing bitstream
  snappy,     ///< nvCOMP Snappy with an NVCOMP_NATIVE self-describing bitstream
};

/**
 * @brief Physical role of a region presented to an expert codec selector.
 */
enum class pack_region_kind {
  data,              ///< Fixed-width or other ordinary column data
  validity,          ///< Null-validity bitmask
  offsets,           ///< String or list offsets
  string_characters  ///< String character bytes
};

/**
 * @brief Read-only description passed to an expert per-region codec selector.
 */
struct pack_region_info {
  std::size_t region_index;        ///< Stable index within this prepared pack operation
  size_type column_index;          ///< Top-level input column owning this region
  pack_region_kind kind;           ///< Physical role of the region
  type_id type;                    ///< Logical/native type used to configure the codec
  std::size_t uncompressed_bytes;  ///< Bytes presented to the selected codec
};

/**
 * @brief Expert codec configuration for one physical packed region.
 *
 * `automatic` applies libcudf's built-in codec policy and permits compact output to fall back to
 * raw bytes when compression misses `minimum_savings_bytes`. A concrete codec forces that codec.
 */
struct pack_region_options {
  pack_compression codec{pack_compression::none};
  std::size_t compression_chunk_bytes{64 * 1024};
  std::size_t minimum_savings_bytes{256};
  int cascaded_num_RLEs{2};
  int cascaded_num_deltas{1};
  bool cascaded_use_bitpacking{true};
};

/**
 * @brief Immutable region description and its mutable expert codec configuration.
 */
struct pack_region {
  pack_region_info const info;
  pack_region_options options;
};

/**
 * @brief Controls whether compressed execution reports the compact size immediately.
 */
enum class compressed_output_mode {
  compact,   ///< Synchronize and report the actual compressed prefix size
  reserved,  ///< Remain asynchronous and retain the complete planned destination capacity
};

/**
 * @brief Options controlling a prepared pack operation.
 */
struct pack_options {
  pack_compression compression{pack_compression::none};
  compressed_output_mode output_mode{compressed_output_mode::compact};
  std::size_t compression_chunk_bytes{64 * 1024};
  std::size_t automatic_min_region_bytes{4 * 1024};
  std::size_t automatic_min_savings_bytes{256};
  int cascaded_num_RLEs{2};
  int cascaded_num_deltas{1};
  bool cascaded_use_bitpacking{true};
};

/**
 * @brief Storage requirements for a prepared pack operation.
 *
 * This is an experimental prototype. The API and representation may change without notice.
 */
struct pack_sizes {
  std::size_t metadata_bytes;     ///< Exact host metadata size
  std::size_t payload_bytes;      ///< Required destination capacity (an upper bound if compressed)
  std::size_t payload_alignment;  ///< Required alignment for the payload base address
  std::size_t uncompressed_payload_bytes;  ///< Exact contiguous size before compression
};

struct pack_result;
class pack_plan_builder;

/**
 * @brief Prepared state for repeatedly packing one table into caller-owned memory.
 *
 * Construction performs layout planning once. The input table or packed columns, their metadata,
 * and all referenced buffers must remain alive and unchanged until every operation using the plan
 * has completed. A plan is bound to the stream passed to `prepare_pack()`.
 *
 * This is an experimental prototype. The API and representation may change without notice.
 */
class pack_plan {
 public:
  pack_plan(pack_plan const&)            = delete;
  pack_plan& operator=(pack_plan const&) = delete;
  pack_plan(pack_plan&&) noexcept;
  pack_plan& operator=(pack_plan&&) noexcept;
  ~pack_plan();

  /**
   * @brief Return the exact storage requirements for this plan.
   */
  [[nodiscard]] pack_sizes sizes() const;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;

  explicit pack_plan(std::unique_ptr<impl>&& implementation);

  friend pack_plan prepare_pack(cudf::table_view const&,
                                cuda::stream_ref,
                                rmm::device_async_resource_ref);
  friend pack_plan prepare_pack(cudf::table_view const&,
                                pack_options const&,
                                cuda::stream_ref,
                                rmm::device_async_resource_ref);
  friend pack_plan prepare_pack(cudf::packed_columns const&,
                                pack_options const&,
                                cuda::stream_ref,
                                rmm::device_async_resource_ref);
  friend class pack_plan_builder;
  friend pack_result pack_into(pack_plan const&, cudf::device_span<uint8_t>);
};

/**
 * @brief Two-stage expert configuration for a prepared pack operation.
 *
 * The builder discovers physical regions once. Callers may edit only `pack_region::options`; the
 * descriptions remain immutable. `build()` finalizes compressor state and destination capacity.
 */
class pack_plan_builder {
 public:
  pack_plan_builder(pack_plan_builder const&)            = delete;
  pack_plan_builder& operator=(pack_plan_builder const&) = delete;
  pack_plan_builder(pack_plan_builder&&) noexcept;
  pack_plan_builder& operator=(pack_plan_builder&&) noexcept;
  ~pack_plan_builder();

  [[nodiscard]] std::span<pack_region> regions();
  [[nodiscard]] std::span<pack_region const> regions() const;
  [[nodiscard]] pack_plan build() &&;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;

  explicit pack_plan_builder(std::unique_ptr<impl>&& implementation);

  friend pack_plan_builder make_pack_plan_builder(cudf::table_view const&,
                                                  pack_options const&,
                                                  cuda::stream_ref,
                                                  rmm::device_async_resource_ref);
  friend pack_plan_builder make_pack_plan_builder(cudf::packed_columns const&,
                                                  pack_options const&,
                                                  cuda::stream_ref,
                                                  rmm::device_async_resource_ref);
};

/**
 * @brief Discover configurable physical regions for expert per-region codec selection.
 *
 * Each region initially inherits the codec and codec parameters in `options`. Callers may edit the
 * returned regions before consuming the builder with `build()`.
 */
pack_plan_builder make_pack_plan_builder(
  cudf::table_view const& input,
  pack_options const& options            = {},
  cuda::stream_ref stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref temp_mr = cudf::get_current_device_resource_ref());

/**
 * @brief Discover configurable regions in an existing uncompressed packed allocation.
 */
pack_plan_builder make_pack_plan_builder(
  cudf::packed_columns const& input,
  pack_options const& options,
  cuda::stream_ref stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref temp_mr = cudf::get_current_device_resource_ref());

/**
 * @brief Prepare an exact, reusable uncompressed pack plan for `input`.
 *
 * @param input View of the table to pack
 * @param stream Stream used for planning and subsequent `pack_into()` operations
 * @param temp_mr Memory resource used for planning scratch allocations
 * @return A move-only plan bound to `input` and `stream`
 */
pack_plan prepare_pack(
  cudf::table_view const& input,
  cuda::stream_ref stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref temp_mr = cudf::get_current_device_resource_ref());

/**
 * @brief Prepare a reusable pack plan with explicit compression options.
 *
 * Compression first creates the normalized contiguous representation, then independently
 * compresses each physical column buffer (data, offsets, characters, or validity) as an nvCOMP
 * native bitstream. Cascaded is configured with the native width and signedness of each region
 * when nvCOMP supports it. `sizes().payload_bytes` is the combined upper-bound capacity;
 * `pack_into()` reports either the compact prefix or reserved capacity selected by `options`.
 *
 * @param input View of the table to pack
 * @param options Compression and codec options
 * @param stream Stream used for planning and subsequent `pack_into()` operations
 * @param temp_mr Memory resource used for planning and compression staging allocations
 * @return A move-only plan bound to `input` and `stream`
 */
pack_plan prepare_pack(
  cudf::table_view const& input,
  pack_options const& options,
  cuda::stream_ref stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref temp_mr = cudf::get_current_device_resource_ref());

/**
 * @brief Prepare compression of an existing uncompressed `cudf::packed_columns` allocation.
 *
 * This overload supports callers that decide whether to compress only after ordinary packing has
 * completed. It borrows `input.gpu_data` as the compression source and therefore avoids copying or
 * repacking the column data. The ordinary pack metadata is retained inside the compressed
 * representation for reconstruction by `materialize()`.
 *
 * `options.compression` must select a compressed representation. The input metadata and device
 * allocation must remain alive and unchanged until every execution using the returned plan has
 * completed.
 *
 * @param input Existing ordinary, uncompressed packed columns
 * @param options Compression and output-layout options
 * @param stream Stream used for planning and subsequent `pack_into()` operations
 * @param temp_mr Memory resource used for codec workspace allocations
 * @return A move-only plan that borrows `input` and is bound to `stream`
 */
pack_plan prepare_pack(
  cudf::packed_columns const& input,
  pack_options const& options,
  cuda::stream_ref stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref temp_mr = cudf::get_current_device_resource_ref());

/**
 * @brief Host metadata and payload size produced by `pack_into()`.
 */
struct pack_result {
  std::vector<uint8_t> metadata;       ///< Metadata describing the packed payload
  std::size_t payload_bytes;           ///< Bytes to retain: actual prefix or reserved capacity
  pack_compression compression;        ///< Representation used by the payload
  compressed_output_mode output_mode;  ///< Size-reporting policy used for this result
};

/**
 * @brief Execute a prepared pack directly into caller-owned device-accessible memory.
 *
 * `destination` may be device memory or mapped pinned-host memory. It must contain at least
 * `plan.sizes().payload_bytes` bytes. Work is submitted to the stream captured by the plan.
 * The caller must preserve the input and destination until that stream reaches the operation.
 *
 * The same plan may be executed repeatedly while its input remains valid and unchanged.
 *
 * @param plan Prepared pack plan
 * @param destination Caller-owned output span
 * @return Host metadata and the number of payload bytes written
 */
pack_result pack_into(pack_plan const& plan, cudf::device_span<uint8_t> destination);

/**
 * @brief Non-owning view of packed host metadata and device-accessible payload bytes.
 */
struct packed_data_view {
  std::span<uint8_t const> metadata;
  cudf::device_span<uint8_t const> payload;
  pack_compression compression{pack_compression::none};
};

/**
 * @brief Construct a zero-copy table view over an uncompressed packed payload.
 *
 * The returned view must not outlive either buffer in `input`.
 * Compressed inputs must be passed to `materialize()` instead.
 *
 * @param input Packed metadata and payload
 * @return A non-owning table view into `input.payload`
 */
table_view unpack_view(packed_data_view input);

/**
 * @brief Materialize an owning table from any supported packed representation.
 *
 * @param input Packed metadata and payload
 * @param stream Stream used for the deep copy
 * @param mr Memory resource for the returned table
 * @return An owning table independent of the packed buffers
 */
std::unique_ptr<table> materialize(
  packed_data_view input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace experimental

/**
 * @brief Produce the metadata used for packing a table stored in a contiguous buffer.
 *
 * The metadata from the `table_view` is copied into a host vector of bytes which can be used to
 * construct a `packed_columns` or `packed_table` structure. The caller is responsible for
 * guaranteeing that all of the columns in the table point into `contiguous_buffer`.
 *
 * @param table View of the table to pack
 * @param contiguous_buffer A contiguous buffer of device memory which contains the data referenced
 *        by the columns in `table`
 * @param buffer_size The size of `contiguous_buffer`
 * @return Vector of bytes representing the metadata used to `unpack` a packed_columns struct
 */
std::vector<uint8_t> pack_metadata(table_view const& table,
                                   uint8_t const* contiguous_buffer,
                                   size_t buffer_size);

/**
 * @brief Deserialize the result of `cudf::pack`.
 *
 * Converts the result of a serialized table into a `table_view` that points to the data stored in
 * the contiguous device buffer contained in `input`.
 *
 * It is the caller's responsibility to ensure that the `table_view` in the output does not outlive
 * the data in the input.
 *
 * No new device memory is allocated in this function.
 *
 * @param input The packed columns to unpack
 * @return The unpacked `table_view`
 */
table_view unpack(packed_columns const& input);

/**
 * @brief Deserialize the result of `cudf::pack`.
 *
 * Converts the result of a serialized table into a `table_view` that points to the data stored in
 * the contiguous device buffer contained in `gpu_data` using the metadata contained in the host
 * buffer `metadata`.
 *
 * It is the caller's responsibility to ensure that the `table_view` in the output does not outlive
 * the data in the input.
 *
 * No new device memory is allocated in this function.
 *
 * @param metadata The host-side metadata buffer resulting from the initial pack() call
 * @param gpu_data The device-side contiguous buffer storing the data that will be referenced by
 *        the resulting `table_view`
 * @return The unpacked `table_view`
 */
table_view unpack(uint8_t const* metadata, uint8_t const* gpu_data);

/**
 * @brief A non-owning view over the host metadata produced by `cudf::pack`.
 *
 * `packed_metadata_view` enables schema introspection — querying column types,
 * sizes, null counts, and nesting structure — without requiring device data
 * and building a `table_view`.
 *
 * The view interprets the serialized `packed_columns::metadata` wire
 * format.
 *
 * @code{.cpp}
 * auto packed = cudf::pack(table);
 * auto view   = cudf::packed_metadata_view(*packed.metadata);
 * std::cout << "columns: " << view.num_columns()
 *           << ", rows: "  << view.num_rows() << "\n";
 * for (cudf::size_type i = 0; i < view.num_columns(); i++) {
 *   auto col = view.column(i);
 *   std::cout << "  type=" << cudf::type_to_name(col.type())
 *             << " children=" << col.num_children() << "\n";
 * }
 * @endcode
 */
class packed_metadata_view {
 public:
  /**
   * @brief A non-owning view of a single column's metadata within packed column data.
   *
   * This lightweight view (two pointers) wraps a single serialized column entry and provides
   * access to its schema information (type, size, null count, children) without requiring
   * device data or building a `column_view`.
   *
   * Instances are obtained from `packed_metadata_view::column()` or
   * `packed_column_metadata::child()`. They remain valid as long as the underlying
   * metadata byte buffer is alive.
   */
  class column_view {
   public:
    /**
     * @brief @return The `data_type` of this column.
     */
    [[nodiscard]] data_type type() const;

    /**
     * @brief @return The number of rows in this column.
     */
    [[nodiscard]] size_type num_rows() const;

    /**
     * @brief @return The null count of this column.
     */
    [[nodiscard]] size_type null_count() const;

    /**
     * @brief @return Byte offset of this column's data in the uncompressed packed payload, or -1
     * if the column has no data buffer.
     */
    [[nodiscard]] int64_t data_offset() const;

    /**
     * @brief @return Byte offset of this column's validity mask in the uncompressed packed
     * payload, or -1 if the column is not nullable.
     */
    [[nodiscard]] int64_t null_mask_offset() const;

    /**
     * @brief @return The number of children of this column.
     */
    [[nodiscard]] size_type num_children() const;

    /**
     * @brief A view of the i-th child column's metadata.
     *
     * @throws std::out_of_range if `i` is not contained in `[0, num_children())`
     * @param i Index of the child column
     * @return A `packed_column_metadata_view` for the i-th child
     */
    [[nodiscard]] column_view child(size_type i) const;

   private:
    friend class packed_metadata_view;
    data_type _type{type_id::EMPTY};
    size_type _size{};
    size_type _null_count{};
    int64_t _data_offset{-1};
    int64_t _null_mask_offset{-1};
    size_type _num_children{};
    // Span from this entry to the end of the metadata buffer (needed for child traversal).
    std::span<std::uint8_t const> _buffer;
    explicit column_view(std::span<std::uint8_t const> buffer);
  };

  /**
   * @brief Construct a view from a metadata byte buffer.
   *
   * @throws cudf::logic_error if the buffer is empty or does not satisfy minimum requirements for
   * describing a valid column tree.
   * @param buffer The metadata bytes (as produced by `cudf::pack`)
   */
  explicit packed_metadata_view(std::span<std::uint8_t const> buffer);

  /**
   * @brief @return The number of top-level columns.
   */
  [[nodiscard]] size_type num_columns() const;

  /**
   * @brief The number of rows in the table.
   *
   * @return The row count
   */
  [[nodiscard]] size_type num_rows() const;

  /**
   * @brief A view of the i-th top-level column's metadata.
   *
   * @throws std::out_of_range if `i` is not contained in `[0, num_columns())`
   * @param i Index of the top-level column
   * @return A `packed_metadata_view::column` for the i-th column
   */
  [[nodiscard]] column_view column(size_type i) const;

 private:
  // Span from the first top-level column entry to the end of the metadata buffer.
  std::span<std::uint8_t const> _entries;
  size_type _num_columns{};
  // Table row count, read directly from the serialized table header.
  size_type _num_rows{};
};

/** @} */
}  // namespace CUDF_EXPORT cudf
