/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/detail/contiguous_split.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace cudf {
namespace {

// Align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr std::size_t split_align = 64;

// The size that contiguous split uses internally as the GPU unit of work.
// The number of `desired_batch_size` batches equals the number of CUDA blocks
// that will be used for the main kernel launch (`copy_partitions`).
static constexpr std::size_t desired_batch_size = 1 * 1024 * 1024;

/**
 * @brief Struct which contains information on a source buffer.
 *
 * The definition of "buffer" used throughout this module is a component piece of a
 * cudf column. So for example, a fixed-width column with validity would have 2 associated
 * buffers : the data itself and the validity buffer.  contiguous_split operates by breaking
 * each column up into it's individual components and copying each one as a separate kernel
 * block.
 */
struct src_buf_info {
  src_buf_info(cudf::type_id _type,
               column_view const offsets,
               int _offset_stack_pos,
               int _parent_offsets_index,
               bool _is_validity,
               size_type _column_offset)
    : type(_type),
      offsets(detail::offsetalator_factory::make_input_iterator(offsets)),
      is_offsets{!offsets.is_empty()},
      offset_stack_pos(_offset_stack_pos),
      parent_offsets_index(_parent_offsets_index),
      is_validity(_is_validity),
      column_offset(_column_offset)
  {
  }

  src_buf_info(cudf::type_id _type,
               int _offset_stack_pos,
               int _parent_offsets_index,
               bool _is_validity,
               size_type _column_offset)
    : type(_type),
      offset_stack_pos(_offset_stack_pos),
      parent_offsets_index(_parent_offsets_index),
      is_validity(_is_validity),
      column_offset(_column_offset)
  {
  }

  cudf::type_id type;
  detail::input_offsetalator offsets{};
  bool is_offsets{false};    // offsets if I am an offset buffer
  int offset_stack_pos;      // position in the offset stack buffer
  int parent_offsets_index;  // immediate parent that has offsets, or -1 if none
  bool is_validity;          // if I am a validity buffer
  size_type column_offset;   // offset in the case of a sliced column
};

/**
 * @brief Struct which contains information on a destination buffer.
 *
 * Similar to src_buf_info, dst_buf_info contains information on a destination buffer we
 * are going to copy to.  If we have N input buffers (which come from X columns), and
 * M partitions, then we have N*M destination buffers.
 */
struct dst_buf_info {
  // constant across all copy commands for this buffer
  std::size_t buf_size;      // total size of buffer, including padding
  std::size_t num_elements;  // # of elements to be copied
  int element_size;          // size of each element in bytes
  std::size_t num_rows;  // # of rows to be copied(which may be different from num_elements in the
                         // case of validity or offset buffers)

  int64_t src_element_index;  // element index to start reading from my associated source buffer
  std::size_t dst_offset;     // my offset into the per-partition allocation
  int64_t value_shift;        // amount to shift values down by (for offset buffers)
  int bit_shift;              // # of bits to shift right by (for validity buffers)
  size_type valid_count;      // validity count for this block of work
  bool is_offsets;            // whether or not this is an offsets buffer

  int src_buf_index;  // source buffer index
  int dst_buf_index;  // destination buffer index
};

/**
 * @brief Copy a single buffer of column data, shifting values (for offset columns),
 * and validity (for validity buffers) as necessary.
 *
 * Copies a single partition of a source column buffer to a destination buffer. Shifts
 * element values by value_shift in the case of a buffer of offsets (value_shift will
 * only ever be > 0 in that case).  Shifts elements bitwise by bit_shift in the case of
 * a validity buffer (bit_shift will only ever be > 0 in that case).  This function assumes
 * value_shift and bit_shift will never be > 0 at the same time.
 *
 * This function expects:
 * - src may be a misaligned address
 * - dst must be an aligned address
 *
 * This function always does the ALU work related to value_shift and bit_shift because it is
 * entirely memory-bandwidth bound.
 *
 * @param dst Destination buffer
 * @param src Source buffer
 * @param t Thread index
 * @param dst_info Destination buffer info containing element count, sizes, shifts, and validity
 * @param stride Size of the kernel block
 */
template <int block_size, bool is_offsets, typename offset_type = int32_t>
__device__ void copy_buffer(uint8_t* __restrict__ dst,
                            uint8_t const* __restrict__ src,
                            int t,
                            dst_buf_info& dst_info,
                            uint32_t stride)
{
  size_type thread_valid_count = 0;
  size_type* valid_count       = dst_info.valid_count > 0 ? &dst_info.valid_count : nullptr;
  auto const value_shift       = dst_info.value_shift;
  auto const bit_shift         = dst_info.bit_shift;
  auto const num_elements      = dst_info.num_elements;
  auto const element_size      = dst_info.element_size;
  auto const num_rows          = dst_info.num_rows;

  src += (dst_info.src_element_index * element_size);

  // handle misalignment. read 16 bytes in 4 byte reads. write in a single 16 byte store.
  std::size_t const num_bytes = num_elements * element_size;
  // how many bytes we're misaligned from 4-byte alignment
  uint32_t const ofs = reinterpret_cast<uintptr_t>(src) % 4;
  std::size_t pos    = t * 16;
  stride *= 16;
  while (pos + 20 <= num_bytes) {
    // read from the nearest aligned address.
    uint32_t const* in32 = reinterpret_cast<uint32_t const*>((src + pos) - ofs);
    uint4 v              = uint4{in32[0], in32[1], in32[2], in32[3]};
    if constexpr (is_offsets) {
      if constexpr (sizeof(offset_type) == 8) {
        ulong2& lv = reinterpret_cast<ulong2&>(v);
        lv.x -= value_shift;
        lv.y -= value_shift;
      } else {
        v.x -= value_shift;
        v.y -= value_shift;
        v.z -= value_shift;
        v.w -= value_shift;
      }
    } else {
      if (ofs || bit_shift) {
        v.x = __funnelshift_r(v.x, v.y, ofs * 8 + bit_shift);
        v.y = __funnelshift_r(v.y, v.z, ofs * 8 + bit_shift);
        v.z = __funnelshift_r(v.z, v.w, ofs * 8 + bit_shift);
        v.w = __funnelshift_r(v.w, in32[4], ofs * 8 + bit_shift);
      }
    }
    reinterpret_cast<uint4*>(dst)[pos / 16] = v;
    if (valid_count) {
      thread_valid_count += (__popc(v.x) + __popc(v.y) + __popc(v.z) + __popc(v.w));
    }
    pos += stride;
  }

  // copy trailing bytes
  if (t == 0) {
    std::size_t remainder;
    if (num_bytes < 16) {
      remainder = num_bytes;
    } else {
      std::size_t const last_bracket = (num_bytes / 16) * 16;
      remainder                      = num_bytes - last_bracket;
      if (remainder < 4) {
        // we had less than 20 bytes for the last possible 16 byte copy, so copy 16 + the extra
        remainder += 16;
      }
    }

    // if we're performing a value shift (offsets)
    if constexpr (is_offsets) {
      std::size_t idx = (num_bytes - remainder) / sizeof(offset_type);
      while (remainder) {
        reinterpret_cast<offset_type*>(dst)[idx] =
          reinterpret_cast<offset_type const*>(src)[idx] - value_shift;
        idx++;
        remainder -= sizeof(offset_type);
      }
    }
    // if we're performing a bit shift (validity)
    else if (bit_shift) {
      // the # of bytes and alignment must be a multiple of 4.
      std::size_t idx = (num_bytes - remainder) / 4;
      uint32_t v      = remainder > 0 ? reinterpret_cast<uint32_t const*>(src)[idx] : 0;

      constexpr size_type rows_per_element = 32;
      auto const have_trailing_bits = ((num_elements * rows_per_element) - num_rows) < bit_shift;
      while (remainder) {
        // if we're at the very last word of a validity copy, we do not always need to read the next
        // word to get the final trailing bits.
        auto const read_trailing_bits = bit_shift > 0 && remainder == 4 && have_trailing_bits;
        uint32_t const next           = (read_trailing_bits || remainder > 4)
                                          ? (reinterpret_cast<uint32_t const*>(src)[idx + 1])
                                          : 0;

        uint32_t const val = (v >> bit_shift) | (next << (32 - bit_shift));
        if (valid_count) { thread_valid_count += __popc(val); }
        reinterpret_cast<uint32_t*>(dst)[idx] = val;
        v                                     = next;
        idx++;
        remainder -= 4;
      }
    } else {
      while (remainder) {
        std::size_t const idx = num_bytes - remainder--;
        uint32_t const val    = reinterpret_cast<uint8_t const*>(src)[idx];
        if (valid_count) { thread_valid_count += __popc(val); }
        reinterpret_cast<uint8_t*>(dst)[idx] = val;
      }
    }
  }

  if (valid_count) {
    if (num_bytes == 0) {
      if (!t) { *valid_count = 0; }
    } else {
      using BlockReduce = cub::BlockReduce<size_type, block_size>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      size_type block_valid_count{BlockReduce(temp_storage).Sum(thread_valid_count)};
      if (!t) {
        // we may have copied more bits than there are actual rows in the output.
        // so we need to subtract off the count of any bits that shouldn't have been
        // considered during the copy step.
        std::size_t const max_row    = (num_bytes * 8);
        std::size_t const slack_bits = max_row > num_rows ? max_row - num_rows : 0;
        auto const slack_mask        = set_most_significant_bits(slack_bits);
        if (slack_mask > 0) {
          uint32_t const last_word = reinterpret_cast<uint32_t*>(dst + (num_bytes - 4))[0];
          block_valid_count -= __popc(last_word & slack_mask);
        }
        *valid_count = block_valid_count;
      }
    }
  }
}

/**
 * @brief Kernel which copies data from multiple source buffers to multiple
 * destination buffers.
 *
 * When doing a contiguous_split on X columns comprising N total internal buffers
 * with M splits, we end up having to copy N*M source/destination buffer pairs.
 * These logical copies are further subdivided to distribute the amount of work
 * to be done as evenly as possible across the multiprocessors on the device.
 * This kernel is arranged such that each block copies 1 source/destination pair.
 *
 * @param index_to_buffer A function that given a `buf_index` returns the destination buffer
 * @param src_bufs Input source buffers
 * @param buf_info Information on the range of values to be copied for each destination buffer
 */
template <int block_size, typename IndexToDstBuf>
CUDF_KERNEL void copy_partitions(IndexToDstBuf index_to_buffer,
                                 uint8_t const** src_bufs,
                                 dst_buf_info* buf_info)
{
  auto const buf_index     = blockIdx.x;
  auto const src_buf_index = buf_info[buf_index].src_buf_index;

  // copy, shifting offsets and validity bits as needed
  auto& dst = buf_info[buf_index];
  if (dst.is_offsets) {
    if (dst.element_size == 4) {
      copy_buffer<block_size, true, int32_t>(index_to_buffer(buf_index) + dst.dst_offset,
                                             src_bufs[src_buf_index],
                                             threadIdx.x,
                                             dst,
                                             blockDim.x);
    }
    // wide offsets (for long strings)
    else {
      copy_buffer<block_size, true, int64_t>(index_to_buffer(buf_index) + dst.dst_offset,
                                             src_bufs[src_buf_index],
                                             threadIdx.x,
                                             dst,
                                             blockDim.x);
    }
  } else {
    copy_buffer<block_size, false>(index_to_buffer(buf_index) + dst.dst_offset,
                                   src_bufs[src_buf_index],
                                   threadIdx.x,
                                   dst,
                                   blockDim.x);
  }
}

// The block of functions below are all related:
//
// compute_offset_stack_size()
// setup_src_buf_data()
// count_src_bufs()
// setup_source_buf_info()
// build_output_columns()
//
// Critically, they all traverse the hierarchy of source columns and their children
// in a specific order to guarantee they produce various outputs in a consistent
// way.  For example, setup_src_buf_info() produces a series of information
// structs that must appear in the same order that setup_src_buf_data() produces
// buffers.
//
// So please be careful if you change the way in which these functions and
// functors traverse the hierarchy.

/**
 * @brief Returns whether or not the specified type is a column that contains offsets.
 */
bool is_offset_type(type_id id) { return (id == type_id::STRING or id == type_id::LIST); }

/**
 * @brief Compute total device memory stack size needed to process nested
 * offsets per-output buffer.
 *
 * When determining the range of rows to be copied for each output buffer
 * we have to recursively apply the stack of offsets from our parent columns
 * (lists or strings).  We want to do this computation on the gpu because offsets
 * are stored in device memory.  However we don't want to do recursion on the gpu, so
 * each destination buffer gets a "stack" of space to work with equal in size to
 * it's offset nesting depth.  This function computes the total size of all of those
 * stacks.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param offset_depth Current offset nesting depth
 *
 * @returns Total offset stack size needed for this range of columns
 */
template <typename InputIter>
std::size_t compute_offset_stack_size(InputIter begin, InputIter end, int offset_depth = 0)
{
  return std::accumulate(begin, end, 0, [offset_depth](auto stack_size, column_view const& col) {
    auto const num_buffers = 1 + (col.nullable() ? 1 : 0);
    return stack_size + (offset_depth * num_buffers) +
           compute_offset_stack_size(
             col.child_begin(), col.child_end(), offset_depth + is_offset_type(col.type().id()));
  });
}

/**
 * @brief Retrieve all buffers for a range of source columns.
 *
 * Retrieve the individual buffers that make up a range of input columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param out_buf Iterator into output buffer infos
 *
 * @returns next output buffer iterator
 */
template <typename InputIter, typename OutputIter>
OutputIter setup_src_buf_data(InputIter begin, InputIter end, OutputIter out_buf)
{
  std::for_each(begin, end, [&out_buf](column_view const& col) {
    if (col.nullable()) {
      *out_buf = reinterpret_cast<uint8_t const*>(col.null_mask());
      out_buf++;
    }
    // NOTE: we're always returning the base pointer here.  column-level offset is accounted
    // for later. Also, for some column types (string, list, struct) this pointer will be null
    // because there is no associated data with the root column.
    *out_buf = col.head<uint8_t>();
    out_buf++;

    out_buf = setup_src_buf_data(col.child_begin(), col.child_end(), out_buf);
  });
  return out_buf;
}

/**
 * @brief Count the total number of source buffers we will be copying
 * from.
 *
 * This count includes buffers for all input columns. For example a
 * fixed-width column with validity would be 2 buffers (data, validity).
 * A string column with validity would be 3 buffers (chars, offsets, validity).
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 *
 * @returns total number of source buffers for this range of columns
 */
template <typename InputIter>
size_type count_src_bufs(InputIter begin, InputIter end)
{
  auto buf_iter = thrust::make_transform_iterator(begin, [](column_view const& col) {
    auto const children_counts = count_src_bufs(col.child_begin(), col.child_end());
    return 1 + (col.nullable() ? 1 : 0) + children_counts;
  });
  return std::accumulate(buf_iter, buf_iter + std::distance(begin, end), 0);
}

/**
 * @brief Computes source buffer information for the copy kernel.
 *
 * For each input column to be split we need to know several pieces of information
 * in the copy kernel.  This function traverses the input columns and prepares this
 * information for the gpu.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param head Beginning of source buffer info array
 * @param current Current source buffer info to be written to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param offset_stack_pos Integer representing our current offset nesting depth
 * (how many list or string levels deep we are)
 * @param parent_offset_index Index into src_buf_info output array indicating our nearest
 * containing list parent. -1 if we have no list parent
 * @param offset_depth Current offset nesting depth (how many list levels deep we are)
 *
 * @returns next src_buf_output after processing this range of input columns
 */
// setup source buf info
template <typename InputIter>
std::pair<src_buf_info*, size_type> setup_source_buf_info(InputIter begin,
                                                          InputIter end,
                                                          src_buf_info* head,
                                                          src_buf_info* current,
                                                          rmm::cuda_stream_view stream,
                                                          int offset_stack_pos    = 0,
                                                          int parent_offset_index = -1,
                                                          int offset_depth        = 0);

/**
 * @brief Functor that builds source buffer information based on input columns.
 *
 * Called by setup_source_buf_info to build information for a single source column.  This function
 * will recursively call setup_source_buf_info in the case of nested types.
 */
struct buf_info_functor {
  src_buf_info* head;

  template <typename T>
  std::pair<src_buf_info*, size_type> operator()(column_view const& col,
                                                 src_buf_info* current,
                                                 int offset_stack_pos,
                                                 int parent_offset_index,
                                                 int offset_depth,
                                                 rmm::cuda_stream_view)
  {
    if (col.nullable()) {
      std::tie(current, offset_stack_pos) =
        add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
    }

    // info for the data buffer
    *current =
      src_buf_info(col.type().id(), offset_stack_pos, parent_offset_index, false, col.offset());

    return {current + 1, offset_stack_pos + offset_depth};
  }

  template <typename T, typename... Args>
  std::pair<src_buf_info*, size_type> operator()(Args&&...)
    requires(std::is_same_v<T, cudf::dictionary32>)
  {
    CUDF_FAIL("Unsupported type");
  }

 private:
  std::pair<src_buf_info*, size_type> add_null_buffer(column_view const& col,
                                                      src_buf_info* current,
                                                      int offset_stack_pos,
                                                      int parent_offset_index,
                                                      int offset_depth)
  {
    // info for the validity buffer
    *current =
      src_buf_info(type_id::INT32, offset_stack_pos, parent_offset_index, true, col.offset());

    return {current + 1, offset_stack_pos + offset_depth};
  }
};

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::string_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth,
  rmm::cuda_stream_view stream)
{
  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // the way strings are arranged, the strings column itself contains char data, but our child
  // offsets column actually contains our offsets. So our parent_offset_index is actually our child.

  // string columns don't necessarily have children if they are empty
  auto const has_offsets_child = col.num_children() > 0;

  // string columns contain the underlying chars data.
  *current = src_buf_info(type_id::STRING,
                          offset_stack_pos,
                          // if I have an offsets child, it's index will be my parent_offset_index
                          has_offsets_child ? ((current + 1) - head) : parent_offset_index,
                          false,
                          col.offset());

  // if I have offsets, I need to include that in the stack size
  offset_stack_pos += has_offsets_child ? offset_depth + 1 : offset_depth;
  current++;

  if (has_offsets_child) {
    CUDF_EXPECTS(col.num_children() == 1, "Encountered malformed string column");
    strings_column_view scv(col);

    // info for the offsets buffer
    auto offset_col = current;
    CUDF_EXPECTS(not scv.offsets().nullable(), "Encountered nullable string offsets column");
    *current = src_buf_info(scv.offsets().type().id(),
                            // note: offsets can be null in the case where the string column
                            // has been created with empty_like().
                            scv.offsets(),
                            offset_stack_pos,
                            parent_offset_index,
                            false,
                            col.offset());

    current++;
    offset_stack_pos += offset_depth;

    // since we are crossing an offset boundary, calculate our new depth and parent offset index.
    offset_depth++;
    parent_offset_index = offset_col - head;
  }

  return {current, offset_stack_pos};
}

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::list_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth,
  rmm::cuda_stream_view stream)
{
  lists_column_view lcv(col);

  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // list columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current =
    src_buf_info(type_id::LIST, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  CUDF_EXPECTS(col.num_children() == 2, "Encountered malformed list column");

  // info for the offsets buffer
  auto offset_col = current;
  *current        = src_buf_info(type_id::INT32,
                          // note: offsets can be null in the case where the lists column
                          // has been created with empty_like().
                          lcv.offsets(),
                          offset_stack_pos,
                          parent_offset_index,
                          false,
                          col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // since we are crossing an offset boundary, calculate our new depth and parent offset index.
  offset_depth++;
  parent_offset_index = offset_col - head;

  return setup_source_buf_info(col.child_begin() + 1,
                               col.child_end(),
                               head,
                               current,
                               stream,
                               offset_stack_pos,
                               parent_offset_index,
                               offset_depth);
}

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::struct_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth,
  rmm::cuda_stream_view stream)
{
  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // struct columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current =
    src_buf_info(type_id::STRUCT, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // recurse on children
  cudf::structs_column_view scv(col);
  std::vector<column_view> sliced_children;
  sliced_children.reserve(scv.num_children());
  std::transform(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{scv.num_children()},
    std::back_inserter(sliced_children),
    [&scv, &stream](size_type child_index) { return scv.get_sliced_child(child_index, stream); });
  return setup_source_buf_info(sliced_children.begin(),
                               sliced_children.end(),
                               head,
                               current,
                               stream,
                               offset_stack_pos,
                               parent_offset_index,
                               offset_depth);
}

template <typename InputIter>
std::pair<src_buf_info*, size_type> setup_source_buf_info(InputIter begin,
                                                          InputIter end,
                                                          src_buf_info* head,
                                                          src_buf_info* current,
                                                          rmm::cuda_stream_view stream,
                                                          int offset_stack_pos,
                                                          int parent_offset_index,
                                                          int offset_depth)
{
  std::for_each(begin, end, [&](column_view const& col) {
    std::tie(current, offset_stack_pos) = cudf::type_dispatcher(col.type(),
                                                                buf_info_functor{head},
                                                                col,
                                                                current,
                                                                offset_stack_pos,
                                                                parent_offset_index,
                                                                offset_depth,
                                                                stream);
  });
  return {current, offset_stack_pos};
}

/**
 * @brief Given a column, processed split buffers, and a metadata builder, populate
 * the metadata for this column in the builder, and return a tuple of:
 * column size, data offset, bitmask offset and null count.
 *
 * @param src column_view to create metadata from
 * @param current_info dst_buf_info pointer reference, pointing to this column's buffer info
 *                     This is a pointer reference because it is updated by this function as the
 *                     columns's validity and data buffers are visited
 * @param mb A metadata_builder instance to update with the column's packed metadata
 * @param use_src_null_count True for the chunked_pack case where current_info has invalid null
 *                           count information. The null count should be taken
 *                           from `src` because this case is restricted to a single partition
 *                           (no splits)
 * @returns a std::tuple containing:
 *          column size, data offset, bitmask offset, and null count
 */
template <typename BufInfo>
std::tuple<std::size_t, int64_t, int64_t, size_type> build_output_column_metadata(
  column_view const& src,
  BufInfo& current_info,
  detail::metadata_builder& mb,
  bool use_src_null_count)
{
  auto [bitmask_offset, null_count] = [&]() {
    if (src.nullable()) {
      // offsets in the existing serialized_column metadata are int64_t
      // that's the reason for the casting in this code.
      int64_t const bitmask_offset =
        current_info->num_elements == 0
          ? -1  // this means that the bitmask buffer pointer should be nullptr
          : static_cast<int64_t>(current_info->dst_offset);

      // use_src_null_count is used for the chunked contig split case, where we have
      // no splits: the null_count is just the source column's null_count
      size_type const null_count = use_src_null_count
                                     ? src.null_count()
                                     : (current_info->num_elements == 0
                                          ? 0
                                          : (current_info->num_rows - current_info->valid_count));

      ++current_info;
      return std::pair(bitmask_offset, null_count);
    }
    return std::pair(static_cast<int64_t>(-1), 0);
  }();

  // size/data pointer for the column
  auto const col_size = [&]() -> std::size_t {
    // if I am a string column, I need to use the number of rows from my child offset column. the
    // number of rows in my dst_buf_info struct will be equal to the number of chars, which is
    // incorrect. this is a quirk of how cudf stores strings.
    if (src.type().id() == type_id::STRING) {
      // if I have no children (no offsets), then I must have a row count of 0
      if (src.num_children() == 0) { return 0; }

      // otherwise my actual number of rows will be the num_rows field of the next dst_buf_info
      // struct (our child offsets column)
      return (current_info + 1)->num_rows;
    }

    // otherwise the number of rows is the number of elements
    return static_cast<std::size_t>(current_info->num_elements);
  }();
  int64_t const data_offset =
    col_size == 0 || src.head() == nullptr ? -1 : static_cast<int64_t>(current_info->dst_offset);

  mb.add_column_info_to_meta(
    src.type(), col_size, null_count, data_offset, bitmask_offset, src.num_children());

  ++current_info;
  return {col_size, data_offset, bitmask_offset, null_count};
}

/**
 * @brief Given a set of input columns and processed split buffers, produce
 * output columns.
 *
 * After performing the split we are left with 1 large buffer per incoming split
 * partition.  We need to traverse this buffer and distribute the individual
 * subpieces that represent individual columns and children to produce the final
 * output columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param info_begin Iterator of dst_buf_info structs containing information about each
 * copied buffer
 * @param out_begin Output iterator of column views
 * @param base_ptr Pointer to the base address of copied data for the working partition
 * @param mb Memory block for the output columns
 *
 * @returns new dst_buf_info iterator after processing this range of input columns
 */
template <typename InputIter, typename BufInfo, typename Output>
BufInfo build_output_columns(InputIter begin,
                             InputIter end,
                             BufInfo info_begin,
                             Output out_begin,
                             uint8_t const* const base_ptr,
                             detail::metadata_builder& mb)
{
  auto current_info = info_begin;
  std::transform(begin, end, out_begin, [&current_info, base_ptr, &mb](column_view const& src) {
    auto [col_size, data_offset, bitmask_offset, null_count] =
      build_output_column_metadata<BufInfo>(src, current_info, mb, false);

    auto const bitmask_ptr =
      base_ptr != nullptr && bitmask_offset != -1
        ? reinterpret_cast<bitmask_type const*>(base_ptr + static_cast<uint64_t>(bitmask_offset))
        : nullptr;

    // size/data pointer for the column
    uint8_t const* data_ptr = base_ptr != nullptr && data_offset != -1
                                ? base_ptr + static_cast<uint64_t>(data_offset)
                                : nullptr;

    // children
    auto children = std::vector<column_view>{};
    children.reserve(src.num_children());

    current_info = build_output_columns(
      src.child_begin(), src.child_end(), current_info, std::back_inserter(children), base_ptr, mb);

    return column_view{src.type(),
                       static_cast<cudf::size_type>(col_size),
                       data_ptr,
                       bitmask_ptr,
                       null_count,
                       0,
                       std::move(children)};
  });

  return current_info;
}

/**
 * @brief Given a set of input columns, processed split buffers, and a metadata_builder,
 * append column metadata using the builder.
 *
 * After performing the split we are left with 1 large buffer per incoming split
 * partition.  We need to traverse this buffer and distribute the individual
 * subpieces that represent individual columns and children to produce the final
 * output columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param info_begin Iterator of dst_buf_info structs containing information about each
 * copied buffer
 * @param mb packed column metadata builder
 *
 * @returns new dst_buf_info iterator after processing this range of input columns
 */
template <typename InputIter, typename BufInfo>
BufInfo populate_metadata(InputIter begin,
                          InputIter end,
                          BufInfo info_begin,
                          detail::metadata_builder& mb)
{
  auto current_info = info_begin;
  std::for_each(begin, end, [&current_info, &mb](column_view const& src) {
    build_output_column_metadata<BufInfo>(src, current_info, mb, true);

    // children
    current_info = populate_metadata(src.child_begin(), src.child_end(), current_info, mb);
  });

  return current_info;
}

/**
 * @brief Functor that retrieves the size of a destination buffer
 */
struct buf_size_functor {
  dst_buf_info const* ci;
  std::size_t operator() __device__(int index) { return ci[index].buf_size; }
};

/**
 * @brief Functor that retrieves the split "key" for a given output
 * buffer index.
 *
 * The key is simply the partition index.
 */
struct split_key_functor {
  int const num_src_bufs;
  int operator() __device__(int buf_index) const { return buf_index / num_src_bufs; }
};

/**
 * @brief Output iterator for writing values to the dst_offset field of the
 * dst_buf_info struct
 */
struct dst_offset_output_iterator {
  dst_buf_info* c;
  using value_type        = std::size_t;
  using difference_type   = std::size_t;
  using pointer           = std::size_t*;
  using reference         = std::size_t&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_offset_output_iterator operator+ __host__ __device__(int i) { return {c + i}; }

  dst_offset_output_iterator& operator++ __host__ __device__()
  {
    c++;
    return *this;
  }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->dst_offset; }
};

/**
 * @brief Output iterator for writing values to the valid_count field of the
 * dst_buf_info struct
 */
struct dst_valid_count_output_iterator {
  dst_buf_info* c;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_valid_count_output_iterator operator+ __host__ __device__(int i) { return {c + i}; }

  dst_valid_count_output_iterator& operator++ __host__ __device__()
  {
    c++;
    return *this;
  }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->valid_count; }
};

/**
 * @brief Functor for computing size of data elements for a given cudf type.
 *
 * Note: columns types which themselves inherently have no data (strings, lists,
 * structs) return 0.
 */
struct size_of_helper {
  template <typename T>
  constexpr int __device__ operator()() const
    requires(!is_fixed_width<T>() && !std::is_same_v<T, cudf::string_view>)
  {
    return 0;
  }

  template <typename T>
  constexpr int __device__ operator()() const
    requires(!is_fixed_width<T>() && std::is_same_v<T, cudf::string_view>)
  {
    return sizeof(cudf::device_storage_type_t<int8_t>);
  }

  template <typename T>
  constexpr int __device__ operator()() const noexcept
    requires(is_fixed_width<T>())
  {
    return sizeof(cudf::device_storage_type_t<T>);
  }
};

/**
 * @brief Functor for returning the number of batches an input buffer is being
 * subdivided into during the repartitioning step.
 *
 * Note: columns types which themselves inherently have no data (strings, lists,
 * structs) return 0.
 */
struct num_batches_func {
  cuda::std::pair<std::size_t, std::size_t> const* const batches;
  __device__ std::size_t operator()(size_type i) const { return cuda::std::get<0>(batches[i]); }
};

/**
 * @brief Get the size in bytes of a batch described by `dst_buf_info`.
 */
struct batch_byte_size_function {
  size_type const num_batches;
  dst_buf_info const* const infos;
  __device__ std::size_t operator()(size_type i) const
  {
    if (i == num_batches) { return 0; }
    auto const& buf = *(infos + i);
    std::size_t const bytes =
      static_cast<std::size_t>(buf.num_elements) * static_cast<std::size_t>(buf.element_size);
    return util::round_up_unsafe(bytes, split_align);
  }
};

/**
 * @brief Get the input buffer index given the output buffer index.
 */
struct out_to_in_index_function {
  size_type const* const batch_offsets;
  int const num_bufs;
  __device__ int operator()(size_type i) const
  {
    return static_cast<size_type>(
             thrust::upper_bound(thrust::seq, batch_offsets, batch_offsets + num_bufs + 1, i) -
             batch_offsets) -
           1;
  }
};

// packed block of memory 1: split indices and src_buf_info structs
struct packed_split_indices_and_src_buf_info {
  packed_split_indices_and_src_buf_info(cudf::table_view const& input,
                                        std::vector<size_type> const& splits,
                                        std::size_t num_partitions,
                                        cudf::size_type num_src_bufs,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref temp_mr)
    : indices_size(cudf::util::round_up_safe((num_partitions + 1) * sizeof(int64_t), split_align)),
      src_buf_info_size(
        cudf::util::round_up_safe(num_src_bufs * sizeof(src_buf_info), split_align)),
      // host-side
      h_indices_and_source_info{
        detail::make_host_vector<uint8_t>(indices_size + src_buf_info_size, stream)},
      h_indices{reinterpret_cast<int64_t*>(h_indices_and_source_info.data())},
      h_src_buf_info{
        reinterpret_cast<src_buf_info*>(h_indices_and_source_info.data() + indices_size)}
  {
    // compute splits -> indices.
    // these are row numbers per split
    h_indices[0]              = 0;
    h_indices[num_partitions] = input.column(0).size();
    std::copy(splits.begin(), splits.end(), std::next(h_indices));

    // setup source buf info
    setup_source_buf_info(input.begin(), input.end(), h_src_buf_info, h_src_buf_info, stream);

    offset_stack_partition_size = compute_offset_stack_size(input.begin(), input.end());
    offset_stack_size           = offset_stack_partition_size * num_partitions * sizeof(size_type);
    // device-side
    // gpu-only : stack space needed for nested list offset calculation
    d_indices_and_source_info =
      rmm::device_buffer(indices_size + src_buf_info_size + offset_stack_size, stream, temp_mr);
    d_indices      = reinterpret_cast<int64_t*>(d_indices_and_source_info.data());
    d_src_buf_info = reinterpret_cast<src_buf_info*>(
      reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) + indices_size);
    d_offset_stack =
      reinterpret_cast<size_type*>(reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) +
                                   indices_size + src_buf_info_size);

    detail::cuda_memcpy_async<uint8_t>(
      device_span<uint8_t>{static_cast<uint8_t*>(d_indices_and_source_info.data()),
                           h_indices_and_source_info.size()},
      h_indices_and_source_info,
      stream);
  }

  size_type const indices_size;
  std::size_t const src_buf_info_size;
  std::size_t offset_stack_size;

  detail::host_vector<uint8_t> h_indices_and_source_info;
  rmm::device_buffer d_indices_and_source_info;

  int64_t* const h_indices;
  src_buf_info* const h_src_buf_info;

  int offset_stack_partition_size;
  int64_t* d_indices;
  src_buf_info* d_src_buf_info;
  size_type* d_offset_stack;
};

// packed block of memory 2: partition buffer sizes and dst_buf_info structs
struct packed_partition_buf_size_and_dst_buf_info {
  packed_partition_buf_size_and_dst_buf_info(std::size_t num_partitions,
                                             std::size_t num_bufs,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref temp_mr)
    : stream(stream),
      buf_sizes_size{cudf::util::round_up_safe(num_partitions * sizeof(std::size_t), split_align)},
      dst_buf_info_size{cudf::util::round_up_safe(num_bufs * sizeof(dst_buf_info), split_align)},
      // host-side
      h_buf_sizes_and_dst_info{
        detail::make_host_vector<uint8_t>(buf_sizes_size + dst_buf_info_size, stream)},
      h_buf_sizes{reinterpret_cast<std::size_t*>(h_buf_sizes_and_dst_info.data())},
      h_dst_buf_info{
        reinterpret_cast<dst_buf_info*>(h_buf_sizes_and_dst_info.data() + buf_sizes_size),
        num_bufs,
        h_buf_sizes_and_dst_info.get_allocator().is_device_accessible()},
      // device-side
      d_buf_sizes_and_dst_info(h_buf_sizes_and_dst_info.size(), stream, temp_mr),
      d_buf_sizes{reinterpret_cast<std::size_t*>(d_buf_sizes_and_dst_info.data())},
      // destination buffer info
      d_dst_buf_info{
        reinterpret_cast<dst_buf_info*>(d_buf_sizes_and_dst_info.data() + buf_sizes_size), num_bufs}
  {
  }

  void copy_to_host()
  {
    // DtoH buf sizes and col info back to the host
    detail::cuda_memcpy_async<uint8_t>(h_buf_sizes_and_dst_info, d_buf_sizes_and_dst_info, stream);
  }

  rmm::cuda_stream_view const stream;

  // buffer sizes and destination info (used in batched copies)
  std::size_t const buf_sizes_size;
  std::size_t const dst_buf_info_size;

  detail::host_vector<uint8_t> h_buf_sizes_and_dst_info;
  std::size_t* const h_buf_sizes;
  host_span<dst_buf_info> const h_dst_buf_info;

  rmm::device_uvector<uint8_t> d_buf_sizes_and_dst_info;
  std::size_t* const d_buf_sizes;
  device_span<dst_buf_info> const d_dst_buf_info;
};

// Packed block of memory 3:
// Pointers to source and destination buffers (and stack space on the
// gpu for offset computation)
struct packed_src_and_dst_pointers {
  packed_src_and_dst_pointers(cudf::table_view const& input,
                              std::size_t num_partitions,
                              cudf::size_type num_src_bufs,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref temp_mr)
    : stream(stream),
      src_bufs_size{cudf::util::round_up_safe(num_src_bufs * sizeof(uint8_t*), split_align)},
      dst_bufs_size{cudf::util::round_up_safe(num_partitions * sizeof(uint8_t*), split_align)},
      // host-side
      h_src_and_dst_buffers{
        detail::make_host_vector<uint8_t>(src_bufs_size + dst_bufs_size, stream)},
      h_src_bufs{reinterpret_cast<uint8_t const**>(h_src_and_dst_buffers.data())},
      h_dst_bufs{reinterpret_cast<uint8_t**>(h_src_and_dst_buffers.data() + src_bufs_size)},
      // device-side
      d_src_and_dst_buffers{h_src_and_dst_buffers.size(), stream, temp_mr},
      d_src_bufs{reinterpret_cast<uint8_t const**>(d_src_and_dst_buffers.data())},
      d_dst_bufs{reinterpret_cast<uint8_t**>(
        reinterpret_cast<uint8_t*>(d_src_and_dst_buffers.data()) + src_bufs_size)}
  {
    // setup src buffers
    setup_src_buf_data(input.begin(), input.end(), h_src_bufs);
  }

  void copy_to_device()
  {
    detail::cuda_memcpy_async<uint8_t>(
      device_span<uint8_t>{static_cast<uint8_t*>(d_src_and_dst_buffers.data()),
                           d_src_and_dst_buffers.size()},
      h_src_and_dst_buffers,
      stream);
  }

  rmm::cuda_stream_view const stream;
  std::size_t const src_bufs_size;
  std::size_t const dst_bufs_size;

  detail::host_vector<uint8_t> h_src_and_dst_buffers;
  uint8_t const** const h_src_bufs;
  uint8_t** const h_dst_bufs;

  rmm::device_buffer d_src_and_dst_buffers;
  uint8_t const** const d_src_bufs;
  uint8_t** const d_dst_bufs;
};

/**
 * @brief Create an instance of `packed_src_and_dst_pointers` populating destination
 * partition buffers (if any) from `out_buffers`. In the chunked_pack case
 * `out_buffers` is empty, and the destination pointer is provided separately
 * to the `copy_partitions` kernel.
 *
 * @param input source table view
 * @param num_partitions the number of partitions (1 meaning no splits)
 * @param num_src_bufs number of buffers for the source columns including children
 * @param out_buffers the destination buffers per partition if in the non-chunked case
 * @param stream Optional CUDA stream on which to execute kernels
 * @param temp_mr A memory resource for temporary and scratch space
 *
 * @returns new unique pointer to packed_src_and_dst_pointers
 */
std::unique_ptr<packed_src_and_dst_pointers> setup_src_and_dst_pointers(
  cudf::table_view const& input,
  std::size_t num_partitions,
  cudf::size_type num_src_bufs,
  std::vector<rmm::device_buffer>& out_buffers,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref temp_mr)
{
  auto src_and_dst_pointers = std::make_unique<packed_src_and_dst_pointers>(
    input, num_partitions, num_src_bufs, stream, temp_mr);

  std::transform(
    out_buffers.begin(), out_buffers.end(), src_and_dst_pointers->h_dst_bufs, [](auto& buf) {
      return static_cast<uint8_t*>(buf.data());
    });

  // copy the struct to device memory to access from the kernel
  src_and_dst_pointers->copy_to_device();

  return src_and_dst_pointers;
}

/**
 * @brief Create an instance of `packed_partition_buf_size_and_dst_buf_info` containing
 * the partition-level dst_buf_info structs for each partition and column buffer.
 *
 * @param input source table view
 * @param splits the numeric value (in rows) for each split, empty for 1 partition
 * @param num_partitions the number of partitions create (1 meaning no splits)
 * @param num_src_bufs number of buffers for the source columns including children
 * @param num_bufs num_src_bufs times the number of partitions
 * @param stream Optional CUDA stream on which to execute kernels
 * @param temp_mr A memory resource for temporary and scratch space
 *
 * @returns new unique pointer to `packed_partition_buf_size_and_dst_buf_info`
 */
std::unique_ptr<packed_partition_buf_size_and_dst_buf_info> compute_splits(
  cudf::table_view const& input,
  std::vector<size_type> const& splits,
  std::size_t num_partitions,
  cudf::size_type num_src_bufs,
  std::size_t num_bufs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref temp_mr)
{
  auto partition_buf_size_and_dst_buf_info =
    std::make_unique<packed_partition_buf_size_and_dst_buf_info>(
      num_partitions, num_bufs, stream, temp_mr);

  auto const d_dst_buf_info = partition_buf_size_and_dst_buf_info->d_dst_buf_info.data();
  auto const d_buf_sizes    = partition_buf_size_and_dst_buf_info->d_buf_sizes;

  auto const split_indices_and_src_buf_info = packed_split_indices_and_src_buf_info(
    input, splits, num_partitions, num_src_bufs, stream, temp_mr);

  auto const d_src_buf_info = split_indices_and_src_buf_info.d_src_buf_info;
  auto const offset_stack_partition_size =
    split_indices_and_src_buf_info.offset_stack_partition_size;
  auto const d_offset_stack = split_indices_and_src_buf_info.d_offset_stack;
  auto const d_indices      = split_indices_and_src_buf_info.d_indices;

  // compute sizes of each column in each partition, including alignment.
  thrust::transform(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator<std::size_t>{num_bufs},
    d_dst_buf_info,
    cuda::proclaim_return_type<dst_buf_info>([d_src_buf_info,
                                              offset_stack_partition_size,
                                              d_offset_stack,
                                              d_indices,
                                              num_src_bufs] __device__(std::size_t t) {
      int const split_index   = t / num_src_bufs;
      int const src_buf_index = t % num_src_bufs;
      auto const& src_info    = d_src_buf_info[src_buf_index];

      // apply nested offsets (lists and string columns).
      //
      // We can't just use the incoming row indices to figure out where to read from in a
      // nested list situation.  We have to apply offsets every time we cross a boundary
      // (list or string).  This loop applies those offsets so that our incoming row_index_start
      // and row_index_end get transformed to our final values.
      //
      int const stack_pos = src_info.offset_stack_pos + (split_index * offset_stack_partition_size);
      size_type* offset_stack    = &d_offset_stack[stack_pos];
      int parent_offsets_index   = src_info.parent_offsets_index;
      int stack_size             = 0;
      int64_t root_column_offset = src_info.column_offset;
      while (parent_offsets_index >= 0) {
        offset_stack[stack_size++] = parent_offsets_index;
        root_column_offset         = d_src_buf_info[parent_offsets_index].column_offset;
        parent_offsets_index       = d_src_buf_info[parent_offsets_index].parent_offsets_index;
      }
      // make sure to include the -column- offset on the root column in our calculation.
      int64_t row_start = d_indices[split_index] + root_column_offset;
      int64_t row_end   = d_indices[split_index + 1] + root_column_offset;
      while (stack_size > 0) {
        stack_size--;
        auto& d_info = d_src_buf_info[offset_stack[stack_size]];
        // this case can happen when you have empty string or list columns constructed with
        // empty_like()
        if (d_info.is_offsets) {
          row_start = d_info.offsets[row_start];
          row_end   = d_info.offsets[row_end];
        }
      }

      // final element indices and row count
      auto const src_element_index = src_info.is_validity ? row_start / 32 : row_start;
      std::size_t const num_rows   = row_end - row_start;
      // if I am an offsets column, all my values need to be shifted
      auto const value_shift = !src_info.is_offsets ? 0 : src_info.offsets[row_start];
      // if I am a validity column, we may need to shift bits
      int const bit_shift = src_info.is_validity ? row_start % 32 : 0;
      // # of rows isn't necessarily the same as # of elements to be copied.
      auto const num_elements = [&]() {
        if (src_info.is_offsets && num_rows > 0) {
          return num_rows + 1;
        } else if (src_info.is_validity) {
          return (num_rows + 31) / 32;
        }
        return num_rows;
      }();
      int const element_size  = cudf::type_dispatcher(data_type{src_info.type}, size_of_helper{});
      std::size_t const bytes = num_elements * static_cast<std::size_t>(element_size);

      return dst_buf_info{util::round_up_unsafe(bytes, split_align),
                          num_elements,
                          element_size,
                          num_rows,
                          src_element_index,
                          0,
                          value_shift,
                          bit_shift,
                          src_info.is_validity ? 1 : 0,
                          src_info.is_offsets,
                          src_buf_index,
                          split_index};
    }));

  // compute total size of each partition
  // key is the split index
  {
    auto const keys = cudf::detail::make_counting_transform_iterator(
      0, split_key_functor{static_cast<int>(num_src_bufs)});
    auto values =
      cudf::detail::make_counting_transform_iterator(0, buf_size_functor{d_dst_buf_info});

    thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                          keys,
                          keys + num_bufs,
                          values,
                          cuda::make_discard_iterator(),
                          d_buf_sizes);
  }

  // compute start offset for each output buffer for each split
  {
    auto const keys = cudf::detail::make_counting_transform_iterator(
      0, split_key_functor{static_cast<int>(num_src_bufs)});
    auto values =
      cudf::detail::make_counting_transform_iterator(0, buf_size_functor{d_dst_buf_info});

    thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                                  keys,
                                  keys + num_bufs,
                                  values,
                                  dst_offset_output_iterator{d_dst_buf_info},
                                  std::size_t{0});
  }

  partition_buf_size_and_dst_buf_info->copy_to_host();

  stream.synchronize();

  return partition_buf_size_and_dst_buf_info;
}

/**
 * @brief Compute the number of source buffers, number of buffers, and splits information.
 *
 * @param input source table view
 * @param splits the numeric value (in rows) for each split, empty for 1 partition
 * @param stream Optional CUDA stream on which to execute kernels
 * @param temp_mr A memory resource for temporary and scratch space
 * @return A tuple containing (num_src_bufs, num_bufs, partition_buf_size_and_dst_buf_info)
 */
std::tuple<size_type, std::size_t, std::unique_ptr<packed_partition_buf_size_and_dst_buf_info>>
compute_num_bufs_and_splits(cudf::table_view const& input,
                            std::vector<size_type> const& splits,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref temp_mr)
{
  std::size_t const num_partitions = splits.size() + 1;
  auto num_src_bufs                = count_src_bufs(input.begin(), input.end());
  auto num_bufs                    = num_src_bufs * num_partitions;

  // First pass over the source tables to generate a `dst_buf_info` per split and column buffer
  // (`num_bufs`). After this, contiguous_split uses `dst_buf_info` to further subdivide the work
  // into 1MB batches in `compute_batches`
  auto partition_buf_size_and_dst_buf_info =
    compute_splits(input, splits, num_partitions, num_src_bufs, num_bufs, stream, temp_mr);

  return std::make_tuple(num_src_bufs, num_bufs, std::move(partition_buf_size_and_dst_buf_info));
}

/**
 * @brief Struct containing information about the actual batches we will send to the
 * `copy_partitions` kernel and the number of iterations we need to carry out this copy.
 *
 * For the non-chunked contiguous_split case, this contains the batched dst_buf_infos and the
 * number of iterations is going to be 1 since the non-chunked case is single pass.
 *
 * For the chunked_pack case, this also contains the batched dst_buf_infos for all
 * iterations in addition to helping keep the state about what batches have been copied so far
 * and what are the sizes (in bytes) of each iteration.
 */
struct chunk_iteration_state {
  chunk_iteration_state(rmm::device_uvector<dst_buf_info> _d_batched_dst_buf_info,
                        rmm::device_uvector<size_type> _d_batch_offsets,
                        std::vector<std::size_t>&& _h_num_buffs_per_iteration,
                        std::vector<std::size_t>&& _h_size_of_buffs_per_iteration,
                        std::size_t total_size)
    : num_iterations(_h_num_buffs_per_iteration.size()),
      current_iteration{0},
      starting_batch{0},
      d_batched_dst_buf_info(std::move(_d_batched_dst_buf_info)),
      d_batch_offsets(std::move(_d_batch_offsets)),
      h_num_buffs_per_iteration(std::move(_h_num_buffs_per_iteration)),
      h_size_of_buffs_per_iteration(std::move(_h_size_of_buffs_per_iteration)),
      total_size(total_size)
  {
  }

  static std::unique_ptr<chunk_iteration_state> create(
    rmm::device_uvector<cuda::std::pair<std::size_t, std::size_t>> const& batches,
    int num_bufs,
    dst_buf_info* d_orig_dst_buf_info,
    std::size_t const* const h_buf_sizes,
    std::size_t num_partitions,
    std::size_t user_buffer_size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);

  /**
   * @brief As of the time of the call, return the starting 1MB batch index, and the
   * number of batches to copy.
   *
   * @return the current iteration's starting_batch and batch count as a pair
   */
  std::pair<std::size_t, std::size_t> get_current_starting_index_and_buff_count() const
  {
    CUDF_EXPECTS(current_iteration < num_iterations,
                 "current_iteration cannot exceed num_iterations");
    auto count_for_current = h_num_buffs_per_iteration[current_iteration];
    return {starting_batch, count_for_current};
  }

  /**
   * @brief Advance the iteration state if there are iterations left, updating the
   * starting batch and returning the amount of bytes were copied in the iteration
   * we just finished.
   * @throws cudf::logic_error If the state was at the last iteration before entering
   * this function.
   * @return size in bytes that were copied in the finished iteration
   */
  std::size_t advance_iteration()
  {
    CUDF_EXPECTS(current_iteration < num_iterations,
                 "current_iteration cannot exceed num_iterations");
    std::size_t bytes_copied = h_size_of_buffs_per_iteration[current_iteration];
    starting_batch += h_num_buffs_per_iteration[current_iteration];
    ++current_iteration;
    return bytes_copied;
  }

  /**
   * Returns true if there are iterations left.
   */
  bool has_more_copies() const { return current_iteration < num_iterations; }

  rmm::device_uvector<dst_buf_info> d_batched_dst_buf_info;  ///< dst_buf_info per 1MB batch
  rmm::device_uvector<size_type> const d_batch_offsets;  ///< Offset within a batch per dst_buf_info
  std::size_t const total_size;                          ///< The aggregate size of all iterations
  int const num_iterations;                              ///< The total number of iterations
  int current_iteration;  ///< Marks the current iteration being worked on

 private:
  std::size_t starting_batch;  ///< Starting batch index for the current iteration
  std::vector<std::size_t> const h_num_buffs_per_iteration;  ///< The count of batches per iteration
  std::vector<std::size_t> const
    h_size_of_buffs_per_iteration;  ///< The size in bytes per iteration
};

std::unique_ptr<chunk_iteration_state> chunk_iteration_state::create(
  rmm::device_uvector<cuda::std::pair<std::size_t, std::size_t>> const& batches,
  int num_bufs,
  dst_buf_info* d_orig_dst_buf_info,
  std::size_t const* const h_buf_sizes,
  std::size_t num_partitions,
  std::size_t user_buffer_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref temp_mr)
{
  rmm::device_uvector<size_type> d_batch_offsets(num_bufs + 1, stream, temp_mr);

  auto const buf_count_iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<std::size_t>(
      [num_bufs, num_batches = num_batches_func{batches.begin()}] __device__(size_type i) {
        return i == num_bufs ? 0 : num_batches(i);
      }));

  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         buf_count_iter,
                         buf_count_iter + num_bufs + 1,
                         d_batch_offsets.begin(),
                         0);

  auto const num_batches_iter =
    cudf::detail::make_counting_transform_iterator(0, num_batches_func{batches.begin()});
  size_type const num_batches = thrust::reduce(
    rmm::exec_policy_nosync(stream, temp_mr), num_batches_iter, num_batches_iter + batches.size());

  auto out_to_in_index = out_to_in_index_function{d_batch_offsets.begin(), num_bufs};

  auto const iter = cuda::counting_iterator<cudf::size_type>{0};

  // load up the batches as d_dst_buf_info
  rmm::device_uvector<dst_buf_info> d_batched_dst_buf_info(num_batches, stream, temp_mr);

  thrust::for_each(
    rmm::exec_policy_nosync(stream, temp_mr),
    iter,
    iter + num_batches,
    [d_orig_dst_buf_info,
     d_batched_dst_buf_info = d_batched_dst_buf_info.data(),
     batches                = batches.data(),
     d_batch_offsets        = d_batch_offsets.begin(),
     out_to_in_index] __device__(size_type i) {
      size_type const in_buf_index = out_to_in_index(i);
      size_type const batch_index  = i - d_batch_offsets[in_buf_index];
      auto const batch_size        = cuda::std::get<1>(batches[in_buf_index]);
      dst_buf_info const& in       = d_orig_dst_buf_info[in_buf_index];

      // adjust info
      dst_buf_info& out = d_batched_dst_buf_info[i];
      out               = in;

      std::size_t const elements_per_batch =
        out.element_size == 0 ? 0 : batch_size / out.element_size;
      out.num_elements = ((batch_index + 1) * elements_per_batch) > in.num_elements
                           ? in.num_elements - (batch_index * elements_per_batch)
                           : elements_per_batch;

      std::size_t const rows_per_batch =
        // if this is a validity buffer, each element is a bitmask_type, which
        // corresponds to 32 rows.
        out.valid_count > 0 ? elements_per_batch * cudf::detail::size_in_bits<bitmask_type>()
                            : elements_per_batch;
      out.num_rows = ((batch_index + 1) * rows_per_batch) > in.num_rows
                       ? in.num_rows - (batch_index * rows_per_batch)
                       : rows_per_batch;

      out.src_element_index = in.src_element_index + (batch_index * elements_per_batch);
      out.dst_offset        = in.dst_offset + (batch_index * batch_size);

      // out.bytes and out.buf_size are unneeded here because they are only used to
      // calculate real output buffer sizes. the data we are generating here is
      // purely intermediate for the purposes of doing more uniform copying of data
      // underneath the final structure of the output
    });

  /**
   * In the chunked case, this is the code that fixes up the offsets of each batch
   * and prepares each iteration. Given the batches computed before, it figures
   * out the number of batches that will fit in an iteration of `user_buffer_size`.
   *
   * Specifically, offsets for batches are reset to the 0th byte when a new iteration
   * of `user_buffer_size` bytes is needed.
   */
  if (user_buffer_size != 0) {
    // copy the batch offsets back to host
    auto const h_offsets = [&] {
      rmm::device_uvector<std::size_t> offsets(num_batches + 1, stream, temp_mr);
      auto const batch_byte_size_iter = cudf::detail::make_counting_transform_iterator(
        0, batch_byte_size_function{num_batches, d_batched_dst_buf_info.begin()});

      thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                             batch_byte_size_iter,
                             batch_byte_size_iter + offsets.size(),
                             offsets.begin());

      // the next part is working on the CPU, so we want to synchronize here
      return detail::make_host_vector(offsets, stream);
    }();

    std::vector<std::size_t> num_batches_per_iteration;
    std::vector<std::size_t> size_of_batches_per_iteration;
    auto accum_size_per_iteration =
      cudf::detail::make_empty_host_vector<std::size_t>(h_offsets.size(), stream);
    std::size_t accum_size = 0;
    {
      auto current_offset_it = h_offsets.begin();
      // figure out how many iterations we need, while fitting batches to iterations
      // with no more than user_buffer_size bytes worth of batches
      while (current_offset_it != h_offsets.end()) {
        // next_iteration_it points to the batch right above the boundary (the batch
        // that didn't fit).
        auto next_iteration_it =
          std::lower_bound(current_offset_it,
                           h_offsets.end(),
                           // We add the cumulative size + 1 because we want to find what would fit
                           // within a buffer of user_buffer_size (up to user_buffer_size).
                           // Since h_offsets is a prefix scan, we add the size we accumulated so
                           // far so we are looking for the next user_buffer_sized boundary.
                           user_buffer_size + accum_size + 1);

        // we subtract 1 from the number of batch here because next_iteration_it points
        // to the batch that didn't fit, so it's one off.
        auto batches_in_iter = std::distance(current_offset_it, next_iteration_it) - 1;

        // to get the amount of bytes in this iteration we get the prefix scan size
        // and subtract the cumulative size so far, leaving the bytes belonging to this
        // iteration
        auto iter_size_bytes = *(current_offset_it + batches_in_iter) - accum_size;
        accum_size += iter_size_bytes;

        num_batches_per_iteration.push_back(batches_in_iter);
        size_of_batches_per_iteration.push_back(iter_size_bytes);
        accum_size_per_iteration.push_back(accum_size);

        if (next_iteration_it == h_offsets.end()) { break; }

        current_offset_it += batches_in_iter;
      }
    }

    // apply changed offset
    {
      auto d_accum_size_per_iteration =
        cudf::detail::make_device_uvector_async(accum_size_per_iteration, stream, temp_mr);

      // we want to update the offset of batches for every iteration, except the first one (because
      // offsets in the first iteration are all 0 based)
      auto num_batches_in_first_iteration = num_batches_per_iteration[0];
      auto const iter                     = cuda::counting_iterator{num_batches_in_first_iteration};
      auto num_iterations                 = accum_size_per_iteration.size();
      thrust::for_each(
        rmm::exec_policy_nosync(stream, temp_mr),
        iter,
        iter + num_batches - num_batches_in_first_iteration,
        [num_iterations,
         d_batched_dst_buf_info     = d_batched_dst_buf_info.data(),
         d_accum_size_per_iteration = d_accum_size_per_iteration.data()] __device__(size_type i) {
          auto prior_iteration_size =
            thrust::upper_bound(thrust::seq,
                                d_accum_size_per_iteration,
                                d_accum_size_per_iteration + num_iterations,
                                d_batched_dst_buf_info[i].dst_offset) -
            1;
          d_batched_dst_buf_info[i].dst_offset -= *prior_iteration_size;
        });
    }
    return std::make_unique<chunk_iteration_state>(std::move(d_batched_dst_buf_info),
                                                   std::move(d_batch_offsets),
                                                   std::move(num_batches_per_iteration),
                                                   std::move(size_of_batches_per_iteration),
                                                   accum_size);

  } else {
    // we instantiate an "iteration state" for the regular single pass contiguous_split
    // consisting of 1 iteration with all of the batches and totalling `total_size` bytes.
    auto const total_size = std::reduce(h_buf_sizes, h_buf_sizes + num_partitions);

    // 1 iteration with the whole size
    return std::make_unique<chunk_iteration_state>(
      std::move(d_batched_dst_buf_info),
      std::move(d_batch_offsets),
      std::move(std::vector<std::size_t>{static_cast<std::size_t>(num_batches)}),
      std::move(std::vector<std::size_t>{total_size}),
      total_size);
  }
}

/**
 * @brief Create an instance of `chunk_iteration_state` containing 1MB batches of work
 * that are further grouped into chunks or iterations.
 *
 * This function handles both the `chunked_pack` case: when `user_buffer_size` is non-zero,
 * and the single-shot `contiguous_split` case.
 *
 * @param num_bufs num_src_bufs times the number of partitions
 * @param d_dst_buf_info dst_buf_info per partition produced in `compute_splits`
 * @param h_buf_sizes size in bytes of a partition (accessible from host)
 * @param num_partitions the number of partitions (1 meaning no splits)
 * @param user_buffer_size if non-zero, it is the size in bytes that 1MB batches should be
 *        grouped in, as different iterations.
 * @param stream Optional CUDA stream on which to execute kernels
 * @param temp_mr A memory resource for temporary and scratch space
 *
 * @returns new unique pointer to `chunk_iteration_state`
 */
std::unique_ptr<chunk_iteration_state> compute_batches(int num_bufs,
                                                       dst_buf_info* const d_dst_buf_info,
                                                       std::size_t const* const h_buf_sizes,
                                                       std::size_t num_partitions,
                                                       std::size_t user_buffer_size,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref temp_mr)
{
  // Since we parallelize at one block per copy, performance is vulnerable to situations where we
  // have small numbers of copies to do (a combination of small numbers of splits and/or columns),
  // so we will take the actual set of outgoing source/destination buffers and further partition
  // them into much smaller batches in order to drive up the number of blocks and overall
  // occupancy.
  rmm::device_uvector<cuda::std::pair<std::size_t, std::size_t>> batches(num_bufs, stream, temp_mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream, temp_mr),
    d_dst_buf_info,
    d_dst_buf_info + num_bufs,
    batches.begin(),
    cuda::proclaim_return_type<cuda::std::pair<std::size_t, std::size_t>>(
      [desired_batch_size = desired_batch_size] __device__(
        dst_buf_info const& buf) -> cuda::std::pair<std::size_t, std::size_t> {
        // Total bytes for this incoming partition
        std::size_t const bytes = buf.num_elements * static_cast<std::size_t>(buf.element_size);

        // This clause handles nested data types (e.g. list or string) that store no data in the row
        // columns, only in their children.
        if (bytes == 0) { return {1, 0}; }

        // The number of batches we want to subdivide this buffer into
        std::size_t const num_batches = cuda::std::max(
          std::size_t{1}, util::round_up_unsafe(bytes, desired_batch_size) / desired_batch_size);

        // NOTE: leaving batch size as a separate parameter for future tuning
        // possibilities, even though in the current implementation it will be a
        // constant.
        return {num_batches, desired_batch_size};
      }));

  return chunk_iteration_state::create(batches,
                                       num_bufs,
                                       d_dst_buf_info,
                                       h_buf_sizes,
                                       num_partitions,
                                       user_buffer_size,
                                       stream,
                                       temp_mr);
}

void copy_data(int num_batches_to_copy,
               int starting_batch,
               uint8_t const** d_src_bufs,
               uint8_t** d_dst_bufs,
               device_span<dst_buf_info> d_dst_buf_info,
               uint8_t* user_buffer,
               rmm::cuda_stream_view stream)
{
  constexpr size_type block_size = 256;
  if (user_buffer != nullptr) {
    auto index_to_buffer = [user_buffer] __device__(unsigned int) { return user_buffer; };
    copy_partitions<block_size><<<num_batches_to_copy, block_size, 0, stream.value()>>>(
      index_to_buffer, d_src_bufs, d_dst_buf_info.data() + starting_batch);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    auto index_to_buffer = [d_dst_bufs,
                            dst_buf_info = d_dst_buf_info.data(),
                            user_buffer] __device__(unsigned int buf_index) {
      auto const dst_buf_index = dst_buf_info[buf_index].dst_buf_index;
      return d_dst_bufs[dst_buf_index];
    };
    copy_partitions<block_size><<<num_batches_to_copy, block_size, 0, stream.value()>>>(
      index_to_buffer, d_src_bufs, d_dst_buf_info.data() + starting_batch);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

/**
 * @brief Function that checks an input table_view and splits for specific edge cases.
 *
 * It will return true if the input is "empty" (no rows or columns), which means
 * special handling has to happen in the calling code.
 *
 * @param input table_view of source table to be split
 * @param splits the splits specified by the user, or an empty vector if no splits
 * @returns true if the input is empty, false otherwise
 */
bool check_inputs(cudf::table_view const& input, std::vector<size_type> const& splits)
{
  auto const num_rows = input.num_rows();
  if (input.num_columns() == 0 && num_rows == 0) { return true; }
  if (splits.size() > 0) {
    CUDF_EXPECTS(
      splits.back() <= num_rows, "splits can't exceed size of input columns", std::out_of_range);
  }
  size_type begin = 0;
  for (auto end : splits) {
    CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.", std::out_of_range);
    CUDF_EXPECTS(
      end >= begin, "End index cannot be smaller than the starting index.", std::invalid_argument);
    CUDF_EXPECTS(end <= num_rows, "Slice range out of bounds.", std::out_of_range);
    begin = end;
  }
  return num_rows == 0 || input.num_columns() == 0;
}

};  // anonymous namespace

namespace detail {

/**
 * @brief A helper struct containing the state of contiguous_split, whether the caller
 * is using the single-pass contiguous_split or chunked_pack.
 *
 * It exposes an iterator-like pattern where contiguous_split_state::has_next()
 * returns true when there is work to be done, and false otherwise.
 *
 * contiguous_split_state::contiguous_split() performs a single-pass contiguous_split
 * and is valid iff contiguous_split_state is instantiated with 0 for the user_buffer_size.
 *
 * contiguous_split_state::contiguous_split_chunk(device_span) is only valid when
 * user_buffer_size > 0. It should be called as long as has_next() returns true. The
 * device_span passed to contiguous_split_chunk must be allocated in stream `stream` by
 * the user.
 *
 * None of the methods are thread safe.
 */
struct contiguous_split_state {
  contiguous_split_state(cudf::table_view const& input,
                         std::size_t user_buffer_size,
                         rmm::cuda_stream_view stream,
                         std::optional<rmm::device_async_resource_ref> mr,
                         rmm::device_async_resource_ref temp_mr)
    : contiguous_split_state(input, {}, user_buffer_size, stream, mr, temp_mr)
  {
  }

  contiguous_split_state(cudf::table_view const& input,
                         std::vector<size_type> const& splits,
                         rmm::cuda_stream_view stream,
                         std::optional<rmm::device_async_resource_ref> mr,
                         rmm::device_async_resource_ref temp_mr)
    : contiguous_split_state(input, splits, 0, stream, mr, temp_mr)
  {
  }

  bool has_next() const { return !is_empty && chunk_iter_state->has_more_copies(); }

  std::size_t get_total_contiguous_size() const
  {
    return is_empty ? 0 : chunk_iter_state->total_size;
  }

  std::vector<packed_table> contiguous_split()
  {
    CUDF_EXPECTS(user_buffer_size == 0, "Cannot contiguous split with a user buffer");
    if (is_empty || input.num_columns() == 0) { return make_packed_tables(); }

    auto const num_batches_total =
      std::get<1>(chunk_iter_state->get_current_starting_index_and_buff_count());

    // perform the copy.
    copy_data(num_batches_total,
              0 /* starting at buffer for single-shot 0*/,
              src_and_dst_pointers->d_src_bufs,
              src_and_dst_pointers->d_dst_bufs,
              chunk_iter_state->d_batched_dst_buf_info,
              nullptr,
              stream);

    // these "orig" dst_buf_info pointers describe the prior-to-batching destination
    // buffers per partition
    auto d_orig_dst_buf_info = partition_buf_size_and_dst_buf_info->d_dst_buf_info;
    auto h_orig_dst_buf_info = partition_buf_size_and_dst_buf_info->h_dst_buf_info;

    // postprocess valid_counts: apply the valid counts computed by copy_data for each
    // batch back to the original dst_buf_infos
    auto const keys = cudf::detail::make_counting_transform_iterator(
      0, out_to_in_index_function{chunk_iter_state->d_batch_offsets.begin(), (int)num_bufs});

    auto values = thrust::make_transform_iterator(
      chunk_iter_state->d_batched_dst_buf_info.begin(),
      cuda::proclaim_return_type<size_type>(
        [] __device__(dst_buf_info const& info) { return info.valid_count; }));

    thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                          keys,
                          keys + num_batches_total,
                          values,
                          cuda::make_discard_iterator(),
                          dst_valid_count_output_iterator{d_orig_dst_buf_info.data()});

    detail::cuda_memcpy<dst_buf_info>(h_orig_dst_buf_info, d_orig_dst_buf_info, stream);

    // not necessary for the non-chunked case, but it makes it so further calls to has_next
    // return false, just in case
    chunk_iter_state->advance_iteration();

    return make_packed_tables();
  }

  std::size_t contiguous_split_chunk(cudf::device_span<uint8_t> const& user_buffer)
  {
    CUDF_FUNC_RANGE();
    CUDF_EXPECTS(
      user_buffer.size() >= user_buffer_size,
      "Cannot use a device span smaller than the output buffer size configured at instantiation!");
    CUDF_EXPECTS(has_next(), "Cannot call contiguous_split_chunk with has_next() == false!");

    auto [starting_batch, num_batches_to_copy] =
      chunk_iter_state->get_current_starting_index_and_buff_count();

    // perform the copy.
    copy_data(num_batches_to_copy,
              starting_batch,
              src_and_dst_pointers->d_src_bufs,
              src_and_dst_pointers->d_dst_bufs,
              chunk_iter_state->d_batched_dst_buf_info,
              user_buffer.data(),
              stream);

    // We do not need to post-process null counts since the null count info is
    // taken from the source table in the contiguous_split_chunk case (no splits)
    return chunk_iter_state->advance_iteration();
  }

  std::unique_ptr<std::vector<uint8_t>> build_packed_column_metadata()
  {
    CUDF_EXPECTS(num_partitions == 1, "build_packed_column_metadata supported only without splits");

    if (input.num_columns() == 0) {
      // A truly empty (0, 0) table has no metadata.
      if (input.num_rows() == 0) { return std::unique_ptr<std::vector<uint8_t>>(); }
      // A zero-column, N-row has metadata-only output recording its row count.
      return std::make_unique<std::vector<uint8_t>>(cudf::pack_metadata(input, nullptr, 0));
    }

    if (is_empty) {
      // this is a bit ugly, but it was done to re-use make_empty_packed_table between the
      // regular contiguous_split and chunked_pack cases.
      auto empty_packed_tables = std::move(make_empty_packed_table().front());
      return std::move(empty_packed_tables.data.metadata);
    }

    auto& h_dst_buf_info  = partition_buf_size_and_dst_buf_info->h_dst_buf_info;
    auto cur_dst_buf_info = h_dst_buf_info.data();
    detail::metadata_builder mb{input.num_columns(), std::nullopt};

    populate_metadata(input.begin(), input.end(), cur_dst_buf_info, mb);

    return std::make_unique<std::vector<uint8_t>>(std::move(mb.build()));
  }

 private:
  contiguous_split_state(cudf::table_view const& input,
                         std::vector<size_type> const& splits,
                         std::size_t user_buffer_size,
                         rmm::cuda_stream_view stream,
                         std::optional<rmm::device_async_resource_ref> mr,
                         rmm::device_async_resource_ref temp_mr)
    : input(input),
      user_buffer_size(user_buffer_size),
      stream(stream),
      mr(mr),
      temp_mr(temp_mr),
      is_empty{check_inputs(input, splits)},
      num_partitions{splits.size() + 1}
  {
    // Per-partition row counts from the split boundaries (0, splits..., num_rows).
    // check_inputs has already validated that the splits are monotonic and in range.
    partition_row_counts.reserve(num_partitions);
    size_type begin = 0;
    for (auto const end : splits) {
      partition_row_counts.push_back(end - begin);
      begin = end;
    }
    partition_row_counts.push_back(input.num_rows() - begin);

    // if the table we are about to contig split is empty, we have special
    // handling where metadata is produced and a 0-byte contiguous buffer
    // is the result.
    if (is_empty) { return; }

    std::tie(num_src_bufs, num_bufs, partition_buf_size_and_dst_buf_info) =
      compute_num_bufs_and_splits(input, splits, stream, temp_mr);

    // Second pass: uses `dst_buf_info` to break down the work into 1MB batches.
    chunk_iter_state = compute_batches(num_bufs,
                                       partition_buf_size_and_dst_buf_info->d_dst_buf_info.data(),
                                       partition_buf_size_and_dst_buf_info->h_buf_sizes,
                                       num_partitions,
                                       user_buffer_size,
                                       stream,
                                       temp_mr);

    // allocate output partition buffers, in the non-chunked case
    if (user_buffer_size == 0) {
      out_buffers.reserve(num_partitions);
      auto h_buf_sizes = partition_buf_size_and_dst_buf_info->h_buf_sizes;
      std::transform(h_buf_sizes,
                     h_buf_sizes + num_partitions,
                     std::back_inserter(out_buffers),
                     [stream = stream, mr = mr.value_or(cudf::get_current_device_resource_ref())](
                       std::size_t bytes) { return rmm::device_buffer{bytes, stream, mr}; });
    }

    src_and_dst_pointers = std::move(setup_src_and_dst_pointers(
      input, num_partitions, num_src_bufs, out_buffers, stream, temp_mr));
  }

  std::vector<packed_table> make_packed_tables()
  {
    if (input.num_columns() == 0) {
      // A truly empty (0, 0) table produces no output.
      if (input.num_rows() == 0) { return std::vector<packed_table>(); }

      // A zero-column, N-row table contains no device data, so each partition is
      // represented as a metadata-only packed table that records its row count.
      std::vector<packed_table> result;
      result.reserve(num_partitions);
      std::transform(
        partition_row_counts.begin(),
        partition_row_counts.end(),
        std::back_inserter(result),
        [](size_type partition_rows) {
          auto partition = cudf::table_view{std::vector<column_view>{}, partition_rows};
          return packed_table{partition,
                              packed_columns{std::make_unique<std::vector<uint8_t>>(
                                               cudf::pack_metadata(partition, nullptr, 0)),
                                             std::make_unique<rmm::device_buffer>()}};
        });
      return result;
    }
    if (is_empty) { return make_empty_packed_table(); }
    std::vector<packed_table> result;
    result.reserve(num_partitions);
    std::vector<column_view> cols;
    cols.reserve(input.num_columns());

    auto& h_dst_buf_info = partition_buf_size_and_dst_buf_info->h_dst_buf_info;
    auto& h_dst_bufs     = src_and_dst_pointers->h_dst_bufs;

    auto cur_dst_buf_info = h_dst_buf_info.data();
    detail::metadata_builder mb(input.num_columns(), std::nullopt);

    for (std::size_t idx = 0; idx < num_partitions; idx++) {
      // traverse the buffers and build the columns.
      cur_dst_buf_info = build_output_columns(input.begin(),
                                              input.end(),
                                              cur_dst_buf_info,
                                              std::back_inserter(cols),
                                              h_dst_bufs[idx],
                                              mb);

      // pack the columns
      result.emplace_back(packed_table{
        cudf::table_view{cols},
        packed_columns{std::make_unique<std::vector<uint8_t>>(mb.build()),
                       std::make_unique<rmm::device_buffer>(std::move(out_buffers[idx]))}});

      cols.clear();
      mb.clear();
    }

    return result;
  }

  std::vector<packed_table> make_empty_packed_table()
  {
    // sanitize the inputs (to handle corner cases like sliced tables)
    std::vector<cudf::column_view> empty_column_views;
    empty_column_views.reserve(input.num_columns());
    std::transform(input.begin(),
                   input.end(),
                   std::back_inserter(empty_column_views),
                   [](column_view const& col) { return cudf::empty_like(col)->view(); });

    table_view empty_inputs(empty_column_views);

    // build the empty results
    std::vector<packed_table> result;
    result.reserve(num_partitions);
    auto const iter = cuda::counting_iterator<std::size_t>{0};
    std::transform(iter,
                   iter + num_partitions,
                   std::back_inserter(result),
                   [&empty_inputs](int partition_index) {
                     return packed_table{empty_inputs,
                                         packed_columns{std::make_unique<std::vector<uint8_t>>(
                                                          pack_metadata(empty_inputs, nullptr, 0)),
                                                        std::make_unique<rmm::device_buffer>()}};
                   });

    return result;
  }

  cudf::table_view const input;        ///< The input table_view to operate on
  std::size_t const user_buffer_size;  ///< The size of the user buffer for the chunked_pack case
  rmm::cuda_stream_view const stream;
  std::optional<rmm::device_async_resource_ref> mr;  ///< The resource for any data returned

  // this resource defaults to `mr` for the contiguous_split case, but it can be useful for the
  // `chunked_pack` case to allocate scratch/temp memory in a pool
  rmm::device_async_resource_ref const temp_mr;  ///< The memory resource for scratch/temp space

  // whether the table was empty to begin with (0 rows or 0 columns) and should be metadata-only
  bool const is_empty;  ///< True if the source table has 0 rows or 0 columns

  // This can be 1 if `contiguous_split` is just packing and not splitting
  std::size_t const num_partitions;  ///< The number of partitions to produce

  // Per-partition row counts derived from `splits` and `input.num_rows()`.
  std::vector<size_type> partition_row_counts;

  size_type num_src_bufs{};  ///< Number of source buffers including children

  std::size_t num_bufs{};  ///< Number of source buffers including children * number of splits

  std::unique_ptr<packed_partition_buf_size_and_dst_buf_info>
    partition_buf_size_and_dst_buf_info;  ///< Per-partition buffer size and destination buffer info

  std::unique_ptr<packed_src_and_dst_pointers>
    src_and_dst_pointers;  ///< Src. and dst. pointers for `copy_partition`

  //
  // State around the chunked pattern
  //

  // chunked_pack will have 1 or more "chunks" to iterate on, defined in chunk_iter_state
  // contiguous_split will have a single "chunk" in chunk_iter_state, so no iteration.
  std::unique_ptr<chunk_iteration_state>
    chunk_iter_state;  ///< State object for chunk iteration state

  // Two API usages are allowed:
  //  - `chunked_pack`: for this mode, the user will provide a buffer that must be at least 1MB.
  //    The behavior is "chunked" in that it will contiguously copy up until the user specified
  //    `user_buffer_size` limit, exposing a next() call for the user to invoke. Note that in this
  //    mode, no partitioning occurs, hence the name "pack".
  //
  //  - `contiguous_split` (default): when the user doesn't provide their own buffer,
  //    `contiguous_split` will allocate a buffer per partition and will place contiguous results in
  //    each buffer.
  //
  std::vector<rmm::device_buffer>
    out_buffers;  ///< Buffers allocated for a regular `contiguous_split`
};

std::vector<packed_table> contiguous_split(cudf::table_view const& input,
                                           std::vector<size_type> const& splits,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  // `temp_mr` is the same as `mr` for contiguous_split as it allocates all
  // of its memory from the default memory resource in cuDF
  auto temp_mr = mr;
  auto state   = contiguous_split_state(input, splits, stream, mr, temp_mr);
  return state.contiguous_split();
}

};  // namespace detail

std::vector<packed_table> contiguous_split(cudf::table_view const& input,
                                           std::vector<size_type> const& splits,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contiguous_split(input, splits, stream, mr);
}

chunked_pack::chunked_pack(cudf::table_view const& input,
                           std::size_t user_buffer_size,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref temp_mr)
{
  CUDF_EXPECTS(user_buffer_size >= desired_batch_size,
               "The output buffer size must be at least 1MB in size");
  // We pass `std::nullopt` for the first `mr` in `contiguous_split_state` to indicate
  // that it does not allocate any user-bound data for the `chunked_pack` case.
  state = std::make_unique<detail::contiguous_split_state>(
    input, user_buffer_size, stream, std::nullopt, temp_mr);
}

// required for the unique_ptr to work with a incomplete type (contiguous_split_state)
chunked_pack::~chunked_pack() = default;

std::size_t chunked_pack::get_total_contiguous_size() const
{
  return state->get_total_contiguous_size();
}

bool chunked_pack::has_next() const { return state->has_next(); }

std::size_t chunked_pack::next(cudf::device_span<uint8_t> const& user_buffer)
{
  return state->contiguous_split_chunk(user_buffer);
}

std::unique_ptr<std::vector<uint8_t>> chunked_pack::build_metadata() const
{
  return state->build_packed_column_metadata();
}

std::unique_ptr<chunked_pack> chunked_pack::create(cudf::table_view const& input,
                                                   std::size_t user_buffer_size,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref temp_mr)
{
  return std::make_unique<chunked_pack>(input, user_buffer_size, stream, temp_mr);
}

std::size_t packed_size(cudf::table_view const& input,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref temp_mr)
{
  // Handle empty table cases
  if (input.num_columns() == 0 || input.num_rows() == 0) { return 0; }

  auto result = compute_num_bufs_and_splits(input, {}, stream, temp_mr);
  auto const& partition_buf_size_and_dst_buf_info = std::get<2>(result);

  // Return the total size for the single partition
  return partition_buf_size_and_dst_buf_info->h_buf_sizes[0];
}

};  // namespace cudf
