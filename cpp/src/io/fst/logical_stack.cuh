/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf_test/print_utilities.cuh>

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace cudf::io::fst {

/**
 * @brief Describes the kind of stack operation.
 */
enum class stack_op_type : int8_t {
  READ  = 0,  ///< Operation reading what is currently on top of the stack
  PUSH  = 1,  ///< Operation pushing a new item on top of the stack
  POP   = 2,  ///< Operation popping the item currently on top of the stack
  RESET = 3   ///< Operation popping all items currently on the stack
};

/**
 * @brief Describes the kind of stack operations supported by the logical stack.
 */
enum class stack_op_support : bool {
  NO_RESET_SUPPORT   = false,  ///< A stack that only supports push(x) and pop() operations
  WITH_RESET_SUPPORT = true    ///< A stack that supports push(x), pop(), and reset() operations
};

namespace detail {

/**
 * @brief A convenience struct that represents a stack operation as a pair, where the stack_level
 * represents the stack's level and the value represents the stack symbol.
 *
 * @tparam StackLevelT The stack level type sufficient to cover all stack levels. Must be signed
 * type as any subsequence of stack operations must be able to be covered. E.g., consider the first
 * 10 operations are all push and the last 10 operations are all pop operations, we need to be able
 * to represent a partial aggregate of the first ten items, which is '+10', just as well as a
 * partial aggregate of the last ten items, which is '-10'.
 * @tparam ValueT The value type that corresponds to the stack symbols (i.e., covers the stack
 * alphabet).
 */
template <typename StackLevelT, typename ValueT>
struct StackOp {
  // Must be signed type as any subsequence of stack operations must be able to be covered.
  static_assert(std::is_signed_v<StackLevelT>, "StackLevelT has to be a signed type");

  StackLevelT stack_level;
  ValueT value;
};

/**
 * @brief Helper class to assist with radix sorting StackOp instances by stack level.
 *
 * @tparam BYTE_SIZE The size of the StackOp.
 */
template <std::size_t BYTE_SIZE>
struct StackOpToUnsigned {
  using UnsignedT = void;
};

template <>
struct StackOpToUnsigned<2U> {
  using UnsignedT = uint16_t;
};

template <>
struct StackOpToUnsigned<4U> {
  using UnsignedT = uint32_t;
};

template <>
struct StackOpToUnsigned<8U> {
  using UnsignedT = uint64_t;
};

/**
 * @brief Alias template to retrieve an unsigned bit-representation that can be used for radix
 * sorting the stack level of a StackOp.
 *
 * @tparam StackOpT The StackOp class template instance for which to get an unsigned
 * bit-representation
 */
template <typename StackOpT>
using UnsignedStackOpType = typename StackOpToUnsigned<sizeof(StackOpT)>::UnsignedT;

/**
 * @brief Function object class template used for converting a stack symbol to a stack
 * operation that has a stack level to which an operation applies.
 *
 * @tparam StackOpT
 * @tparam StackSymbolToStackOpTypeT
 */
template <typename StackOpT, typename StackSymbolToStackOpTypeT>
struct StackSymbolToStackOp {
  template <typename StackSymbolT>
  constexpr CUDF_HOST_DEVICE StackOpT operator()(StackSymbolT const& stack_symbol) const
  {
    stack_op_type stack_op = symbol_to_stack_op_type(stack_symbol);
    // PUSH => +1, POP => -1, READ => 0
    int32_t level_delta = (stack_op == stack_op_type::PUSH)  ? 1
                          : (stack_op == stack_op_type::POP) ? -1
                                                             : 0;
    return StackOpT{static_cast<decltype(StackOpT::stack_level)>(level_delta), stack_symbol};
  }

  /// Function object returning a stack operation type for a given stack symbol
  StackSymbolToStackOpTypeT symbol_to_stack_op_type;
};

/**
 * @brief Function object that maps a stack `reset` operation to `1`.
 */
template <typename StackSymbolToStackOpTypeT>
struct NewlineToResetStackSegmentOp {
  template <typename StackSymbolT>
  constexpr CUDF_HOST_DEVICE uint32_t operator()(StackSymbolT const& stack_symbol) const
  {
    stack_op_type stack_op = symbol_to_stack_op_type(stack_symbol);

    // Every reset operation marks the beginning of a new segment
    return (stack_op == stack_op_type::RESET) ? 1 : 0;
  }

  /// Function object returning a stack operation type for a given stack symbol
  StackSymbolToStackOpTypeT symbol_to_stack_op_type;
};

/**
 * @brief Function object that wraps around for values that exceed the largest value of `TargetT`
 */
template <typename TargetT>
struct ModToTargetTypeOpT {
  template <typename T>
  constexpr CUDF_HOST_DEVICE TargetT operator()(T const& val) const
  {
    return static_cast<TargetT>(
      val % (static_cast<T>(cuda::std::numeric_limits<TargetT>::max()) + static_cast<T>(1)));
  }
};

/**
 * @brief Binary reduction operator to compute the absolute stack level from relative stack levels
 * (i.e., +1 for a PUSH, -1 for a POP operation).
 */
template <typename StackSymbolToStackOpTypeT>
struct AddStackLevelFromStackOp {
  template <typename StackLevelT, typename ValueT>
  constexpr CUDF_HOST_DEVICE StackOp<StackLevelT, ValueT> operator()(
    StackOp<StackLevelT, ValueT> const& lhs, StackOp<StackLevelT, ValueT> const& rhs) const
  {
    StackLevelT new_level = lhs.stack_level + rhs.stack_level;
    return StackOp<StackLevelT, ValueT>{new_level, rhs.value};
  }

  /// Function object returning a stack operation type for a given stack symbol
  StackSymbolToStackOpTypeT symbol_to_stack_op_type;
};

/**
 * @brief Binary reduction operator that propagates a write operation for a specific stack level to
 * all reads of that same stack level. That is, if the stack level of LHS compares equal to the
 * stack level of the RHS and if the RHS is a read and the LHS is a write operation type, then we
 * return LHS, otherwise we return the RHS.
 */
template <typename StackSymbolToStackOpTypeT>
struct PopulatePopWithPush {
  template <typename StackLevelT, typename ValueT>
  constexpr CUDF_HOST_DEVICE StackOp<StackLevelT, ValueT> operator()(
    StackOp<StackLevelT, ValueT> const& lhs, StackOp<StackLevelT, ValueT> const& rhs) const
  {
    // If RHS is a read, then we need to figure out whether we can propagate the value from the LHS
    bool is_rhs_read = symbol_to_stack_op_type(rhs.value) != stack_op_type::PUSH;

    // Whether LHS is a matching write (i.e., the push operation that is on top of the stack for the
    // RHS's read)
    bool is_lhs_matching_write = (lhs.stack_level == rhs.stack_level) &&
                                 symbol_to_stack_op_type(lhs.value) == stack_op_type::PUSH;

    return (is_rhs_read && is_lhs_matching_write) ? lhs : rhs;
  }

  /// Function object returning a stack operation type for a given stack symbol
  StackSymbolToStackOpTypeT symbol_to_stack_op_type;
};

/**
 * @brief Binary reduction operator that is used to replace each read_symbol occurrence with the
 * last non-read_symbol that precedes such read_symbol.
 */
template <typename StackSymbolT>
struct PropagateLastWrite {
  constexpr CUDF_HOST_DEVICE StackSymbolT operator()(StackSymbolT const& lhs,
                                                     StackSymbolT const& rhs) const
  {
    // If RHS is a yet-to-be-propagated, then we need to check whether we can use the LHS to fill
    bool is_rhs_read = (rhs == read_symbol);

    // We propagate the write from the LHS if it's a write
    bool is_lhs_write = (lhs != read_symbol);

    return (is_rhs_read && is_lhs_write) ? lhs : rhs;
  }

  /// The read_symbol that is supposed to be replaced
  StackSymbolT read_symbol;
};

/**
 * @brief Helper function object class to convert a StackOp to the stack symbol of that
 * StackOp.
 */
struct StackOpToStackSymbol {
  template <typename StackLevelT, typename ValueT>
  constexpr CUDF_HOST_DEVICE ValueT operator()(StackOp<StackLevelT, ValueT> const& kv_op) const
  {
    return kv_op.value;
  }
};

/**
 * @brief Replaces all operations that apply to stack level '0' with the empty stack symbol
 */
template <typename StackOpT>
struct RemapEmptyStack {
  constexpr CUDF_HOST_DEVICE StackOpT operator()(StackOpT const& kv_op) const
  {
    return kv_op.stack_level == 0 ? empty_stack_symbol : kv_op;
  }
  StackOpT empty_stack_symbol;
};

}  // namespace detail

/**
 * @brief Takes a sparse representation of a sequence of stack operations that either push something
 * onto the stack or pop something from the stack and resolves the symbol that is on top of the
 * stack.
 *
 * @tparam SupportResetOperation Whether the logical stack also supports `reset` operations that
 * reset the stack to the empty stack
 * @tparam StackLevelT Signed integer type that must be sufficient to cover [-max_stack_level,
 * max_stack_level] for the given sequence of stack operations. Must be signed as it needs to cover
 * the stack level of any arbitrary subsequence of stack operations.
 * @tparam StackSymbolItT An input iterator type that provides the sequence of symbols that
 * represent stack operations
 * @tparam SymbolPositionT The index that this stack operation is supposed to apply to
 * @tparam StackSymbolToStackOpTypeT Function object class to transform items from StackSymbolItT to
 * stack_op_type
 * @tparam TopOfStackOutItT Output iterator type to which StackSymbolT are being assigned
 * @tparam StackSymbolT The internal type being used (usually corresponding to StackSymbolItT's
 * value_type)
 * @tparam OffsetT Signed or unsigned integer type large enough to index into both the sparse input
 * sequence and the top-of-stack output sequence
 *
 * @param[in] d_symbols Sequence of symbols that represent stack operations. Memory may alias with
 * \p d_top_of_stack
 * @param[in,out] d_symbol_positions Sequence of symbol positions (for a sparse representation),
 * sequence must be ordered in ascending order. Note, the memory of this array is repurposed for
 * double-buffering.
 * @param[in] symbol_to_stack_op Function object that returns a stack operation type (push, pop, or
 * read) for a given symbol from \p d_symbols
 * @param[out] d_top_of_stack A random access output iterator that will be populated with
 * what-is-on-top-of-the-stack for the given sequence of stack operations \p d_symbols
 * @param[in] empty_stack_symbol The symbol that will be written to top_of_stack whenever the stack
 * was empty
 * @param[in] read_symbol A symbol that may not be confused for a symbol that would push to the
 * stack
 * @param[in] num_symbols_out The number of symbols that are supposed to be filled with
 * what-is-on-top-of-the-stack
 * @param[in] stream The cuda stream to which to dispatch the work
 */
template <stack_op_support SupportResetOperation,
          typename StackLevelT,
          typename StackSymbolItT,
          typename SymbolPositionT,
          typename StackSymbolToStackOpTypeT,
          typename TopOfStackOutItT,
          typename StackSymbolT>
void sparse_stack_op_to_top_of_stack(StackSymbolItT d_symbols,
                                     device_span<SymbolPositionT> d_symbol_positions,
                                     StackSymbolToStackOpTypeT symbol_to_stack_op,
                                     TopOfStackOutItT d_top_of_stack,
                                     StackSymbolT const empty_stack_symbol,
                                     StackSymbolT const read_symbol,
                                     std::size_t const num_symbols_out,
                                     rmm::cuda_stream_view stream)
{
  rmm::device_buffer temp_storage{};

  // Type used to hold pairs of (stack_level, value) pairs
  using StackOpT = detail::StackOp<StackLevelT, StackSymbolT>;

  // Type used to mark *-by-key segments after `reset` operations
  using StackSegmentT = uint8_t;

  // The unsigned integer type that we use for radix sorting items of type StackOpT
  using StackOpUnsignedT = detail::UnsignedStackOpType<StackOpT>;
  static_assert(!std::is_void<StackOpUnsignedT>(), "unsupported StackOpT size");

  // Transforming sequence of stack symbols to stack operations
  using StackSymbolToStackOpT = detail::StackSymbolToStackOp<StackOpT, StackSymbolToStackOpTypeT>;

  // transform_iterator converting stack symbols to stack operations
  using TransformInputItT = thrust::transform_iterator<StackSymbolToStackOpT, StackSymbolItT>;

  constexpr bool supports_reset_op = SupportResetOperation == stack_op_support::WITH_RESET_SUPPORT;

  auto const num_symbols_in = d_symbol_positions.size();

  // Converting a stack symbol that may either push or pop to a stack operation:
  // stack_symbol -> ([+1,0,-1], stack_symbol)
  StackSymbolToStackOpT stack_sym_to_kv_op{symbol_to_stack_op};
  TransformInputItT stack_symbols_in(d_symbols, stack_sym_to_kv_op);

  // Double-buffer for sorting along the given sequence of symbol positions (the sparse
  // representation)
  cub::DoubleBuffer<SymbolPositionT> d_symbol_positions_db{nullptr, nullptr};

  // Double-buffer for sorting the stack operations by the stack level to which such operation
  // applies
  cub::DoubleBuffer<StackOpT> d_kv_operations{nullptr, nullptr};

  // A double-buffer that aliases memory from d_kv_operations with unsigned types in order to
  // be able to perform a radix sort
  cub::DoubleBuffer<StackOpUnsignedT> d_kv_operations_unsigned{nullptr, nullptr};

  constexpr std::size_t bits_per_byte = 8;
  constexpr std::size_t begin_bit     = offsetof(StackOpT, stack_level) * bits_per_byte;
  constexpr std::size_t end_bit       = begin_bit + (sizeof(StackOpT::stack_level) * bits_per_byte);

  // The stack operation that makes sure that reads for stack level '0' will be populated
  // with the empty_stack_symbol
  StackOpT const empty_stack{0, empty_stack_symbol};

  thrust::transform_iterator<detail::RemapEmptyStack<StackOpT>, StackOpT*> kv_ops_scan_in(
    nullptr, detail::RemapEmptyStack<StackOpT>{empty_stack});
  StackOpT* kv_ops_scan_out = nullptr;

  std::size_t stack_level_scan_bytes      = 0;
  std::size_t stack_level_sort_bytes      = 0;
  std::size_t match_level_scan_bytes      = 0;
  std::size_t propagate_writes_scan_bytes = 0;

  // Getting temporary storage requirements for the prefix sum of the stack level after each
  // operation
  if constexpr (supports_reset_op) {
    // Iterator that returns `1` for every symbol that corresponds to a `reset` operation
    auto reset_segments_it = thrust::make_transform_iterator(
      d_symbols,
      detail::NewlineToResetStackSegmentOp<StackSymbolToStackOpTypeT>{symbol_to_stack_op});

    auto const fake_key_segment_it      = static_cast<StackSegmentT*>(nullptr);
    std::size_t gen_segments_scan_bytes = 0;
    std::size_t scan_by_key_bytes       = 0;
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
      nullptr,
      gen_segments_scan_bytes,
      reset_segments_it,
      cuda::make_transform_output_iterator(fake_key_segment_it,
                                           detail::ModToTargetTypeOpT<StackSegmentT>{}),
      num_symbols_in,
      stream));
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveScanByKey(
      nullptr,
      scan_by_key_bytes,
      fake_key_segment_it,
      stack_symbols_in,
      d_kv_operations.Current(),
      detail::AddStackLevelFromStackOp<StackSymbolToStackOpTypeT>{symbol_to_stack_op},
      num_symbols_in,
      cuda::std::equal_to{},
      stream));
    stack_level_scan_bytes = std::max(gen_segments_scan_bytes, scan_by_key_bytes);
  } else {
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveScan(
      nullptr,
      stack_level_scan_bytes,
      stack_symbols_in,
      d_kv_operations.Current(),
      detail::AddStackLevelFromStackOp<StackSymbolToStackOpTypeT>{symbol_to_stack_op},
      num_symbols_in,
      stream));
  }

  // Getting temporary storage requirements for the stable radix sort (sorting by stack level of the
  // operations)
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                stack_level_sort_bytes,
                                                d_kv_operations_unsigned,
                                                d_symbol_positions_db,
                                                num_symbols_in,
                                                begin_bit,
                                                end_bit,
                                                stream));

  // Getting temporary storage requirements for the scan to match pop operations with the latest
  // push of the same level
  CUDF_CUDA_TRY(cub::DeviceScan::InclusiveScan(
    nullptr,
    match_level_scan_bytes,
    kv_ops_scan_in,
    kv_ops_scan_out,
    detail::PopulatePopWithPush<StackSymbolToStackOpTypeT>{symbol_to_stack_op},
    num_symbols_in,
    stream));

  // Getting temporary storage requirements for the scan to propagate top-of-stack for spots that
  // didn't push or pop
  CUDF_CUDA_TRY(
    cub::DeviceScan::ExclusiveScan(nullptr,
                                   propagate_writes_scan_bytes,
                                   d_top_of_stack,
                                   d_top_of_stack,
                                   detail::PropagateLastWrite<StackSymbolT>{read_symbol},
                                   empty_stack_symbol,
                                   num_symbols_out,
                                   stream));

  // Scratch memory required by the algorithms
  auto total_temp_storage_bytes = std::max({stack_level_scan_bytes,
                                            stack_level_sort_bytes,
                                            match_level_scan_bytes,
                                            propagate_writes_scan_bytes});

  if (temp_storage.size() < total_temp_storage_bytes) {
    temp_storage.resize(total_temp_storage_bytes, stream);
  }
  // Actual device buffer size, as we need to pass in an lvalue-ref to cub algorithms as
  // temp_storage_bytes
  total_temp_storage_bytes = temp_storage.size();

  rmm::device_uvector<SymbolPositionT> d_symbol_position_alt{num_symbols_in, stream};
  rmm::device_uvector<StackOpT> d_kv_ops_current{num_symbols_in, stream};
  rmm::device_uvector<StackOpT> d_kv_ops_alt{num_symbols_in, stream};

  //------------------------------------------------------------------------------
  // ALGORITHM
  //------------------------------------------------------------------------------
  // Initialize double-buffer for sorting the indexes of the sequence of sparse stack operations
  d_symbol_positions_db =
    cub::DoubleBuffer<SymbolPositionT>{d_symbol_positions.data(), d_symbol_position_alt.data()};

  // Initialize double-buffer for sorting the indexes of the sequence of sparse stack operations
  d_kv_operations = cub::DoubleBuffer<StackOpT>{d_kv_ops_current.data(), d_kv_ops_alt.data()};

  // Compute prefix sum of the stack level after each operation
  if constexpr (supports_reset_op) {
    // Iterator that returns `1` for every symbol that corresponds to a `reset` operation
    auto reset_segments_it = thrust::make_transform_iterator(
      d_symbols,
      detail::NewlineToResetStackSegmentOp<StackSymbolToStackOpTypeT>{symbol_to_stack_op});

    rmm::device_uvector<StackSegmentT> key_segments{num_symbols_in, stream};
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
      temp_storage.data(),
      total_temp_storage_bytes,
      reset_segments_it,
      cuda::make_transform_output_iterator(key_segments.data(),
                                           detail::ModToTargetTypeOpT<StackSegmentT>{}),
      num_symbols_in,
      stream));
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveScanByKey(
      temp_storage.data(),
      total_temp_storage_bytes,
      key_segments.data(),
      stack_symbols_in,
      d_kv_operations.Current(),
      detail::AddStackLevelFromStackOp<StackSymbolToStackOpTypeT>{symbol_to_stack_op},
      num_symbols_in,
      cuda::std::equal_to{},
      stream));
  } else {
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveScan(
      temp_storage.data(),
      total_temp_storage_bytes,
      stack_symbols_in,
      d_kv_operations.Current(),
      detail::AddStackLevelFromStackOp<StackSymbolToStackOpTypeT>{symbol_to_stack_op},
      num_symbols_in,
      stream));
  }

  // Check if the last element of d_kv_operations is 0. If not, then we have a problem.
  if (num_symbols_in && !supports_reset_op) {
    StackOpT last_symbol = d_kv_ops_current.element(num_symbols_in - 1, stream);
    CUDF_EXPECTS(last_symbol.stack_level == 0, "The logical stack is not empty!");
  }

  // Stable radix sort, sorting by stack level of the operations
  d_kv_operations_unsigned = cub::DoubleBuffer<StackOpUnsignedT>{
    reinterpret_cast<StackOpUnsignedT*>(d_kv_operations.Current()),
    reinterpret_cast<StackOpUnsignedT*>(d_kv_operations.Alternate())};
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(temp_storage.data(),
                                                total_temp_storage_bytes,
                                                d_kv_operations_unsigned,
                                                d_symbol_positions_db,
                                                num_symbols_in,
                                                begin_bit,
                                                end_bit,
                                                stream));

  // transform_iterator that remaps all operations on stack level 0 to the empty stack symbol
  kv_ops_scan_in  = {reinterpret_cast<StackOpT*>(d_kv_operations_unsigned.Current()),
                     detail::RemapEmptyStack<StackOpT>{empty_stack}};
  kv_ops_scan_out = reinterpret_cast<StackOpT*>(d_kv_operations_unsigned.Alternate());

  // Inclusive scan to match pop operations with the latest push operation of that level
  CUDF_CUDA_TRY(cub::DeviceScan::InclusiveScan(
    temp_storage.data(),
    total_temp_storage_bytes,
    kv_ops_scan_in,
    kv_ops_scan_out,
    detail::PopulatePopWithPush<StackSymbolToStackOpTypeT>{symbol_to_stack_op},
    num_symbols_in,
    stream));

  // Fill the output tape with read-symbol
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               thrust::device_ptr<StackSymbolT>{d_top_of_stack},
               thrust::device_ptr<StackSymbolT>{d_top_of_stack + num_symbols_out},
               read_symbol);

  // transform_iterator the stack operations to the stack symbol they represent
  thrust::transform_iterator<detail::StackOpToStackSymbol, StackOpT*> kv_op_to_stack_sym_it(
    kv_ops_scan_out, detail::StackOpToStackSymbol{});

  // Scatter the stack symbols to the output tape (spots that are not scattered to have been
  // pre-filled with the read-symbol)
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  kv_op_to_stack_sym_it,
                  kv_op_to_stack_sym_it + num_symbols_in,
                  d_symbol_positions_db.Current(),
                  d_top_of_stack);

  // We perform an exclusive scan in order to fill the items at the very left that may
  // be reading the empty stack before there's the first push occurrence in the sequence.
  // Also, we're interested in the top-of-the-stack symbol before the operation was applied.
  CUDF_CUDA_TRY(
    cub::DeviceScan::ExclusiveScan(temp_storage.data(),
                                   total_temp_storage_bytes,
                                   d_top_of_stack,
                                   d_top_of_stack,
                                   detail::PropagateLastWrite<StackSymbolT>{read_symbol},
                                   empty_stack_symbol,
                                   num_symbols_out,
                                   stream));
}

}  // namespace cudf::io::fst
