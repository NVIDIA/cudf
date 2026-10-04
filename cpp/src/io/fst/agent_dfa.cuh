/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "in_reg_array.cuh"

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/array>
#include <cuda/std/type_traits>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

namespace cudf::io::fst::detail {

/// Type used to enumerate (and index) into the states defined by a DFA
using StateIndexT = uint32_t;

/**
 * @brief Implements an associative composition operation for state transition vectors to be used
 * with a prefix scan.
 *
 * Read the following table as follows: c = op(l,r), where op is the composition operator.
 * For row 0: l maps 0 to 2. r maps 2 to 2. Hence, the result for 0 is 2.
 * For row 1: l maps 1 to 1. r maps 1 to 2. Hence, the result for 1 is 2.
 * For row 2: l maps 2 to 0. r maps 0 to 1. Hence, the result for 2 is 1.
 *
 *     l   r  = c  (     s->l->r)
 * 0: [2] [1]  [2] (i.e. 0->2->2)
 * 1: [1] [2]  [2] (i.e. 1->1->2)
 * 2: [0] [2]  [1] (i.e. 2->0->1)
 * @tparam NUM_ITEMS The number of items stored within a vector
 */
template <int32_t NUM_ITEMS>
struct VectorCompositeOp {
  template <typename VectorT>
  __device__ __forceinline__ VectorT operator()(VectorT const& lhs, VectorT const& rhs)
  {
    VectorT res{};
#pragma unroll
    for (int32_t i = 0; i < NUM_ITEMS; ++i) {
      res.Set(i, rhs.Get(lhs.Get(i)));
    }
    return res;
  }
};

/**
 * @brief A class whose ReadSymbol member function is invoked for each symbol being read from the
 * input tape. The wrapper class looks up whether a state transition caused by a symbol is supposed
 * to emit any output symbol (the "transduced" output) and, if so, keeps track of *how many* symbols
 * it intends to write out.
 */
template <typename TransducerTableT>
class DFACountCallbackWrapper {
 public:
  __device__ __forceinline__ DFACountCallbackWrapper(TransducerTableT transducer_table)
    : transducer_table(transducer_table)
  {
  }

  template <typename OffsetT>
  __device__ __forceinline__ void Init(OffsetT const&)
  {
    out_count = 0;
  }

  template <typename CharIndexT, typename StateIndexT, typename SymbolIndexT, typename SymbolT>
  __device__ __forceinline__ void ReadSymbol(CharIndexT const character_index,
                                             StateIndexT const old_state,
                                             StateIndexT const new_state,
                                             SymbolIndexT const symbol_id,
                                             SymbolT const read_symbol)
  {
    uint32_t const count = transducer_table(old_state, symbol_id, read_symbol);
    out_count += count;
  }

  __device__ __forceinline__ void TearDown() {}
  TransducerTableT const transducer_table;
  uint32_t out_count{};
};

/**
 * @brief A class whose ReadSymbol member function is invoked for each symbol being read from the
 * input tape. The wrapper class looks up whether a state transition caused by a symbol is supposed
 * to emit any output symbol (the "transduced" output) and, if so, writes out such symbols to the
 * given output iterators.
 *
 * @tparam MaxTranslatedOutChars The maximum number of symbols that are written on a any given state
 * transition
 * @tparam TransducerTableT The type implementing a transducer table that can be used for looking up
 * the symbols that are supposed to be emitted on a given state transition.
 * @tparam TransducedOutItT A random-access output iterator type to which symbols returned by the
 * transducer table are assignable.
 * @tparam TransducedIndexOutItT A random-access output iterator type to which indexes are written.
 */
template <int MaxTranslatedOutChars,
          typename TransducerTableT,
          typename TransducedOutItT,
          typename TransducedIndexOutItT>
class DFAWriteCallbackWrapper {
 public:
  __device__ __forceinline__ DFAWriteCallbackWrapper(TransducerTableT transducer_table,
                                                     TransducedOutItT out_it,
                                                     TransducedIndexOutItT out_idx_it,
                                                     uint32_t out_offset,
                                                     uint32_t /*tile_out_offset*/,
                                                     uint32_t /*tile_in_offset*/,
                                                     uint32_t /*tile_out_count*/)
    : transducer_table(transducer_table),
      out_it(out_it),
      out_idx_it(out_idx_it),
      out_offset(out_offset)
  {
  }

  template <typename OffsetT>
  __device__ __forceinline__ void Init(OffsetT const& in_offset)
  {
    this->in_offset = in_offset;
  }

  template <typename CharIndexT,
            typename StateIndexT,
            typename SymbolIndexT,
            typename SymbolT,
            int MaxTranslatedOutChars_>
  __device__ __forceinline__
    typename ::cuda::std::enable_if<(MaxTranslatedOutChars_ <= 2), void>::type
    ReadSymbol(CharIndexT const character_index,
               StateIndexT const old_state,
               StateIndexT const new_state,
               SymbolIndexT const symbol_id,
               SymbolT const read_symbol,
               cuda::std::integral_constant<int, MaxTranslatedOutChars_> /*MaxTranslatedOutChars*/)
  {
    uint32_t const count = transducer_table(old_state, symbol_id, read_symbol);

#pragma unroll
    for (uint32_t out_char = 0; out_char < MaxTranslatedOutChars_; out_char++) {
      if (out_char < count) {
        out_it[out_offset + out_char] =
          transducer_table(old_state, symbol_id, out_char, read_symbol);
        out_idx_it[out_offset + out_char] = in_offset + character_index;
      }
    }
    out_offset += count;
  }

  template <typename CharIndexT,
            typename StateIndexT,
            typename SymbolIndexT,
            typename SymbolT,
            int MaxTranslatedOutChars_>
  __device__ __forceinline__
    typename ::cuda::std::enable_if<(MaxTranslatedOutChars_ > 2), void>::type
    ReadSymbol(CharIndexT const character_index,
               StateIndexT const old_state,
               StateIndexT const new_state,
               SymbolIndexT const symbol_id,
               SymbolT const read_symbol,
               cuda::std::integral_constant<int, MaxTranslatedOutChars_>)
  {
    uint32_t const count = transducer_table(old_state, symbol_id, read_symbol);

    for (uint32_t out_char = 0; out_char < count; out_char++) {
      out_it[out_offset + out_char] = transducer_table(old_state, symbol_id, out_char, read_symbol);
      out_idx_it[out_offset + out_char] = in_offset + character_index;
    }
    out_offset += count;
  }

  template <typename CharIndexT, typename StateIndexT, typename SymbolIndexT, typename SymbolT>
  __device__ __forceinline__ void ReadSymbol(CharIndexT const character_index,
                                             StateIndexT const old_state,
                                             StateIndexT const new_state,
                                             SymbolIndexT const symbol_id,
                                             SymbolT const read_symbol)
  {
    ReadSymbol(character_index,
               old_state,
               new_state,
               symbol_id,
               read_symbol,
               cuda::std::integral_constant<int, MaxTranslatedOutChars>{});
  }

  __device__ __forceinline__ void TearDown() {}

 public:
  TransducerTableT const transducer_table;
  TransducedOutItT out_it;
  TransducedIndexOutItT out_idx_it;
  uint32_t out_offset;
  uint32_t in_offset;
};

/**
 * @brief A class whose ReadSymbol member function is invoked for each symbol being read from the
 * input tape. The wrapper class looks up whether a state transition caused by a symbol is supposed
 * to emit any output symbol (the "transduced" output) and, if so, writes out such symbols to the
 * given output iterators. This class uses a shared memory-backed write buffer to coalesce writes to
 * global memory.
 *
 * @tparam DiscardIndexOutput Whether to discard the indexes instead of writing them to the given
 * output iterator
 * @tparam DiscardTranslatedOutput Whether to discard the translated output symbols instead of
 * writing them to the given output iterator
 * @tparam NumWriteBufferItems The number of items to allocate in shared memory for the write
 * buffer.
 * @tparam OutputT The type of the translated items
 * @tparam TransducerTableT The type implementing a transducer table that can be used for looking up
 * the symbols that are supposed to be emitted on a given state transition.
 * @tparam TransducedOutItT A random-access output iterator type to which symbols returned by the
 * transducer table are assignable.
 * @tparam TransducedIndexOutItT A random-access output iterator type to which indexes are written.
 */
template <bool DiscardIndexOutput,
          bool DiscardTranslatedOutput,
          int NumWriteBufferItems,
          typename OutputT,
          typename TransducerTableT,
          typename TransducedOutItT,
          typename TransducedIndexOutItT>
class WriteCoalescingCallbackWrapper {
  struct TempStorage_Offsets {
    uint16_t compacted_offset[NumWriteBufferItems];
  };
  struct TempStorage_Symbols {
    OutputT compacted_symbols[NumWriteBufferItems];
  };
  using offset_cache_t =
    ::cuda::std::conditional_t<DiscardIndexOutput, cub::NullType, TempStorage_Offsets>;
  using symbol_cache_t = ::cuda::std::
    conditional_t<DiscardTranslatedOutput, cub::Uninitialized<cub::NullType>, TempStorage_Symbols>;
  struct TempStorage_ : offset_cache_t, symbol_cache_t {};

  __device__ __forceinline__ TempStorage_& PrivateStorage()
  {
    __shared__ TempStorage private_storage;
    return private_storage.Alias();
  }
  TempStorage_& temp_storage;

 public:
  struct TempStorage : cub::Uninitialized<TempStorage_> {};

  __device__ __forceinline__ WriteCoalescingCallbackWrapper(TransducerTableT transducer_table,
                                                            TransducedOutItT out_it,
                                                            TransducedIndexOutItT out_idx_it,
                                                            uint32_t thread_out_offset,
                                                            uint32_t tile_out_offset,
                                                            uint32_t tile_in_offset,
                                                            uint32_t tile_out_count)
    : temp_storage(PrivateStorage()),
      transducer_table(transducer_table),
      out_it(out_it),
      out_idx_it(out_idx_it),
      thread_out_offset(thread_out_offset),
      tile_out_offset(tile_out_offset),
      tile_in_offset(tile_in_offset),
      tile_out_count(tile_out_count)
  {
  }

  template <typename OffsetT>
  __device__ __forceinline__ void Init(OffsetT const& offset)
  {
    this->in_offset = offset;
  }

  template <typename CharIndexT, typename StateIndexT, typename SymbolIndexT, typename SymbolT>
  __device__ __forceinline__ void ReadSymbol(CharIndexT const character_index,
                                             StateIndexT const old_state,
                                             StateIndexT const new_state,
                                             SymbolIndexT const symbol_id,
                                             SymbolT const read_symbol)
  {
    uint32_t const count = transducer_table(old_state, symbol_id, read_symbol);
    for (uint32_t out_char = 0; out_char < count; out_char++) {
      if constexpr (!DiscardIndexOutput) {
        temp_storage.compacted_offset[thread_out_offset + out_char - tile_out_offset] =
          in_offset + character_index - tile_in_offset;
      }
      if constexpr (!DiscardTranslatedOutput) {
        temp_storage.compacted_symbols[thread_out_offset + out_char - tile_out_offset] =
          transducer_table(old_state, symbol_id, out_char, read_symbol);
      }
    }
    thread_out_offset += count;
  }

  __device__ __forceinline__ void TearDown()
  {
    __syncthreads();
    if constexpr (!DiscardTranslatedOutput) {
      for (thread_index_type out_char = threadIdx.x; out_char < tile_out_count;
           out_char += blockDim.x) {
        out_it[tile_out_offset + out_char] = temp_storage.compacted_symbols[out_char];
      }
    }
    if constexpr (!DiscardIndexOutput) {
      for (thread_index_type out_char = threadIdx.x; out_char < tile_out_count;
           out_char += blockDim.x) {
        out_idx_it[tile_out_offset + out_char] =
          temp_storage.compacted_offset[out_char] + tile_in_offset;
      }
    }
    __syncthreads();
  }

 public:
  TransducerTableT const transducer_table;
  TransducedOutItT out_it;
  TransducedIndexOutItT out_idx_it;
  uint32_t thread_out_offset;
  uint32_t tile_out_offset;
  uint32_t tile_in_offset;
  uint32_t in_offset;
  uint32_t tile_out_count;
};

/**
 * @brief Helper class that transitions the state of multiple DFA instances simultaneously whenever
 * a symbol is read.
 *
 * @tparam NUM_INSTANCES The number of DFA instances to keep track of
 * @tparam TransitionTableT The transition table type used for looking up the new state for a
 * current_state and a read_symbol.
 */
template <int32_t NUM_INSTANCES, typename TransitionTableT>
class StateVectorTransitionOp {
 public:
  __device__ __forceinline__
  StateVectorTransitionOp(TransitionTableT const& transition_table,
                          cuda::std::array<StateIndexT, NUM_INSTANCES>& state_vector)
    : transition_table(transition_table), state_vector(state_vector)
  {
  }

  template <typename CharIndexT, typename SymbolIndexT, typename SymbolT>
  __device__ __forceinline__ void ReadSymbol(CharIndexT const& character_index,
                                             SymbolIndexT const& read_symbol_id,
                                             SymbolT const& read_symbol) const
  {
#pragma unroll
    for (int32_t i = 0; i < NUM_INSTANCES; ++i) {
      state_vector[i] = transition_table(state_vector[i], read_symbol_id);
    }
  }

 public:
  cuda::std::array<StateIndexT, NUM_INSTANCES>& state_vector;
  TransitionTableT const& transition_table;
};

template <typename CallbackOpT, typename TransitionTableT>
struct StateTransitionOp {
  StateIndexT state;
  TransitionTableT const& transition_table;
  CallbackOpT& callback_op;

  __device__ __forceinline__ StateTransitionOp(TransitionTableT const& transition_table,
                                               StateIndexT state,
                                               CallbackOpT& callback_op)
    : transition_table(transition_table), state(state), callback_op(callback_op)
  {
  }

  template <typename CharIndexT, typename SymbolIndexT, typename SymbolT>
  __device__ __forceinline__ void ReadSymbol(CharIndexT const& character_index,
                                             SymbolIndexT const& read_symbol_id,
                                             SymbolT const& read_symbol)
  {
    // Remember what state we were in before we made the transition
    StateIndexT previous_state = state;

    state = transition_table(state, read_symbol_id);
    callback_op.ReadSymbol(character_index, previous_state, state, read_symbol_id, read_symbol);
  }
};

template <typename AgentDFAPolicy, typename SymbolItT, typename OffsetT>
struct AgentDFA {
  using SymbolIndexT = uint32_t;
  using AliasedLoadT = uint32_t;
  using CharT        = typename std::iterator_traits<SymbolItT>::value_type;

  //------------------------------------------------------------------------------
  // DERIVED CONFIGS
  //------------------------------------------------------------------------------
  static constexpr uint32_t BLOCK_THREADS    = AgentDFAPolicy::BLOCK_THREADS;
  static constexpr uint32_t ITEMS_PER_THREAD = AgentDFAPolicy::ITEMS_PER_THREAD;

  // The number of symbols per thread
  static constexpr uint32_t SYMBOLS_PER_THREAD = ITEMS_PER_THREAD;
  static constexpr uint32_t SYMBOLS_PER_BLOCK  = BLOCK_THREADS * SYMBOLS_PER_THREAD;

  static constexpr uint32_t MIN_UINTS_PER_BLOCK =
    cuda::ceil_div<uint32_t>(SYMBOLS_PER_BLOCK, sizeof(AliasedLoadT));
  static constexpr uint32_t UINTS_PER_THREAD = cuda::ceil_div(MIN_UINTS_PER_BLOCK, BLOCK_THREADS);
  static constexpr uint32_t UINTS_PER_BLOCK  = UINTS_PER_THREAD * BLOCK_THREADS;
  static constexpr uint32_t SYMBOLS_PER_UINT_BLOCK = UINTS_PER_BLOCK * sizeof(AliasedLoadT);

  //------------------------------------------------------------------------------
  // TYPEDEFS
  //------------------------------------------------------------------------------
  struct _TempStorage {
    // For aliased loading of characters into shared memory
    union {
      CharT chars[SYMBOLS_PER_BLOCK];
      AliasedLoadT uints[UINTS_PER_BLOCK];
    };
  };

  struct TempStorage : cub::Uninitialized<_TempStorage> {};

  //------------------------------------------------------------------------------
  // MEMBER VARIABLES
  //------------------------------------------------------------------------------
  _TempStorage& temp_storage;

  //------------------------------------------------------------------------------
  // CONSTRUCTOR
  //------------------------------------------------------------------------------
  __device__ __forceinline__ AgentDFA(TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias())
  {
  }

  template <int32_t NUM_SYMBOLS, typename SymbolMatcherT, typename CallbackOpT, bool IS_FULL_BLOCK>
  __device__ __forceinline__ static void ThreadParse(SymbolMatcherT const& symbol_matcher,
                                                     CharT const* chars,
                                                     SymbolIndexT const& max_num_chars,
                                                     CallbackOpT callback_op,
                                                     cuda::std::bool_constant<IS_FULL_BLOCK>)
  {
    // Iterate over symbols
#pragma unroll
    for (int32_t i = 0; i < NUM_SYMBOLS; ++i) {
      if (IS_FULL_BLOCK || threadIdx.x * SYMBOLS_PER_THREAD + i < max_num_chars) {
        auto matched_id = symbol_matcher(chars[i]);
        callback_op.ReadSymbol(i, matched_id, chars[i]);
      }
    }
  }

  template <int32_t NUM_SYMBOLS,
            typename SymbolMatcherT,
            typename StateTransitionOpT,
            bool IS_FULL_BLOCK>
  __device__ __forceinline__ void GetThreadStateTransitions(SymbolMatcherT const& symbol_matcher,
                                                            CharT const* chars,
                                                            SymbolIndexT const& max_num_chars,
                                                            StateTransitionOpT& state_transition_op,
                                                            cuda::std::bool_constant<IS_FULL_BLOCK>)
  {
    ThreadParse<NUM_SYMBOLS>(symbol_matcher,
                             chars,
                             max_num_chars,
                             state_transition_op,
                             cuda::std::bool_constant<IS_FULL_BLOCK>());
  }

  //---------------------------------------------------------------------
  // LOADING FULL BLOCK OF CHARACTERS, NON-ALIASED
  //---------------------------------------------------------------------
  template <typename CharInItT>
  __device__ __forceinline__ void LoadBlock(CharInItT d_chars,
                                            OffsetT const block_offset,
                                            OffsetT const num_total_symbols,
                                            cuda::std::true_type /*IS_FULL_BLOCK*/,
                                            cuda::std::integral_constant<int, 1> /*ALIGNMENT*/)
  {
    CharT thread_chars[SYMBOLS_PER_THREAD];

    CharInItT d_block_symbols = d_chars + block_offset;
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_block_symbols, thread_chars);

#pragma unroll
    for (int32_t i = 0; i < SYMBOLS_PER_THREAD; ++i) {
      temp_storage.chars[threadIdx.x + i * BLOCK_THREADS] = thread_chars[i];
    }
  }

  //---------------------------------------------------------------------
  // LOADING PARTIAL BLOCK OF CHARACTERS, NON-ALIASED
  //---------------------------------------------------------------------
  template <typename CharInItT>
  __device__ __forceinline__ void LoadBlock(CharInItT d_chars,
                                            OffsetT const block_offset,
                                            OffsetT const num_total_symbols,
                                            cuda::std::false_type /*IS_FULL_BLOCK*/,
                                            cuda::std::integral_constant<int, 1> /*ALIGNMENT*/)
  {
    CharT thread_chars[SYMBOLS_PER_THREAD];

    if (num_total_symbols <= block_offset) return;

    // Last unit to be loaded is IDIV_CEIL(#SYM, SYMBOLS_PER_UNIT)
    OffsetT num_total_chars = num_total_symbols - block_offset;

    CharInItT d_block_symbols = d_chars + block_offset;
    cub::LoadDirectStriped<BLOCK_THREADS>(
      threadIdx.x, d_block_symbols, thread_chars, num_total_chars);

#pragma unroll
    for (int32_t i = 0; i < SYMBOLS_PER_THREAD; ++i) {
      temp_storage.chars[threadIdx.x + i * BLOCK_THREADS] = thread_chars[i];
    }
  }

  //---------------------------------------------------------------------
  // LOADING FULL BLOCK OF CHARACTERS, ALIASED
  //---------------------------------------------------------------------
  __device__ __forceinline__ void LoadBlock(
    CharT const* d_chars,
    OffsetT const block_offset,
    OffsetT const num_total_symbols,
    cuda::std::true_type /*IS_FULL_BLOCK*/,
    cuda::std::integral_constant<int, sizeof(AliasedLoadT)> /*ALIGNMENT*/)
  {
    AliasedLoadT thread_units[UINTS_PER_THREAD];

    AliasedLoadT const* d_block_symbols =
      reinterpret_cast<AliasedLoadT const*>(d_chars + block_offset);
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_block_symbols, thread_units);

#pragma unroll
    for (int32_t i = 0; i < UINTS_PER_THREAD; ++i) {
      temp_storage.uints[threadIdx.x + i * BLOCK_THREADS] = thread_units[i];
    }
  }

  //---------------------------------------------------------------------
  // LOADING PARTIAL BLOCK OF CHARACTERS, ALIASED
  //---------------------------------------------------------------------
  __device__ __forceinline__ void LoadBlock(
    CharT const* d_chars,
    OffsetT const block_offset,
    OffsetT const num_total_symbols,
    cuda::std::false_type /*IS_FULL_BLOCK*/,
    cuda::std::integral_constant<int, sizeof(AliasedLoadT)> /*ALIGNMENT*/)
  {
    AliasedLoadT thread_units[UINTS_PER_THREAD];

    if (num_total_symbols <= block_offset) return;

    // Last unit to be loaded is IDIV_CEIL(#SYM, SYMBOLS_PER_UNIT)
    OffsetT num_total_units =
      cuda::ceil_div(num_total_symbols - block_offset, sizeof(AliasedLoadT));

    AliasedLoadT const* d_block_symbols =
      reinterpret_cast<AliasedLoadT const*>(d_chars + block_offset);
    cub::LoadDirectStriped<BLOCK_THREADS>(
      threadIdx.x, d_block_symbols, thread_units, num_total_units);

#pragma unroll
    for (int32_t i = 0; i < UINTS_PER_THREAD; ++i) {
      temp_storage.uints[threadIdx.x + i * BLOCK_THREADS] = thread_units[i];
    }
  }

  //---------------------------------------------------------------------
  // LOADING BLOCK OF CHARACTERS: DISPATCHER
  //---------------------------------------------------------------------
  __device__ __forceinline__ void LoadBlock(CharT const* d_chars,
                                            OffsetT const block_offset,
                                            OffsetT const num_total_symbols)
  {
    // Check if pointer is aligned to four bytes
    if (((uintptr_t)(void const*)(d_chars + block_offset) % 4) == 0) {
      if (block_offset + SYMBOLS_PER_UINT_BLOCK < num_total_symbols) {
        LoadBlock(d_chars,
                  block_offset,
                  num_total_symbols,
                  cuda::std::true_type(),
                  cuda::std::integral_constant<int, 4>());
      } else {
        LoadBlock(d_chars,
                  block_offset,
                  num_total_symbols,
                  cuda::std::false_type(),
                  cuda::std::integral_constant<int, 1>());
      }
    } else {
      if (block_offset + SYMBOLS_PER_UINT_BLOCK < num_total_symbols) {
        LoadBlock(d_chars,
                  block_offset,
                  num_total_symbols,
                  cuda::std::true_type(),
                  cuda::std::integral_constant<int, 1>());
      } else {
        LoadBlock(d_chars,
                  block_offset,
                  num_total_symbols,
                  cuda::std::false_type(),
                  cuda::std::integral_constant<int, 1>());
      }
    }
  }

  template <typename CharInItT>
  __device__ __forceinline__ void LoadBlock(CharInItT d_chars,
                                            OffsetT const block_offset,
                                            OffsetT const num_total_symbols)
  {
    // Check if we are loading a full tile of data
    if (block_offset + SYMBOLS_PER_UINT_BLOCK < num_total_symbols) {
      LoadBlock(d_chars,
                block_offset,
                num_total_symbols,
                cuda::std::true_type(),
                cuda::std::integral_constant<int, 1>());
    } else {
      LoadBlock(d_chars,
                block_offset,
                num_total_symbols,
                cuda::std::false_type(),
                cuda::std::integral_constant<int, 1>());
    }
  }

  template <int32_t NUM_STATES, typename SymbolMatcherT, typename TransitionTableT>
  __device__ __forceinline__ void GetThreadStateTransitionVector(
    SymbolMatcherT const& symbol_matcher,
    TransitionTableT const& transition_table,
    SymbolItT d_chars,
    OffsetT const block_offset,
    OffsetT const num_total_symbols,
    cuda::std::array<StateIndexT, NUM_STATES>& state_vector)
  {
    using StateVectorTransitionOpT = StateVectorTransitionOp<NUM_STATES, TransitionTableT>;

    // Start parsing and to transition states
    StateVectorTransitionOpT transition_op(transition_table, state_vector);

    // Load characters into shared memory
    LoadBlock(d_chars, block_offset, num_total_symbols);

    // If this is a full block (i.e., all threads can parse all their symbols)
    OffsetT num_block_chars = num_total_symbols - block_offset;
    bool is_full_block      = (num_block_chars >= SYMBOLS_PER_BLOCK);

    // Ensure characters have been loaded
    __syncthreads();

    // Thread's symbols
    CharT const* t_chars = &temp_storage.chars[threadIdx.x * SYMBOLS_PER_THREAD];

    // Parse thread's symbols and transition the state-vector
    if (is_full_block) {
      GetThreadStateTransitions<SYMBOLS_PER_THREAD>(
        symbol_matcher, t_chars, num_block_chars, transition_op, cuda::std::true_type());
    } else {
      GetThreadStateTransitions<SYMBOLS_PER_THREAD>(
        symbol_matcher, t_chars, num_block_chars, transition_op, cuda::std::false_type());
    }
  }

  template <bool BYPASS_LOAD,
            typename SymbolMatcherT,
            typename TransitionTableT,
            typename CallbackOpT>
  __device__ __forceinline__ void GetThreadStateTransitions(
    SymbolMatcherT const& symbol_matcher,
    TransitionTableT const& transition_table,
    SymbolItT d_chars,
    OffsetT const block_offset,
    OffsetT const num_total_symbols,
    StateIndexT& state,
    CallbackOpT& callback_op,
    cuda::std::bool_constant<BYPASS_LOAD>)
  {
    using StateTransitionOpT = StateTransitionOp<CallbackOpT, TransitionTableT>;

    // Start parsing and to transition states
    StateTransitionOpT transition_op(transition_table, state, callback_op);

    // Load characters into shared memory
    if (!BYPASS_LOAD) LoadBlock(d_chars, block_offset, num_total_symbols);

    // If this is a full block (i.e., all threads can parse all their symbols)
    OffsetT num_block_chars = num_total_symbols - block_offset;
    bool is_full_block      = (num_block_chars >= SYMBOLS_PER_BLOCK);

    // Ensure characters have been loaded
    __syncthreads();

    // Thread's symbols
    CharT* t_chars = &temp_storage.chars[threadIdx.x * SYMBOLS_PER_THREAD];

    // Initialize callback
    callback_op.Init(block_offset + threadIdx.x * SYMBOLS_PER_THREAD);

    // Parse thread's symbols and transition the state-vector
    if (is_full_block) {
      GetThreadStateTransitions<SYMBOLS_PER_THREAD>(
        symbol_matcher, t_chars, num_block_chars, transition_op, cuda::std::true_type());
    } else {
      GetThreadStateTransitions<SYMBOLS_PER_THREAD>(
        symbol_matcher, t_chars, num_block_chars, transition_op, cuda::std::false_type());
    }

    callback_op.TearDown();
  }
};

template <bool IS_TRANS_VECTOR_PASS,
          bool IS_SINGLE_PASS,
          typename DfaT,
          typename TileStateT,
          typename AgentDFAPolicy,
          typename SymbolItT,
          typename OffsetT,
          typename StateVectorT,
          typename OutOffsetScanTileState,
          typename TransducedOutItT,
          typename TransducedIndexOutItT,
          typename TransducedCountOutItT>
__launch_bounds__(int32_t(AgentDFAPolicy::BLOCK_THREADS)) CUDF_KERNEL
  void SimulateDFAKernel(DfaT dfa,
                         SymbolItT d_chars,
                         OffsetT const num_chars,
                         StateIndexT seed_state,
                         StateVectorT* __restrict__ d_thread_state_transition,
                         TileStateT tile_state,
                         OutOffsetScanTileState offset_tile_state,
                         TransducedOutItT transduced_out_it,
                         TransducedIndexOutItT transduced_out_idx_it,
                         TransducedCountOutItT d_num_transduced_out_it)
{
  using AgentDfaSimT = AgentDFA<AgentDFAPolicy, SymbolItT, OffsetT>;

  static constexpr int32_t NUM_STATES = DfaT::MAX_NUM_STATES;

  constexpr uint32_t BLOCK_THREADS     = AgentDFAPolicy::BLOCK_THREADS;
  constexpr uint32_t SYMBOLS_PER_BLOCK = AgentDfaSimT::SYMBOLS_PER_BLOCK;

  // Shared memory required by the DFA simulation algorithm
  __shared__ typename AgentDfaSimT::TempStorage dfa_storage;

  // Shared memory required by the symbol group lookup table
  __shared__ typename DfaT::SymbolGroupStorageT symbol_matcher_storage;

  // Shared memory required by the transition table
  __shared__ typename DfaT::TransitionTableStorageT transition_table_storage;

  // Shared memory required by the transducer table
  __shared__ typename DfaT::TranslationTableStorageT transducer_table_storage;

  // Initialize symbol group lookup table
  auto symbol_matcher = dfa.InitSymbolGroupLUT(symbol_matcher_storage);

  // Initialize transition table
  auto transition_table = dfa.InitTransitionTable(transition_table_storage);

  // Initialize transition table
  auto transducer_table = dfa.InitTranslationTable(transducer_table_storage);

  // Set up DFA
  AgentDfaSimT agent_dfa(dfa_storage);

  // The state transition vector passed on to the second stage of the algorithm
  StateVectorT out_state_vector;

  using OutSymbolT = typename DfaT::OutSymbolT;
  // static constexpr int32_t MIN_TRANSLATED_OUT = DfaT::MIN_TRANSLATED_OUT;
  static constexpr int32_t num_max_translated_out = DfaT::MAX_TRANSLATED_OUT;
  static constexpr bool discard_out_index =
    ::cuda::std::is_same<TransducedIndexOutItT, cuda::discard_iterator>::value;
  static constexpr bool discard_out_it =
    ::cuda::std::is_same<TransducedOutItT, cuda::discard_iterator>::value;
  using NonWriteCoalescingT =
    DFAWriteCallbackWrapper<num_max_translated_out,
                            decltype(dfa.InitTranslationTable(transducer_table_storage)),
                            TransducedOutItT,
                            TransducedIndexOutItT>;

  using WriteCoalescingT =
    WriteCoalescingCallbackWrapper<discard_out_index,
                                   discard_out_it,
                                   num_max_translated_out * SYMBOLS_PER_BLOCK,
                                   OutSymbolT,
                                   decltype(dfa.InitTranslationTable(transducer_table_storage)),
                                   TransducedOutItT,
                                   TransducedIndexOutItT>;

  static constexpr bool is_translation_pass = (!IS_TRANS_VECTOR_PASS) || IS_SINGLE_PASS;

  // Use write-coalescing only if the worst-case output size per tile fits into shared memory
  static constexpr bool can_use_smem_cache =
    (sizeof(typename WriteCoalescingT::TempStorage) + sizeof(typename AgentDfaSimT::TempStorage) +
     sizeof(typename DfaT::SymbolGroupStorageT) + sizeof(typename DfaT::TransitionTableStorageT) +
     sizeof(typename DfaT::TranslationTableStorageT)) < (48 * 1024);
  static constexpr bool use_smem_cache =
    is_translation_pass and
    (sizeof(typename WriteCoalescingT::TempStorage) <= AgentDFAPolicy::SMEM_THRESHOLD) and
    can_use_smem_cache;

  using DFASimulationCallbackWrapperT =
    cuda::std::conditional_t<use_smem_cache, WriteCoalescingT, NonWriteCoalescingT>;

  // Stage 1: Compute the state-transition vector
  if (IS_TRANS_VECTOR_PASS || IS_SINGLE_PASS) {
    // Keeping track of the state for each of the <NUM_STATES> state machines
    cuda::std::array<StateIndexT, NUM_STATES> state_vector;

    // Initialize the seed state transition vector with the identity vector
    thrust::sequence(thrust::seq, cuda::std::begin(state_vector), cuda::std::end(state_vector));

    // Compute the state transition vector
    agent_dfa.GetThreadStateTransitionVector<NUM_STATES>(symbol_matcher,
                                                         transition_table,
                                                         d_chars,
                                                         blockIdx.x * SYMBOLS_PER_BLOCK,
                                                         num_chars,
                                                         state_vector);

    // Initialize the state transition vector passed on to the second stage
#pragma unroll
    for (int32_t i = 0; i < NUM_STATES; ++i) {
      out_state_vector.Set(i, state_vector[i]);
    }

    // Write out state-transition vector
    if (!IS_SINGLE_PASS) {
      d_thread_state_transition[blockIdx.x * BLOCK_THREADS + threadIdx.x] = out_state_vector;
    }
  }

  // Stage 2: Perform FSM simulation
  if ((!IS_TRANS_VECTOR_PASS) || IS_SINGLE_PASS) {
    StateIndexT state = 0;

    //------------------------------------------------------------------------------
    // SINGLE-PASS:
    // -> block-wide inclusive prefix scan on the state transition vector
    // -> first block/tile: write out block aggregate as the "tile's" inclusive (i.e., the one that
    // incorporates all preceding blocks/tiles results)
    //------------------------------------------------------------------------------
    if constexpr (IS_SINGLE_PASS) {
      uint32_t tile_idx             = blockIdx.x;
      using StateVectorCompositeOpT = VectorCompositeOp<NUM_STATES>;

      using PrefixCallbackOpT_ =
        cub::TilePrefixCallbackOp<StateVectorT, StateVectorCompositeOpT, TileStateT>;

      using ItemsBlockScan =
        cub::BlockScan<StateVectorT, BLOCK_THREADS, cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>;

      __shared__ typename ItemsBlockScan::TempStorage scan_temp_storage;
      __shared__ typename PrefixCallbackOpT_::TempStorage prefix_callback_temp_storage;

      // STATE-TRANSITION IDENTITY VECTOR
      StateVectorT state_identity_vector;
      for (int32_t i = 0; i < NUM_STATES; ++i) {
        state_identity_vector.Set(i, i);
      }
      StateVectorCompositeOpT state_vector_scan_op;

      //
      if (tile_idx == 0) {
        StateVectorT block_aggregate;
        ItemsBlockScan(scan_temp_storage)
          .ExclusiveScan(out_state_vector,
                         out_state_vector,
                         state_identity_vector,
                         state_vector_scan_op,
                         block_aggregate);

        if (threadIdx.x == 0 /*and not IS_LAST_TILE*/) {
          tile_state.SetInclusive(0, block_aggregate);
        }
      } else {
        auto prefix_op = PrefixCallbackOpT_(
          tile_state, prefix_callback_temp_storage, state_vector_scan_op, tile_idx);

        ItemsBlockScan(scan_temp_storage)
          .ExclusiveScan(out_state_vector, out_state_vector, state_vector_scan_op, prefix_op);
      }
      __syncthreads();
      state = out_state_vector.Get(seed_state);
    } else {
      state = d_thread_state_transition[blockIdx.x * BLOCK_THREADS + threadIdx.x].Get(seed_state);
    }

    // Perform finite-state machine simulation, computing size of transduced output
    DFACountCallbackWrapper count_chars_callback_op{transducer_table};

    StateIndexT t_start_state = state;
    agent_dfa.GetThreadStateTransitions(symbol_matcher,
                                        transition_table,
                                        d_chars,
                                        blockIdx.x * SYMBOLS_PER_BLOCK,
                                        num_chars,
                                        state,
                                        count_chars_callback_op,
                                        cuda::std::bool_constant<IS_SINGLE_PASS>());

    __syncthreads();

    using OffsetPrefixScanCallbackOpT_ =
      cub::TilePrefixCallbackOp<OffsetT, cuda::std::plus<>, OutOffsetScanTileState>;

    using OutOffsetBlockScan =
      cub::BlockScan<OffsetT, BLOCK_THREADS, cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>;

    __shared__ typename OutOffsetBlockScan::TempStorage scan_temp_storage;
    __shared__ typename OffsetPrefixScanCallbackOpT_::TempStorage prefix_callback_temp_storage;

    uint32_t tile_idx = blockIdx.x;
    uint32_t tile_out_offset{};
    uint32_t tile_out_count{};
    uint32_t thread_out_offset{};
    if (tile_idx == 0) {
      OffsetT block_aggregate = 0;
      OutOffsetBlockScan(scan_temp_storage)
        .ExclusiveScan(count_chars_callback_op.out_count,
                       thread_out_offset,
                       static_cast<OffsetT>(0),
                       cuda::std::plus{},
                       block_aggregate);
      tile_out_count = block_aggregate;
      if (threadIdx.x == 0 /*and not IS_LAST_TILE*/) {
        offset_tile_state.SetInclusive(0, block_aggregate);
      }

      if (tile_idx == gridDim.x - 1 && threadIdx.x == 0) {
        *d_num_transduced_out_it = block_aggregate;
      }
    } else {
      auto prefix_op = OffsetPrefixScanCallbackOpT_(
        offset_tile_state, prefix_callback_temp_storage, cuda::std::plus{}, tile_idx);

      OutOffsetBlockScan(scan_temp_storage)
        .ExclusiveScan(
          count_chars_callback_op.out_count, thread_out_offset, cuda::std::plus{}, prefix_op);
      tile_out_offset = prefix_op.GetExclusivePrefix();
      tile_out_count  = prefix_op.GetBlockAggregate();
      if (tile_idx == gridDim.x - 1 && threadIdx.x == 0) {
        *d_num_transduced_out_it = prefix_op.GetInclusivePrefix();
      }
    }

    DFASimulationCallbackWrapperT write_translated_callback_op{transducer_table,
                                                               transduced_out_it,
                                                               transduced_out_idx_it,
                                                               thread_out_offset,
                                                               tile_out_offset,
                                                               blockIdx.x * SYMBOLS_PER_BLOCK,
                                                               tile_out_count};
    agent_dfa.GetThreadStateTransitions(symbol_matcher,
                                        transition_table,
                                        d_chars,
                                        blockIdx.x * SYMBOLS_PER_BLOCK,
                                        num_chars,
                                        t_start_state,
                                        write_translated_callback_op,
                                        cuda::std::true_type());
  }
}

}  // namespace cudf::io::fst::detail
