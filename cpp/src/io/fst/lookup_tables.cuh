/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/fst/device_dfa.cuh"
#include "io/utilities/hostdevice_vector.hpp"

#include <cudf/types.hpp>

#include <cub/cub.cuh>
#include <cuda/std/iterator>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

namespace cudf::io::fst::detail {

/**
 * @brief Helper function object that delegates a lookup to a given lookup table without mapping any
 * of the given arguments.
 */
struct IdentityOp {
  template <typename LookUpTableT, typename... Args>
  __host__ __device__ __forceinline__ auto operator()(LookUpTableT const& lookup_table,
                                                      Args&&... args) const
  {
    return lookup_table.lookup(std::forward<Args>(args)...);
  }
};

/**
 * @brief Class template that can be plugged into the finite-state machine to look up the symbol
 * group index for a given symbol. Class template does not support multi-symbol lookups (i.e., no
 * look-ahead). The class uses shared memory for the lookups.
 *
 * @tparam SymbolT The symbol type being passed in to lookup the corresponding symbol group id
 * @tparam PreMapOpT A function object that is invoked with `(lut, symbol)` and must return the
 * symbol group index of `symbol`.  `lut` is an instance of the lookup table and `symbol` is the
 * symbol for which to get the symbol group index. If no particular mapping is needed, an instance
 * of `IdentityOp` can be used.
 */
template <typename SymbolT, typename PreMapOpT>
class SingleSymbolSmemLUT {
 private:
  // Type used for representing a symbol group id (i.e., what we return for a given symbol)
  using SymbolGroupIdT = uint8_t;

  // Number of entries for every lookup (e.g., for 8-bit Symbol this is 256)
  static constexpr uint32_t NUM_ENTRIES_PER_LUT = 0x01U << (sizeof(SymbolT) * 8U);

  struct _TempStorage {
    // sym_to_sgid[symbol] -> symbol group index
    SymbolGroupIdT sym_to_sgid[NUM_ENTRIES_PER_LUT];
  };

 public:
  using TempStorage = cub::Uninitialized<_TempStorage>;

  struct KernelParameter {
    using LookupTableT = SingleSymbolSmemLUT<SymbolT, PreMapOpT>;

    // sym_to_sgid[min(symbol,num_valid_entries)] -> symbol group index
    uint32_t num_valid_entries;

    // sym_to_sgid[symbol] -> symbol group index
    SymbolGroupIdT sym_to_sgid[NUM_ENTRIES_PER_LUT];

    // Function object that transforms a symbol to a symbol group id
    PreMapOpT pre_map_op;
  };

  /**
   * @brief Initializes the given \p sgid_init with the symbol group lookups defined by \p
   * symbol_strings.
   *
   * @param symbol_strings Array of strings, where the i-th string holds all symbols
   * (characters!) that correspond to the i-th symbol group index
   * @param pre_map_op Function object that transforms a symbol to a symbol group id
   * @return
   */
  template <typename SymbolGroupItT>
  static KernelParameter InitDeviceSymbolGroupIdLut(SymbolGroupItT const& symbol_strings,
                                                    PreMapOpT pre_map_op)
  {
    KernelParameter init_data{};

    // The symbol group index to be returned if none of the given symbols match
    SymbolGroupIdT no_match_id = symbol_strings.size();

    // The symbol with the largest value that is mapped to a symbol group id
    SymbolGroupIdT max_lookup_index = 0;

    // Initialize all entries: by default we return the no-match-id
    std::fill(&init_data.sym_to_sgid[0], &init_data.sym_to_sgid[NUM_ENTRIES_PER_LUT], no_match_id);

    // Set up lookup table
    uint32_t sg_id = 0;
    // Iterate over the symbol groups
    for (auto const& sg_symbols : symbol_strings) {
      // Iterate over all symbols that belong to the current symbol group
      for (auto const& sg_symbol : sg_symbols) {
        max_lookup_index = std::max(max_lookup_index, static_cast<SymbolGroupIdT>(sg_symbol));
        init_data.sym_to_sgid[static_cast<int32_t>(sg_symbol)] = sg_id;
      }
      sg_id++;
    }

    // Initialize the out-of-bounds lookup: sym_to_sgid[max_lookup_index+1] -> no_match_id
    auto const oob_match_index             = max_lookup_index + 1;
    init_data.sym_to_sgid[oob_match_index] = no_match_id;

    // The number of valid entries in the table (including the entry for the out-of-bounds symbol
    // group id)
    init_data.num_valid_entries = oob_match_index + 1;
    init_data.pre_map_op        = pre_map_op;

    return init_data;
  }

  _TempStorage& temp_storage;
  SymbolGroupIdT num_valid_entries;
  PreMapOpT pre_map_op;

  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  /**
   * @brief Initializes the lookup table, primarily to be invoked from within device code but also
   * provides host-side implementation for verification.
   * @note Synchronizes the thread block, if called from device, and, hence, requires all threads
   * of the thread block to call the constructor
   */
  constexpr CUDF_HOST_DEVICE SingleSymbolSmemLUT(KernelParameter const& kernel_param,
                                                 TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias()), num_valid_entries(kernel_param.num_valid_entries)
  {
    // GPU-side init
#if CUB_PTX_ARCH > 0
    for (int32_t i = threadIdx.x; i < kernel_param.num_valid_entries; i += blockDim.x) {
      this->temp_storage.sym_to_sgid[i] = kernel_param.sym_to_sgid[i];
    }
    __syncthreads();

#else
    // CPU-side init
    std::copy_n(kernel_param.sym_to_sgid, kernel_param.num_luts, this->temp_storage.sym_to_sgid);
#endif
  }

  template <typename SymbolT_>
  constexpr CUDF_HOST_DEVICE int32_t operator()(SymbolT_ const symbol) const
  {
    // Look up the symbol group for given symbol
    return pre_map_op(*this, symbol);
  }

  constexpr CUDF_HOST_DEVICE int32_t lookup(SymbolT const symbol) const
  {
    // Look up the symbol group for given symbol
    return temp_storage
      .sym_to_sgid[min(static_cast<SymbolGroupIdT>(symbol), num_valid_entries - 1U)];
  }
};

/**
 * @brief A simple symbol group lookup wrapper that uses a simple function object to
 * retrieve the symbol group id for a symbol.
 *
 * @tparam SymbolGroupLookupOpT The function object type to return the symbol group for a given
 * symbol
 */
template <typename SymbolGroupLookupOpT>
class SymbolGroupLookupOp {
 private:
  struct _TempStorage {};

 public:
  using TempStorage = cub::Uninitialized<_TempStorage>;

  struct KernelParameter {
    // Declare the member type that the DFA is going to instantiate
    using LookupTableT = SymbolGroupLookupOp<SymbolGroupLookupOpT>;
    SymbolGroupLookupOpT sgid_lookup_op;
  };

  static KernelParameter InitDeviceSymbolGroupIdLut(SymbolGroupLookupOpT sgid_lookup_op)
  {
    return KernelParameter{sgid_lookup_op};
  }

 private:
  _TempStorage& temp_storage;
  SymbolGroupLookupOpT sgid_lookup_op;

  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

 public:
  CUDF_HOST_DEVICE SymbolGroupLookupOp(KernelParameter const& kernel_param,
                                       TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias()), sgid_lookup_op(kernel_param.sgid_lookup_op)
  {
  }

  template <typename SymbolT_>
  constexpr CUDF_HOST_DEVICE int32_t operator()(SymbolT_ const symbol) const
  {
    // Look up the symbol group for given symbol
    return sgid_lookup_op(symbol);
  }
};

/**
 * @brief Prepares a simple symbol group lookup wrapper that uses a simple function object to
 * retrieve the symbol group id for a symbol.
 *
 * @tparam FunctorT A function object type that must implement the signature `int32_t
 * operator()(symbol)`, where `symbol` is a symbol from the input type.
 * @param sgid_lookup_op A function object that must implement the signature `int32_t
 * operator()(symbol)`, where `symbol` is a symbol from the input type.
 * @return The kernel parameter of type SymbolGroupLookupOp::KernelParameter that is used to
 * initialize a simple symbol group id lookup wrapper
 */
template <typename FunctorT>
auto make_symbol_group_lookup_op(FunctorT sgid_lookup_op)
{
  return SymbolGroupLookupOp<FunctorT>::InitDeviceSymbolGroupIdLut(sgid_lookup_op);
}

/**
 * @brief Creates a symbol group lookup table of type `SingleSymbolSmemLUT` that uses a two-staged
 * lookup approach. @p pre_map_op is a function object invoked with `(lut, symbol)` that must return
 * the symbol group id for the given `symbol`. `lut` is an instance of the lookup table
 * and `symbol` is a symbol from the input tape. Usually, @p pre_map_op first maps a symbol from
 * the input tape to an integral that is convertible to `symbol_t`. In a second stage, @p pre_map_op
 * uses `lut`'s `lookup(mapped_symbol)` that maps that integral to the symbol group id.
 *
 * @tparam symbol_t Must be an integral type
 * @tparam NUM_SYMBOL_GROUPS The number of symbol groups, excluding the catchall symbol group (aka
 * "other" symbol group)
 * @tparam pre_map_op_t A unary function object type that returns the symbol group id
 * @param symbol_strings An array of vectors, where all the symbols in the i-th vector are mapped to
 * the i-th symbol group
 * @param pre_map_op A unary function object type that returns the symbol group id for a symbol
 * @return A symbol group lookup table
 */
template <typename symbol_t, std::size_t NUM_SYMBOL_GROUPS, typename pre_map_op_t>
auto make_symbol_group_lut(
  std::array<std::vector<symbol_t>, NUM_SYMBOL_GROUPS> const& symbol_strings,
  pre_map_op_t pre_map_op)
{
  using lookup_table_t = SingleSymbolSmemLUT<symbol_t, pre_map_op_t>;
  return lookup_table_t::InitDeviceSymbolGroupIdLut(symbol_strings, pre_map_op);
}

/**
 * @brief Creates a symbol group lookup table of type `SingleSymbolSmemLUT` that uses a two-staged
 * lookup approach. @p pre_map_op is a function object invoked with `(lut, symbol)` that must return
 * the symbol group id for the given `symbol`. `lut` is an instance of the lookup table
 * and `symbol` is a symbol from the input tape. Usually, @p pre_map_op first maps a symbol from
 * the input tape to an integral that is convertible to `symbol_t`. In a second stage, @p pre_map_op
 * uses `lut`'s `lookup(mapped_symbol)` that maps that integral to the symbol group id.
 *
 * @tparam symbol_t The type returned by @p pre_map_op must be assignable to `char`
 * @tparam NUM_SYMBOL_GROUPS The number of symbol groups, excluding the catchall symbol group (aka
 * "other" symbol group)
 * @tparam pre_map_op_t A unary function object type that returns the symbol group id for a symbol
 * @param symbol_strings An array of strings, where all the characters in the i-th string are mapped
 * to the i-th symbol group
 * @param pre_map_op A unary function object type that returns the symbol group id for a symbol
 * @return A symbol group lookup table
 */
template <std::size_t NUM_SYMBOL_GROUPS, typename pre_map_op_t>
auto make_symbol_group_lut(std::array<std::string, NUM_SYMBOL_GROUPS> const& symbol_strings,
                           pre_map_op_t pre_map_op)
{
  using symbol_t       = char;
  using lookup_table_t = SingleSymbolSmemLUT<symbol_t, pre_map_op_t>;
  return lookup_table_t::InitDeviceSymbolGroupIdLut(symbol_strings, pre_map_op);
}

/**
 * @brief Creates a symbol group lookup table that maps a symbol to a symbol group id, requiring the
 * symbol type from the input tape to be assignable to `symbol_t` and `symbol_t` to be of integral
 * type.
 *
 * @tparam symbol_t The input tape's symbol type must be assignable to this type
 * @tparam NUM_SYMBOL_GROUPS The number of symbol groups, excluding the catchall symbol group (aka
 * "other" symbol group)
 * @param symbol_strings An array of vectors, where all the symbols in the i-th vector are mapped to
 * the i-th symbol group
 * @return A symbol group lookup table
 */
template <typename symbol_t, std::size_t NUM_SYMBOL_GROUPS>
auto make_symbol_group_lut(
  std::array<std::vector<symbol_t>, NUM_SYMBOL_GROUPS> const& symbol_strings)
{
  return make_symbol_group_lut(symbol_strings, IdentityOp{});
}

/**
 * @brief Creates a symbol group lookup table that maps a symbol to a symbol group id, requiring the
 * symbol type from the input tape to be assignable to `symbol_t` and `symbol_t` to be of integral
 * type.
 *
 * @tparam symbol_t The input tape's symbol type must be assignable to this type
 * @tparam NUM_SYMBOL_GROUPS The number of symbol groups, excluding the catchall symbol group (aka
 * "other" symbol group)
 * @param symbol_strings An array of strings, where all the characters in the i-th string are mapped
 * to the i-th symbol group
 * @return A symbol group lookup table
 */
template <std::size_t NUM_SYMBOL_GROUPS>
auto make_symbol_group_lut(std::array<std::string, NUM_SYMBOL_GROUPS> const& symbol_strings)
{
  return make_symbol_group_lut(symbol_strings, IdentityOp{});
}

/**
 * @brief Lookup table mapping (old_state, symbol_group_id) transitions to a new target state. The
 * class uses shared memory for the lookups.
 *
 * @tparam MAX_NUM_SYMBOLS The maximum number of symbols being output by a single state transition
 * @tparam MAX_NUM_STATES The maximum number of states that this lookup table shall support
 */
template <int32_t MAX_NUM_SYMBOLS, int32_t MAX_NUM_STATES>
class TransitionTable {
 private:
  // Type used
  using ItemT = char;

  struct _TempStorage {
    ItemT transitions[MAX_NUM_STATES * MAX_NUM_SYMBOLS];
  };

 public:
  static constexpr int32_t NUM_STATES = MAX_NUM_STATES;
  using TempStorage                   = cub::Uninitialized<_TempStorage>;

  struct KernelParameter {
    using LookupTableT = TransitionTable<MAX_NUM_SYMBOLS, MAX_NUM_STATES>;

    ItemT transitions[MAX_NUM_STATES * MAX_NUM_SYMBOLS];
  };

  template <typename StateIdT>
  static KernelParameter InitDeviceTransitionTable(
    std::array<std::array<StateIdT, MAX_NUM_SYMBOLS>, MAX_NUM_STATES> const& transition_table)
  {
    KernelParameter init_data{};
    // transition_table[state][symbol] -> new state
    for (std::size_t state = 0; state < transition_table.size(); ++state) {
      for (std::size_t symbol = 0; symbol < transition_table[state].size(); ++symbol) {
        CUDF_EXPECTS(
          static_cast<int64_t>(transition_table[state][symbol]) <=
            std::numeric_limits<ItemT>::max(),
          "Target state index value exceeds value representable by the transition table's type");
        init_data.transitions[symbol * MAX_NUM_STATES + state] =
          static_cast<ItemT>(transition_table[state][symbol]);
      }
    }

    return init_data;
  }

  constexpr CUDF_HOST_DEVICE TransitionTable(KernelParameter const& kernel_param,
                                             TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias())
  {
#if CUB_PTX_ARCH > 0
    for (int i = threadIdx.x; i < MAX_NUM_STATES * MAX_NUM_SYMBOLS; i += blockDim.x) {
      this->temp_storage.transitions[i] = kernel_param.transitions[i];
    }
    __syncthreads();
#else
    std::copy_n(
      kernel_param.transitions, MAX_NUM_STATES * MAX_NUM_SYMBOLS, this->temp_storage.transitions);
#endif
  }

  /**
   * @brief Returns a random-access iterator to lookup all the state transitions for one specific
   * symbol from an arbitrary old_state, i.e., it[old_state] -> new_state.
   *
   * @param state_id The DFA's current state index from which we'll transition
   * @param match_id The symbol group id of the symbol that we just read in
   * @return
   */
  template <typename StateIndexT, typename SymbolIndexT>
  constexpr CUDF_HOST_DEVICE int32_t operator()(StateIndexT const state_id,
                                                SymbolIndexT const match_id) const
  {
    return temp_storage.transitions[match_id * MAX_NUM_STATES + state_id];
  }

 private:
  _TempStorage& temp_storage;

  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;

    return private_storage;
  }
};

/**
 * @brief Creates a transition table of type `TransitionTable` that maps `(state_id, match_id)`
 * pairs to the new target state for the given `(state_id, match_id)`-combination.
 *
 * @tparam StateIdT An integral type used to represent state indexes
 * @tparam MAX_NUM_SYMBOLS The maximum number of symbols being output by a single state transition
 * @tparam MAX_NUM_STATES The maximum number of states that this lookup table shall support
 * @param transition_table The transition table
 * @return A transition table of type `TransitionTable`
 */
template <typename StateIdT, std::size_t MAX_NUM_SYMBOLS, std::size_t MAX_NUM_STATES>
auto make_transition_table(
  std::array<std::array<StateIdT, MAX_NUM_SYMBOLS>, MAX_NUM_STATES> const& transition_table)
{
  using transition_table_t = TransitionTable<MAX_NUM_SYMBOLS, MAX_NUM_STATES>;
  return transition_table_t::InitDeviceTransitionTable(transition_table);
}

/**
 * @brief Compile-time reflection to check if `OpT` type has the `TempStorage` and
 * `KernelParameter` type members.
 */
template <typename OpT, typename = void>
struct is_complex_op : std::false_type {};

template <typename OpT>
struct is_complex_op<OpT, std::void_t<typename OpT::TempStorage, typename OpT::KernelParameter>>
  : std::true_type {};

/**
 * @brief The device view that is passed to the finite-state transducer algorithm. Each of the
 * lookup tables can either be a simple function object that defines the `operator()` required for
 * respective lookup table or a complex class.
 *
 * @tparam SymbolGroupIdLookupT
 * @tparam TransitionTableT
 * @tparam TranslationTableT
 * @tparam NUM_STATES
 */
template <typename SymbolGroupIdLookupT,
          typename TransitionTableT,
          typename TranslationTableT,
          int32_t NUM_STATES>
class dfa_device_view {
 private:
  // Complex symbol group lookup operators need to declare a `TempStorage` and `KernelParameter`
  // type member that is passed during device-side initialization.
  using sgid_lut_init_t = std::conditional_t<is_complex_op<SymbolGroupIdLookupT>::value,
                                             typename SymbolGroupIdLookupT::KernelParameter,
                                             SymbolGroupIdLookupT>;

  // Complex transition table lookup operators need to declare a `TempStorage` and
  // `KernelParameter` type member that is passed during device-side initialization.
  using transition_table_init_t = std::conditional_t<is_complex_op<TransitionTableT>::value,
                                                     typename TransitionTableT::KernelParameter,
                                                     TransitionTableT>;

  // Complex translation table lookup operators need to declare a `TempStorage` and
  // `KernelParameter` type member that is passed during device-side initialization.
  using translation_table_init_t = std::conditional_t<is_complex_op<TranslationTableT>::value,
                                                      typename TranslationTableT::KernelParameter,
                                                      TranslationTableT>;

 public:
  // The maximum number of states supported by this DFA instance
  // This is a value queried by the DFA simulation algorithm
  static constexpr int32_t MAX_NUM_STATES = NUM_STATES;

  using OutSymbolT                            = typename TranslationTableT::OutSymbolT;
  static constexpr int32_t MIN_TRANSLATED_OUT = TranslationTableT::MIN_TRANSLATED_OUT;
  static constexpr int32_t MAX_TRANSLATED_OUT = TranslationTableT::MAX_TRANSLATED_OUT;

  using SymbolGroupStorageT      = std::conditional_t<is_complex_op<SymbolGroupIdLookupT>::value,
                                                      typename SymbolGroupIdLookupT::TempStorage,
                                                      typename cub::NullType>;
  using TransitionTableStorageT  = std::conditional_t<is_complex_op<TransitionTableT>::value,
                                                      typename TransitionTableT::TempStorage,
                                                      typename cub::NullType>;
  using TranslationTableStorageT = std::conditional_t<is_complex_op<TranslationTableT>::value,
                                                      typename TranslationTableT::TempStorage,
                                                      typename cub::NullType>;

  __device__ auto InitSymbolGroupLUT(SymbolGroupStorageT& temp_storage)
  {
    return SymbolGroupIdLookupT(*d_sgid_lut_init, temp_storage);
  }

  __device__ auto InitTransitionTable(TransitionTableStorageT& temp_storage)
  {
    return TransitionTableT(*d_transition_table_init, temp_storage);
  }

  __device__ auto InitTranslationTable(TranslationTableStorageT& temp_storage)
  {
    return TranslationTableT(*d_translation_table_init, temp_storage);
  }

  dfa_device_view(sgid_lut_init_t const* d_sgid_lut_init,
                  transition_table_init_t const* d_transition_table_init,
                  translation_table_init_t const* d_translation_table_init)
    : d_sgid_lut_init(d_sgid_lut_init),
      d_transition_table_init(d_transition_table_init),
      d_translation_table_init(d_translation_table_init)
  {
  }

 private:
  sgid_lut_init_t const* d_sgid_lut_init;
  transition_table_init_t const* d_transition_table_init;
  translation_table_init_t const* d_translation_table_init;
};

/**
 * @brief Lookup table mapping (old_state, symbol_group_id) transitions to a sequence of symbols
 * that the finite-state transducer is supposed to output for each transition. The class uses
 * shared memory for the lookups.
 *
 * @tparam OutSymbolT The symbol type being output
 * @tparam OutSymbolOffsetT Type sufficiently large to index into the lookup table of output
 * symbols
 * @tparam MAX_NUM_SYMBOLS The maximum number of symbol groups supported by this lookup table
 * @tparam MAX_NUM_STATES The maximum number of states that this lookup table shall support
 * @tparam MIN_TRANSLATED_OUT_ The minimum number of symbols being output by a single state
 * transition
 * @tparam MAX_TRANSLATED_OUT_ The maximum number of symbols being output by a single state
 * transition
 * @tparam MAX_TABLE_SIZE The maximum number of items in the lookup table of output symbols
 */
template <typename OutSymbolT_,
          typename OutSymbolOffsetT,
          int32_t MAX_NUM_SYMBOLS,
          int32_t MAX_NUM_STATES,
          int32_t MIN_TRANSLATED_OUT_,
          int32_t MAX_TRANSLATED_OUT_,
          int32_t MAX_TABLE_SIZE = (MAX_NUM_SYMBOLS * MAX_NUM_STATES)>
class TransducerLookupTable {
 private:
  struct _TempStorage {
    OutSymbolOffsetT out_offset[MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1];
    OutSymbolT_ out_symbols[MAX_TABLE_SIZE];
  };

 public:
  using OutSymbolT                            = OutSymbolT_;
  static constexpr int32_t MIN_TRANSLATED_OUT = MIN_TRANSLATED_OUT_;
  static constexpr int32_t MAX_TRANSLATED_OUT = MAX_TRANSLATED_OUT_;

  using TempStorage = cub::Uninitialized<_TempStorage>;

  struct KernelParameter {
    using LookupTableT = TransducerLookupTable<OutSymbolT,
                                               OutSymbolOffsetT,
                                               MAX_NUM_SYMBOLS,
                                               MAX_NUM_STATES,
                                               MIN_TRANSLATED_OUT,
                                               MAX_TRANSLATED_OUT,
                                               MAX_TABLE_SIZE>;

    OutSymbolOffsetT d_out_offsets[MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1];
    OutSymbolT d_out_symbols[MAX_TABLE_SIZE];
  };

  /**
   * @brief Initializes the lookup table, primarily to be invoked from within device code but also
   * provides host-side implementation for verification.
   * @note Synchronizes the thread block, if called from device, and, hence, requires all threads
   * of the thread block to call the constructor
   */
  static KernelParameter InitDeviceTranslationTable(
    std::array<std::array<std::vector<OutSymbolT>, MAX_NUM_SYMBOLS>, MAX_NUM_STATES> const&
      translation_table)
  {
    KernelParameter init_data;
    std::vector<OutSymbolT> out_symbols;
    out_symbols.reserve(MAX_TABLE_SIZE);
    std::vector<OutSymbolOffsetT> out_symbol_offsets;
    out_symbol_offsets.reserve(MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1);
    out_symbol_offsets.push_back(0);

    // Iterate over the states in the transition table
    for (auto const& state_trans : translation_table) {
      uint32_t num_added = 0;
      // Iterate over the symbols in the transition table
      for (auto const& symbol_out : state_trans) {
        // Insert the output symbols for this specific (state, symbol) transition
        out_symbols.insert(std::end(out_symbols), std::begin(symbol_out), std::end(symbol_out));
        out_symbol_offsets.push_back(out_symbols.size());
        num_added++;
      }

      // Copy the last offset for all symbols (to guarantee a proper lookup for omitted symbols of
      // this state)
      if (MAX_NUM_SYMBOLS > num_added) {
        int32_t count = MAX_NUM_SYMBOLS - num_added;
        auto begin_it = std::prev(std::end(out_symbol_offsets));
        std::fill_n(begin_it, count, out_symbol_offsets[0]);
      }
    }

    // Check whether runtime-provided table size exceeds the compile-time given max. table size
    CUDF_EXPECTS(out_symbols.size() <= MAX_TABLE_SIZE, "Unsupported translation table");

    // Prepare host-side data to be copied and passed to the device
    std::copy(
      std::cbegin(out_symbol_offsets), std::cend(out_symbol_offsets), init_data.d_out_offsets);
    std::copy(std::cbegin(out_symbols), std::cend(out_symbols), init_data.d_out_symbols);

    return init_data;
  }

 private:
  _TempStorage& temp_storage;

  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

 public:
  /**
   * @brief Initializes the lookup table, primarily to be invoked from within device code but also
   * provides host-side implementation for verification.
   * @note Synchronizes the thread block, if called from device, and, hence, requires all threads
   * of the thread block to call the constructor
   */
  CUDF_HOST_DEVICE TransducerLookupTable(KernelParameter const& kernel_param,
                                         TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias())
  {
    constexpr uint32_t num_offsets = MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1;
#if CUB_PTX_ARCH > 0
    for (int i = threadIdx.x; i < num_offsets; i += blockDim.x) {
      this->temp_storage.out_offset[i] = kernel_param.d_out_offsets[i];
    }
    // Make sure all threads in the block can read out_symbol_offsets[num_offsets - 1] from shared
    // memory
    __syncthreads();
    for (int i = threadIdx.x; i < this->temp_storage.out_offset[num_offsets - 1]; i += blockDim.x) {
      this->temp_storage.out_symbols[i] = kernel_param.d_out_symbols[i];
    }
    __syncthreads();
#else
    std::copy_n(kernel_param.d_out_offsets, num_offsets, this->temp_storage.out_symbol_offsets);
    std::copy_n(kernel_param.d_out_symbols,
                this->temp_storage.out_symbol_offsets,
                this->temp_storage.out_symbols);
#endif
  }

  template <typename StateIndexT, typename SymbolIndexT, typename RelativeOffsetT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE auto operator()(StateIndexT const state_id,
                                             SymbolIndexT const match_id,
                                             RelativeOffsetT const relative_offset,
                                             SymbolT const /*read_symbol*/) const
  {
    auto offset = temp_storage.out_offset[state_id * MAX_NUM_SYMBOLS + match_id] + relative_offset;
    return temp_storage.out_symbols[offset];
  }

  template <typename StateIndexT, typename SymbolIndexT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE OutSymbolOffsetT operator()(StateIndexT const state_id,
                                                         SymbolIndexT const match_id,
                                                         SymbolT const /*read_symbol*/) const
  {
    return temp_storage.out_offset[state_id * MAX_NUM_SYMBOLS + match_id + 1] -
           temp_storage.out_offset[state_id * MAX_NUM_SYMBOLS + match_id];
  }
};

/**
 * @brief Creates a translation table that maps (old_state, symbol_group_id) transitions to a
 * sequence of symbols that the finite-state transducer is supposed to output for each transition.
 *
 * @tparam MAX_TABLE_SIZE The maximum number of items in the lookup table of output symbols
 * @tparam MIN_TRANSLATED_OUT The minimum number of symbols being output by a single state
 * transition
 * @tparam MAX_TRANSLATED_OUT The maximum number of symbols being output by a single state
 * transition
 * @tparam OutSymbolT The symbol type being output
 * @tparam MAX_NUM_SYMBOLS The maximum number of symbol groups supported by this lookup table
 * @tparam MAX_NUM_STATES The maximum number of states that this lookup table shall support
 * @param translation_table The translation table
 * @return A translation table of type `TransducerLookupTable`.
 */
template <std::size_t MAX_TABLE_SIZE,
          std::size_t MIN_TRANSLATED_OUT,
          std::size_t MAX_TRANSLATED_OUT,
          typename OutSymbolT,
          std::size_t MAX_NUM_SYMBOLS,
          std::size_t MAX_NUM_STATES>
auto make_translation_table(std::array<std::array<std::vector<OutSymbolT>, MAX_NUM_SYMBOLS>,
                                       MAX_NUM_STATES> const& translation_table)
{
  using OutSymbolOffsetT    = int32_t;
  using translation_table_t = TransducerLookupTable<OutSymbolT,
                                                    OutSymbolOffsetT,
                                                    MAX_NUM_SYMBOLS,
                                                    MAX_NUM_STATES,
                                                    MIN_TRANSLATED_OUT,
                                                    MAX_TRANSLATED_OUT,
                                                    MAX_TABLE_SIZE>;
  return translation_table_t::InitDeviceTranslationTable(translation_table);
}

template <typename TranslationOpT,
          typename OutSymbolT_,
          std::int32_t MIN_TRANSLATED_OUT_,
          std::int32_t MAX_TRANSLATED_OUT_>
class TranslationOp {
 private:
  struct _TempStorage {};

 public:
  using OutSymbolT                            = OutSymbolT_;
  static constexpr int32_t MIN_TRANSLATED_OUT = MIN_TRANSLATED_OUT_;
  static constexpr int32_t MAX_TRANSLATED_OUT = MAX_TRANSLATED_OUT_;

  using TempStorage = cub::Uninitialized<_TempStorage>;

  struct KernelParameter {
    using LookupTableT =
      TranslationOp<TranslationOpT, OutSymbolT, MIN_TRANSLATED_OUT, MAX_TRANSLATED_OUT>;
    TranslationOpT translation_op;
  };

  /**
   * @brief Initializes the lookup table, primarily to be invoked from within device code but also
   * provides host-side implementation for verification.
   * @note Synchronizes the thread block, if called from device, and, hence, requires all threads
   * of the thread block to call the constructor
   */
  static KernelParameter InitDeviceTranslationTable(TranslationOpT translation_op)
  {
    return KernelParameter{translation_op};
  }

 private:
  _TempStorage& temp_storage;
  TranslationOpT translation_op;

  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

 public:
  CUDF_HOST_DEVICE TranslationOp(KernelParameter const& kernel_param, TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias()), translation_op(kernel_param.translation_op)
  {
  }

  template <typename StateIndexT, typename SymbolIndexT, typename RelativeOffsetT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE auto operator()(StateIndexT const state_id,
                                             SymbolIndexT const match_id,
                                             RelativeOffsetT const relative_offset,
                                             SymbolT const read_symbol) const
  {
    return translation_op(state_id, match_id, relative_offset, read_symbol);
  }

  template <typename StateIndexT, typename SymbolIndexT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE auto operator()(StateIndexT const state_id,
                                             SymbolIndexT const match_id,
                                             SymbolT const read_symbol) const
  {
    return translation_op(state_id, match_id, read_symbol);
  }
};

/**
 * @brief Creates a simple translation table that uses a simple function object to retrieve the
 *
 * @tparam FunctorT A function object type that must implement two signatures: (1) with `(state_id,
 * match_id, read_symbol)` and (2) with `(state_id, match_id, relative_offset, read_symbol)`
 * @tparam MIN_TRANSLATED_SYMBOLS The minimum number of translated output symbols for any given
 * input symbol
 * @tparam MAX_TRANSLATED_SYMBOLS The maximum number of translated output symbols for any given
 * input symbol
 * @param map_op A function object that must implement two signatures: (1) with `(state_id,
 * match_id, read_symbol)` and (2) with `(state_id, match_id, relative_offset, read_symbol)`.
 * Invocations of the first signature, (1), must return the number of symbols that are emitted for
 * the given transition. The second signature, (2), must return the i-th symbol to be emitted for
 * that transition, where `i` corresponds to `relative_offse`
 * @return A translation table of type `TranslationO`
 */
template <typename OutSymbolT,
          std::size_t MIN_TRANSLATED_OUT,
          std::size_t MAX_TRANSLATED_OUT,
          typename FunctorT>
auto make_translation_functor(FunctorT map_op)
{
  return TranslationOp<FunctorT, OutSymbolT, MIN_TRANSLATED_OUT, MAX_TRANSLATED_OUT>::
    InitDeviceTranslationTable(map_op);
}

/**
 * @brief Helper class to facilitate the specification and instantiation of a DFA (i.e., the
 * transition table and its number of states, the mapping of symbols to symbol groups, and the
 * translation table that specifies which state transitions cause which output to be written).
 *
 * @tparam OutSymbolT The symbol type being output by the finite-state transducer
 * @tparam NUM_SYMBOLS The number of symbol groups amongst which to differentiate including the
 * wildcard symbol group (one dimension of the transition table)
 * @tparam NUM_STATES The number of states defined by the DFA (the other dimension of the
 * transition table)
 */
template <typename SymbolGroupIdInitT,
          typename TransitionTableInitT,
          typename TranslationTableInitT>
class Dfa {
  static constexpr int32_t single_item = 1;

 public:
  auto get_device_view()
  {
    return dfa_device_view<typename SymbolGroupIdInitT::LookupTableT,
                           typename TransitionTableInitT::LookupTableT,
                           typename TranslationTableInitT::LookupTableT,
                           TransitionTableInitT::LookupTableT::NUM_STATES>{
      &init_data.d_begin()->sgid_lut_init,
      &init_data.d_begin()->transition_table_init,
      &init_data.d_begin()->translation_table_init};
  }

  Dfa(SymbolGroupIdInitT const& sgid_lut_init,
      TransitionTableInitT const& transition_table_init,
      TranslationTableInitT const& translation_table_init,
      cuda::stream_ref stream)
    : init_data{single_item, stream}
  {
    *init_data.host_ptr() = {sgid_lut_init, transition_table_init, translation_table_init};
    init_data.host_to_device_async(stream);
  }

  /**
   * @brief Dispatches the finite-state transducer algorithm to the GPU.
   *
   * @tparam SymbolT The atomic symbol type from the input tape
   * @tparam TransducedOutItT Random-access output iterator to which the transduced output will be
   * written
   * @tparam TransducedIndexOutItT Random-access output iterator type to which the input symbols'
   * indexes are written.
   * @tparam TransducedCountOutItT A single-item output iterator type to which the total number of
   * output symbols is written
   * @tparam OffsetT A type large enough to index into either of both: (a) the input symbols and
   * (b) the output symbols
   * @param d_chars_it Pointer to the input string of symbols
   * @param num_chars The total number of input symbols to process
   * @param d_out_it Random-access output iterator to which the transduced output is
   * written
   * @param d_out_idx_it Random-access output iterator to which, the index i is written
   * iff the i-th input symbol caused some output to be written
   * @param d_num_transduced_out_it A single-item output iterator type to which the total number
   * of output symbols is written
   * @param seed_state The DFA's starting state. For streaming DFAs this corresponds to the
   * "end-state" of the previous invocation of the algorithm.
   * @param stream CUDA stream to launch kernels within. Default is the null-stream.
   */
  template <typename SymbolItT,
            typename TransducedOutItT,
            typename TransducedIndexOutItT,
            typename TransducedCountOutItT,
            typename OffsetT>
  void Transduce(SymbolItT d_chars_it,
                 OffsetT num_chars,
                 TransducedOutItT d_out_it,
                 TransducedIndexOutItT d_out_idx_it,
                 TransducedCountOutItT d_num_transduced_out_it,
                 uint32_t const seed_state,
                 cuda::stream_ref stream)
  {
    std::size_t temp_storage_bytes = 0;
    rmm::device_buffer temp_storage{};
    DeviceTransduce(nullptr,
                    temp_storage_bytes,
                    this->get_device_view(),
                    d_chars_it,
                    num_chars,
                    d_out_it,
                    d_out_idx_it,
                    d_num_transduced_out_it,
                    seed_state,
                    stream.get());

    if (temp_storage.size() < temp_storage_bytes) {
      temp_storage.resize(temp_storage_bytes, stream);
    }

    DeviceTransduce(temp_storage.data(),
                    temp_storage_bytes,
                    this->get_device_view(),
                    d_chars_it,
                    num_chars,
                    d_out_it,
                    d_out_idx_it,
                    d_num_transduced_out_it,
                    seed_state,
                    stream.get());
  }

 private:
  struct host_device_data {
    SymbolGroupIdInitT sgid_lut_init;
    TransitionTableInitT transition_table_init;
    TranslationTableInitT translation_table_init;
  };
  cudf::detail::hostdevice_vector<host_device_data> init_data{};
};

/**
 * @brief Creates a deterministic finite automaton (DFA) as specified by the triple of (symbol
 * group, transition, translation)-lookup tables to be used with the finite-state transducer
 * algorithm.
 *
 * @param sgid_lut_init Object used to initialize the symbol group lookup table
 * @param transition_table_init Object used to initialize the transition table
 * @param translation_table_init Object used to initialize the translation table
 * @param stream The stream used to allocate and initialize device-side memory that is used to
 * initialize the lookup tables
 * @return A DFA of type `Dfa`.
 */
template <typename SymbolGroupIdInitT,
          typename TransitionTableInitT,
          typename TranslationTableInitT>
auto make_fst(SymbolGroupIdInitT const& sgid_lut_init,
              TransitionTableInitT const& transition_table_init,
              TranslationTableInitT const& translation_table_init,
              cuda::stream_ref stream)
{
  return Dfa<SymbolGroupIdInitT, TransitionTableInitT, TranslationTableInitT>(
    sgid_lut_init, transition_table_init, translation_table_init, stream);
}

}  // namespace cudf::io::fst::detail
