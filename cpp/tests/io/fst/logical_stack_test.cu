/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/hostdevice_vector.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <src/io/fst/logical_stack.cuh>

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stack>
#include <vector>

namespace {
namespace fst = cudf::io::fst;

/**
 * @brief Generates the sparse representation of stack operations to feed into the logical
 * stack
 *
 * @param begin Forward input iterator to the first item of symbols that are checked for whether
 * they push or pop
 * @param end Forward input iterator to one one past the last item of symbols that are checked for
 * whether they push or pop
 * @param to_stack_op A function object that takes an instance of InputItT's value type and
 * returns the kind of stack operation such item represents (i.e., of type stack_op_type)
 * @param stack_symbol_out Forward output iterator to which symbols that either push or pop are
 * assigned
 * @param stack_op_index_out Forward output iterator to which the indexes of symbols that either
 * push or pop are assigned
 * @return Pair of iterators to one past the last item of the items written to \p stack_symbol_out
 * and \p stack_op_index_out, respectively
 */
template <typename InputItT,
          typename ToStackOpTypeT,
          typename StackSymbolOutItT,
          typename StackOpIndexOutItT>
std::pair<StackSymbolOutItT, StackOpIndexOutItT> to_sparse_stack_symbols(
  InputItT begin,
  InputItT end,
  ToStackOpTypeT to_stack_op,
  StackSymbolOutItT stack_symbol_out,
  StackOpIndexOutItT stack_op_index_out)
{
  std::size_t index = 0;
  for (auto it = begin; it < end; it++) {
    fst::stack_op_type op_type = to_stack_op(*it);
    if (op_type == fst::stack_op_type::PUSH || op_type == fst::stack_op_type::POP) {
      *stack_symbol_out   = *it;
      *stack_op_index_out = index;
      stack_symbol_out++;
      stack_op_index_out++;
    }
    index++;
  }
  return std::make_pair(stack_symbol_out, stack_op_index_out);
}

/**
 * @brief Reads in a sequence of items that represent stack operations, applies these operations to
 * a stack, and, for every operation being read in, outputs what was the symbol on top of the stack
 * before the operations was applied. In case the stack is empty before any operation,
 * \p empty_stack will be output instead.
 *
 * @tparam InputItT Forward input iterator type to items representing stack operations
 * @tparam ToStackOpTypeT A transform function object class that maps an item representing a stack
 * operation to the stack_op_type of such item
 * @tparam StackSymbolT Type representing items being pushed onto the stack
 * @tparam TopOfStackOutItT A forward output iterator type being assigned items of StackSymbolT
 * @param[in] begin Forward iterator to the beginning of the items representing stack operations
 * @param[in] end Iterator to one past the last item representing the stack operation
 * @param[in] to_stack_op A function object that takes an instance of InputItT's value type and
 * returns the kind of stack operation such item represents (i.e., of type stack_op_type)
 * @param[in] empty_stack A symbol that will be written to top_of_stack_out_it whenever the stack
 * was empty
 * @param[out] top_of_stack The output iterator to which the item will be written to
 * @return TopOfStackOutItT Iterators to one past the last element that was written
 */
template <typename InputItT,
          typename ToStackOpTypeT,
          typename StackSymbolT,
          typename TopOfStackOutItT>
TopOfStackOutItT to_top_of_stack(InputItT begin,
                                 InputItT end,
                                 ToStackOpTypeT to_stack_op,
                                 StackSymbolT empty_stack,
                                 TopOfStackOutItT top_of_stack_out_it)
{
  // This is the data structure that keeps track of the full stack state for each input symbol
  std::stack<StackSymbolT> stack_state;

  for (auto it = begin; it < end; it++) {
    // Write what is currently on top of the stack when reading in the current symbol
    *top_of_stack_out_it = stack_state.empty() ? empty_stack : stack_state.top();
    top_of_stack_out_it++;

    auto const& current        = *it;
    fst::stack_op_type op_type = to_stack_op(current);

    // Check whether this symbol corresponds to a push or pop operation and modify the stack
    // accordingly
    if (op_type == fst::stack_op_type::PUSH) {
      stack_state.push(current);
    } else if (op_type == fst::stack_op_type::POP) {
      stack_state.pop();
    }
  }
  return top_of_stack_out_it;
}

/**
 * @brief Function object used to filter for brackets and braces that represent push and pop
 * operations
 *
 */
struct JSONToStackOp {
  template <typename StackSymbolT>
  constexpr CUDF_HOST_DEVICE fst::stack_op_type operator()(StackSymbolT const& stack_symbol) const
  {
    return (stack_symbol == '{' || stack_symbol == '[')   ? fst::stack_op_type::PUSH
           : (stack_symbol == '}' || stack_symbol == ']') ? fst::stack_op_type::POP
                                                          : fst::stack_op_type::READ;
  }
};
}  // namespace

// Base test fixture for tests
struct LogicalStackTest : public cudf::test::BaseFixture {};

TEST_F(LogicalStackTest, GroundTruth)
{
  // Type sufficient to cover any stack level (must be a signed type)
  using StackLevelT   = int8_t;
  using SymbolT       = char;
  using SymbolOffsetT = uint32_t;

  // The stack symbol that we'll fill everywhere where there's nothing on the stack
  constexpr SymbolT empty_stack_symbol = '_';

  // This just has to be a stack symbol that may not be confused with a symbol that would push
  constexpr SymbolT read_symbol = 'x';

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  cuda::stream_ref stream_view{stream.value()};

  // Test input,
  std::string input = R"(  {)"
                      R"(category": "reference",)"
                      R"("index:" [4,12,42],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  )"
                      R"({)"
                      R"("category": "reference",)"
                      R"("index:" [4,{},null,{"a":[]}],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  {} [] [ ])";

  // Repeat input sample 1024x
  for (std::size_t i = 0; i < 10; i++)
    input += input;

  // Input's size
  std::size_t string_size = input.size();

  // Getting the symbols that actually modify the stack (i.e., symbols that push or pop)
  std::string stack_symbols{};
  std::vector<SymbolOffsetT> stack_op_indexes;
  stack_op_indexes.reserve(string_size);

  // Get the sparse representation of stack operations
  to_sparse_stack_symbols(std::cbegin(input),
                          std::cend(input),
                          JSONToStackOp{},
                          std::back_inserter(stack_symbols),
                          std::back_inserter(stack_op_indexes));

  rmm::device_uvector<SymbolT> d_stack_ops{stack_symbols.size(), stream_view};
  rmm::device_uvector<SymbolOffsetT> d_stack_op_indexes{stack_op_indexes.size(), stream_view};
  cudf::detail::hostdevice_vector<SymbolT> top_of_stack_gpu{string_size, stream_view};
  cudf::device_span<SymbolOffsetT> d_stack_op_idx_span{d_stack_op_indexes};

  CUDF_CUDA_TRY(cudaMemcpyAsync(d_stack_ops.data(),
                                stack_symbols.data(),
                                stack_symbols.size() * sizeof(SymbolT),
                                cudaMemcpyDefault,
                                stream.value()));

  CUDF_CUDA_TRY(cudaMemcpyAsync(d_stack_op_indexes.data(),
                                stack_op_indexes.data(),
                                stack_op_indexes.size() * sizeof(SymbolOffsetT),
                                cudaMemcpyDefault,
                                stream.value()));

  // Run algorithm
  fst::sparse_stack_op_to_top_of_stack<fst::stack_op_support::NO_RESET_SUPPORT, StackLevelT>(
    d_stack_ops.data(),
    d_stack_op_idx_span,
    JSONToStackOp{},
    top_of_stack_gpu.device_ptr(),
    empty_stack_symbol,
    read_symbol,
    string_size,
    stream.value());

  // Async copy results from device to host
  top_of_stack_gpu.device_to_host_async(stream_view);

  // Get CPU-side results for verification
  std::string top_of_stack_cpu{};
  top_of_stack_cpu.reserve(string_size);
  to_top_of_stack(std::cbegin(input),
                  std::cend(input),
                  JSONToStackOp{},
                  empty_stack_symbol,
                  std::back_inserter(top_of_stack_cpu));

  // Make sure results have been copied back to host
  stream.synchronize();

  // Verify results
  ASSERT_EQ(string_size, top_of_stack_cpu.size());
  ASSERT_EQ(top_of_stack_gpu.size(), top_of_stack_cpu.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(top_of_stack_gpu.host_ptr(), top_of_stack_cpu, string_size);
}

CUDF_TEST_PROGRAM_MAIN()
