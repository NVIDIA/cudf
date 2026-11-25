/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <string>
#include <vector>

namespace {

struct compare_case {
  cudf::size_type lhs_offset;
  cudf::size_type lhs_size;
  cudf::size_type rhs_offset;
  cudf::size_type rhs_size;
};

struct compare_fn {
  char const* d_chars;
  compare_case const* d_cases;

  __device__ int operator()(cudf::size_type idx) const
  {
    auto const c      = d_cases[idx];
    auto const lhs    = cudf::string_view(d_chars + c.lhs_offset, c.lhs_size);
    auto const result = lhs.compare(cudf::string_view(d_chars + c.rhs_offset, c.rhs_size));
    return (result > 0) - (result < 0);
  }
};

// Expected sign of comparing lhs with rhs as unsigned bytes, then by length
int expected_sign(std::string const& lhs, std::string const& rhs)
{
  auto const n = std::min(lhs.size(), rhs.size());
  for (std::size_t i = 0; i < n; ++i) {
    auto const l = static_cast<unsigned char>(lhs[i]);
    auto const r = static_cast<unsigned char>(rhs[i]);
    if (l != r) { return l < r ? -1 : 1; }
  }
  if (lhs.size() == rhs.size()) { return 0; }
  return lhs.size() < rhs.size() ? -1 : 1;
}

// Appends str to chars so that it starts at the given alignment (mod 4)
cudf::size_type append_aligned(std::vector<char>& chars, std::string const& str, int alignment)
{
  while (chars.size() % 4 != static_cast<std::size_t>(alignment)) {
    chars.push_back('#');
  }
  auto const offset = static_cast<cudf::size_type>(chars.size());
  chars.insert(chars.end(), str.begin(), str.end());
  return offset;
}

}  // namespace

struct StringViewCompareTest : public cudf::test::BaseFixture {};

TEST_F(StringViewCompareTest, AlignmentsAndMismatchPositions)
{
  // Pairs of strings that are equal, differ at every byte position, or where one is a prefix of
  // the other. Mismatching bytes include values >= 0x80 to check that bytes compare as unsigned.
  std::vector<std::pair<std::string, std::string>> pairs;
  for (int size : {0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
    std::string base(size, 'a');
    for (int i = 0; i < size; ++i) {
      base[i] = static_cast<char>('a' + (i % 26));
    }
    pairs.emplace_back(base, base);
    for (int pos = 0; pos < size; ++pos) {
      for (auto [l, r] :
           {std::pair{'a', 'b'}, std::pair{'\x01', '\xff'}, std::pair{'\x7f', '\x80'}}) {
        auto lhs = base;
        auto rhs = base;
        lhs[pos] = l;
        rhs[pos] = r;
        pairs.emplace_back(lhs, rhs);
        pairs.emplace_back(rhs, lhs);
      }
    }
    for (int prefix : {0, 1, 4, 7, 8, 9}) {
      if (prefix < size) {
        pairs.emplace_back(base.substr(0, size - prefix - 1), base);
        pairs.emplace_back(base, base.substr(0, size - prefix - 1));
      }
    }
  }

  std::vector<char> chars;
  std::vector<compare_case> cases;
  std::vector<int> expected;
  for (auto const& [lhs, rhs] : pairs) {
    for (int lhs_alignment = 0; lhs_alignment < 4; ++lhs_alignment) {
      for (int rhs_alignment = 0; rhs_alignment < 4; ++rhs_alignment) {
        auto const lhs_offset = append_aligned(chars, lhs, lhs_alignment);
        auto const rhs_offset = append_aligned(chars, rhs, rhs_alignment);
        cases.push_back({lhs_offset,
                         static_cast<cudf::size_type>(lhs.size()),
                         rhs_offset,
                         static_cast<cudf::size_type>(rhs.size())});
        expected.push_back(expected_sign(lhs, rhs));
      }
    }
  }
  // Compare a string with itself at the same address, and with an empty prefix of itself
  auto const last = cases.back();
  cases.push_back({last.lhs_offset, last.lhs_size, last.lhs_offset, last.lhs_size});
  expected.push_back(0);
  cases.push_back({last.lhs_offset, last.lhs_size, last.lhs_offset, 0});
  expected.push_back(last.lhs_size > 0 ? 1 : 0);

  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const d_chars = cudf::detail::make_device_uvector(chars, stream, mr);
  auto const d_cases = cudf::detail::make_device_uvector(cases, stream, mr);
  auto d_results     = rmm::device_uvector<int>(cases.size(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::counting_iterator<cudf::size_type>(cases.size()),
                    d_results.begin(),
                    compare_fn{d_chars.data(), d_cases.data()});
  auto const results = cudf::detail::make_std_vector(d_results, stream);

  for (std::size_t i = 0; i < cases.size(); ++i) {
    ASSERT_EQ(results[i], expected[i])
      << "case " << i << ": lhs offset " << cases[i].lhs_offset << " size " << cases[i].lhs_size
      << ", rhs offset " << cases[i].rhs_offset << " size " << cases[i].rhs_size;
  }
}
