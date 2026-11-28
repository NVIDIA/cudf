/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/dictionary/encode.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/limits>
#include <thrust/transform.h>

#include <vector>

namespace {

namespace lex = cudf::detail::row::lexicographic;

template <typename Comparator>
struct ordered_row_pairs {
  Comparator compare;
  cudf::size_type rows;

  __device__ bool operator()(cudf::size_type pair) const
  {
    return compare(pair / rows, pair % rows);
  }
};

template <typename Comparator>
void expect_self_ordering(cudf::table_view input,
                          std::vector<bool> const& expected,
                          bool equivalent,
                          Comparator comparator,
                          cuda::stream_ref stream,
                          cudf::memory_resources mr)
{
  auto const rows        = input.num_rows();
  auto output            = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                          rows * rows,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr.get_output_mr());
  auto const self        = lex::self_comparator{input, {}, {}, stream, mr};
  auto const check_nulls = cudf::nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const transform   = [&](auto compare) {
    thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator<cudf::size_type>{rows * rows},
                      output->mutable_view().data<bool>(),
                      ordered_row_pairs{compare, rows});
  };
  auto const dispatch = [&]<bool nested>() {
    if (equivalent)
      transform(self.less_equivalent<nested>(check_nulls, comparator));
    else
      transform(self.less<nested>(check_nulls, comparator));
  };
  if (cudf::has_nested_columns(input))
    dispatch.template operator()<true>();
  else
    dispatch.template operator()<false>();
  auto const reference =
    cudf::test::fixed_width_column_wrapper<bool>{expected.begin(), expected.end(), stream, mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    reference, output->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

template <typename Rep>
std::vector<bool> ordered_reference(std::vector<Rep> const& values,
                                    std::vector<int> const& validity,
                                    bool equivalent)
{
  std::vector<bool> result;
  for (std::size_t i = 0; i < values.size(); ++i) {
    for (std::size_t j = 0; j < values.size(); ++j) {
      auto const left  = validity.empty() || validity[i];
      auto const right = validity.empty() || validity[j];
      result.push_back(left && right ? (equivalent ? values[i] <= values[j] : values[i] < values[j])
                                     : (!left && right) || (equivalent && !left && !right));
    }
  }
  return result;
}

struct scale_observing_ordering {
  numeric::scale_type scale;

  template <typename T>
  __device__ cudf::detail::weak_ordering operator()(T lhs, T rhs) const
  {
    if constexpr (cudf::is_fixed_point<T>()) {
      if (lhs.scale() == scale && rhs.scale() == scale)
        return cudf::detail::compare_elements(lhs.value(), rhs.value());
    }
    // Custom comparators must still receive logical decimal values, not storage values.
    return cudf::detail::weak_ordering::GREATER;
  }
};

template <typename T>
struct EqualScaleOrderingTest : cudf::test::BaseFixtureWithHarness {};
TYPED_TEST_SUITE(EqualScaleOrderingTest, cudf::test::FixedPointTypes);

TYPED_TEST(EqualScaleOrderingTest, RepresentationOrderAndCustomComparator)
{
  using Rep         = cudf::device_storage_type_t<TypeParam>;
  auto const stream = this->stream();
  auto const mr     = this->resources();
  std::vector<Rep> const values{
    cuda::std::numeric_limits<Rep>::lowest(), -7, 0, 42, cuda::std::numeric_limits<Rep>::max(), 0};
  for (auto scale : {cuda::std::numeric_limits<int32_t>::lowest(),
                     -3,
                     0,
                     7,
                     cuda::std::numeric_limits<int32_t>::max()}) {
    auto const input = cudf::test::fixed_point_column_wrapper<Rep>{
      values.begin(), values.end(), numeric::scale_type{scale}, stream, mr};
    for (bool equivalent : {false, true}) {
      auto const reference = ordered_reference(values, {}, equivalent);
      expect_self_ordering(cudf::table_view{{input}},
                           reference,
                           equivalent,
                           lex::physical_element_comparator{},
                           stream,
                           mr);
      expect_self_ordering(cudf::table_view{{input}},
                           reference,
                           equivalent,
                           lex::sorting_physical_element_comparator{},
                           stream,
                           mr);
      expect_self_ordering(cudf::table_view{{input}},
                           reference,
                           equivalent,
                           scale_observing_ordering{numeric::scale_type{scale}},
                           stream,
                           mr);
    }
  }
}

TYPED_TEST(EqualScaleOrderingTest, NestedDictionaryAndNullableSlices)
{
  using Rep         = cudf::device_storage_type_t<TypeParam>;
  auto const stream = this->stream();
  auto const mr     = this->resources();
  auto input        = cudf::test::fixed_point_column_wrapper<Rep>{
    {99, 4, 5, 4, 7, 99}, {1, 1, 0, 1, 1, 1}, numeric::scale_type{-2}, stream, mr};
  auto dictionary =
    cudf::dictionary::encode(input, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto structures = cudf::test::structs_column_wrapper{{input}, {}, stream, mr};
  auto offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{{0, 1, 2, 3, 4, 5, 6}, stream, mr};
  auto lists = cudf::make_lists_column(
    6,
    offsets.release(),
    std::make_unique<cudf::column>(
      static_cast<cudf::column_view>(input), stream, mr.get_output_mr()),
    0,
    cuda::device_buffer<cudf::bitmask_type>{stream, mr.get_output_mr(), 0, cuda::no_init});
  for (auto view : {static_cast<cudf::column_view>(input),
                    dictionary->view(),
                    static_cast<cudf::column_view>(structures),
                    lists->view()}) {
    auto const sliced = cudf::slice(view, {1, 5}, stream).front();
    for (bool equivalent : {false, true}) {
      auto const reference =
        ordered_reference(std::vector<Rep>{4, 5, 4, 7}, {1, 0, 1, 1}, equivalent);
      expect_self_ordering(cudf::table_view{{sliced}},
                           reference,
                           equivalent,
                           lex::sorting_physical_element_comparator{},
                           stream,
                           mr);
    }
  }
}

}  // namespace
