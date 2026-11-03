/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/batch_gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Helpers (global namespace per the cudf test conventions).
// ============================================================================

std::unique_ptr<cudf::table> run_batch_gather(
  cudf::table_view const& source,
  cudf::column_view const& map,
  cudf::out_of_bounds_policy bounds = cudf::out_of_bounds_policy::DONT_CHECK,
  cudf::negative_index_policy neg   = cudf::negative_index_policy::NOT_ALLOWED)
{
  return cudf::detail::batch_gather(
    source,
    map,
    bounds,
    neg,
    cudf::get_default_stream(),
    cudf::memory_resources{cudf::get_current_device_resource_ref()});
}

std::unique_ptr<cudf::table> run_batch_gather_tile(cudf::table_view const& source,
                                                   cudf::column_view const& map,
                                                   cudf::size_type tile_cols);

std::unique_ptr<cudf::table> run_reference(
  cudf::table_view const& source,
  cudf::column_view const& map,
  cudf::out_of_bounds_policy bounds = cudf::out_of_bounds_policy::DONT_CHECK,
  cudf::negative_index_policy neg   = cudf::negative_index_policy::NOT_ALLOWED)
{
  return cudf::gather(source,
                      map,
                      bounds,
                      neg,
                      cudf::get_default_stream(),
                      cudf::memory_resources{cudf::get_current_device_resource_ref()});
}

// Builds a column of any FixedWidthTypes member (decimals ride their representation type at
// scale 0) with element i = value_of(i) and validity valid_of(i).
template <typename T, typename ValueFn, typename ValidFn>
std::unique_ptr<cudf::column> make_fixed_column(cudf::size_type size,
                                                ValueFn value_of,
                                                ValidFn valid_of)
{
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [valid_of](cudf::size_type i) { return static_cast<bool>(valid_of(i)); });
  if constexpr (cudf::is_fixed_point<T>()) {
    using Rep = typename T::rep;
    std::vector<Rep> reps;
    reps.reserve(size);
    for (cudf::size_type i = 0; i < size; ++i) {
      reps.push_back(static_cast<Rep>(value_of(i)));
    }
    return cudf::test::fixed_point_column_wrapper<Rep>(
             reps.begin(), reps.end(), validity, numeric::scale_type{0})
      .release();
  } else {
    std::vector<T> values;
    values.reserve(size);
    for (cudf::size_type i = 0; i < size; ++i) {
      if constexpr (cudf::is_timestamp<T>()) {
        values.push_back(T{typename T::duration{static_cast<typename T::rep>(value_of(i))}});
      } else if constexpr (cudf::is_duration<T>()) {
        values.push_back(T{static_cast<typename T::rep>(value_of(i))});
      } else {
        values.push_back(static_cast<T>(value_of(i)));
      }
    }
    return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end(), validity)
      .release();
  }
}

// Non-nullable variant of `make_fixed_column`: no validity iterator reaches the wrapper, so no
// mask is allocated (the validity-iterator ctor always allocates one, even for all-true input).
template <typename T, typename ValueFn>
std::unique_ptr<cudf::column> make_fixed_column_without_validity(cudf::size_type size,
                                                                 ValueFn value_of)
{
  if constexpr (cudf::is_fixed_point<T>()) {
    using Rep = typename T::rep;
    std::vector<Rep> reps;
    reps.reserve(size);
    for (cudf::size_type i = 0; i < size; ++i) {
      reps.push_back(static_cast<Rep>(value_of(i)));
    }
    return cudf::test::fixed_point_column_wrapper<Rep>(
             reps.begin(), reps.end(), numeric::scale_type{0})
      .release();
  } else {
    std::vector<T> values;
    values.reserve(size);
    for (cudf::size_type i = 0; i < size; ++i) {
      values.push_back(static_cast<T>(value_of(i)));
    }
    return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end()).release();
  }
}

std::vector<cudf::size_type> shuffled_iota(cudf::size_type size, unsigned seed = 0)
{
  std::vector<cudf::size_type> result(size);
  std::iota(result.begin(), result.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(result.begin(), result.end(), rng);
  return result;
}

// ============================================================================
// §5.1 Typed happy path over all fixed-width types (incl. fixed-point).
// ============================================================================

template <typename T>
class BatchGatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(BatchGatherTest, cudf::test::FixedWidthTypes);

TYPED_TEST(BatchGatherTest, Identity)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto gather_map = make_fixed_column<cudf::size_type>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(BatchGatherTest, ReverseIdentity)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto gather_map = make_fixed_column<cudf::size_type>(
    source_size,
    [](cudf::size_type i) { return source_size - 1 - i; },
    [](cudf::size_type) { return true; });
  auto expected = make_fixed_column<TypeParam>(
    source_size,
    [](cudf::size_type i) { return source_size - 1 - i; },
    [](cudf::size_type) { return true; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TYPED_TEST(BatchGatherTest, ShuffledMap)
{
  constexpr cudf::size_type source_size{1000};
  auto const host_map = shuffled_iota(source_size);

  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({source_column->view()});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TYPED_TEST(BatchGatherTest, NullablePartialNulls)
{
  constexpr cudf::size_type source_size{1000};  // nulls at even rows, valid at odd rows
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; });
  auto gather_map = make_fixed_column<cudf::size_type>(
    source_size / 2,
    [](cudf::size_type i) { return i * 2 + 1; },
    [](cudf::size_type) { return true; });
  auto expected = make_fixed_column<TypeParam>(
    source_size / 2,
    [](cudf::size_type i) { return i * 2 + 1; },
    [](cudf::size_type) { return true; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TYPED_TEST(BatchGatherTest, AllNullSource)
{
  constexpr cudf::size_type source_size{1000};
  auto const host_map = shuffled_iota(source_size);

  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return false; });
  auto expected = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type) { return 0; }, [](cudf::size_type) { return false; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TYPED_TEST(BatchGatherTest, NullableNoNullsSource)
{
  constexpr cudf::size_type source_size{1000};
  // Validity iterator over all-true values: nullable column with null_count() == 0.
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  ASSERT_EQ(source_column->null_count(), 0);
  auto gather_map = make_fixed_column<cudf::size_type>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(BatchGatherTest, NullifyOutOfBounds)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {0, source_size, source_size + 5, 2}};
  auto expected = make_fixed_column<TypeParam>(
    4,
    [](cudf::size_type i) { return i == 3 ? 2 : 0; },
    [](cudf::size_type i) { return i == 0 or i == 3; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TYPED_TEST(BatchGatherTest, NullifyNegativeWrap)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {-1, -source_size, 0, source_size - 1, -source_size - 1}};
  auto expected = make_fixed_column<TypeParam>(
    5,
    [](cudf::size_type i) { return (i == 0 or i == 3) ? source_size - 1 : 0; },
    [](cudf::size_type i) { return i != 4; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::NULLIFY,
                                 cudf::negative_index_policy::ALLOWED);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

// ============================================================================
// §5.2 Sliced sources (non-zero offset): the data<T>()/mask-offset regression class.
// ============================================================================

class BatchGatherSlicedTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherSlicedTest, SlicedDataAndMask)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<int32_t>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; });

  auto sliced = cudf::slice(source_column->view(), {100, 600});
  ASSERT_EQ(sliced.size(), 1);
  auto gather_map = make_fixed_column<cudf::size_type>(
    500, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({sliced[0]});
  auto result = run_batch_gather(source_table, gather_map->view());

  auto expected = make_fixed_column<int32_t>(
    500,
    [](cudf::size_type i) { return 100 + i; },
    [](cudf::size_type i) { return (100 + i) % 2 != 0; });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TEST_F(BatchGatherSlicedTest, SlicedMixedWidths)
{
  constexpr cudf::size_type source_size{100};
  constexpr cudf::size_type slice_begin{10};
  constexpr cudf::size_type slice_end{60};

  auto col_i8 = make_fixed_column<int8_t>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto col_i64 = make_fixed_column<int64_t>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto col_dec = make_fixed_column<numeric::decimal128>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  auto sliced_i8  = cudf::slice(col_i8->view(), {slice_begin, slice_end});
  auto sliced_i64 = cudf::slice(col_i64->view(), {slice_begin, slice_end});
  auto sliced_dec = cudf::slice(col_dec->view(), {slice_begin, slice_end});
  cudf::table_view source_table({sliced_i8[0], sliced_i64[0], sliced_dec[0]});

  auto gather_map = make_fixed_column<cudf::size_type>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return i; },
    [](cudf::size_type) { return true; });
  auto result = run_batch_gather(source_table, gather_map->view());

  auto expected_i8 = make_fixed_column<int8_t>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return slice_begin + i; },
    [](cudf::size_type) { return true; });
  auto expected_i64 = make_fixed_column<int64_t>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return slice_begin + i; },
    [](cudf::size_type) { return true; });
  auto expected_dec = make_fixed_column<numeric::decimal128>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return slice_begin + i; },
    [](cudf::size_type) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_i8->view(), result->view().column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_i64->view(), result->view().column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_dec->view(), result->view().column(2));
}

// Regression for the ALL_NULL null-count semantics: a SLICED all-null source has
// `src.size() < gather_map.size()`, so the destination null count is the map size —
// not the source slice size. Asserts the null_count explicitly (table equality alone
// can mask a wrong count).
TEST_F(BatchGatherSlicedTest, AllNullSlicedSourceParity)
{
  constexpr cudf::size_type source_size{1000};
  constexpr cudf::size_type slice_begin{100};
  constexpr cudf::size_type slice_end{600};
  constexpr cudf::size_type num_rows{500};

  auto source_column = make_fixed_column<int32_t>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return false; });

  auto sliced = cudf::slice(source_column->view(), {slice_begin, slice_end});
  ASSERT_EQ(sliced.size(), 1);
  ASSERT_EQ(sliced[0].size(), slice_end - slice_begin);
  auto gather_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({sliced[0]});
  auto result = run_batch_gather(source_table, gather_map->view());

  ASSERT_EQ(result->view().column(0).null_count(), num_rows);
  auto expected = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type) { return 0; }, [](cudf::size_type) { return false; });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

// Regression for the ALL_NULL null-count semantics across the three carve consumer classes:
// a fixed-width child inside a STRUCT (consumes plan.null_count through the struct path), a
// string-keyed DICTIONARY (eager parent mask path), and a plain STRINGS column (staged
// null_count path). A map SHORTER than the source discriminates src.size() from num_rows: on
// the pre-fix tree every column here would report 1000 (src.size()) instead of 500, so the
// per-column null_count assertion below — not table equality alone — is what fails.
std::unique_ptr<cudf::column> make_independent_struct(
  std::vector<std::unique_ptr<cudf::column>> children, std::vector<bool> const& parent_valid);
std::vector<std::string> periodic_strings(cudf::size_type num_rows, int const stride);

TEST_F(BatchGatherSlicedTest, AllNullMixedColumnsShortMapParity)
{
  constexpr cudf::size_type source_size{1000};
  constexpr cudf::size_type num_rows{500};

  auto all_null_child = make_fixed_column<int32_t>(
    source_size, [](cudf::size_type) { return 0; }, [](cudf::size_type) { return false; });
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(std::move(all_null_child));
  auto col_struct =
    make_independent_struct(std::move(struct_children), std::vector<bool>(source_size, false));

  std::vector<std::string> dict_values(source_size);
  std::generate(dict_values.begin(), dict_values.end(), [n = 0]() mutable {
    return std::string(1, static_cast<char>('a' + (n++) % 4));
  });
  auto dict_invalid =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type) { return false; });
  cudf::test::dictionary_column_wrapper<std::string> col_dict(
    dict_values.begin(), dict_values.end(), dict_invalid);

  auto str_invalid =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type) { return false; });
  auto const strings = periodic_strings(source_size, 7);
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), str_invalid);

  cudf::table_view source_table({col_struct->view(), col_dict, col_str});
  auto gather_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  auto result    = run_batch_gather(source_table, gather_map->view());
  auto reference = run_reference(source_table, gather_map->view());

  ASSERT_EQ(result->num_columns(), 3);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  for (cudf::size_type c = 0; c < 3; ++c) {
    EXPECT_EQ(result->view().column(c).null_count(), num_rows) << "column " << c;
  }
}

// ============================================================================
// §5.2b Preset-mask fill parity: the batched constant-mask fills (ALL_VALID 0xff /
// ALL_NULL 0x00 riding pattern staging bands, one memset per band) must produce masks
// byte-identical to the stock per-mask memsets — including the Arrow-padded tail words,
// which column comparisons do read through the null-count/mask equality path. The band
// path engages only above `kMaskBandPresetGate` (8) preset destination masks per call;
// fixtures here carry 12+ preset columns to stay on the band path (and one fixture sits
// exactly AT the gate to pin the stock per-mask path).
// ============================================================================

class BatchGatherPresetMaskTest : public cudf::test::BaseFixture {};

// Boundary shapes at the padded-word tails: 511/512/513-bit masks straddle the last
// word's valid/padding boundary, where a fill that missed padding words or inverted the
// tail word would corrupt the mask (padding bits are never read back, but the
// ALL_NULL-with-padding case must still zero them for byte parity with stock). 12 preset
// columns keep every fixture above the `kMaskBandPresetGate` band gate.
TEST_F(BatchGatherPresetMaskTest, AllValidMaskWordTailParity)
{
  for (auto const num_rows : {511, 512, 513}) {
    std::vector<std::unique_ptr<cudf::column>> preset_cols;
    preset_cols.reserve(12);
    for (int i = 0; i < 12; ++i) {
      preset_cols.push_back(make_fixed_column<int32_t>(
        num_rows,
        [i](cudf::size_type r) { return i * 1000 + r; },
        [](cudf::size_type) { return true; }));
    }
    auto views = std::vector<cudf::column_view>();
    views.reserve(preset_cols.size());
    for (auto const& c : preset_cols) {
      views.push_back(c->view());
    }
    cudf::table_view source_table(views);
    auto gather_map = make_fixed_column<cudf::size_type>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

    auto result    = run_batch_gather(source_table, gather_map->view());
    auto reference = run_reference(source_table, gather_map->view());

    CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
    for (cudf::size_type c = 0; c < 12; ++c) {
      EXPECT_EQ(result->view().column(c).null_count(), 0) << "column " << c;
    }
  }
}

TEST_F(BatchGatherPresetMaskTest, AllNullMaskWordTailParity)
{
  for (auto const num_rows : {511, 512, 513}) {
    std::vector<std::unique_ptr<cudf::column>> preset_cols;
    preset_cols.reserve(12);
    for (int i = 0; i < 12; ++i) {
      preset_cols.push_back(make_fixed_column<int32_t>(
        num_rows, [](cudf::size_type) { return 0; }, [](cudf::size_type) { return false; }));
    }
    auto views = std::vector<cudf::column_view>();
    views.reserve(preset_cols.size());
    for (auto const& c : preset_cols) {
      views.push_back(c->view());
    }
    cudf::table_view source_table(views);
    auto gather_map = make_fixed_column<cudf::size_type>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

    auto result    = run_batch_gather(source_table, gather_map->view());
    auto reference = run_reference(source_table, gather_map->view());

    CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
    for (cudf::size_type c = 0; c < 12; ++c) {
      EXPECT_EQ(result->view().column(c).null_count(), num_rows) << "column " << c;
    }
  }
}

// Sliced all-valid and all-null sources: the preset plan keys off the DESTINATION row
// count, so a sliced source whose underlying mask is larger must still fill exactly the
// destination extent — and the null count must be the map size, not the slice size. 6+6
// columns stay above the `kMaskBandPresetGate` band gate.
TEST_F(BatchGatherPresetMaskTest, SlicedPresetSourceParity)
{
  constexpr cudf::size_type source_size{1000};
  constexpr cudf::size_type slice_begin{100};
  constexpr cudf::size_type num_rows{500};

  auto all_valid = make_fixed_column<int32_t>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto all_null = make_fixed_column<int32_t>(
    source_size, [](cudf::size_type) { return 0; }, [](cudf::size_type) { return false; });
  auto sliced_valid = cudf::slice(all_valid->view(), {slice_begin, slice_begin + num_rows});
  auto sliced_null  = cudf::slice(all_null->view(), {slice_begin, slice_begin + num_rows});
  ASSERT_EQ(sliced_valid.size(), 1);
  ASSERT_EQ(sliced_null.size(), 1);
  auto views = std::vector<cudf::column_view>();
  views.reserve(12);
  for (int i = 0; i < 6; ++i) {
    views.push_back(sliced_valid[0]);
  }
  for (int i = 0; i < 6; ++i) {
    views.push_back(sliced_null[0]);
  }
  auto gather_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table(views);
  auto result    = run_batch_gather(source_table, gather_map->view());
  auto reference = run_reference(source_table, gather_map->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  for (cudf::size_type c = 0; c < 6; ++c) {
    EXPECT_EQ(result->view().column(c).null_count(), 0) << "column " << c;
  }
  for (cudf::size_type c = 6; c < 12; ++c) {
    EXPECT_EQ(result->view().column(c).null_count(), num_rows) << "column " << c;
  }
}

// Zero-row map with preset sources: the ALL_VALID destination stays nullable-but-valid
// (empty mask, null_count 0) and the ALL_NULL destination reports null_count 0 for an
// empty column — stock parity, and the band memsets must not fire on empty arenas.
TEST_F(BatchGatherPresetMaskTest, ZeroRowPresetSources)
{
  std::vector<std::unique_ptr<cudf::column>> preset_cols;
  preset_cols.reserve(12);
  for (int i = 0; i < 6; ++i) {
    preset_cols.push_back(make_fixed_column<int32_t>(
      10, [i](cudf::size_type r) { return i * 10 + r; }, [](cudf::size_type) { return true; }));
  }
  for (int i = 0; i < 6; ++i) {
    preset_cols.push_back(make_fixed_column<int32_t>(
      10, [](cudf::size_type r) { return r; }, [](cudf::size_type) { return false; }));
  }
  auto views = std::vector<cudf::column_view>();
  views.reserve(preset_cols.size());
  for (auto const& c : preset_cols) {
    views.push_back(c->view());
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{};

  cudf::table_view source_table(views);
  auto result = run_batch_gather(source_table, gather_map);

  ASSERT_EQ(result->num_rows(), 0);
  for (cudf::size_type c = 0; c < 6; ++c) {
    EXPECT_EQ(result->view().column(c).null_count(), 0) << "column " << c;
  }
}

// Both patterns in ONE call (the two-band case): a mixed table where every preset class
// is present, plus a kernel-masked column (NULLIFY) to prove the kernel-written masks
// never ride a band. Compared against the stock reference exactly. 15 preset columns
// (9 all-valid + 4 all-null + the all-valid struct node + its all-valid child) stay
// above the band gate.
TEST_F(BatchGatherPresetMaskTest, MixedPresetClassesParity)
{
  constexpr cudf::size_type num_rows{1000};

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(15);
  for (int i = 0; i < 9; ++i) {
    cols.push_back(make_fixed_column<int32_t>(
      num_rows,
      [i](cudf::size_type r) { return i * 1000 + r; },
      [](cudf::size_type) { return true; }));
  }
  for (int i = 0; i < 4; ++i) {
    cols.push_back(make_fixed_column<int64_t>(
      num_rows, [](cudf::size_type) { return 0; }, [](cudf::size_type) { return false; }));
  }
  auto kernel_masked = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 3 != 0; });
  // A struct node rides the band through the struct path when its source is all-valid.
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return 2 * i; }, [](cudf::size_type) { return true; }));
  auto col_struct =
    make_independent_struct(std::move(struct_children), std::vector<bool>(num_rows, true));

  cols.push_back(std::move(kernel_masked));
  cols.push_back(std::move(col_struct));

  auto views = std::vector<cudf::column_view>();
  views.reserve(cols.size());
  for (auto const& c : cols) {
    views.push_back(c->view());
  }
  auto gather_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table(views);
  auto result    = run_batch_gather(source_table, gather_map->view());
  auto reference = run_reference(source_table, gather_map->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  for (cudf::size_type c = 0; c < 9; ++c) {
    EXPECT_EQ(result->view().column(c).null_count(), 0) << "column " << c;
  }
  for (cudf::size_type c = 9; c < 13; ++c) {
    EXPECT_EQ(result->view().column(c).null_count(), num_rows) << "column " << c;
  }
  EXPECT_EQ(result->view().column(14).null_count(), 0);  // the struct node
}

// Below the `kMaskBandPresetGate` band gate (8 preset masks in one call): the fill falls
// back to the stock per-mask carve + per-mask memset path — the fixed two-band-fill cost
// (~48 µs) would exceed the replaced per-mask drip (~7 µs/mask submission) at few columns.
// Byte parity with stock must hold identically on this path, and the path choice is
// observable here only through the parity itself: 8 preset masks is exactly AT the gate.
TEST_F(BatchGatherPresetMaskTest, BandGateBelowThresholdPerMaskParity)
{
  constexpr cudf::size_type num_rows{1000};

  // 8 preset columns exactly at the gate (gate fires only ABOVE 8) + one kernel-masked
  // column: every preset mask takes the stock per-mask path.
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(9);
  for (int i = 0; i < 6; ++i) {
    cols.push_back(make_fixed_column<int32_t>(
      num_rows,
      [i](cudf::size_type r) { return i * 1000 + r; },
      [](cudf::size_type) { return true; }));
  }
  for (int i = 0; i < 2; ++i) {
    cols.push_back(make_fixed_column<int64_t>(
      num_rows, [](cudf::size_type) { return 0; }, [](cudf::size_type) { return false; }));
  }
  cols.push_back(make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 3 != 0; }));

  auto views = std::vector<cudf::column_view>();
  views.reserve(cols.size());
  for (auto const& c : cols) {
    views.push_back(c->view());
  }
  auto gather_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table(views);
  auto result    = run_batch_gather(source_table, gather_map->view());
  auto reference = run_reference(source_table, gather_map->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  for (cudf::size_type c = 0; c < 6; ++c) {
    EXPECT_EQ(result->view().column(c).null_count(), 0) << "column " << c;
  }
  for (cudf::size_type c = 6; c < 8; ++c) {
    EXPECT_EQ(result->view().column(c).null_count(), num_rows) << "column " << c;
  }
}

// ============================================================================
// §5.3 Degenerate shapes.
// ============================================================================

class BatchGatherDegenerateTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherDegenerateTest, ZeroRowMap)
{
  auto source_column = make_fixed_column<int32_t>(
    10, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{};

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map);

  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->num_columns(), 1);
  EXPECT_EQ(result->view().column(0).type(), source_column->view().type());
}

// A zero-ROW string source under NULLIFY: every gathered row is out-of-bounds, so stock
// produces an all-null column (null_count == num_rows). The zero-row source is excluded
// from the live cohort in the assemble step, so its kernel-written (UNINITIALIZED) plan
// would otherwise never reach the kernel — the completion would build an uninitialized
// mask with a placeholder 0 null count. The build must take the inline all-null path.
// The mixed cohort rides beside a zero-row fixed-width column: a non-empty string column
// is structurally unrunnable here because table_view requires equal row counts across
// columns; a 0-row fixed-width column keeps the size match while still pinning the
// mixed-cohort shape.
TEST_F(BatchGatherDegenerateTest, ZeroRowStringSourceNullifyParity)
{
  auto empty_str = cudf::test::strings_column_wrapper{}.release();
  cudf::test::fixed_width_column_wrapper<cudf::size_type> empty_col{};

  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{{0, 1, 2, 3}};

  cudf::table_view single({empty_str->view()});
  auto result    = run_batch_gather(single, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  auto reference = run_reference(single, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  EXPECT_EQ(result->view().column(0).null_count(), 4);

  cudf::table_view mixed({empty_str->view(), empty_col});
  auto result_mixed    = run_batch_gather(mixed, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  auto reference_mixed = run_reference(mixed, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference_mixed->view(), result_mixed->view());
  EXPECT_EQ(result_mixed->view().column(0).null_count(), 4);
}

TEST_F(BatchGatherDegenerateTest, ZeroColumnTable)
{
  cudf::table_view source{std::vector<cudf::column_view>{}, 5};
  auto gather_map = make_fixed_column<cudf::size_type>(
    4, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  auto result = run_batch_gather(source, gather_map->view());

  EXPECT_EQ(result->num_columns(), 0);
  EXPECT_EQ(result->num_rows(), 4);
}

TEST_F(BatchGatherDegenerateTest, OneRowTable)
{
  auto source_column = make_fixed_column<int32_t>(
    1, [](cudf::size_type) { return 42; }, [](cudf::size_type) { return true; });
  cudf::table_view source_table({source_column->view()});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> identity_map{{0}};
  auto result = run_batch_gather(source_table, identity_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());

  cudf::test::fixed_width_column_wrapper<cudf::size_type> wrap_map{{-1}};
  auto wrapped = run_batch_gather(source_table,
                                  wrap_map,
                                  cudf::out_of_bounds_policy::NULLIFY,
                                  cudf::negative_index_policy::ALLOWED);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, wrapped->view());
}

// ============================================================================
// §5.4 Width/shape scaling.
// ============================================================================

class BatchGatherShapeTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherShapeTest, WideTable512Columns)
{
  constexpr cudf::size_type num_rows{1000};
  constexpr cudf::size_type num_cols{512};

  std::vector<std::unique_ptr<cudf::column>> source_columns;
  source_columns.reserve(num_cols);
  for (cudf::size_type c = 0; c < num_cols; ++c) {
    source_columns.push_back(make_fixed_column<int32_t>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  }
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  for (auto const& col : source_columns) {
    views.push_back(col->view());
  }
  cudf::table_view source_table(views);

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherShapeTest, YCapColumnLoop)
{
  // 66'001 columns span 8'251 tiles per block.y and force a PARTIAL final tile (1 column);
  // the y-stride wrap past the 65535 gridDim.y cap only engages above 524'280 columns
  // (65'535 tiles x 8), which is not host-feasible to build here — the wrap shares the
  // x-stride code shape and its cap math is exercised via the grid-y computation.
  constexpr cudf::size_type num_rows{100};
  constexpr cudf::size_type num_cols{66'001};

  std::vector<std::unique_ptr<cudf::column>> source_columns;
  source_columns.reserve(num_cols);
  for (cudf::size_type c = 0; c < num_cols; ++c) {
    source_columns.push_back(make_fixed_column<int32_t>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  }
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  for (auto const& col : source_columns) {
    views.push_back(col->view());
  }
  cudf::table_view source_table(views);

  auto gather_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TEST_F(BatchGatherShapeTest, TallTable)
{
  constexpr cudf::size_type num_rows{1'048'576};
  constexpr cudf::size_type num_cols{8};

  std::vector<std::unique_ptr<cudf::column>> source_columns;
  source_columns.reserve(num_cols);
  for (cudf::size_type c = 0; c < num_cols; ++c) {
    source_columns.push_back(make_fixed_column<int32_t>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  }
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  for (auto const& col : source_columns) {
    views.push_back(col->view());
  }
  cudf::table_view source_table(views);

  auto identity_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto identity_result = run_batch_gather(source_table, identity_map->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, identity_result->view());

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> shuffled_map(host_map.begin(),
                                                                       host_map.end());
  auto shuffled_result = run_batch_gather(source_table, shuffled_map);
  auto reference       = run_reference(source_table, shuffled_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), shuffled_result->view());
}

// ============================================================================
// §5.5 Bounds and policies.
// ============================================================================

class BatchGatherPolicyTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherPolicyTest, DontCheckPolicyAllValid)
{
  constexpr cudf::size_type num_rows{100};
  // Non-nullable source: DONT_CHECK must not fabricate a mask (§3.4 row 1).
  auto source_column =
    make_fixed_column_without_validity<int32_t>(num_rows, [](cudf::size_type i) { return i; });
  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map);

  ASSERT_FALSE(source_column->nullable());
  EXPECT_FALSE(result->view().column(0).nullable());
  auto expected = make_fixed_column_without_validity<int32_t>(
    num_rows, [&host_map](cudf::size_type i) { return host_map[i]; });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TEST_F(BatchGatherPolicyTest, DontCheckPolicyNullableSource)
{
  constexpr cudf::size_type num_rows{100};
  auto source_column = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; });
  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map);

  EXPECT_TRUE(result->view().column(0).nullable());
  auto expected = make_fixed_column<int32_t>(
    num_rows,
    [&host_map](cudf::size_type i) { return host_map[i]; },
    [&host_map](cudf::size_type i) { return host_map[i] % 2 != 0; });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));

  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherPolicyTest, NullifyMakesNullable)
{
  constexpr cudf::size_type num_rows{100};
  auto source_column =
    make_fixed_column_without_validity<int32_t>(num_rows, [](cudf::size_type i) { return i; });
  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);

  EXPECT_TRUE(result->view().column(0).nullable());
  EXPECT_EQ(result->view().column(0).null_count(), 0);
  auto expected = make_fixed_column<int32_t>(
    num_rows,
    [&host_map](cudf::size_type i) { return host_map[i]; },
    [](cudf::size_type) { return true; });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TEST_F(BatchGatherPolicyTest, ExplicitWideTileMaskedDifferential)
{
  // The K=8 instantiation's in-kernel mask path (per-column warp_valid[k] + valid_counts) is
  // only reachable via the explicit tile_cols overload on a null-bearing table.
  constexpr cudf::size_type num_rows{1000};
  auto source_column = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; });
  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({source_column->view()});
  auto result    = run_batch_gather_tile(source_table, gather_map, 8);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  EXPECT_TRUE(result->view().column(0).nullable());
}

// ============================================================================
// §5.6 Mixed tables: fused + fallback must agree with the stock gather.
// ============================================================================

class BatchGatherMixedTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherMixedTest, MixedFixedWidthAndStrings)
{
  constexpr cudf::size_type num_rows{100};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back("s" + std::to_string(i));
  }
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());
  auto col_i64 = make_fixed_column<int64_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str, col_i64->view()});

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherMixedTest, MixedWithListsAndStructs)
{
  constexpr cudf::size_type num_rows{10};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::lists_column_wrapper<int32_t> col_list{
    {1, 2}, {}, {3}, {4, 5}, {}, {6}, {7, 8}, {}, {9}, {10}};
  auto child_i64 = cudf::test::fixed_width_column_wrapper<int64_t>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto child_str =
    cudf::test::strings_column_wrapper{{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}};
  cudf::test::structs_column_wrapper col_struct{
    {child_i64, child_str}, {true, true, false, true, true, true, false, true, true, true}};

  cudf::table_view source_table({col_i32->view(), col_list, col_struct});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {9, 0, 3, 7, 1, 5, 2, 8, 4, 6}};

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherMixedTest, MixedNullifyOOB)
{
  constexpr cudf::size_type num_rows{10};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::lists_column_wrapper<int32_t> col_list{
    {1, 2}, {}, {3}, {4, 5}, {}, {6}, {7, 8}, {}, {9}, {10}};
  auto child_i64 = cudf::test::fixed_width_column_wrapper<int64_t>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto child_str =
    cudf::test::strings_column_wrapper{{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}};
  cudf::test::structs_column_wrapper col_struct{
    {child_i64, child_str}, {true, true, false, true, true, true, false, true, true, true}};

  cudf::table_view source_table({col_i32->view(), col_list, col_struct});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {0, num_rows, 3, num_rows + 5, 1, 5, 2, 8, 4, 6}};

  auto result    = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  auto reference = run_reference(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherMixedTest, MixedNullableStrings)
{
  constexpr cudf::size_type num_rows{100};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 3 != 0; });
  std::vector<std::string> strings;
  std::vector<bool> valid;
  strings.reserve(num_rows);
  valid.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(i % 5 == 0 ? "" : "v" + std::to_string(i));  // empty string != null string
    valid.push_back(i % 4 != 0);
  }
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [&valid](cudf::size_type i) { return valid[i]; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_i32->view(), col_str});

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Row lengths straddling the block-copy gate (32 bytes): the uint4 interior, the scalar edge
// bytes and the byte-parallel fallback all see rows on both sides of the seam.
TEST_F(BatchGatherMixedTest, StringRowsAcrossBlockCopyGate)
{
  constexpr cudf::size_type num_rows{20};
  cudf::size_type const lengths[num_rows] = {0,  1,  3,  4,  15, 16,  17,  31,  32,   33,
                                             47, 48, 63, 64, 65, 127, 128, 129, 1023, 4097};

  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    auto const size = static_cast<std::size_t>(lengths[i]);
    std::string value(size, '\0');
    for (std::size_t j = 0; j < size; ++j) {
      value[j] = static_cast<char>('a' + (i + static_cast<cudf::size_type>(j)) % 26);
    }
    strings.push_back(std::move(value));
  }
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());
  auto col_i64 = make_fixed_column<int64_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_str, col_i64->view()});

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Every row is above the gate and the table spans several block groups, so the published slots
// are recycled across group iterations; nullable source rows keep the skipped-row path live.
TEST_F(BatchGatherMixedTest, StringRowsAboveGateAcrossGroups)
{
  constexpr cudf::size_type num_rows{600};
  std::vector<std::string> strings(num_rows);
  std::vector<bool> valid(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings[i].assign(static_cast<std::size_t>(33 + i % 97), static_cast<char>('A' + i % 26));
    valid[i] = i % 11 != 0;
  }
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [&valid](cudf::size_type i) { return valid[i]; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_str});

  auto const host_map = shuffled_iota(num_rows, 1234);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Fewer rows than one block group: the warps without a row must still reach both barriers.
TEST_F(BatchGatherMixedTest, StringRowsAboveGateFewerThanOneGroup)
{
  cudf::test::strings_column_wrapper col_str{
    {std::string(33, 'a'), std::string(1024, 'b'), std::string(40, 'c')}};

  cudf::table_view source_table({col_str});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{{2, 0, 1}};

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// An all-short group takes the char-parallel fan: every live byte of the group is written by
// exactly one thread via the binary search over the staged scratch boundaries. Rows of length
// 0 (dead and empty), 1, and the 32-byte gate boundary interleave.
TEST_F(BatchGatherMixedTest, StringShortRowsCharParallelFan)
{
  constexpr cudf::size_type num_rows{100};
  std::vector<std::string> strings;
  std::vector<bool> valid;
  strings.reserve(num_rows);
  valid.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    auto const size = static_cast<std::size_t>(i % 33);  // 0..32: every row fits a warp span
    strings.push_back(std::string(size, static_cast<char>('a' + i % 26)));
    valid.push_back(i % 7 != 0);
  }
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [&valid](cudf::size_type i) { return valid[i]; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_str});

  auto const host_map = shuffled_iota(num_rows, 77);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// A long row anywhere in its block group keeps the whole group on the slot machinery: the long
// row takes the block-cooperative copy and the short rows of the SAME group keep the warp
// copy (the fan never runs in a mixed group).
TEST_F(BatchGatherMixedTest, StringMixedGroupKeepsSlotMachinery)
{
  constexpr cudf::size_type num_rows{24};  // 3 block groups of 8 warps; rows 5 and 20 are long
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    // Row 5 (group 0) and row 20 (group 2) are long; everything else is short.
    strings.push_back(
      std::string(i == 5 or i == 20 ? 500 : 8 + i % 16, static_cast<char>('A' + i % 26)));
  }
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());

  cudf::table_view source_table({col_str});

  auto const host_map = shuffled_iota(num_rows, 42);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// The last group is partial and its final row is the column's byte-range tail: the fan's tail
// sentinel must close the last live row's span with that row's +1 scratch entry.
TEST_F(BatchGatherMixedTest, StringShortRowsTailGroupSentinel)
{
  constexpr cudf::size_type num_rows{19};  // 2 full 8-warp groups + a 3-row tail group
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(
      std::string(static_cast<std::size_t>(1 + (i * 7) % 32), static_cast<char>('a' + i % 26)));
  }
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());

  cudf::table_view source_table({col_str});

  auto const host_map = shuffled_iota(num_rows, 7);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Out-of-bounds map entries inside an all-short fan group: NULLIFY gives those rows no
// payload (the scan left their scratch spans zero-width), so the fan must skip their bytes
// while still copying the live rows around them.
TEST_F(BatchGatherMixedTest, StringShortRowsFanNullifyOutOfBounds)
{
  constexpr cudf::size_type num_rows{50};
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(
      std::string(static_cast<std::size_t>(1 + (i * 11) % 32), static_cast<char>('a' + i % 26)));
  }
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());

  cudf::table_view source_table({col_str});

  // 10 destinations gather out-of-bounds (NULLIFY); the rest are a shuffle of live rows.
  std::vector<cudf::size_type> host_map(num_rows);
  std::iota(host_map.begin(), host_map.end(), 0);
  std::shuffle(host_map.begin(), host_map.end(), std::mt19937{9});
  for (cudf::size_type i = 0; i < num_rows; i += 5) {
    host_map[i] = num_rows + 1 + i;
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  auto reference = run_reference(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Negative map entries with negative_index_policy::ALLOWED wrap into the source from the end;
// the fan's resolve must apply the same wrap for its byte lookups.
TEST_F(BatchGatherMixedTest, StringShortRowsFanNegativeWrap)
{
  constexpr cudf::size_type num_rows{40};
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(
      std::string(static_cast<std::size_t>(1 + (i * 5) % 32), static_cast<char>('a' + i % 26)));
  }
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());

  cudf::table_view source_table({col_str});

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(i % 3 == 0 ? static_cast<cudf::size_type>(-1 - static_cast<int64_t>(i))
                                  : (num_rows - 1 - i));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::ALLOWED);
  auto reference = run_reference(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::ALLOWED);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Repeated map entries: several destination rows share one source row; the fan must copy the
// same source bytes to each of their disjoint destination spans.
TEST_F(BatchGatherMixedTest, StringShortRowsFanRepeatedSource)
{
  constexpr cudf::size_type num_rows{60};
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(
      std::string(static_cast<std::size_t>(1 + (i * 13) % 32), static_cast<char>('a' + i % 26)));
  }
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());

  cudf::table_view source_table({col_str});

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(i % 7);
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Below the row gate the batched-memcpy routing must stay OFF: short tables keep the payload
// kernel's chars machinery (the engine split mirrors `strings::detail::gather`, whose threshold
// is 524288 rows).
TEST_F(BatchGatherMixedTest, StringBelowBatchMemcpyGateUsesKernelChars)
{
  constexpr cudf::size_type num_rows{524'287};  // one below the gate
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(
      std::string(static_cast<std::size_t>(1 + (i * 17) % 32), static_cast<char>('a' + i % 26)));
  }
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});

  auto const host_map = shuffled_iota(num_rows, 3);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// At and above the gate every all-short column routes to cub::DeviceMemcpy::Batched: this
// exercises the flat (column, row) address build (shuffled map, nulls, repeated sources), the
// boundary rows (0/1/32-byte lengths), a routed column next to an UNROUTED one (rows above the
// block-copy gate keep the kernel's chars path), and sliced sources. The kernel still owns
// both columns' masks and offsets. Source columns are periodic, so the map exercises every
// length class across the full row count.
TEST_F(BatchGatherMixedTest, StringAtBatchMemcpyGateRoutesRoutedAndUnrouted)
{
  constexpr cudf::size_type num_rows{524'288};  // exactly the gate
  auto make_col = [](int const seed, bool const nullable, bool const short_rows) {
    std::vector<std::string> strings;
    std::vector<bool> valid;
    strings.reserve(num_rows);
    valid.reserve(num_rows);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      // short_rows: sizes in [0, 32] (the block-copy gate covers the whole column, so the
      // average is <= 32 B and the routing predicate fires). Otherwise: every row is above
      // the gate (>= 33 B), so the column must keep the kernel's chars path.
      auto const size = short_rows ? static_cast<std::size_t>((i * seed + 3) % 33)
                                   : static_cast<std::size_t>(33 + (i * seed + 5) % 97);
      strings.push_back(std::string(size, static_cast<char>('a' + (i * seed + i / 7) % 26)));
      valid.push_back(!nullable or i % 6 != 0);
    }
    auto validity = cudf::detail::make_counting_transform_iterator(
      0, [&valid](cudf::size_type i) { return valid[i]; });
    return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), validity);
  };
  // seed 1 all-short: average <= 32 B -> routed to Batched. seed 101 all-long: every row is
  // above the block-copy gate -> stays on the kernel's chars path.
  auto routed_short  = make_col(1, true, true);
  auto unrouted_long = make_col(101, true, false);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), unrouted_long, routed_short});

  auto const host_map = shuffled_iota(num_rows, 11);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Deep below the gate the whole routing machinery must be inert: no routing walk, no CUB
// call, no per-item flag effect — the table gathers exactly like the reference path. This
// pins the staging bypass so a future unconditional host-side addition cannot silently
// reintroduce the small-row overhead.
TEST_F(BatchGatherMixedTest, StringSubThresholdBypassMatchesReference)
{
  constexpr cudf::size_type num_rows{65'536};  // 1/8 of the gate
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(
      std::string(static_cast<std::size_t>(1 + (i * 23) % 40), static_cast<char>('a' + i % 26)));
  }
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows,
    [](cudf::size_type i) { return i * 7; },
    [](cudf::size_type i) { return i % 3 != 0; });

  cudf::table_view source_table({col_str, col_i32->view()});

  auto const host_map = shuffled_iota(num_rows, 5);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherMixedTest, MixedNullableFixedWidthWithNullFreeLists)
{
  // Divergence regression (test-first RED): a fixed-width column WITH nulls next to a nullable
  // list (no nulls) whose elements are nullable (no nulls). Stock gates the mask phase on the
  // WHOLE table, so the list takes the gather_bitmask path and its gatherer-produced child is
  // left alone; a fallback-subset gate sees no possible nulls and fabricates all-valid child
  // masks via recursive set_all_valid_null_masks — an observable child-nullability difference.
  constexpr cudf::size_type num_rows{10};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; });

  // Nullable list column (ALL_VALID mask, zero nulls) whose element child is nullable with
  // zero nulls: offsets give 10 lists over 13 elements, several empty.
  constexpr cudf::size_type num_elements{13};
  auto elements = make_fixed_column<int32_t>(
    num_elements, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{{0, 2, 2, 5, 5, 8, 8, 9, 10, 13, 13}}
      .release();
  auto list_col =
    cudf::make_lists_column(num_rows,
                            std::move(offsets),
                            std::move(elements),
                            0,
                            cudf::create_null_mask(num_rows, cudf::mask_state::ALL_VALID));

  cudf::table_view source_table({col_i32->view(), list_col->view()});
  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// The b3 deferred counts readback: nullable string sources (~1% nulls at the
// default datagen profile, here seeded deterministically) take the
// kernel-written mask path, whose per-column null count is read back AFTER the
// assembly loop instead of blocking the host on the payload kernel. The counts
// reach their consumer via the post-hoc set_null_count patch, so this pins the
// patched result (values AND null masks) against the stock gather — a deferral
// that loses or mis-times the readback produces wrong null counts here.
TEST_F(BatchGatherMixedTest, DeferredStringCountsReadbackMatchesReference)
{
  constexpr cudf::size_type num_rows{65536};
  std::mt19937 rng(7);
  auto valid_of = [&rng](cudf::size_type) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng) >= 0.01;  // ~1% nulls
  };
  auto value_of = [&rng](cudf::size_type) {
    return std::uniform_int_distribution<int64_t>(-1000, 1000)(rng);
  };
  auto col_i32 = make_fixed_column<int32_t>(num_rows, value_of, valid_of);

  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(std::string(1 + i % 24, static_cast<char>('a' + i % 26)));
  }
  auto validity = cudf::detail::make_counting_transform_iterator(0, valid_of);
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_i32->view(), col_str});

  auto const host_map = shuffled_iota(num_rows, 7);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  // The patch must leave the kernel-masked string column's null count exact
  // (not the placeholder 0 the assembly loop constructs with).
  EXPECT_EQ(result->view().column(1).null_count(), reference->view().column(1).null_count());
  EXPECT_GT(reference->view().column(1).null_count(), 0);  // ~1% of 65536 ⇒ ~650
}

// ============================================================================
// §5.6b String seam: tall string columns between the fused fast path and the stock per-column
// fallback. Strings below `kBatchMemcpyRowThreshold` stay fused — short-row shapes ride the
// fused short-string engine (the former seam band [262144, 524288) re-enters the fused path
// with it, where its cells are measured engine wins) and ≥524288-row strings run the
// tall-str1 routed-metadata path; only long-row below-gate shapes take the payload kernel's
// group walk. The tests pin parity (CUDF_TEST_EXPECT_TABLES_EQUAL against a run_reference
// gather) at every boundary.
// ============================================================================

class BatchGatherTallSeamTest : public cudf::test::BaseFixture {};

std::vector<std::string> periodic_strings(cudf::size_type num_rows, int const stride)
{
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(std::string(static_cast<std::size_t>(1 + (i * stride) % 32),
                                  static_cast<char>('a' + i % 26)));
  }
  return strings;
}

// Class 1: below the threshold the N3 fast path stays pinned — a strings-only table (and a
// 2-column small case) gathers exactly like stock.
TEST_F(BatchGatherTallSeamTest, BelowThresholdStringsFastPathPinned)
{
  constexpr cudf::size_type num_rows{65'536};
  auto const strings = periodic_strings(num_rows, 23);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_str});
  auto const host_map = shuffled_iota(num_rows, 5);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherTallSeamTest, SmallTwoColumnFastPathPinned)
{
  constexpr cudf::size_type num_rows{10'000};
  auto const strings = periodic_strings(num_rows, 17);
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows,
    [](cudf::size_type i) { return i * 7; },
    [](cudf::size_type i) { return i % 3 != 0; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 9);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Class 2: at 262144 rows (the band's lower edge) short-average strings stay on the fused
// path — the band re-enters the fused path, so this call is engine-admitted. Parity with
// stock is still the contract. The names stay (they pin the same row counts); only the route
// changed.
TEST_F(BatchGatherTallSeamTest, AtThresholdStringsFallbackPinned)
{
  constexpr cudf::size_type num_rows{262'144};
  auto const strings = periodic_strings(num_rows, 17);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 6 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_str});
  auto const host_map = shuffled_iota(num_rows, 7);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Guard pin for the band interior: at 262144 rows the string column rides the fused path's
// band route — engine-admitted for short-average rows (the band's lower edge is a measured
// engine win cell) — while fixed-width siblings stay on the descriptor path. Parity covers
// correctness; the routing is host-side on `kBatchMemcpyRowThreshold` alone, pinned by this
// exact row count (a re-tightened band exclusion would route this call back to the stock
// fallback and is caught against stock semantics).
TEST_F(BatchGatherTallSeamTest, SeamBandStringStaysFusedPath)
{
  constexpr cudf::size_type num_rows{262'144};
  auto const strings = periodic_strings(num_rows, 17);  // short-average rows: engine-shaped
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 6 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  // A short-average string column beside fixed-width siblings: only the string column's
  // routing is in question; the fixed-width columns stay on the descriptor path either way.
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 7);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// At 300000 rows (band interior) short-average strings stay fused: the band re-enters the
// fused path, so the call is engine-admitted. Parity with stock is the contract.
TEST_F(BatchGatherTallSeamTest, AboveThresholdStringsFallbackPinned)
{
  constexpr cudf::size_type num_rows{300'000};
  auto const strings = periodic_strings(num_rows, 13);
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 3);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Class 3: off-by-one boundary pin — 262143 and 262144 rows are both fused (engine-admitted
// for short-average rows) and must BOTH match stock; the pair fails if the comparison is off
// by one.
TEST_F(BatchGatherTallSeamTest, BoundaryBelowByOneUsesFastPath)
{
  constexpr cudf::size_type num_rows{262'143};
  auto const strings = periodic_strings(num_rows, 19);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 5 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 21);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Class 4a: NULLIFY policy with out-of-bounds entries in the band — exercises the fused
// string build's NULLIFY leg next to fused fixed-width columns in the same call.
TEST_F(BatchGatherTallSeamTest, NullifyOutOfBoundsAtThreshold)
{
  constexpr cudf::size_type num_rows{262'144};
  auto const strings = periodic_strings(num_rows, 11);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 7 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  std::mt19937 rng(13);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    // Mostly in-bounds; every 64th entry is out-of-bounds (source row count + offset).
    host_map.push_back(i % 64 == 0 ? num_rows + static_cast<cudf::size_type>(rng() % 1024)
                                   : static_cast<cudf::size_type>(rng() % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  auto reference = run_reference(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Class 4c: an all-null string source in the band — the fused build's skip_data inline
// build must match the stock path's null handling, parity on both sides of the gate.
TEST_F(BatchGatherTallSeamTest, AllNullStringsAtThreshold)
{
  constexpr cudf::size_type num_rows{262'144};
  auto const strings = periodic_strings(num_rows, 29);
  auto all_invalid =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type) { return false; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), all_invalid);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 15);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Class 5: negative-index policy ALLOWED in the band — the fused string resolve must take
// the same wrap transform-iterator branch stock does.
TEST_F(BatchGatherTallSeamTest, NegativeIndexWrapAtThreshold)
{
  constexpr cudf::size_type num_rows{262'144};
  auto const strings = periodic_strings(num_rows, 31);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 8 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    // Every 8th entry is negative (wraps to num_src_rows + neg), the rest in-bounds.
    host_map.push_back(i % 8 == 0 ? -(1 + static_cast<cudf::size_type>(i / 8) % num_rows)
                                  : static_cast<cudf::size_type>((i * 31) % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::ALLOWED);
  auto reference = run_reference(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::ALLOWED);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Class 6: mixed-table composition in the band (m1 style) — strings fused, lists unchanged
// (already fallback), fixed-width fused; whole-table equality vs stock.
TEST_F(BatchGatherTallSeamTest, MixedCompositionAtThreshold)
{
  constexpr cudf::size_type num_rows{262'144};
  auto const strings = periodic_strings(num_rows, 37);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 9 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows,
    [](cudf::size_type i) { return i * 3; },
    [](cudf::size_type i) { return i % 2 != 0; });

  // One list<int32> element per row (variable length), non-nullable parent.
  std::vector<cudf::size_type> offsets;
  offsets.reserve(num_rows + 1);
  offsets.push_back(0);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    offsets.push_back(offsets.back() + static_cast<cudf::size_type>(1 + i % 4));
  }
  auto elements = make_fixed_column<int32_t>(
    offsets.back(), [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto col_list = cudf::make_lists_column(
    num_rows,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(offsets.begin(), offsets.end())
      .release(),
    std::move(elements),
    0,
    cudf::create_null_mask(num_rows, cudf::mask_state::ALL_VALID));

  cudf::table_view source_table({col_i32->view(), col_str, col_list->view()});
  auto const host_map = shuffled_iota(num_rows, 17);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Class 7: empty map with a strings table — the gate is off at num_rows == 0; the zero-row
// empty-column branch of the fast path still handles it (stock parity).
TEST_F(BatchGatherTallSeamTest, EmptyMapStrings)
{
  auto strings = cudf::test::strings_column_wrapper{{"a", "", "ccc", "dd"}};
  cudf::table_view source_table({strings});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{};

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Class 8: sliced (offset != 0) string source at the threshold — the stock path's addressing
// is offset-safe by construction; guards any future gate-time offset assumption.
TEST_F(BatchGatherTallSeamTest, SlicedStringsAtThreshold)
{
  constexpr cudf::size_type num_rows{262'144};
  auto const strings = periodic_strings(num_rows + 1000, 41);
  cudf::test::strings_column_wrapper full_col(strings.begin(), strings.end());
  auto sliced       = cudf::slice(full_col, {500, 500 + num_rows});
  auto col_i32_full = make_fixed_column<int32_t>(
    num_rows + 1000, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto col_i32 = cudf::slice(col_i32_full->view(), {500, 500 + num_rows});

  cudf::table_view source_table({col_i32[0], sliced[0]});
  auto const host_map = shuffled_iota(num_rows, 19);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// The N3 fast path's destination offsets children and chars buffers are carved from ONE
// call-wide arena resource. The carve resource must stay correct when the call's locals die:
// each returned column owns value copies of the shared resource, so the arena lives exactly
// as long as the columns do. This test runs the gather with a tracking output resource whose
// ONLY reference dies before the result is inspected — if a carved buffer still referenced
// the call-scope arena instead of a shared copy, arena teardown would free live column
// memory (or an ownership gap would leak the block) and the readback/comparison below would
// catch it.
TEST_F(BatchGatherTallSeamTest, CarvedStringsOutliveCallScopeArena)
{
  constexpr cudf::size_type num_rows{65'536};
  auto const strings = periodic_strings(num_rows, 23);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);
  // An all-short column routes its chars through the carve arena on the fast path; the
  // kernel-masked mask (nullable source) is planned separately and stays a stock allocation.
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 == 0; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 5);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  // The tracking resource is a block-scope local: it is destroyed at the end of this loop
  // iteration, well before the column comparison runs. Its counters also prove the fast path
  // actually allocated (the carve arena + masks) — a 0-allocation result would mean the test
  // never exercised the carved path.
  auto const result = [&] {
    auto output_mr = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
    auto const reference = run_reference(source_table, gather_map);  // stock path, default resource
    auto gathered =
      cudf::detail::batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::NOT_ALLOWED,
                                 cudf::get_default_stream(),
                                 cudf::memory_resources{rmm::device_async_resource_ref{output_mr}});
    CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), gathered->view());
    auto const arena_allocations = output_mr.get_allocations_counter().value;
    EXPECT_GT(arena_allocations, 0);
    return gathered;
  }();

  // The arena resource is gone here; the carved columns must still read back correctly.
  CUDF_TEST_EXPECT_TABLES_EQUAL(run_reference(source_table, gather_map)->view(), result->view());
}

// Both carve arenas' shared-ownership lifetime: a MIXED batch (strings + fixed-width columns,
// the fixed-width class including plain leaves, a struct's leaf, and a string-keyed dictionary
// whose deferred assembly carves its parent mask) must read back correctly AFTER the call's
// tracking output resource is destroyed. Each carved buffer stores a value copy of its
// arena's shared resource, so a buffer that instead referenced a call-scope arena would read
// freed memory in the post-scope comparison below.
TEST_F(BatchGatherTallSeamTest, CarvedMixedColumnsOutliveCallScopeArena)
{
  constexpr cudf::size_type num_rows{65'536};
  auto const strings = periodic_strings(num_rows, 23);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 == 0; });
  // A struct leaf + a string-keyed dictionary exercise the struct-node mask carve and the
  // deferred-assembly dictionary carve (the parent mask rides the fw arena through the
  // staged record).
  auto col_i64 = make_fixed_column<int64_t>(
    num_rows,
    [](cudf::size_type i) { return 2 * i; },
    [](cudf::size_type i) { return i % 3 != 0; });
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(std::move(col_i64));
  cudf::test::structs_column_wrapper col_struct{std::move(struct_children)};
  std::vector<std::string> dict_values(num_rows);
  std::generate(dict_values.begin(), dict_values.end(), [n = 0]() mutable {
    return std::string(1, static_cast<char>('a' + (n++) % 3));
  });
  auto dict_validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 5 != 0; });
  cudf::test::dictionary_column_wrapper<std::string> col_dict(
    dict_values.begin(), dict_values.end(), dict_validity);

  cudf::table_view source_table({col_i32->view(), col_str, col_struct, col_dict});
  auto const host_map = shuffled_iota(num_rows, 5);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  // The tracking resource is a block-scope local: it is destroyed at the end of this loop
  // iteration, well before the comparison runs. Its counters prove both carve arenas and the
  // stock string-mask allocations actually happened — a 0-allocation result would mean the
  // test never exercised the carved paths.
  auto const result = [&] {
    auto output_mr = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
    auto const reference = run_reference(source_table, gather_map);  // stock path, default resource
    auto gathered =
      cudf::detail::batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::NOT_ALLOWED,
                                 cudf::get_default_stream(),
                                 cudf::memory_resources{rmm::device_async_resource_ref{output_mr}});
    CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), gathered->view());
    auto const arena_allocations = output_mr.get_allocations_counter().value;
    EXPECT_GT(arena_allocations, 0);
    return gathered;
  }();

  // Both arena resources are gone here; the carved columns must still read back correctly.
  CUDF_TEST_EXPECT_TABLES_EQUAL(run_reference(source_table, gather_map)->view(), result->view());
}

// ============================================================================
// §5.6c String routed-metadata parity: tall short-row calls route per-column chars to
// cub::DeviceMemcpy::Batched and their offsets/masks/counts to the row-parallel
// string_routed_metadata_kernel. Rows above 524288 exercise the routed path; below the gate
// the fused short-string engine serves short-row shapes, so a 262144-row seam call is
// engine-admitted rather than on the fallback. Every test compares exactly
// against a run_reference gather.
// ============================================================================

// The staging carve arenas (device landing buffers for the string path's H2D uploads and the
// descriptor upload) are temp-scoped: every carved buffer must be finished by the launch block
// that enqueued it BEFORE the arena's stream-ordered free fires. Sources are SLICED (the
// staging sizes derive from the sliced views' sizes, not the parent's) and one string source is
// EMPTY (zero rows: its upload is skipped by the live filter, so the carve must still size
// correctly) — the envelope's parity requirements for the carve-bound scratch. The call runs
// under a block-scoped tracked output resource and compares against the stock gather both
// in-scope and after the tracking resource dies.
TEST_F(BatchGatherTallSeamTest, StagingCarveSlicedSourceParity)
{
  constexpr cudf::size_type parent_size{1000};
  constexpr cudf::size_type slice_begin{100};
  constexpr cudf::size_type slice_end{600};
  constexpr cudf::size_type num_rows{500};

  auto strings_parent = periodic_strings(parent_size, 23);
  auto str_validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 0; });
  cudf::test::strings_column_wrapper str_parent(
    strings_parent.begin(), strings_parent.end(), str_validity);
  auto sliced_str = cudf::slice(str_parent, {slice_begin, slice_end});
  ASSERT_EQ(sliced_str.size(), 1);
  // A column of EMPTY strings exercises the zero-size trap: every row's byte total is 0, so
  // the routing walk sends it to the unrouted cohort and no chars copy exists for it — the
  // carve must still size correctly (a zero-length column contributes only its staging item).
  std::vector<std::string> const empty_strings(num_rows, "");
  cudf::test::strings_column_wrapper empty_str(empty_strings.begin(), empty_strings.end());
  auto col_i32 = make_fixed_column<int32_t>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return i; },
    [](cudf::size_type i) { return i % 2 == 0; });
  auto col_i64 = make_fixed_column<int64_t>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return 2 * i; },
    [](cudf::size_type i) { return i % 3 != 0; });
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(std::move(col_i64));
  cudf::test::structs_column_wrapper col_struct{std::move(struct_children)};

  std::vector<std::string> dict_values(slice_end - slice_begin);
  std::generate(dict_values.begin(), dict_values.end(), [n = 0]() mutable {
    return std::string(1, static_cast<char>('a' + (n++) % 3));
  });
  auto dict_validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 5 != 0; });
  cudf::test::dictionary_column_wrapper<std::string> col_dict(
    dict_values.begin(), dict_values.end(), dict_validity);

  cudf::table_view source_table({sliced_str[0], empty_str, col_i32->view(), col_struct, col_dict});
  auto const host_map = shuffled_iota(num_rows, 5);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto const result = [&] {
    auto output_mr = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
    auto const reference = run_reference(source_table, gather_map);  // stock path, default resource
    auto gathered =
      cudf::detail::batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::NOT_ALLOWED,
                                 cudf::get_default_stream(),
                                 cudf::memory_resources{rmm::device_async_resource_ref{output_mr}});
    CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), gathered->view());
    auto const arena_allocations = output_mr.get_allocations_counter().value;
    EXPECT_GT(arena_allocations, 0);
    return gathered;
  }();

  CUDF_TEST_EXPECT_TABLES_EQUAL(run_reference(source_table, gather_map)->view(), result->view());
}

class StringRoutedMetadataParity : public cudf::test::BaseFixture {};

// Tall single column: the whole table routes (average gathered row 16 B <= 32 B), the
// routed kernel owns every offsets/mask/count byte, and CUB owns the chars.
TEST_F(StringRoutedMetadataParity, TallSingleColumnAllRouted)
{
  constexpr cudf::size_type num_rows{524'288};
  auto const strings = periodic_strings(num_rows, 23);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_str});
  auto const host_map = shuffled_iota(num_rows, 5);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// A genuinely long-row generator: row i has 512 + (i % 64) bytes — the average is far
// above the 32 B routing gate, so every column built from it stays unrouted (periodic_strings
// alone cannot: any stride mod 32 cycles all residue classes, giving ~16 B rows).
std::vector<std::string> long_strings(cudf::size_type num_rows)
{
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(
      std::string(static_cast<std::size_t>(512 + i % 64), static_cast<char>('a' + i % 26)));
  }
  return strings;
}

// Mixed routed + unrouted columns in both orders: one short (routed) and one long
// (unrouted) string column, each cohort's metadata written by its own engine.
TEST_F(StringRoutedMetadataParity, MixedRoutedUnroutedShortFirst)
{
  constexpr cudf::size_type num_rows{524'288};
  auto const short_strings = periodic_strings(num_rows, 7);  // avg ~16 B -> routed
  auto const long_column   = long_strings(num_rows);         // avg ~543 B -> unrouted
  auto validity_short =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 3 != 0; });
  auto validity_long =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 5 != 0; });
  cudf::test::strings_column_wrapper col_short(
    short_strings.begin(), short_strings.end(), validity_short);
  cudf::test::strings_column_wrapper col_long(
    long_column.begin(), long_column.end(), validity_long);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_short, col_long});
  auto const host_map = shuffled_iota(num_rows, 11);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(StringRoutedMetadataParity, MixedRoutedUnroutedLongFirst)
{
  constexpr cudf::size_type num_rows{524'288};
  auto const short_strings = periodic_strings(num_rows, 13);
  auto const long_column   = long_strings(num_rows);
  auto validity_short =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 7 != 0; });
  cudf::test::strings_column_wrapper col_short(
    short_strings.begin(), short_strings.end(), validity_short);
  cudf::test::strings_column_wrapper col_long(long_column.begin(), long_column.end());

  cudf::table_view source_table({col_long, col_short});
  auto const host_map = shuffled_iota(num_rows, 3);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// The N+1 sentinel edge and num_rows % 256 tails: 524288 is divisible by 256, so a second
// tall size off the tile boundary exercises the tail tile (524288 + 7 rows) and the
// unique-sentinel ownership at the last row.
TEST_F(StringRoutedMetadataParity, TailTileSentinelOwnership)
{
  constexpr cudf::size_type num_rows{524'295};
  auto const strings = periodic_strings(num_rows, 31);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 6 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 17);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// NULLIFY with out-of-bounds entries on a routed call: dead rows resolve to no payload and
// must contribute zero bytes (offsets delta 0) and zero validity — the routed kernel's mask
// ballots and count reduction must match the stock NULLIFY null_count exactly.
TEST_F(StringRoutedMetadataParity, NullifyDeadRowsRouted)
{
  constexpr cudf::size_type num_rows{524'288};
  auto const strings = periodic_strings(num_rows, 19);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 8 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  std::mt19937 rng(23);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(i % 32 == 0 ? num_rows + static_cast<cudf::size_type>(rng() % 4096)
                                   : static_cast<cudf::size_type>(rng() % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  auto reference = run_reference(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Negative-index wrap on a routed call: the routed kernel's resolve must reproduce the
// kernel chars path's normalization exactly (the same resolve_gathered_row call).
TEST_F(StringRoutedMetadataParity, NegativeIndexWrapRouted)
{
  constexpr cudf::size_type num_rows{524'288};
  auto const strings = periodic_strings(num_rows, 37);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 9 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(i % 8 == 0 ? -(1 + static_cast<cudf::size_type>(i / 8) % num_rows)
                                  : static_cast<cudf::size_type>((i * 31) % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::ALLOWED);
  auto reference = run_reference(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::ALLOWED);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// A tall all-empty (every gathered row zero bytes but valid) column: bytes == 0, so the
// column stays UNROUTED (routing requires bytes > 0), and the zero-total scratch path must
// produce an all-zero offsets child with the sentinel — same behavior as before this
// branch, pinned here because the cohort split changed the kernel's flat domain.
TEST_F(StringRoutedMetadataParity, AllEmptyRowsZeroTotalUnrouted)
{
  constexpr cudf::size_type num_rows{524'288};
  std::vector<std::string> empty_strings(num_rows, "");
  cudf::test::strings_column_wrapper col_str(empty_strings.begin(), empty_strings.end());

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 29);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Empty SOURCE column beside a routed one: the empty column's offsets complete host-side
// (no kernel slot) while the tall column routes; the cohort split must not disturb the
// empty-source completion. A zero-ROW source column cannot legally share a table with
// 524'288-row columns (table_view rejects mixed sizes), so "empty source" here is a
// zero-BYTES source: every gathered row is empty and its total (scratch[N]) is 0.
TEST_F(StringRoutedMetadataParity, EmptySourceBesideRouted)
{
  constexpr cudf::size_type num_rows{524'288};
  auto const strings = periodic_strings(num_rows, 41);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);
  std::vector<std::string> all_empty(num_rows, "");
  cudf::test::strings_column_wrapper col_empty(all_empty.begin(), all_empty.end());

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_empty, col_str});
  auto const host_map = shuffled_iota(num_rows, 33);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Sliced tall source on a routed call: the routed kernel indexes the source through the
// item's mask_bit_offset/offsets-view offset exactly like the CUB iterators do.
TEST_F(StringRoutedMetadataParity, SlicedSourceRouted)
{
  constexpr cudf::size_type num_rows{524'288};
  auto const strings = periodic_strings(num_rows + 1000, 43);
  cudf::test::strings_column_wrapper full_col(strings.begin(), strings.end());
  auto sliced       = cudf::slice(full_col, {700, 700 + num_rows});
  auto col_i32_full = make_fixed_column<int32_t>(
    num_rows + 1000, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto col_i32 = cudf::slice(col_i32_full->view(), {700, 700 + num_rows});

  cudf::table_view source_table({col_i32[0], sliced[0]});
  auto const host_map = shuffled_iota(num_rows, 37);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// The routing gate band below the metadata engine's reach: a 262143-row SHORT-average call
// rides the fused short-string engine (below the gate the engine replaces the group walk,
// including the whole former seam band); this parity pin stays so any future gate
// re-tightening that moves the cell back to the stock fallback is caught against stock
// semantics.
TEST_F(StringRoutedMetadataParity, GateBandStaysFallback)
{
  constexpr cudf::size_type num_rows{262'143};
  auto const strings = periodic_strings(num_rows, 13);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 5 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 21);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// ============================================================================
// §5.6c Fused short-string engine (below-gate warp-tile path): parity against the stock
// gather. The engine replaces the payload kernel for below-gate short-average calls —
// including the whole former seam band [262144, 524288), whose cells are measured engine
// wins — while ≥524288-row strings run the routed-metadata path. Every test here compares
// against `run_reference` so the engine's offsets/masks/counts/chars must match stock exactly.
// ============================================================================

class StringFusedEngineParity : public cudf::test::BaseFixture {};

// The band-cell parity class (the measured win cell class): short-average rows at a large row
// count — one warp covers 32 rows of one column; tile tails (num_rows % 32 != 0) exercise the
// ballot's dead-lane path. 262143 pins the band's lower edge minus one; the band interior is
// pinned by `SeamBandStringStaysFusedPath` above.
TEST_F(StringFusedEngineParity, SeamBandShortRowsSingleColumn)
{
  constexpr cudf::size_type num_rows{262'143};  // 1 mod 32: one tail tile
  auto const strings = periodic_strings(num_rows, 17);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 6 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_str});
  auto const host_map = shuffled_iota(num_rows, 7);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Multi-column: warps map flat slots across column tiles; the sentinel (N+1) writer must be
// per-column and the masks/counts per-column too. Mixed widths across columns. Row count just
// under the seam band's lower edge so the call stays engine-admitted.
TEST_F(StringFusedEngineParity, MultiColumnTilesAndSentinels)
{
  constexpr cudf::size_type num_rows{262'142};            // 262142 % 32 != 0: one tail tile
  auto const strings_a = periodic_strings(num_rows, 7);   // avg ~16 B
  auto const strings_b = periodic_strings(num_rows, 31);  // avg ~16 B, different stride
  auto validity_a =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 3 != 0; });
  auto validity_b =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 0; });
  cudf::test::strings_column_wrapper col_a(strings_a.begin(), strings_a.end(), validity_a);
  cudf::test::strings_column_wrapper col_b(strings_b.begin(), strings_b.end(), validity_b);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_a, col_b});
  auto const host_map = shuffled_iota(num_rows, 11);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Long-row safety inside an engine call: most rows short, a few rows far above the block-copy
// gate — the average must stay at or under 32 B so the engine is admitted, and the long rows
// must ride the full-warp copy path byte-exactly (the 4-lane subgroup loop skips them by the
// row_len gate). 31 short rows (<= 12 B) plus one 300 B row average ~29 B <= 32 B per tile.
TEST_F(StringFusedEngineParity, ShortAverageColumnWithLongOutliers)
{
  constexpr cudf::size_type num_rows{262'143};
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    // Every 32nd row is 300+ B; the rest are 1-12 B. Average = (31 * 6 + 320) / 32 ~ 15 B.
    strings.push_back(
      std::string(i % 32 == 7 ? std::size_t{300} + (i / 32) % 50 : std::size_t{1} + (i * 7) % 12,
                  static_cast<char>('a' + i % 26)));
  }
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 5 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 23);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Exact engine-shape boundary pins: every row length class the subgroup loop must handle —
// 0 (valid+empty), 1, 3, 4, 16 (subgroup boundary), 17 (full-warp boundary), 32 (gate), 33
// (above gate) bytes. The engine is admitted (average <= 32 B) and must be byte-exact on all.
TEST_F(StringFusedEngineParity, RowLengthClassBoundaries)
{
  constexpr cudf::size_type num_rows{8 * 16};  // 128 rows: 16 of each length class
  std::array<cudf::size_type, 8> const lengths{{0, 1, 3, 4, 16, 17, 32, 33}};
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(std::string(static_cast<std::size_t>(lengths[i % lengths.size()]),
                                  static_cast<char>('a' + i % 26)));
  }
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 7 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 29);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// NULLIFY with out-of-bounds entries on an engine call: dead rows contribute zero bytes and
// zero validity through the engine's ballots exactly as the payload kernel's mask phase did.
TEST_F(StringFusedEngineParity, NullifyDeadRowsEngine)
{
  constexpr cudf::size_type num_rows{262'143};
  auto const strings = periodic_strings(num_rows, 19);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 8 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  std::mt19937 rng(23);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(i % 32 == 0 ? num_rows + static_cast<cudf::size_type>(rng() % 4096)
                                   : static_cast<cudf::size_type>(rng() % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  auto reference = run_reference(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Negative-index wrap on an engine call: the engine's resolve must reproduce the kernel
// chars path's normalization exactly (the same resolve_gathered_row call).
TEST_F(StringFusedEngineParity, NegativeIndexWrapEngine)
{
  constexpr cudf::size_type num_rows{262'143};
  auto const strings = periodic_strings(num_rows, 37);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 9 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(i % 8 == 0 ? -(1 + static_cast<cudf::size_type>(i / 8) % num_rows)
                                  : static_cast<cudf::size_type>((i * 31) % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::ALLOWED);
  auto reference = run_reference(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::ALLOWED);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// All-empty rows: every gathered row zero bytes but valid — the engine's subgroup loop
// skips them (len 0) while offsets must be all-zero with the sentinel, masks all-valid.
TEST_F(StringFusedEngineParity, AllEmptyRowsEngine)
{
  constexpr cudf::size_type num_rows{262'143};
  std::vector<std::string> empty_strings(num_rows, "");
  cudf::test::strings_column_wrapper col_str(empty_strings.begin(), empty_strings.end());

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 31);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Sliced tall source on an engine call: the engine indexes the source through the item's
// mask_bit_offset/offsets-view offset exactly like the group walk does.
TEST_F(StringFusedEngineParity, SlicedSourceEngine)
{
  constexpr cudf::size_type num_rows{262'143};
  auto const strings = periodic_strings(num_rows + 1000, 43);
  cudf::test::strings_column_wrapper full_col(strings.begin(), strings.end());
  auto sliced       = cudf::slice(full_col, {700, 700 + num_rows});
  auto col_i32_full = make_fixed_column<int32_t>(
    num_rows + 1000, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto col_i32 = cudf::slice(col_i32_full->view(), {700, 700 + num_rows});

  cudf::table_view source_table({col_i32[0], sliced[0]});
  auto const host_map = shuffled_iota(num_rows, 37);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Repeated map entries on an engine call: the same source row gathered many times — the
// subgroup copies must not alias (each row's bytes land at its own scratch offset).
TEST_F(StringFusedEngineParity, RepeatedSourceRowsEngine)
{
  constexpr cudf::size_type num_rows{262'143};
  auto const strings = periodic_strings(num_rows, 11);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});

  // Repeat every 32nd source row at every destination row of its tile — dense aliasing.
  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(i % 2 == 0 ? (i / 32) * 32 % num_rows : static_cast<cudf::size_type>(i));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// A below-gate long-average column must NOT be engine-admitted (the short-average check
// rejects it) and must still match stock — the payload kernel's group walk serves it. The
// 262143-row count keeps the call fused (below the routing gate); the tall seam tests pin
// the ≥524288 side.
TEST_F(StringFusedEngineParity, BelowGateLongRowsKeepPayloadKernel)
{
  constexpr cudf::size_type num_rows{262'143};
  auto const strings = long_strings(num_rows);  // avg ~543 B: average above the 32 B gate
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 5 != 0; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str});
  auto const host_map = shuffled_iota(num_rows, 21);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// ============================================================================
// §5.6d Lean CUB virtual iterators: at the batched-memcpy gate the routed columns' chars ride
// cub::DeviceMemcpy::Batched over virtual iterators whose destination/size come from the
// scanned prefixes and whose resolve chain runs once per buffer. These shapes are ≥524288
// rows, so the routed-metadata path serves them and the lean iterators are LIVE: every test
// compares the gate-exact shape against the stock gather (the CUB engine is stock's own
// engine at these shapes, so CUDF_TEST_EXPECT_TABLES_EQUAL is exact).
// ============================================================================

class StringLeanIteratorsParity : public cudf::test::BaseFixture {};

// Column factory for gate shapes: rows are all at or under the block-copy gate (so a column
// whose total passes the routing predicate is routed to CUB), with sizes in [min_len,
// min_len + spread] and an optional per-row validity pattern.
cudf::test::strings_column_wrapper gate_shape_strings(cudf::size_type num_rows,
                                                      int const seed,
                                                      bool const nullable,
                                                      int const min_len,
                                                      int const spread)
{
  std::vector<std::string> strings;
  std::vector<bool> valid;
  strings.reserve(num_rows);
  valid.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(std::string(static_cast<std::size_t>(min_len + (i * seed + i / 97) % spread),
                                  static_cast<char>('a' + (i * seed) % 26)));
    valid.push_back(not nullable or i % 5 != 0);
  }
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [&valid](cudf::size_type i) { return valid[i]; });
  return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), validity);
}

// The core lean-iterator shape: two routed short columns, shuffled map with repeated indices
// (duplicate source rows land under distinct destinations), nulls in both columns.
TEST_F(StringLeanIteratorsParity, RoutedAllShortStringsParity)
{
  constexpr cudf::size_type num_rows{524'288};  // exactly the gate
  auto col_a = gate_shape_strings(num_rows, 3, true, 0, 33);
  auto col_b = gate_shape_strings(num_rows, 11, true, 0, 33);

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  std::mt19937 rng(5);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    // Shuffled with deliberate repeats: every 3rd row re-reads the previous destination's
    // source (the prefix deltas of repeated rows are identical).
    host_map.push_back(i % 3 == 0 and i > 0 ? host_map[i - 1]
                                            : static_cast<cudf::size_type>(rng() % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({col_a, col_b});
  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Routed column with a dense band of EMPTY valid rows interleaved with short rows: the
// prefix delta is 0 for empty rows, so their src/dst operators return nullptr (never
// dereferenced) while the column's nonzero total keeps dst_chars allocated for the live
// rows — the zero-delta and resolve gates must agree with stock's outcome.
TEST_F(StringLeanIteratorsParity, RoutedZeroSizeRowsParity)
{
  constexpr cudf::size_type num_rows{524'288};
  std::vector<std::string> strings;
  std::vector<bool> valid;
  strings.reserve(num_rows);
  valid.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    // Every other row empty; the rest 1-8 bytes: average well under the gate.
    strings.push_back(i % 2 == 0 ? std::string{}
                                 : std::string(static_cast<std::size_t>(1 + i % 8),
                                               static_cast<char>('a' + i % 26)));
    valid.push_back(i % 7 != 0);
  }
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [&valid](cudf::size_type i) { return valid[i]; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_str});
  auto const host_map = shuffled_iota(num_rows, 13);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// A routed column whose every OTHER row is DEAD under NULLIFY (out-of-bounds map)
// interleaves zero-delta dead rows with live rows: the dead rows' prefix deltas are 0
// (their src/dst operators return nullptr, never dereferenced) while the live rows ride
// the single resolve chain — all against stock's identical outcome.
// (An ALL-dead routed column is impossible by construction: an all-dead column's total is
// 0, which fails the routing predicate and keeps the column out of the CUB domain.)
TEST_F(StringLeanIteratorsParity, RoutedInterleavedDeadRowsParity)
{
  constexpr cudf::size_type num_rows{524'288};
  auto col_a = gate_shape_strings(num_rows, 7, true, 0, 33);
  auto col_b = gate_shape_strings(num_rows, 9, false, 0, 33);

  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(i % 2 == 0 ? static_cast<cudf::size_type>(i)
                                  : num_rows + static_cast<cudf::size_type>(i % 991));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({col_a, col_b});
  auto result    = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  auto reference = run_reference(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Duplicate-map amplification of ONE source row across the whole routed column: every
// destination row carries the same prefix delta, and the single resolve chain reads the
// same source span once per buffer.
TEST_F(StringLeanIteratorsParity, RoutedDuplicatesParity)
{
  constexpr cudf::size_type num_rows{524'288};
  cudf::test::strings_column_wrapper col_src{"nine-bytes"};

  std::vector<cudf::size_type> host_map(num_rows, 0);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({col_src});
  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Sliced source at the gate: the offsets iterator carries the view offset while chars are
// addressed from the unsliced base — the lean iterators inherit that addressing unchanged.
TEST_F(StringLeanIteratorsParity, RoutedSlicedSourceParity)
{
  constexpr cudf::size_type num_rows{524'288};
  auto const full = gate_shape_strings(num_rows + 1000, 17, true, 0, 33);
  auto sliced     = cudf::slice(full, {500, 500 + num_rows});

  cudf::table_view source_table({sliced[0]});
  auto const host_map = shuffled_iota(num_rows, 23);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// NULLIFY out-of-bounds and negative-index-ALLOWED maps over routed columns: both policy
// branches run inside the single resolve of the lean src operator.
TEST_F(StringLeanIteratorsParity, RoutedNullifyAndNegativeParity)
{
  constexpr cudf::size_type num_rows{524'288};
  auto col_str = gate_shape_strings(num_rows, 21, true, 0, 33);

  // NULLIFY: every 16th entry out of bounds.
  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  std::mt19937 rng(29);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(i % 16 == 0 ? num_rows + static_cast<cudf::size_type>(rng() % 2048)
                                   : static_cast<cudf::size_type>(rng() % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> map_nullify(host_map.begin(),
                                                                      host_map.end());
  cudf::table_view source_table({col_str});
  auto result_nullify =
    run_batch_gather(source_table, map_nullify, cudf::out_of_bounds_policy::NULLIFY);
  auto reference_nullify =
    run_reference(source_table, map_nullify, cudf::out_of_bounds_policy::NULLIFY);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference_nullify->view(), result_nullify->view());

  // Negative ALLOWED: every 8th entry wraps below zero.
  std::vector<cudf::size_type> host_map_neg;
  host_map_neg.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map_neg.push_back(i % 8 == 0 ? -(1 + static_cast<cudf::size_type>(i / 8) % num_rows)
                                      : static_cast<cudf::size_type>((i * 37) % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> map_neg(host_map_neg.begin(),
                                                                  host_map_neg.end());
  auto result_neg    = run_batch_gather(source_table,
                                     map_neg,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     cudf::negative_index_policy::ALLOWED);
  auto reference_neg = run_reference(source_table,
                                     map_neg,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     cudf::negative_index_policy::ALLOWED);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference_neg->view(), result_neg->view());
}

// Routed (short rows) next to UNROUTED (long rows) at the gate: CUB serves one column's
// chars through the lean iterators while the payload kernel serves the other's — the
// per-item engine split rides the same items table both engines read.
TEST_F(StringLeanIteratorsParity, RoutedMixedRoutedUnroutedParity)
{
  constexpr cudf::size_type num_rows{524'288};
  auto col_short = gate_shape_strings(num_rows, 2, true, 0, 33);   // avg <= 32 B: routed
  auto col_long  = gate_shape_strings(num_rows, 4, true, 40, 17);  // avg > 32 B: kernel

  cudf::table_view source_table({col_short, col_long});
  auto const host_map = shuffled_iota(num_rows, 31);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// Boundary triple around the batched-memcpy gate: one below, at, and one above — the lean
// iterators must behave exactly like the pre-lean engine at and above the gate and stay
// out of the picture below it.
TEST_F(StringLeanIteratorsParity, RoutedBoundaryRowsParity)
{
  for (auto const num_rows :
       {cudf::size_type{524'287}, cudf::size_type{524'288}, cudf::size_type{524'289}}) {
    SCOPED_TRACE("num_rows=" + std::to_string(num_rows));
    auto col_str = gate_shape_strings(num_rows, 8, true, 0, 33);
    auto col_i32 = make_fixed_column<int32_t>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

    cudf::table_view source_table({col_i32->view(), col_str});
    auto const host_map = shuffled_iota(num_rows, 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                       host_map.end());

    auto result    = run_batch_gather(source_table, gather_map);
    auto reference = run_reference(source_table, gather_map);
    CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  }
}

// Separate temp/output resources: the CUB env takes the temporary resource (the engine's
// scratch) while every destination buffer rides the output resource; allocations must
// release with the call. The validator's oracle is a stock gather built on the SETUP
// resource (so the failing current resource never sees it), not the source column — a
// random map gathers a permutation, not the source itself.
TEST_F(StringLeanIteratorsParity, RoutedSeparateResourcesParity)
{
  auto harness = cudf::test::memory_resource_test_harness{};
  auto stream  = cudf::test::get_default_stream();
  auto setup   = cudf::memory_resources{harness.setup_mr(), harness.setup_mr()};

  constexpr cudf::size_type num_rows{524'288};
  std::vector<cudf::size_type> host_map;
  host_map.reserve(num_rows);
  std::mt19937 rng(11);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host_map.push_back(static_cast<cudf::size_type>(rng() % num_rows));
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(
    host_map.begin(), host_map.end(), stream, setup);
  auto source_column = gate_shape_strings(num_rows, 6, true, 0, 33);
  cudf::table_view source_table({source_column});

  // Stock oracle: DONT_CHECK + NOT_ALLOWED, same as the batch_gather call below.
  auto expected = run_reference(source_table, gather_map);

  cudf::test::expect_api_uses_memory_resources(
    harness,
    [&](cudf::memory_resources mr) {
      return cudf::detail::batch_gather(source_table,
                                        gather_map,
                                        cudf::out_of_bounds_policy::DONT_CHECK,
                                        cudf::negative_index_policy::NOT_ALLOWED,
                                        stream,
                                        mr);
    },
    [](auto const& result) { return result->alloc_size(); },
    [&](auto const& result) { CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view()); },
    {cudf::test::output_allocation_expectation::AT_LEAST_LIVE,
     cudf::test::temporary_allocation_expectation::SOME},
    stream);
}

// §5.7 Randomized differential against the stock gather (reference oracle).
// ============================================================================

class BatchGatherRandomizedTest : public cudf::test::BaseFixture {};

std::vector<cudf::type_id> const& random_candidate_types()
{
  static std::vector<cudf::type_id> const candidates{cudf::type_id::INT8,
                                                     cudf::type_id::INT16,
                                                     cudf::type_id::INT32,
                                                     cudf::type_id::INT64,
                                                     cudf::type_id::UINT32,
                                                     cudf::type_id::FLOAT32,
                                                     cudf::type_id::FLOAT64,
                                                     cudf::type_id::BOOL8,
                                                     cudf::type_id::DECIMAL32,
                                                     cudf::type_id::DECIMAL64,
                                                     cudf::type_id::DECIMAL128,
                                                     cudf::type_id::TIMESTAMP_NANOSECONDS,
                                                     cudf::type_id::DURATION_MICROSECONDS,
                                                     cudf::type_id::STRING};
  return candidates;
}

template <typename T>
std::unique_ptr<cudf::column> make_random_fixed(cudf::size_type num_rows,
                                                double null_probability,
                                                std::mt19937& rng)
{
  auto value_of = [&rng](cudf::size_type) {
    return std::uniform_int_distribution<int64_t>(-1000, 1000)(rng);
  };
  auto valid_of = [&rng, null_probability](cudf::size_type) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng) >= null_probability;
  };
  return make_fixed_column<T>(num_rows, value_of, valid_of);
}

std::unique_ptr<cudf::column> make_random_column(cudf::type_id id,
                                                 cudf::size_type num_rows,
                                                 double null_probability,
                                                 std::mt19937& rng)
{
  switch (id) {
    case cudf::type_id::INT8: return make_random_fixed<int8_t>(num_rows, null_probability, rng);
    case cudf::type_id::INT16: return make_random_fixed<int16_t>(num_rows, null_probability, rng);
    case cudf::type_id::INT32: return make_random_fixed<int32_t>(num_rows, null_probability, rng);
    case cudf::type_id::INT64: return make_random_fixed<int64_t>(num_rows, null_probability, rng);
    case cudf::type_id::UINT32: return make_random_fixed<uint32_t>(num_rows, null_probability, rng);
    case cudf::type_id::FLOAT32: return make_random_fixed<float>(num_rows, null_probability, rng);
    case cudf::type_id::FLOAT64: return make_random_fixed<double>(num_rows, null_probability, rng);
    case cudf::type_id::BOOL8: return make_random_fixed<bool>(num_rows, null_probability, rng);
    case cudf::type_id::DECIMAL32:
      return make_random_fixed<numeric::decimal32>(num_rows, null_probability, rng);
    case cudf::type_id::DECIMAL64:
      return make_random_fixed<numeric::decimal64>(num_rows, null_probability, rng);
    case cudf::type_id::DECIMAL128:
      return make_random_fixed<numeric::decimal128>(num_rows, null_probability, rng);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return make_random_fixed<cudf::timestamp_ns>(num_rows, null_probability, rng);
    case cudf::type_id::DURATION_MICROSECONDS:
      return make_random_fixed<cudf::duration_us>(num_rows, null_probability, rng);
    case cudf::type_id::STRING: {
      std::vector<std::string> strings;
      std::vector<bool> valid;
      strings.reserve(num_rows);
      valid.reserve(num_rows);
      for (cudf::size_type i = 0; i < num_rows; ++i) {
        strings.push_back(std::to_string(std::uniform_int_distribution<int>(-1000, 1000)(rng)));
        valid.push_back(std::uniform_real_distribution<double>(0.0, 1.0)(rng) >= null_probability);
      }
      auto validity = cudf::detail::make_counting_transform_iterator(
        0, [&valid](cudf::size_type i) { return valid[i]; });
      return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), validity).release();
    }
    default: CUDF_FAIL("Unsupported type id in randomized test");
  }
}

TEST_F(BatchGatherRandomizedTest, RandomizedDifferential)
{
  for (int seed = 0; seed < 10; ++seed) {
    SCOPED_TRACE("iteration seed=" + std::to_string(seed));
    std::mt19937 rng(static_cast<unsigned>(seed));

    auto const num_rows = std::uniform_int_distribution<cudf::size_type>(1, 5000)(rng);
    auto const num_cols = std::uniform_int_distribution<cudf::size_type>(1, 64)(rng);
    auto const null_density =
      std::vector<double>{0.0, 0.1, 0.5, 1.0}[std::uniform_int_distribution<int>(0, 3)(rng)];

    std::vector<std::unique_ptr<cudf::column>> source_columns;
    source_columns.reserve(num_cols);
    for (cudf::size_type c = 0; c < num_cols; ++c) {
      auto const type = random_candidate_types()[std::uniform_int_distribution<std::size_t>(
        0, random_candidate_types().size() - 1)(rng)];
      source_columns.push_back(make_random_column(type, num_rows, null_density, rng));
    }
    std::vector<cudf::column_view> views;
    views.reserve(num_cols);
    for (auto const& col : source_columns) {
      views.push_back(col->view());
    }
    cudf::table_view source_table(views);

    auto const map_mode = std::uniform_int_distribution<int>(0, 2)(rng);
    std::vector<cudf::size_type> host_map(num_rows);
    switch (map_mode) {
      case 0: std::iota(host_map.begin(), host_map.end(), 0); break;
      case 1: {
        std::iota(host_map.begin(), host_map.end(), 0);
        std::shuffle(host_map.begin(), host_map.end(), rng);
        break;
      }
      default: {
        std::uniform_int_distribution<cudf::size_type> uniform_idx(0, num_rows - 1);
        std::generate(host_map.begin(), host_map.end(), [&]() { return uniform_idx(rng); });
        break;
      }
    }
    cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                       host_map.end());

    auto const bounds = std::uniform_int_distribution<int>(0, 1)(rng) == 0
                          ? cudf::out_of_bounds_policy::DONT_CHECK
                          : cudf::out_of_bounds_policy::NULLIFY;
    auto const neg    = std::uniform_int_distribution<int>(0, 1)(rng) == 0
                          ? cudf::negative_index_policy::NOT_ALLOWED
                          : cudf::negative_index_policy::ALLOWED;

    auto result    = run_batch_gather(source_table, gather_map, bounds, neg);
    auto reference = run_reference(source_table, gather_map, bounds, neg);

    CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  }
}

// ============================================================================
// §5.8 Map types: input_indexalator normalization.
// ============================================================================

class BatchGatherMapTypesTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherMapTypesTest, MapTypesInt8ToInt64)
{
  constexpr cudf::size_type num_rows{50};
  auto source_column = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::table_view source_table({source_column->view()});

  // Same values in every index type; batch_gather must normalize them identically.
  auto iota_map = [](auto type_tag) {
    using MapType = decltype(type_tag);
    std::vector<MapType> values(num_rows);
    std::iota(values.begin(), values.end(), 0);
    return cudf::test::fixed_width_column_wrapper<MapType>(values.begin(), values.end()).release();
  };

  auto first  = run_batch_gather(source_table, iota_map(int8_t{})->view());
  auto second = run_batch_gather(source_table, iota_map(int16_t{})->view());
  auto third  = run_batch_gather(source_table, iota_map(int32_t{})->view());
  auto fourth = run_batch_gather(source_table, iota_map(int64_t{})->view());
  auto fifth  = run_batch_gather(source_table, iota_map(uint32_t{})->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, first->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(first->view(), second->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(first->view(), third->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(first->view(), fourth->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(first->view(), fifth->view());
}

// ============================================================================
// §5.9 Launch-count verification bodies (profiled externally with nsys).
// ============================================================================

class BatchGatherLaunchCount : public cudf::test::BaseFixture {};

namespace {
constexpr cudf::size_type launch_count_rows{10'000};
constexpr cudf::size_type launch_count_cols{512};

std::vector<cudf::column_view> wide_fixed_width_views(
  std::vector<std::unique_ptr<cudf::column>>& columns)
{
  for (cudf::size_type c = 0; c < launch_count_cols; ++c) {
    columns.push_back(make_fixed_column<int32_t>(
      launch_count_rows,
      [](cudf::size_type i) { return i; },
      [](cudf::size_type) { return true; }));
  }
  auto views = std::vector<cudf::column_view>{};
  views.reserve(columns.size());
  for (auto const& col : columns) {
    views.push_back(col->view());
  }
  return views;
}
}  // namespace

TEST_F(BatchGatherLaunchCount, WideFixedWidth)
{
  // Profiled recipe: nsys must show exactly one batch_gather_kernel launch and no thrust kernels.
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto views        = wide_fixed_width_views(columns);
  auto source_table = cudf::table_view(views);
  auto gather_map   = make_fixed_column<cudf::size_type>(
    launch_count_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  auto result = run_batch_gather(source_table, gather_map->view());

  ASSERT_EQ(result->num_rows(), launch_count_rows);
  ASSERT_EQ(result->num_columns(), launch_count_cols);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TEST_F(BatchGatherLaunchCount, GatherLaunchCount)
{
  // Stock-gather twin: nsys must show num_cols + 1 kernel launches on the same input.
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto views        = wide_fixed_width_views(columns);
  auto source_table = cudf::table_view(views);
  auto gather_map   = make_fixed_column<cudf::size_type>(
    launch_count_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  auto result = run_reference(source_table, gather_map->view());

  ASSERT_EQ(result->num_rows(), launch_count_rows);
  ASSERT_EQ(result->num_columns(), launch_count_cols);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

// ============================================================================
// §5.10 Map-locality routing: classifier decisions + routed-path behavior parity.
// ============================================================================

class BatchGatherRoutingTest : public cudf::test::BaseFixture {};

std::unique_ptr<cudf::table> run_batch_gather_tile(cudf::table_view const& source,
                                                   cudf::column_view const& map,
                                                   cudf::size_type tile_cols)
{
  return cudf::detail::batch_gather(source,
                                    map,
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    cudf::negative_index_policy::NOT_ALLOWED,
                                    cudf::get_default_stream(),
                                    cudf::memory_resources{cudf::get_current_device_resource_ref()},
                                    tile_cols);
}

TEST_F(BatchGatherRoutingTest, ClassifierWorkGateSamplesTallFewColumnTables)
{
  // 1048576 rows x 8 fused columns crosses both sampling floors: the sampler runs and an
  // ascending map routes to the wide tile.
  constexpr cudf::size_type num_rows{1'048'576};
  auto const source_table = cudf::table_view{std::vector<cudf::column_view>{}, num_rows};
  auto gather_map         = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(
              source_table, gather_map->view(), 8, 4, cudf::get_default_stream()),
            8);
}

TEST_F(BatchGatherRoutingTest, ClassifierUnsampledRegimesPreferK2WithinL2)
{
  // Below either sampling floor the decision is unsampled: K2 while two fused columns plus the
  // map fit half of L2 with margin on any realistic device (65536 rows x 8 x 4B doubled is
  // ~0.8MB — two orders below every GPU L2).
  constexpr cudf::size_type mid_rows{65'536};
  auto const mid_source = cudf::table_view{std::vector<cudf::column_view>{}, mid_rows};
  auto mid_map          = make_fixed_column<cudf::size_type>(
    mid_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(
              mid_source, mid_map->view(), 8, 4, cudf::get_default_stream()),
            2);

  // Wide shape under the row floor: K-insensitive regime, tiny L2 footprint -> K2.
  constexpr cudf::size_type wide_rows{65'536};
  auto const wide_source = cudf::table_view{std::vector<cudf::column_view>{}, wide_rows};
  auto wide_map          = make_fixed_column<cudf::size_type>(
    wide_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(
              wide_source, wide_map->view(), 512, 4, cudf::get_default_stream()),
            2);

  // Short table: same unsampled K2 (rule: num_fused_cols >= 2).
  constexpr cudf::size_type small_rows{100};
  auto const small_source = cudf::table_view{std::vector<cudf::column_view>{}, small_rows};
  auto small_map          = make_fixed_column<cudf::size_type>(
    small_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(
              small_source, small_map->view(), 8, 4, cudf::get_default_stream()),
            2);
}

TEST_F(BatchGatherRoutingTest, ClassifierUnsampledFallsBackToK1BeyondL2)
{
  // A source footprint no device L2 halves (2 x 8M x 8B = 128MB) forces the safe K1 even with
  // an ascending map.
  constexpr cudf::size_type huge_src_rows{8'000'000};
  auto const huge_source = cudf::table_view{std::vector<cudf::column_view>{}, huge_src_rows};
  constexpr cudf::size_type num_rows{100};
  auto gather_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(
              huge_source, gather_map->view(), 8, 8, cudf::get_default_stream()),
            1);
}

TEST_F(BatchGatherRoutingTest, ClassifierRoutesShuffledMapToSingleColumn)
{
  // 1048576-row shuffled map: sampled (both floors crossed), classified random -> K1.
  constexpr cudf::size_type num_rows{1'048'576};
  auto const source_table = cudf::table_view{std::vector<cudf::column_view>{}, num_rows};
  auto const host_map     = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(
              source_table, gather_map, 8, 4, cudf::get_default_stream()),
            1);
}

TEST_F(BatchGatherRoutingTest, ClassifierKeepsSingleColumnTablesAtK1)
{
  // 0/1 fused columns always take K1 regardless of shape.
  constexpr cudf::size_type num_rows{1'048'576};
  auto const source_table = cudf::table_view{std::vector<cudf::column_view>{}, num_rows};
  auto gather_map         = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(
              source_table, gather_map->view(), 1, 4, cudf::get_default_stream()),
            1);
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(
              source_table, gather_map->view(), 0, 4, cudf::get_default_stream()),
            1);
}

TEST_F(BatchGatherRoutingTest, RoutedPathMatchesExplicitTileWidths)
{
  // 8 fixed-width columns x 65536 rows: below the sampling floors both maps take the unsampled
  // K2 path (the L2 footprint fits), and results are tile-width independent — the routed
  // result must agree with BOTH explicit widths.
  constexpr cudf::size_type num_rows{65536};
  constexpr cudf::size_type num_cols{8};
  std::vector<std::unique_ptr<cudf::column>> source_columns;
  source_columns.reserve(num_cols);
  for (cudf::size_type c = 0; c < num_cols; ++c) {
    source_columns.push_back(make_fixed_column<int32_t>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  }
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  for (auto const& col : source_columns) {
    views.push_back(col->view());
  }
  cudf::table_view source_table(views);

  auto ascending_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto routed_ascending = run_batch_gather(source_table, ascending_map->view());
  auto explicit_wide    = run_batch_gather_tile(source_table, ascending_map->view(), 8);
  auto explicit_single  = run_batch_gather_tile(source_table, ascending_map->view(), 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_ascending->view(), explicit_wide->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_ascending->view(), explicit_single->view());

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> shuffled_map(host_map.begin(),
                                                                       host_map.end());
  auto routed_shuffled    = run_batch_gather(source_table, shuffled_map);
  auto wide_on_shuffled   = run_batch_gather_tile(source_table, shuffled_map, 8);
  auto single_on_shuffled = run_batch_gather_tile(source_table, shuffled_map, 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_shuffled->view(), wide_on_shuffled->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_shuffled->view(), single_on_shuffled->view());
}

TEST_F(BatchGatherRoutingTest, RoutedNestedTablesMatchExplicitTileWidths)
{
  // Nested-heavy tables must route by the flattened descriptor count, not the top-level
  // column count: a struct with 7 leaves is 8 descriptors, so a 1'048'576-row ascending map
  // crosses both sampling floors (rows >= 262144, work = 8 x 1048576 >= 4M) and routes to the
  // wide tile — the top-level count of 1 would have taken the unsampled K1. Tile-width
  // independence holds for both map shapes regardless of the routed verdict.
  constexpr cudf::size_type num_rows{1'048'576};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(7);
  for (int c = 0; c < 7; ++c) {
    children.push_back(make_fixed_column<int32_t>(
      num_rows, [c](cudf::size_type i) { return i + c; }, [](cudf::size_type) { return true; }));
  }
  cudf::test::structs_column_wrapper col_struct{std::move(children)};
  cudf::table_view source_table({col_struct});

  auto ascending_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto routed_ascending  = run_batch_gather(source_table, ascending_map->view());
  auto explicit_wide     = run_batch_gather_tile(source_table, ascending_map->view(), 8);
  auto explicit_narrow   = run_batch_gather_tile(source_table, ascending_map->view(), 1);
  auto reference_results = run_reference(source_table, ascending_map->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_ascending->view(), reference_results->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_ascending->view(), explicit_wide->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_ascending->view(), explicit_narrow->view());

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> shuffled_map(host_map.begin(),
                                                                       host_map.end());
  auto routed_shuffled  = run_batch_gather(source_table, shuffled_map);
  auto wide_on_shuffled = run_batch_gather_tile(source_table, shuffled_map, 8);
  auto ref_shuffled     = run_reference(source_table, shuffled_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_shuffled->view(), ref_shuffled->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_shuffled->view(), wide_on_shuffled->view());
}

TEST_F(BatchGatherRoutingTest, RoutedMixedNestedAndFlatTablesMatchExplicitTileWidths)
{
  // One plain column + a struct: flattened count 6 (3 struct nodes' mask-only descriptors
  // + 2 leaves + 1 plain) keeps the fused path off the forced-K1 regime the top-level count
  // of 2 would have taken, and the map rows stay under the sampling floors so the unsampled
  // K2 applies.
  constexpr cudf::size_type num_rows{65'536};
  auto col_plain = make_fixed_column<int64_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  std::vector<std::unique_ptr<cudf::column>> outer_children;
  outer_children.reserve(2);
  for (int c = 0; c < 2; ++c) {
    std::vector<std::unique_ptr<cudf::column>> leaves;
    leaves.push_back(make_fixed_column<int32_t>(
      num_rows, [c](cudf::size_type i) { return i * c; }, [](cudf::size_type) { return true; }));
    outer_children.push_back(cudf::create_structs_hierarchy(
      num_rows,
      std::move(leaves),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, cudf::get_default_stream()),
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref()));
  }
  cudf::test::structs_column_wrapper col_struct{std::move(outer_children)};
  cudf::table_view source_table({col_plain->view(), col_struct});

  auto ascending_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto routed_ascending    = run_batch_gather(source_table, ascending_map->view());
  auto reference_ascending = run_reference(source_table, ascending_map->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_ascending->view(), reference_ascending->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(
    routed_ascending->view(),
    run_batch_gather_tile(source_table, ascending_map->view(), 2)->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(
    routed_ascending->view(),
    run_batch_gather_tile(source_table, ascending_map->view(), 1)->view());

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> shuffled_map(host_map.begin(),
                                                                       host_map.end());
  auto routed_shuffled = run_batch_gather(source_table, shuffled_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_shuffled->view(),
                                run_reference(source_table, shuffled_map)->view());
}

// ============================================================================
// §5.12 Nested types on the descriptor path: struct flattening (N1) and
// dictionary fusion (N2). Struct tests are differential against the stock
// gather; dictionaries additionally assert catalog/shape properties the row
// comparator cannot observe.
// ============================================================================

class BatchGatherStructTest : public cudf::test::BaseFixture {};

// Builds a struct source column whose parent mask is NOT superimposed into the children
// (assembled with `create_structs_hierarchy`, the same reconstruction the fused path uses), so
// child validity stays independent of parent validity. The stock wrapper superimposes, which
// would mask the very behaviors these tests probe.
std::unique_ptr<cudf::column> make_independent_struct(
  std::vector<std::unique_ptr<cudf::column>> children, std::vector<bool> const& parent_valid)
{
  auto [mask, null_count] =
    cudf::test::detail::make_null_mask(parent_valid.begin(), parent_valid.end());
  return cudf::create_structs_hierarchy(static_cast<cudf::size_type>(parent_valid.size()),
                                        std::move(children),
                                        null_count,
                                        std::move(mask),
                                        cudf::get_default_stream(),
                                        cudf::get_current_device_resource_ref());
}

TEST_F(BatchGatherStructTest, FlatStructDifferential)
{
  // Nullable struct parent over children: the parent mask rides a MASK-ONLY descriptor, the
  // children ride plain data descriptors. The wrapper superimposes the parent's nulls into the
  // children, so the leaves gather through kernel-written masks of their own.
  constexpr cudf::size_type num_rows{500};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  children.push_back(make_fixed_column<int64_t>(
    num_rows, [](cudf::size_type i) { return 2 * i; }, [](cudf::size_type) { return true; }));
  children.push_back(make_fixed_column<float>(
    num_rows, [](cudf::size_type i) { return i + 0.5f; }, [](cudf::size_type) { return true; }));
  std::vector<bool> parent_valid(num_rows);
  std::generate(
    parent_valid.begin(), parent_valid.end(), [n = 0]() mutable { return n++ % 3 != 0; });
  cudf::test::structs_column_wrapper col_struct{std::move(children), parent_valid};

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({col_struct});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  ASSERT_EQ(result->num_columns(), 1);
  EXPECT_EQ(result->view().column(0).type().id(), cudf::type_id::STRUCT);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherStructTest, ParentAndChildValidityStayIndependent)
{
  // A source whose parent mask is NOT superimposed into the children (built with
  // create_structs_hierarchy): the gather must keep each node's OWN mask. ANDing ancestor
  // validity into children would inflate the child null count under an identity map.
  constexpr cudf::size_type num_rows{200};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; }));
  children.push_back(make_fixed_column<float>(
    num_rows,
    [](cudf::size_type i) { return i + 0.25f; },
    [](cudf::size_type i) { return i % 5 != 0; }));
  std::vector<bool> parent_valid(num_rows, true);
  for (cudf::size_type i = 0; i < num_rows; i += 7) {
    parent_valid[i] = false;
  }
  auto [parent_mask, parent_null_count] = cudf::test::detail::make_null_mask(
    parent_valid.begin(), parent_valid.end(), cudf::get_default_stream());
  auto source_struct = cudf::create_structs_hierarchy(num_rows,
                                                      std::move(children),
                                                      parent_null_count,
                                                      std::move(parent_mask),
                                                      cudf::get_default_stream(),
                                                      cudf::get_current_device_resource_ref());
  ASSERT_EQ(source_struct->null_count(), parent_null_count);

  cudf::table_view source_table({source_struct->view()});
  // Under an identity map the gathered masks equal the source masks, so an ancestor-AND would
  // show up as an inflated child null count (union with the parent's nulls).
  std::vector<cudf::size_type> identity(num_rows);
  std::iota(identity.begin(), identity.end(), 0);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> identity_map(identity.begin(),
                                                                       identity.end());
  auto identity_result    = run_batch_gather(source_table, identity_map);
  auto identity_reference = run_reference(source_table, identity_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(identity_reference->view().column(0),
                                 identity_result->view().column(0));

  // Child null count must equal the source child's own count (100 even rows), not the
  // union with the parent's nulls (which would be 114).
  auto result_child = identity_result->view().column(0).child(0);
  EXPECT_EQ(result_child.null_count(), 100);
  EXPECT_EQ(identity_result->view().column(0).null_count(), parent_null_count);

  auto const host_map = shuffled_iota(num_rows, 42);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> shuffled_map(host_map.begin(),
                                                                       host_map.end());
  auto shuffled_result    = run_batch_gather(source_table, shuffled_map);
  auto shuffled_reference = run_reference(source_table, shuffled_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(shuffled_reference->view().column(0),
                                 shuffled_result->view().column(0));
}

TEST_F(BatchGatherStructTest, NullFreeChildrenStayNonNullable)
{
  // A nullable parent with nulls next to non-nullable children (no superimposition): the
  // children must NOT grow masks; only the parent's mask is kernel-written.
  constexpr cudf::size_type num_rows{100};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(
    make_fixed_column_without_validity<int32_t>(num_rows, [](cudf::size_type i) { return i; }));
  children.push_back(
    make_fixed_column_without_validity<int64_t>(num_rows, [](cudf::size_type i) { return i; }));
  std::vector<bool> parent_valid(num_rows);
  std::generate(
    parent_valid.begin(), parent_valid.end(), [n = 0]() mutable { return n++ % 4 != 1; });
  auto source_struct = make_independent_struct(std::move(children), parent_valid);

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({source_struct->view()});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

  EXPECT_EQ(
    result->view().column(0).null_count(),
    static_cast<cudf::size_type>(std::count(parent_valid.begin(), parent_valid.end(), false)));
  EXPECT_FALSE(result->view().column(0).child(0).nullable());
  EXPECT_FALSE(result->view().column(0).child(1).nullable());
}

TEST_F(BatchGatherStructTest, NestedStructDifferential)
{
  // struct<struct<i16, i64>, i8> with nulls at every level: each node emits its own
  // mask-only descriptor and the hierarchy is rebuilt bottom-up.
  constexpr cudf::size_type num_rows{300};
  std::vector<std::unique_ptr<cudf::column>> inner_children;
  inner_children.push_back(make_fixed_column<int16_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; }));
  inner_children.push_back(make_fixed_column<int64_t>(
    num_rows, [](cudf::size_type i) { return 3 * i; }, [](cudf::size_type) { return true; }));
  std::vector<bool> inner_valid(num_rows);
  std::generate(inner_valid.begin(), inner_valid.end(), [n = 0]() mutable { return n++ % 5 != 2; });
  cudf::test::structs_column_wrapper inner_struct{std::move(inner_children), inner_valid};

  std::vector<std::unique_ptr<cudf::column>> outer_children;
  outer_children.push_back(inner_struct.release());
  outer_children.push_back(make_fixed_column<int8_t>(
    num_rows, [](cudf::size_type i) { return i % 97; }, [](cudf::size_type) { return true; }));
  std::vector<bool> outer_valid(num_rows);
  std::generate(outer_valid.begin(), outer_valid.end(), [n = 0]() mutable { return n++ % 3 != 0; });
  cudf::test::structs_column_wrapper outer_struct{std::move(outer_children), outer_valid};

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({outer_struct});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  ASSERT_EQ(result->view().column(0).num_children(), 2);
  EXPECT_EQ(result->view().column(0).child(0).type().id(), cudf::type_id::STRUCT);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherStructTest, SlicedStructDifferential)
{
  // Parent-applied child offset applied exactly once: the leaf data base and the mask bit
  // offset both come from the sliced parent's offset, never from the raw child.
  constexpr cudf::size_type source_size{1000};
  constexpr cudf::size_type slice_begin{100};
  constexpr cudf::size_type slice_end{600};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(make_fixed_column<int32_t>(
    source_size,
    [](cudf::size_type i) { return i; },
    [](cudf::size_type i) { return i % 2 != 0; }));
  children.push_back(make_fixed_column<int64_t>(
    source_size, [](cudf::size_type i) { return 10 * i; }, [](cudf::size_type) { return true; }));
  std::vector<bool> parent_valid(source_size);
  std::generate(
    parent_valid.begin(), parent_valid.end(), [n = 0]() mutable { return n++ % 3 != 0; });
  cudf::test::structs_column_wrapper col_struct{std::move(children), parent_valid};

  auto const sliced = cudf::slice(col_struct, {slice_begin, slice_end});
  ASSERT_EQ(sliced.size(), 1);
  auto gather_map = make_fixed_column<cudf::size_type>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return i; },
    [](cudf::size_type) { return true; });

  cudf::table_view source_table({sliced[0]});
  auto result    = run_batch_gather(source_table, gather_map->view());
  auto reference = run_reference(source_table, gather_map->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

  // Explicit: row i of the output holds source row slice_begin + i, nulls included.
  auto result_child = result->view().column(0).child(0);
  EXPECT_EQ(result_child.size(), slice_end - slice_begin);
  EXPECT_EQ(result->view().column(0).null_count(),
            static_cast<cudf::size_type>(std::count(
              parent_valid.begin() + slice_begin, parent_valid.begin() + slice_end, false)));
}

TEST_F(BatchGatherStructTest, SlicedNestedStructDifferential)
{
  // Two levels of slicing arithmetic: the outer struct's offset shifts both the inner struct's
  // mask and the inner leaves' data bases.
  constexpr cudf::size_type source_size{800};
  constexpr cudf::size_type slice_begin{217};
  constexpr cudf::size_type slice_end{733};
  std::vector<std::unique_ptr<cudf::column>> inner_children;
  inner_children.push_back(make_fixed_column<int16_t>(
    source_size,
    [](cudf::size_type i) { return i; },
    [](cudf::size_type i) { return i % 4 != 0; }));
  inner_children.push_back(make_fixed_column<int64_t>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  std::vector<bool> inner_valid(source_size);
  std::generate(inner_valid.begin(), inner_valid.end(), [n = 0]() mutable { return n++ % 5 != 3; });
  cudf::test::structs_column_wrapper inner_struct{std::move(inner_children), inner_valid};

  std::vector<std::unique_ptr<cudf::column>> outer_children;
  outer_children.push_back(inner_struct.release());
  outer_children.push_back(make_fixed_column<int8_t>(
    source_size, [](cudf::size_type i) { return i % 89; }, [](cudf::size_type) { return true; }));
  std::vector<bool> outer_valid(source_size);
  std::generate(outer_valid.begin(), outer_valid.end(), [n = 0]() mutable { return n++ % 3 != 1; });
  cudf::test::structs_column_wrapper outer_struct{std::move(outer_children), outer_valid};

  auto const sliced = cudf::slice(outer_struct, {slice_begin, slice_end});
  ASSERT_EQ(sliced.size(), 1);
  auto gather_map = make_fixed_column<cudf::size_type>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return (i * 7) % (slice_end - slice_begin); },
    [](cudf::size_type) { return true; });

  cudf::table_view source_table({sliced[0]});
  auto result    = run_batch_gather(source_table, gather_map->view());
  auto reference = run_reference(source_table, gather_map->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherStructTest, AllNullParentKeepsChildrenGathered)
{
  // An all-null parent pre-sets an ALL_NULL mask and emits no descriptor; children are still
  // gathered with their own (absent) masks — the parent's nulls must not leak into them.
  constexpr cudf::size_type num_rows{120};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(
    make_fixed_column_without_validity<int32_t>(num_rows, [](cudf::size_type i) { return i; }));
  children.push_back(
    make_fixed_column_without_validity<float>(num_rows, [](cudf::size_type i) { return i; }));
  auto source_struct =
    make_independent_struct(std::move(children), std::vector<bool>(num_rows, false));

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({source_struct->view()});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

  EXPECT_EQ(result->view().column(0).null_count(), num_rows);
  EXPECT_EQ(result->view().column(0).child(0).null_count(), 0);
  EXPECT_EQ(result->view().column(0).child(1).null_count(), 0);
}

TEST_F(BatchGatherStructTest, NullifyOutOfBoundsAndNegativeWrap)
{
  // Under NULLIFY every level of the struct gets a kernel-written mask; OOB and negative
  // (wrapped) map entries nullify consistently at every depth.
  constexpr cudf::size_type num_rows{50};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  children.push_back(make_fixed_column<int64_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 4 != 0; }));
  std::vector<bool> parent_valid(num_rows);
  std::generate(
    parent_valid.begin(), parent_valid.end(), [n = 0]() mutable { return n++ % 3 != 0; });
  cudf::test::structs_column_wrapper col_struct{std::move(children), parent_valid};

  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {0, num_rows, num_rows + 3, -1, -num_rows, 7, 8, 9, 10, 11}};
  cudf::table_view source_table({col_struct});

  auto result    = run_batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::NULLIFY,
                                 cudf::negative_index_policy::ALLOWED);
  auto reference = run_reference(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::NULLIFY,
                                 cudf::negative_index_policy::ALLOWED);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

  EXPECT_TRUE(result->view().column(0).nullable());
  EXPECT_TRUE(result->view().column(0).child(0).nullable());
  EXPECT_TRUE(result->view().column(0).child(1).nullable());
}

TEST_F(BatchGatherStructTest, StructWithStringChildStaysFallback)
{
  // A struct with a non-fixed-width child is not flattened: the whole column gathers through
  // the stock path with identical results.
  constexpr cudf::size_type num_rows{80};
  auto child_i64 = make_fixed_column<int64_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  std::vector<std::string> strings;
  std::vector<bool> str_valid;
  strings.reserve(num_rows);
  str_valid.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back("s" + std::to_string(i));
    str_valid.push_back(i % 3 != 0);
  }
  auto str_validity = cudf::detail::make_counting_transform_iterator(
    0, [&str_valid](cudf::size_type i) { return str_valid[i]; });
  cudf::test::strings_column_wrapper child_str(strings.begin(), strings.end(), str_validity);

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(child_i64));
  children.push_back(child_str.release());
  std::vector<bool> parent_valid(num_rows);
  std::generate(
    parent_valid.begin(), parent_valid.end(), [n = 0]() mutable { return n++ % 5 != 1; });
  cudf::test::structs_column_wrapper col_struct{std::move(children), parent_valid};

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({col_struct});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherStructTest, TileWidthIndependent)
{
  // MASK-ONLY and leaf descriptors must produce identical output under every tile width.
  constexpr cudf::size_type num_rows{400};
  std::vector<std::unique_ptr<cudf::column>> inner_children;
  inner_children.push_back(make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  std::vector<bool> inner_valid(num_rows);
  std::generate(inner_valid.begin(), inner_valid.end(), [n = 0]() mutable { return n++ % 4 != 2; });
  cudf::test::structs_column_wrapper inner_struct{std::move(inner_children), inner_valid};

  std::vector<std::unique_ptr<cudf::column>> outer_children;
  outer_children.push_back(inner_struct.release());
  outer_children.push_back(make_fixed_column<double>(
    num_rows, [](cudf::size_type i) { return i * 1.5; }, [](cudf::size_type) { return true; }));
  std::vector<bool> outer_valid(num_rows);
  std::generate(outer_valid.begin(), outer_valid.end(), [n = 0]() mutable { return n++ % 6 != 4; });
  cudf::test::structs_column_wrapper outer_struct{std::move(outer_children), outer_valid};

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({outer_struct});

  auto routed = run_batch_gather(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed->view(),
                                run_batch_gather_tile(source_table, gather_map, 1)->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed->view(),
                                run_batch_gather_tile(source_table, gather_map, 2)->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed->view(),
                                run_batch_gather_tile(source_table, gather_map, 4)->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed->view(),
                                run_batch_gather_tile(source_table, gather_map, 8)->view());
}

TEST_F(BatchGatherStructTest, ZeroRowMap)
{
  constexpr cudf::size_type num_rows{10};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  cudf::test::structs_column_wrapper col_struct{std::move(children), std::vector<bool>{}};

  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{};
  cudf::table_view source_table({col_struct});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->view().column(0).type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(result->view().column(0).num_children(), reference->view().column(0).num_children());
}

class BatchGatherDictionaryTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherDictionaryTest, Int32KeysDifferential)
{
  // Dictionary over int32 values with duplicate references and null rows: the annotated
  // indices copy through a data descriptor, the mask lands on the parent.
  constexpr cudf::size_type num_rows{400};
  std::vector<int32_t> values;
  std::vector<bool> valid;
  values.reserve(num_rows);
  valid.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values.push_back(static_cast<int32_t>(i % 17));  // duplicates throughout
    valid.push_back(i % 4 != 0);
  }
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [&valid](cudf::size_type i) { return valid[i]; });
  cudf::test::dictionary_column_wrapper<int32_t> col_dict(values.begin(), values.end(), validity);

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({col_dict});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

  // The key catalog is a verbatim deep copy: same type, size, and values as the source keys.
  auto const source_keys = cudf::dictionary_column_view{source_table.column(0)}.keys();
  auto const result_keys = cudf::dictionary_column_view{result->view().column(0)}.keys();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(source_keys, result_keys);
  EXPECT_FALSE(result->view().column(0).child(0).nullable());  // indices stay mask-free; the parent
                                                               // carries the mask
}

TEST_F(BatchGatherDictionaryTest, Uint32KeysDifferential)
{
  constexpr cudf::size_type num_rows{256};
  std::vector<uint32_t> values;
  values.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values.push_back(static_cast<uint32_t>(i % 9) * 1'000'000u);  // duplicates throughout
  }
  cudf::test::dictionary_column_wrapper<uint32_t> col_dict(values.begin(), values.end());

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({col_dict});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

  auto const source_keys = cudf::dictionary_column_view{source_table.column(0)}.keys();
  auto const result_keys = cudf::dictionary_column_view{result->view().column(0)}.keys();
  EXPECT_EQ(result_keys.type().id(), source_keys.type().id());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(source_keys, result_keys);
}

TEST_F(BatchGatherDictionaryTest, UnusedKeysRetained)
{
  // Gathering a subset that never references some keys must still copy the FULL catalog:
  // no dedup, no remap, full key order.
  cudf::test::dictionary_column_wrapper<int32_t> col_dict{{5, 3, 3, 9, 5, 3, 7, 3, 5, 9}};
  auto const source_keys = static_cast<cudf::dictionary_column_view>(col_dict).keys();
  ASSERT_EQ(source_keys.size(), 4);  // {3, 5, 7, 9}

  // Map only onto rows whose value is 3 (source rows 1, 2, 5, 7): keys 5, 7, 9 go unreferenced.
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{{1, 5, 7, 2}};
  cudf::table_view source_table({col_dict});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(reference->view().column(0), result->view().column(0));

  auto const result_keys = cudf::dictionary_column_view{result->view().column(0)}.keys();
  EXPECT_EQ(result_keys.size(), 4);  // unused keys {5, 7, 9} retained
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(source_keys, result_keys);

  auto const result_indices = cudf::dictionary_column_view{result->view().column(0)}.indices();
  EXPECT_EQ(result_indices.size(), 4);
  EXPECT_FALSE(result_indices.nullable());
}

TEST_F(BatchGatherDictionaryTest, SlicedSourceDifferential)
{
  // The annotated-indices convention: index element i lives at indices_base + parent.offset(),
  // and the mask bit comes from the parent's own mask at the same offset.
  constexpr cudf::size_type source_size{500};
  constexpr cudf::size_type slice_begin{40};
  constexpr cudf::size_type slice_end{340};
  std::vector<int32_t> values;
  std::vector<bool> valid;
  values.reserve(source_size);
  valid.reserve(source_size);
  for (cudf::size_type i = 0; i < source_size; ++i) {
    values.push_back(static_cast<int32_t>(i % 23));
    valid.push_back(i % 6 != 0);
  }
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [&valid](cudf::size_type i) { return valid[i]; });
  cudf::test::dictionary_column_wrapper<int32_t> col_dict(values.begin(), values.end(), validity);

  auto const sliced =
    cudf::slice(static_cast<cudf::column_view>(col_dict), {slice_begin, slice_end});
  ASSERT_EQ(sliced.size(), 1);
  auto gather_map = make_fixed_column<cudf::size_type>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return (i * 11) % (slice_end - slice_begin); },
    [](cudf::size_type) { return true; });

  cudf::table_view source_table({sliced[0]});
  auto result    = run_batch_gather(source_table, gather_map->view());
  auto reference = run_reference(source_table, gather_map->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

  auto const source_keys =
    cudf::dictionary_column_view{static_cast<cudf::column_view>(col_dict)}.keys();
  auto const result_keys = cudf::dictionary_column_view{result->view().column(0)}.keys();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(source_keys, result_keys);
}

TEST_F(BatchGatherDictionaryTest, NullifyOutOfBounds)
{
  constexpr cudf::size_type num_rows{10};
  cudf::test::dictionary_column_wrapper<int32_t> col_dict{{0, 1, 2, 0, 1, 2, 0, 1, 2, 0}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {0, num_rows, num_rows + 5, 2, -1, -num_rows, 3, 4, 5, 6}};
  cudf::table_view source_table({col_dict});

  auto result    = run_batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::NULLIFY,
                                 cudf::negative_index_policy::ALLOWED);
  auto reference = run_reference(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::NULLIFY,
                                 cudf::negative_index_policy::ALLOWED);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(reference->view().column(0), result->view().column(0));

  EXPECT_TRUE(result->view().column(0).nullable());
  EXPECT_FALSE(result->view().column(0).child(0).nullable());
  EXPECT_EQ(result->view().column(0).null_count(),
            2);  // map rows 1 (== num_rows) and 2 (> num_rows)
}

TEST_F(BatchGatherDictionaryTest, IndexWidthsFuse)
{
  // The descriptor path accepts any signed index width: encode the same values into INT8,
  // INT16, INT32, and INT64 indices and gather each differentially.
  constexpr cudf::size_type num_rows{300};
  std::vector<int64_t> values;
  values.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values.push_back(static_cast<int64_t>(i % 11));
  }
  auto values_col = cudf::test::fixed_width_column_wrapper<int64_t>(values.begin(), values.end());

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  for (auto const indices_type :
       {cudf::type_id::INT8, cudf::type_id::INT16, cudf::type_id::INT32, cudf::type_id::INT64}) {
    auto col_dict = cudf::dictionary::encode(
      values_col, cudf::data_type{indices_type}, cudf::get_default_stream());
    cudf::table_view source_table({col_dict->view()});

    auto result    = run_batch_gather(source_table, gather_map);
    auto reference = run_reference(source_table, gather_map);
    CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

    auto const result_indices = cudf::dictionary_column_view{result->view().column(0)}.indices();
    EXPECT_EQ(result_indices.type().id(), indices_type);
  }
}

TEST_F(BatchGatherDictionaryTest, StringKeysDifferential)
{
  // Dictionary over STRING keys: the catalog deep copy must survive the deferred assembly —
  // sliced-and-rebased offsets, exact chars byte range, verbatim key catalog (order, values,
  // no dedup), and the empty-string key (zero-length chars range) round-trip exactly as the
  // stock copy_slice produces them. The source dictionary is built by encoding a strings
  // column with duplicate values, including an empty string.
  constexpr cudf::size_type num_rows{250};
  std::vector<std::string> values;
  values.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values.push_back(i % 19 == 0 ? std::string{} : "key_" + std::to_string(i % 19));
  }
  auto values_col = cudf::test::strings_column_wrapper(values.begin(), values.end());

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto col_dict = cudf::dictionary::encode(
    values_col, cudf::data_type{cudf::type_id::INT32}, cudf::get_default_stream());
  cudf::table_view source_table({col_dict->view()});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

  auto const source_keys = cudf::dictionary_column_view{source_table.column(0)}.keys();
  auto const result_keys = cudf::dictionary_column_view{result->view().column(0)}.keys();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(source_keys, result_keys);
  EXPECT_EQ(result_keys.size(), source_keys.size());
  EXPECT_FALSE(result->view().column(0).child(0).nullable());  // parent carries the mask
}

TEST_F(BatchGatherDictionaryTest, TwoStringKeyedCatalogsOneSync)
{
  // Two string-keyed dictionary columns in one call: both catalogs come from ONE batched
  // offsets read plus ONE synchronization (the per-catalog reads of the stock path are gone).
  // Correctness is asserted differentially; the sync-count property is covered by the
  // implementation comment and the profiler, not the comparator.
  constexpr cudf::size_type num_rows{300};
  auto make_values = [](int modulo, int offset) {
    std::vector<std::string> values;
    values.reserve(num_rows);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      values.push_back(i % modulo == 0 ? std::string{}
                                       : "cat_" + std::to_string(i % modulo + offset));
    }
    return values;
  };
  auto values_a = make_values(11, 0);
  auto values_b = make_values(7, 100);
  auto col_a =
    cudf::dictionary::encode(cudf::test::strings_column_wrapper(values_a.begin(), values_a.end()),
                             cudf::data_type{cudf::type_id::INT32},
                             cudf::get_default_stream());
  auto col_b =
    cudf::dictionary::encode(cudf::test::strings_column_wrapper(values_b.begin(), values_b.end()),
                             cudf::data_type{cudf::type_id::INT32},
                             cudf::get_default_stream());
  cudf::table_view source_table({col_a->view(), col_b->view()});

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());

  for (size_t c : {size_t{0}, size_t{1}}) {
    auto const source_keys = cudf::dictionary_column_view{source_table.column(c)}.keys();
    auto const result_keys = cudf::dictionary_column_view{result->view().column(c)}.keys();
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(source_keys, result_keys);
  }
}

TEST_F(BatchGatherDictionaryTest, NullableZeroNullsTakesAllValidPath)
{
  // A nullable dictionary with ZERO nulls: the mask plan pre-sets the destination mask
  // ALL_VALID — no kernel mask, no count readback, and the output stays nullable with zero
  // nulls. The gathered values must still match the stock gather.
  constexpr cudf::size_type num_rows{150};
  std::vector<int32_t> values;
  values.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values.push_back(static_cast<int32_t>(i % 12));
  }
  auto always_valid =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type) { return true; });
  cudf::test::dictionary_column_wrapper<int32_t> col_dict(
    values.begin(), values.end(), always_valid);

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({col_dict});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  EXPECT_TRUE(result->view().column(0).nullable());  // source was nullable; mask preserved
  EXPECT_EQ(result->view().column(0).null_count(), 0);
}

TEST_F(BatchGatherDictionaryTest, AllNullParent)
{
  constexpr cudf::size_type num_rows{60};
  std::vector<int32_t> values(num_rows, 4);
  auto never_valid =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type) { return false; });
  cudf::test::dictionary_column_wrapper<int32_t> col_dict(
    values.begin(), values.end(), never_valid);

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({col_dict});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  EXPECT_EQ(result->view().column(0).null_count(), num_rows);
}

TEST_F(BatchGatherDictionaryTest, ZeroRowMap)
{
  cudf::test::dictionary_column_wrapper<int32_t> col_dict{{1, 2, 2, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{};
  cudf::table_view source_table({col_dict});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->view().column(0).type().id(), cudf::type_id::DICTIONARY32);
  EXPECT_EQ(result->view().column(0).num_children(), reference->view().column(0).num_children());
}

TEST_F(BatchGatherMixedTest, NestedTypesFusedWithFallbackColumns)
{
  // Fused fixed-width + flattened struct + fused dictionary next to fallback strings: the
  // fallback's whole-table mask predicates still see every column, fused or not.
  constexpr cudf::size_type num_rows{200};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 3 != 0; });

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(make_fixed_column<int64_t>(
    num_rows, [](cudf::size_type i) { return 2 * i; }, [](cudf::size_type) { return true; }));
  struct_children.push_back(make_fixed_column<float>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 7 != 0; }));
  std::vector<bool> parent_valid(num_rows);
  std::generate(
    parent_valid.begin(), parent_valid.end(), [n = 0]() mutable { return n++ % 5 != 1; });
  cudf::test::structs_column_wrapper col_struct{std::move(struct_children), parent_valid};

  std::vector<int32_t> values;
  values.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values.push_back(static_cast<int32_t>(i % 13));
  }
  auto value_validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 4 != 2; });
  cudf::test::dictionary_column_wrapper<int32_t> col_dict(
    values.begin(), values.end(), value_validity);

  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(i % 9 == 0 ? "" : "row" + std::to_string(i));
  }
  auto str_validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 8 != 5; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), str_validity);

  cudf::table_view source_table(
    {col_i32->view(), col_struct, static_cast<cudf::column_view>(col_dict), col_str});

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}
