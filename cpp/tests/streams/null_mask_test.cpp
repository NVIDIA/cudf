/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>

class NullMaskTest : public cudf::test::BaseFixture {};

TEST_F(NullMaskTest, CreateNullMask)
{ cudf::create_null_mask(10, cudf::mask_state::ALL_VALID, cudf::test::get_default_stream()); }

TEST_F(NullMaskTest, SetNullMask)
{
  cudf::test::fixed_width_column_wrapper<bool> col({0, 1, 0, 1, 1},
                                                   {true, false, true, false, false});

  cudf::set_null_mask(static_cast<cudf::mutable_column_view>(col).null_mask(),
                      0,
                      3,
                      false,
                      cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, CopyBitmask)
{
  cudf::test::fixed_width_column_wrapper<bool> const col({0, 1, 0, 1, 1},
                                                         {true, false, true, false, false});

  cudf::copy_bitmask(
    static_cast<cudf::column_view>(col).null_mask(), 0, 3, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, CopyBitmaskFromColumn)
{
  cudf::test::fixed_width_column_wrapper<bool> const col({0, 1, 0, 1, 1},
                                                         {true, false, true, false, false});

  cudf::copy_bitmask(col, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, BitMaskAnd)
{
  cudf::test::fixed_width_column_wrapper<bool> const col1({0, 1, 0, 1, 1},
                                                          {true, false, true, false, false});
  cudf::test::fixed_width_column_wrapper<bool> const col2({0, 1, 0, 1, 1},
                                                          {true, true, false, false, true});

  auto tbl = cudf::table_view{{col1, col2}};
  cudf::bitmask_and(tbl, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, BitMaskOr)
{
  cudf::test::fixed_width_column_wrapper<bool> const col1({0, 1, 0, 1, 1},
                                                          {true, false, true, false, false});
  cudf::test::fixed_width_column_wrapper<bool> const col2({0, 1, 0, 1, 1},
                                                          {true, true, false, false, true});

  auto tbl = cudf::table_view{{col1, col2}};
  cudf::bitmask_or(tbl, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, NullCount)
{
  cudf::test::fixed_width_column_wrapper<bool> const col({0, 1, 0, 1, 1},
                                                         {true, true, false, false, true});

  cudf::null_count(
    static_cast<cudf::column_view>(col).null_mask(), 0, 4, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, SegmentedCount)
{
  auto stream        = cudf::test::get_default_stream();
  auto mask          = rmm::device_uvector<cudf::bitmask_type>(10, stream);
  auto const indices = std::vector<cudf::size_type>{0, 320, 0, 320};

  cudf::segmented_null_count(mask.data(), indices, stream);
  cudf::segmented_valid_count(mask.data(), indices, stream);
}

TEST_F(NullMaskTest, SegmentedAnd)
{
  auto col1 = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 1, 1}, {0, 1, 1, 1, 0});
  auto col2 = cudf::test::fixed_width_column_wrapper<bool>({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});
  auto col3 = cudf::test::fixed_width_column_wrapper<bool>({0, 2, 1, 0, 255}, {1, 1, 1, 1, 1});

  std::vector<cudf::column_view> const colviews({col1, col2, col3});
  std::vector<cudf::size_type> const segment_offsets{0, 1};
  cudf::segmented_bitmask_and(colviews, segment_offsets, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
