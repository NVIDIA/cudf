/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/iterator>

#include <memory>
#include <type_traits>

// fixed_width, dict, string, list, struct
template <typename T>
std::unique_ptr<cudf::column> example_column()
  requires(cudf::is_fixed_width<T>())
{
  auto begin = cuda::counting_iterator<int32_t>{1};
  auto end   = cuda::counting_iterator<int32_t>{16};
  return cudf::test::fixed_width_column_wrapper<T>(begin, end).release();
}

template <typename T>
std::unique_ptr<cudf::column> example_column()
  requires(cudf::is_dictionary<T>())
{
  return cudf::test::dictionary_column_wrapper<std::string>(
           {"fff", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "", ""},
           {true, true, true, true, true, true, true, true, false})
    .release();
}

template <typename T>
std::unique_ptr<cudf::column> example_column()

  requires(std::is_same_v<T, std::string> or std::is_same_v<T, cudf::string_view>)
{
  return cudf::test::strings_column_wrapper(
           {"fff", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "", ""})
    .release();
}

template <typename T>
std::unique_ptr<cudf::column> example_column()
  requires(std::is_same_v<T, cudf::list_view>)
{ return cudf::test::lists_column_wrapper<int>({{1, 2, 3}, {4, 5}, {}, {6, 7, 8}}).release(); }

template <typename T>
std::unique_ptr<cudf::column> example_column()
  requires(std::is_same_v<T, cudf::struct_view>)
{
  auto begin    = cuda::counting_iterator<int32_t>{1};
  auto end      = cuda::counting_iterator<int32_t>{16};
  auto member_0 = cudf::test::fixed_width_column_wrapper<int32_t>(begin, end);
  auto member_1 = cudf::test::fixed_width_column_wrapper<int32_t>(begin + 10, end + 10);
  return cudf::test::structs_column_wrapper({member_0, member_1}).release();
}

template <typename T>
struct ColumnViewShallowTests : public cudf::test::BaseFixture {};

using AllTypes = cudf::test::Concat<cudf::test::AllTypes, cudf::test::CompoundTypes>;
TYPED_TEST_SUITE(ColumnViewShallowTests, AllTypes);

// Test for fixed_width, dict, string, list, struct
// column_view, column_view = same hash.
// column_view, make a copy = same hash.
// new column_view from column = same hash
// column_view, copy column = diff hash
// column_view, diff column = diff hash.
//
// column_view old, update data + new column_view     = same hash.
// column_view old, add null_mask + new column_view   = diff hash.
// column_view old, update nulls + new column_view    = same hash.
// column_view old, set_null_count + new column_view  = same hash.
//
// column_view, sliced[0, size) = same hash (for split too)
// column_view, sliced[n:)      = diff hash (for split too)
// column_view, bit_cast        = diff hash
//
// mutable_column_view, column_view = same hash
// mutable_column_view, modified mutable_column_view = same hash
//
// update the children column data  = same hash
// update the children column_views = diff hash

TYPED_TEST(ColumnViewShallowTests, shallow_hash_basic)
{
  using namespace cudf::detail;
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};
  // same = same hash
  {
    EXPECT_EQ(shallow_hash(col_view), shallow_hash(col_view));
  }
  // copy column_view = same hash
  {
    auto col_view_copy = col_view;
    EXPECT_EQ(shallow_hash(col_view), shallow_hash(col_view_copy));
  }

  // new column_view from column = same hash
  {
    auto col_view_new = cudf::column_view{*col};
    EXPECT_EQ(shallow_hash(col_view), shallow_hash(col_view_new));
  }

  // copy column = diff hash
  {
    auto col_new       = std::make_unique<cudf::column>(*col);
    auto col_view_copy = col_new->view();
    EXPECT_NE(shallow_hash(col_view), shallow_hash(col_view_copy));
  }

  // column_view, diff column = diff hash.
  {
    auto col_diff      = example_column<TypeParam>();
    auto col_view_diff = cudf::column_view{*col_diff};
    EXPECT_NE(shallow_hash(col_view), shallow_hash(col_view_diff));
  }
}
TYPED_TEST(ColumnViewShallowTests, shallow_hash_update_data)
{
  using namespace cudf::detail;
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};
  // update data + new column_view = same hash.
  {
    // update data by modifying some bits: fixed_width, string, dict, list, struct
    if constexpr (cudf::is_fixed_width<TypeParam>()) {
      // Update data
      auto data = reinterpret_cast<cudf::bitmask_type*>(col->mutable_view().head());
      cudf::set_null_mask(data, 2, 64, true);
    } else {
      // Update child(0).data
      auto data = reinterpret_cast<cudf::bitmask_type*>(col->child(0).mutable_view().head());
      cudf::set_null_mask(data, 2, 64, true);
    }
    auto col_view_new = cudf::column_view{*col};
    EXPECT_EQ(shallow_hash(col_view), shallow_hash(col_view_new));
  }
  // add null_mask + new column_view = diff hash.
  {
    col->set_null_mask(cudf::create_null_mask(col->size(), cudf::mask_state::ALL_VALID), 0);
    auto col_view_new = cudf::column_view{*col};
    EXPECT_NE(shallow_hash(col_view), shallow_hash(col_view_new));
    [[maybe_unused]] auto const nulls = col_view_new.null_count();
    EXPECT_NE(shallow_hash(col_view), shallow_hash(col_view_new));
    auto col_view_new2 = cudf::column_view{*col};
    EXPECT_EQ(shallow_hash(col_view_new), shallow_hash(col_view_new2));
  }
  col_view = cudf::column_view{*col};  // updating after adding null_mask
  // update nulls + new column_view = same hash.
  {
    cudf::set_null_mask(col->mutable_view().null_mask(), 2, 4, false);
    auto col_view_new = cudf::column_view{*col};
    EXPECT_EQ(shallow_hash(col_view), shallow_hash(col_view_new));
  }
  // set_null_count + new column_view = same hash.
  {
    col->set_null_count(col->size());
    auto col_view_new2 = cudf::column_view{*col};
    EXPECT_EQ(shallow_hash(col_view), shallow_hash(col_view_new2));
  }
}

TYPED_TEST(ColumnViewShallowTests, shallow_hash_slice)
{
  using namespace cudf::detail;
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};
  // column_view, sliced[0, size)  = same hash (for split too)
  {
    auto col_sliced = cudf::slice(col_view, {0, col_view.size()});
    EXPECT_EQ(shallow_hash(col_view), shallow_hash(col_sliced[0]));
    auto col_split = cudf::split(col_view, {0});
    EXPECT_NE(shallow_hash(col_view), shallow_hash(col_split[0]));
    EXPECT_EQ(shallow_hash(col_view), shallow_hash(col_split[1]));
  }
  // column_view, sliced[n:]       = diff hash (for split too)
  {
    auto col_sliced = cudf::slice(col_view, {1, col_view.size()});
    EXPECT_NE(shallow_hash(col_view), shallow_hash(col_sliced[0]));
    auto col_split = cudf::split(col_view, {1});
    EXPECT_NE(shallow_hash(col_view), shallow_hash(col_split[0]));
    EXPECT_NE(shallow_hash(col_view), shallow_hash(col_split[1]));
  }
  // column_view, col copy sliced[0, 0)  = same hash (empty column)
  {
    auto col_new        = std::make_unique<cudf::column>(*col);
    auto col_new_view   = col_new->view();
    auto col_sliced     = cudf::slice(col_view, {0, 0, 1, 1, col_view.size(), col_view.size()});
    auto col_new_sliced = cudf::slice(col_new_view, {0, 0, 1, 1, col_view.size(), col_view.size()});

    EXPECT_EQ(shallow_hash(col_sliced[0]), shallow_hash(col_sliced[1]));
    EXPECT_EQ(shallow_hash(col_sliced[1]), shallow_hash(col_sliced[2]));
    EXPECT_EQ(shallow_hash(col_sliced[0]), shallow_hash(col_new_sliced[0]));
    EXPECT_EQ(shallow_hash(col_sliced[1]), shallow_hash(col_new_sliced[1]));
    EXPECT_EQ(shallow_hash(col_sliced[2]), shallow_hash(col_new_sliced[2]));
  }

  // column_view, bit_cast         = diff hash
  {
    if constexpr (std::is_integral_v<TypeParam> and not std::is_same_v<TypeParam, bool>) {
      using newType    = std::conditional_t<std::is_signed_v<TypeParam>,
                                            std::make_unsigned_t<TypeParam>,
                                            std::make_signed_t<TypeParam>>;
      auto new_type    = cudf::data_type(cudf::type_to_id<newType>());
      auto col_bitcast = cudf::bit_cast(col_view, new_type);
      EXPECT_NE(shallow_hash(col_view), shallow_hash(col_bitcast));
    }
  }
}

TYPED_TEST(ColumnViewShallowTests, shallow_hash_mutable)
{
  using namespace cudf::detail;
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};
  // mutable_column_view, column_view = same hash
  {
    auto col_mutable = cudf::mutable_column_view{*col};
    EXPECT_EQ(shallow_hash(col_mutable), shallow_hash(col_view));
  }
  // mutable_column_view, modified mutable_column_view = same hash
  // update the children column data = same hash
  {
    auto col_mutable = cudf::mutable_column_view{*col};
    if constexpr (cudf::is_fixed_width<TypeParam>()) {
      // Update data
      auto data = reinterpret_cast<cudf::bitmask_type*>(col->mutable_view().head());
      cudf::set_null_mask(data, 1, 32, false);
    } else {
      // Update child(0).data
      auto data = reinterpret_cast<cudf::bitmask_type*>(col->child(0).mutable_view().head());
      cudf::set_null_mask(data, 1, 32, false);
    }
    EXPECT_EQ(shallow_hash(col_view), shallow_hash(col_mutable));
    auto col_mutable_new = cudf::mutable_column_view{*col};
    EXPECT_EQ(shallow_hash(col_mutable), shallow_hash(col_mutable_new));
  }
  // update the children column_views = diff hash
  {
    if constexpr (cudf::is_nested<TypeParam>()) {
      col->child(0).set_null_mask(
        cudf::create_null_mask(col->child(0).size(), cudf::mask_state::ALL_NULL),
        col->child(0).size());
      auto col_child_updated = cudf::mutable_column_view{*col};
      EXPECT_NE(shallow_hash(col_view), shallow_hash(col_child_updated));
    }
  }
}

TYPED_TEST(ColumnViewShallowTests, is_shallow_equivalent_basic)
{
  using namespace cudf::detail;
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};
  // same = same hash
  {
    EXPECT_TRUE(is_shallow_equivalent(col_view, col_view));
  }
  // copy column_view = same hash
  {
    auto col_view_copy = col_view;
    EXPECT_TRUE(is_shallow_equivalent(col_view, col_view_copy));
  }

  // new column_view from column = same hash
  {
    auto col_view_new = cudf::column_view{*col};
    EXPECT_TRUE(is_shallow_equivalent(col_view, col_view_new));
  }

  // copy column = diff hash
  {
    auto col_new       = std::make_unique<cudf::column>(*col);
    auto col_view_copy = col_new->view();
    EXPECT_FALSE(is_shallow_equivalent(col_view, col_view_copy));
  }

  // column_view, diff column = diff hash.
  {
    auto col_diff      = example_column<TypeParam>();
    auto col_view_diff = cudf::column_view{*col_diff};
    EXPECT_FALSE(is_shallow_equivalent(col_view, col_view_diff));
  }
}
TYPED_TEST(ColumnViewShallowTests, is_shallow_equivalent_update_data)
{
  using namespace cudf::detail;
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};
  // update data + new column_view = same hash.
  {
    // update data by modifying some bits: fixed_width, string, dict, list, struct
    if constexpr (cudf::is_fixed_width<TypeParam>()) {
      // Update data
      auto data = reinterpret_cast<cudf::bitmask_type*>(col->mutable_view().head());
      cudf::set_null_mask(data, 2, 64, true);
    } else {
      // Update child(0).data
      auto data = reinterpret_cast<cudf::bitmask_type*>(col->child(0).mutable_view().head());
      cudf::set_null_mask(data, 2, 64, true);
    }
    auto col_view_new = cudf::column_view{*col};
    EXPECT_TRUE(is_shallow_equivalent(col_view, col_view_new));
  }
  // add null_mask + new column_view = diff hash.
  {
    col->set_null_mask(cudf::create_null_mask(col->size(), cudf::mask_state::ALL_VALID), 0);
    auto col_view_new = cudf::column_view{*col};
    EXPECT_FALSE(is_shallow_equivalent(col_view, col_view_new));
    [[maybe_unused]] auto const nulls = col_view_new.null_count();
    EXPECT_FALSE(is_shallow_equivalent(col_view, col_view_new));
    auto col_view_new2 = cudf::column_view{*col};
    EXPECT_TRUE(is_shallow_equivalent(col_view_new, col_view_new2));
  }
  col_view = cudf::column_view{*col};  // updating after adding null_mask
  // update nulls + new column_view = same hash.
  {
    cudf::set_null_mask(col->mutable_view().null_mask(), 2, 4, false);
    auto col_view_new = cudf::column_view{*col};
    EXPECT_TRUE(is_shallow_equivalent(col_view, col_view_new));
  }
  // set_null_count + new column_view = same hash.
  {
    col->set_null_count(col->size());
    auto col_view_new2 = cudf::column_view{*col};
    EXPECT_TRUE(is_shallow_equivalent(col_view, col_view_new2));
  }
}

TYPED_TEST(ColumnViewShallowTests, is_shallow_equivalent_slice)
{
  using namespace cudf::detail;
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};
  // column_view, sliced[0, size)  = same hash (for split too)
  {
    auto col_sliced = cudf::slice(col_view, {0, col_view.size()});
    EXPECT_TRUE(is_shallow_equivalent(col_view, col_sliced[0]));
    auto col_split = cudf::split(col_view, {0});
    EXPECT_FALSE(is_shallow_equivalent(col_view, col_split[0]));
    EXPECT_TRUE(is_shallow_equivalent(col_view, col_split[1]));
  }
  // column_view, sliced[n:]       = diff hash (for split too)
  {
    auto col_sliced = cudf::slice(col_view, {1, col_view.size()});
    EXPECT_FALSE(is_shallow_equivalent(col_view, col_sliced[0]));
    auto col_split = cudf::split(col_view, {1});
    EXPECT_FALSE(is_shallow_equivalent(col_view, col_split[0]));
    EXPECT_FALSE(is_shallow_equivalent(col_view, col_split[1]));
  }
  // column_view, col copy sliced[0, 0)  = same hash (empty column)
  {
    auto col_new        = std::make_unique<cudf::column>(*col);
    auto col_new_view   = col_new->view();
    auto col_sliced     = cudf::slice(col_view, {0, 0, 1, 1, col_view.size(), col_view.size()});
    auto col_new_sliced = cudf::slice(col_new_view, {0, 0, 1, 1, col_view.size(), col_view.size()});

    EXPECT_TRUE(is_shallow_equivalent(col_sliced[0], col_sliced[1]));
    EXPECT_TRUE(is_shallow_equivalent(col_sliced[1], col_sliced[2]));
    EXPECT_TRUE(is_shallow_equivalent(col_sliced[0], col_new_sliced[0]));
    EXPECT_TRUE(is_shallow_equivalent(col_sliced[1], col_new_sliced[1]));
    EXPECT_TRUE(is_shallow_equivalent(col_sliced[2], col_new_sliced[2]));
  }

  // column_view, bit_cast         = diff hash
  {
    if constexpr (std::is_integral_v<TypeParam> and not std::is_same_v<TypeParam, bool>) {
      using newType    = std::conditional_t<std::is_signed_v<TypeParam>,
                                            std::make_unsigned_t<TypeParam>,
                                            std::make_signed_t<TypeParam>>;
      auto new_type    = cudf::data_type(cudf::type_to_id<newType>());
      auto col_bitcast = cudf::bit_cast(col_view, new_type);
      EXPECT_FALSE(is_shallow_equivalent(col_view, col_bitcast));
    }
  }
}

TYPED_TEST(ColumnViewShallowTests, is_shallow_equivalent_mutable)
{
  using namespace cudf::detail;
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};
  // mutable_column_view, column_view = same hash
  {
    auto col_mutable = cudf::mutable_column_view{*col};
    EXPECT_TRUE(is_shallow_equivalent(col_mutable, col_view));
  }
  // mutable_column_view, modified mutable_column_view = same hash
  // update the children column data = same hash
  {
    auto col_mutable = cudf::mutable_column_view{*col};
    if constexpr (cudf::is_fixed_width<TypeParam>()) {
      // Update data
      auto data = reinterpret_cast<cudf::bitmask_type*>(col->mutable_view().head());
      cudf::set_null_mask(data, 1, 32, false);
    } else {
      // Update child(0).data
      auto data = reinterpret_cast<cudf::bitmask_type*>(col->child(0).mutable_view().head());
      cudf::set_null_mask(data, 1, 32, false);
    }
    EXPECT_TRUE(is_shallow_equivalent(col_view, col_mutable));
    auto col_mutable_new = cudf::mutable_column_view{*col};
    EXPECT_TRUE(is_shallow_equivalent(col_mutable, col_mutable_new));
  }
  // update the children column_views = diff hash
  {
    if constexpr (cudf::is_nested<TypeParam>()) {
      col->child(0).set_null_mask(
        cudf::create_null_mask(col->child(0).size(), cudf::mask_state::ALL_NULL), col->size());
      auto col_child_updated = cudf::mutable_column_view{*col};
      EXPECT_FALSE(is_shallow_equivalent(col_view, col_child_updated));
    }
  }
}
