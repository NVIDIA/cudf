/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <numeric>
#include <random>
#include <vector>

template <typename T>
struct TypedColumnTest : public cudf::test::BaseFixture {
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }

  TypedColumnTest(cuda::stream_ref stream = cudf::get_default_stream())
    : data{_num_elements * sizeof(T), stream},
      mask{cudf::bitmask_allocation_size_bytes(_num_elements), stream}
  {
    std::vector<char> h_data(std::max(data.size(), mask.size()));
    std::iota(h_data.begin(), h_data.end(), 0);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(data.data(), h_data.data(), data.size(), cudaMemcpyDefault, stream.get()));
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(mask.data(), h_data.data(), mask.size(), cudaMemcpyDefault, stream.get()));
  }

  cudf::size_type num_elements() { return _num_elements; }

  std::random_device r;
  std::default_random_engine generator{r()};
  std::uniform_int_distribution<cudf::size_type> distribution{200, 1000};
  cudf::size_type _num_elements{distribution(generator)};
  rmm::device_buffer data{};
  rmm::device_buffer mask{};
  rmm::device_buffer all_valid_mask{create_null_mask(num_elements(), cudf::mask_state::ALL_VALID)};
  rmm::device_buffer all_null_mask{create_null_mask(num_elements(), cudf::mask_state::ALL_NULL)};
};

TYPED_TEST_SUITE(TypedColumnTest, cudf::test::Types<int32_t>);

/**
 * @brief Verifies equality of the properties and data of a `column`'s views.
 *
 * @param col The `column` to verify
 */
void verify_column_views(cudf::column col)
{
  cudf::column_view view                 = col;
  cudf::mutable_column_view mutable_view = col;
  EXPECT_EQ(col.type(), view.type());
  EXPECT_EQ(col.type(), mutable_view.type());
  EXPECT_EQ(col.size(), view.size());
  EXPECT_EQ(col.size(), mutable_view.size());
  EXPECT_EQ(col.null_count(), view.null_count());
  EXPECT_EQ(col.null_count(), mutable_view.null_count());
  EXPECT_EQ(col.nullable(), view.nullable());
  EXPECT_EQ(col.nullable(), mutable_view.nullable());
  EXPECT_EQ(col.num_children(), view.num_children());
  EXPECT_EQ(col.num_children(), mutable_view.num_children());
  EXPECT_EQ(view.head(), mutable_view.head());
  EXPECT_EQ(view.data<char>(), mutable_view.data<char>());
  EXPECT_EQ(view.offset(), mutable_view.offset());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountNoMask)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};
  EXPECT_FALSE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountEmptyMask)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};
  EXPECT_FALSE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountAllValid)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  EXPECT_TRUE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, ExplicitNullCountAllValid)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  EXPECT_TRUE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountAllNull)
{
  cudf::column col{this->type(),
                   this->num_elements(),
                   std::move(this->data),
                   std::move(this->all_null_mask),
                   this->num_elements()};
  EXPECT_TRUE(col.nullable());
  EXPECT_TRUE(col.has_nulls());
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, ExplicitNullCountAllNull)
{
  cudf::column col{this->type(),
                   this->num_elements(),
                   std::move(this->data),
                   std::move(this->all_null_mask),
                   this->num_elements()};
  EXPECT_TRUE(col.nullable());
  EXPECT_TRUE(col.has_nulls());
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, SetNullCountNoMask)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};
  EXPECT_THROW(col.set_null_count(1), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetEmptyNullMaskNonZeroNullCount)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};
  rmm::device_buffer empty_null_mask{};
  EXPECT_THROW(col.set_null_mask(std::move(empty_null_mask), this->num_elements()),
               cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetInvalidSizeNullMaskNonZeroNullCount)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};
  auto invalid_size_null_mask =
    create_null_mask(std::min(this->num_elements() - 50, 0), cudf::mask_state::ALL_VALID);
  EXPECT_THROW(
    col.set_null_mask(invalid_size_null_mask, this->num_elements(), cudf::get_default_stream()),
    cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetNullCountEmptyMask)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};
  EXPECT_THROW(col.set_null_count(1), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetNullCountAllValid)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  EXPECT_NO_THROW(col.set_null_count(0));
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, SetNullCountAllNull)
{
  cudf::column col{this->type(),
                   this->num_elements(),
                   std::move(this->data),
                   std::move(this->all_null_mask),
                   this->num_elements()};
  EXPECT_NO_THROW(col.set_null_count(this->num_elements()));
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, ResetNullCountAllNull)
{
  cudf::column col{this->type(),
                   this->num_elements(),
                   std::move(this->data),
                   std::move(this->all_null_mask),
                   this->num_elements()};

  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, ResetNullCountAllValid)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, CopyDataNoMask)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_FALSE(col.nullable());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify deep copy
  cudf::column_view v = col;
  EXPECT_NE(v.head(), this->data.data());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(v.head(), this->data.data(), this->data.size());
}

TYPED_TEST(TypedColumnTest, MoveDataNoMask)
{
  void* original_data = this->data.data();
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_FALSE(col.nullable());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify shallow copy
  cudf::column_view v = col;
  EXPECT_EQ(v.head(), original_data);
}

TYPED_TEST(TypedColumnTest, CopyDataAndMask)
{
  cudf::column col{this->type(),
                   this->num_elements(),
                   rmm::device_buffer{this->data, cudf::get_default_stream()},
                   rmm::device_buffer{this->all_valid_mask, cudf::get_default_stream()},
                   0};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_TRUE(col.nullable());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify deep copy
  cudf::column_view v = col;
  EXPECT_NE(v.head(), this->data.data());
  EXPECT_NE(v.null_mask(), this->all_valid_mask.data());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(v.head(), this->data.data(), this->data.size());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(v.null_mask(), this->all_valid_mask.data(), this->mask.size());
}

TYPED_TEST(TypedColumnTest, MoveDataAndMask)
{
  void* original_data = this->data.data();
  void* original_mask = this->all_valid_mask.data();
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_TRUE(col.nullable());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify shallow copy
  cudf::column_view v = col;
  EXPECT_EQ(v.head(), original_data);
  EXPECT_EQ(v.null_mask(), original_mask);
}

TYPED_TEST(TypedColumnTest, CopyConstructorNoMask)
{
  cudf::column original{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};
  cudf::column copy{original};
  verify_column_views(copy);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(original, copy);

  // Verify deep copy
  cudf::column_view original_view = original;
  cudf::column_view copy_view     = copy;
  EXPECT_NE(original_view.head(), copy_view.head());
}

TYPED_TEST(TypedColumnTest, CopyConstructorWithMask)
{
  cudf::column original{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  cudf::column copy{original};
  verify_column_views(copy);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(original, copy);

  // Verify deep copy
  cudf::column_view original_view = original;
  cudf::column_view copy_view     = copy;
  EXPECT_NE(original_view.head(), copy_view.head());
  EXPECT_NE(original_view.null_mask(), copy_view.null_mask());
}

TYPED_TEST(TypedColumnTest, MoveConstructorNoMask)
{
  cudf::column original{
    this->type(), this->num_elements(), std::move(this->data), rmm::device_buffer{}, 0};

  auto original_data = original.view().head();

  cudf::column moved_to{std::move(original)};

  EXPECT_EQ(0, original.size());  // NOLINT
  EXPECT_EQ(cudf::data_type{cudf::type_id::EMPTY}, original.type());

  verify_column_views(moved_to);

  // Verify move
  cudf::column_view moved_to_view = moved_to;
  EXPECT_EQ(original_data, moved_to_view.head());
}

TYPED_TEST(TypedColumnTest, MoveConstructorWithMask)
{
  cudf::column original{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  auto original_data = original.view().head();
  auto original_mask = original.view().null_mask();
  cudf::column moved_to{std::move(original)};
  verify_column_views(moved_to);

  EXPECT_EQ(0, original.size());  // NOLINT
  EXPECT_EQ(cudf::data_type{cudf::type_id::EMPTY}, original.type());

  // Verify move
  cudf::column_view moved_to_view = moved_to;
  EXPECT_EQ(original_data, moved_to_view.head());
  EXPECT_EQ(original_mask, moved_to_view.null_mask());
}

TYPED_TEST(TypedColumnTest, DeviceUvectorConstructorNoMask)
{
  auto data = cudf::device_span<TypeParam const>(static_cast<TypeParam*>(this->data.data()),
                                                 this->num_elements());

  auto original = cudf::detail::make_device_uvector_async(
    data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto original_data = original.data();
  cudf::column moved_to{std::move(original), rmm::device_buffer{}, 0};
  verify_column_views(moved_to);

  // Verify move
  cudf::column_view moved_to_view = moved_to;
  EXPECT_EQ(original_data, moved_to_view.head());
}

TYPED_TEST(TypedColumnTest, DeviceUvectorConstructorWithMask)
{
  auto data = cudf::device_span<TypeParam const>(static_cast<TypeParam*>(this->data.data()),
                                                 this->num_elements());

  auto original = cudf::detail::make_device_uvector_async(
    data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto original_data = original.data();
  auto original_mask = this->all_valid_mask.data();
  cudf::column moved_to{std::move(original), std::move(this->all_valid_mask), 0};
  verify_column_views(moved_to);

  // Verify move
  cudf::column_view moved_to_view = moved_to;
  EXPECT_EQ(original_data, moved_to_view.head());
  EXPECT_EQ(original_mask, moved_to_view.null_mask());
}

TYPED_TEST(TypedColumnTest, ConstructWithChildren)
{
  std::vector<std::unique_ptr<cudf::column>> children;

  children.emplace_back(std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT8},
    42,
    rmm::device_buffer{this->data, cudf::get_default_stream()},
    rmm::device_buffer{this->all_valid_mask, cudf::get_default_stream()},
    0));
  children.emplace_back(std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::FLOAT64},
    314,
    rmm::device_buffer{this->data, cudf::get_default_stream()},
    rmm::device_buffer{this->all_valid_mask, cudf::get_default_stream()},
    0));
  cudf::column col{this->type(),
                   this->num_elements(),
                   rmm::device_buffer{this->data, cudf::get_default_stream()},
                   rmm::device_buffer{this->all_valid_mask, cudf::get_default_stream()},
                   0,
                   std::move(children)};

  verify_column_views(col);
  EXPECT_EQ(2, col.num_children());
  EXPECT_EQ(cudf::data_type{cudf::type_id::INT8}, col.child(0).type());
  EXPECT_EQ(42, col.child(0).size());
  EXPECT_EQ(cudf::data_type{cudf::type_id::FLOAT64}, col.child(1).type());
  EXPECT_EQ(314, col.child(1).size());
}

TYPED_TEST(TypedColumnTest, ReleaseNoChildren)
{
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  auto original_data = col.view().head();
  auto original_mask = col.view().null_mask();

  cudf::column::contents contents = col.release();
  EXPECT_EQ(original_data, contents.data->data());
  EXPECT_EQ(original_mask, contents.null_mask->data());
  EXPECT_EQ(0u, contents.children.size());
  EXPECT_EQ(0, col.size());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(cudf::data_type{cudf::type_id::EMPTY}, col.type());
  EXPECT_EQ(0, col.num_children());
}

TYPED_TEST(TypedColumnTest, ReleaseWithChildren)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(std::make_unique<cudf::column>(
    this->type(),
    this->num_elements(),
    rmm::device_buffer{this->data, cudf::get_default_stream()},
    rmm::device_buffer{this->all_valid_mask, cudf::get_default_stream()},
    0));
  children.emplace_back(std::make_unique<cudf::column>(
    this->type(),
    this->num_elements(),
    rmm::device_buffer{this->data, cudf::get_default_stream()},
    rmm::device_buffer{this->all_valid_mask, cudf::get_default_stream()},
    0));
  cudf::column col{this->type(),
                   this->num_elements(),
                   rmm::device_buffer{this->data, cudf::get_default_stream()},
                   rmm::device_buffer{this->all_valid_mask, cudf::get_default_stream()},
                   0,
                   std::move(children)};

  auto original_data = col.view().head();
  auto original_mask = col.view().null_mask();

  cudf::column::contents contents = col.release();
  EXPECT_EQ(original_data, contents.data->data());
  EXPECT_EQ(original_mask, contents.null_mask->data());
  EXPECT_EQ(2u, contents.children.size());
  EXPECT_EQ(0, col.size());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(cudf::data_type{cudf::type_id::EMPTY}, col.type());
  EXPECT_EQ(0, col.num_children());
}

TYPED_TEST(TypedColumnTest, ColumnViewConstructorWithMask)
{
  cudf::column original{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  cudf::column_view original_view = original;
  cudf::column copy{original_view};
  verify_column_views(copy);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(original, copy);

  // Verify deep copy
  cudf::column_view copy_view = copy;
  EXPECT_NE(original_view.head(), copy_view.head());
  EXPECT_NE(original_view.null_mask(), copy_view.null_mask());
}

template <typename T>
struct ListsColumnTest : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(ListsColumnTest, NumericTypesNotBool);

TYPED_TEST(ListsColumnTest, ListsColumnViewConstructor)
{
  cudf::test::lists_column_wrapper<TypeParam> list{{1, 2}, {3, 4}, {5, 6, 7}, {8, 9}};

  auto result = std::make_unique<cudf::column>(list);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(list, result->view());
}

TYPED_TEST(ListsColumnTest, ListsSlicedColumnViewConstructor)
{
  cudf::test::lists_column_wrapper<TypeParam> list{{1, 2}, {3, 4}, {5, 6, 7}, {8, 9}};
  cudf::test::lists_column_wrapper<TypeParam> expect{{3, 4}, {5, 6, 7}};

  auto sliced = cudf::slice(list, {1, 3}).front();
  auto result = std::make_unique<cudf::column>(sliced);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, result->view());
}

TYPED_TEST(ListsColumnTest, ListsSlicedIncludesEmpty)
{
  cudf::test::lists_column_wrapper<TypeParam> list{{1, 2}, {}, {3, 4}, {8, 9}};
  cudf::test::lists_column_wrapper<TypeParam> expect{{}, {3, 4}};

  auto sliced = cudf::slice(list, {1, 3}).front();
  auto result = std::make_unique<cudf::column>(sliced);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, result->view());
}

TYPED_TEST(ListsColumnTest, ListsSlicedNonNestedEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;

  // Column of List<int>
  LCW list{{1, 2}, {}, {3, 4}, {8, 9}};
  // Column of 1 row, an empty List<int>
  LCW expect{LCW{}};

  auto sliced = cudf::slice(list, {1, 2}).front();
  auto result = std::make_unique<cudf::column>(sliced);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, result->view());
}

TYPED_TEST(ListsColumnTest, ListsSlicedNestedEmpty)
{
  using LCW     = cudf::test::lists_column_wrapper<TypeParam>;
  using FWCW_SZ = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  // Column of List<List<int>>, with incomplete hierarchy
  LCW list{{LCW{1}, LCW{2}},
           {},  // < ----------- empty List<List<int>>, slice this
           {LCW{3}, LCW{4, 5}}};

  // Make 1-row column of type List<List<int>>, the row data contains 0 element.
  // Well-formed memory layout:
  // type: List<List<int>>
  // Length: 1
  // Mask: 1
  // Offsets: 0, 0
  //    List<int>
  //    Length: 0
  //    Offset:
  //        INT
  //        Length: 0
  auto leaf      = std::make_unique<cudf::column>(cudf::column(LCW{}));
  auto offset    = std::make_unique<cudf::column>(cudf::column(FWCW_SZ{0, 0}));
  auto null_mask = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED);
  auto expect =
    cudf::make_lists_column(1, std::move(offset), std::move(leaf), 0, std::move(null_mask));

  auto sliced = cudf::slice(list, {1, 2}).front();
  auto result = std::make_unique<cudf::column>(sliced);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expect, result->view());
}

TYPED_TEST(ListsColumnTest, ListsSlicedZeroSliceLengthNested)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;

  // Column of List<List<int>>, with incomplete hierarchy
  LCW list{{LCW{1}, LCW{2}}, {}, {LCW{3}, LCW{4, 5}}};

  auto expect = cudf::empty_like(list);

  auto sliced = cudf::slice(list, {0, 0}).front();
  auto result = std::make_unique<cudf::column>(sliced);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expect, result->view());
}

TYPED_TEST(ListsColumnTest, ListsSlicedZeroSliceLengthNonNested)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;

  LCW list{{1, 2}, {}, {3, 4}, {8, 9}};

  auto expect = cudf::empty_like(list);

  auto sliced = cudf::slice(list, {0, 0}).front();
  auto result = std::make_unique<cudf::column>(sliced);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expect, result->view());
}

TYPED_TEST(ListsColumnTest, ListsSlicedColumnViewConstructorWithNulls)
{
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  auto expect_valids = cudf::test::iterators::nulls_at_multiples_of(2);

  using LCW = cudf::test::lists_column_wrapper<TypeParam>;

  cudf::test::lists_column_wrapper<TypeParam> list{
    {{{{1, 2}, {3, 4}}, valids}, LCW{}, {{{5, 6, 7}, LCW{}, {8, 9}}, valids}, LCW{}, LCW{}},
    valids};

  cudf::test::lists_column_wrapper<TypeParam> expect{
    {LCW{}, {{{5, 6, 7}, LCW{}, {8, 9}}, valids}, LCW{}, LCW{}}, expect_valids};

  auto sliced = cudf::slice(list, {1, 5}).front();
  auto result = std::make_unique<cudf::column>(sliced);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, result->view());

  // TODO: null mask equality is being checked separately because
  // expect_columns_equal doesn't do the check for lists columns.
  // This is fixed in https://github.com/rapidsai/cudf/pull/5904,
  // so we should remove this check after that's merged:
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    cudf::mask_to_bools(result->view().null_mask(), 0, 4)->view(),
    cudf::mask_to_bools(static_cast<cudf::column_view>(expect).null_mask(), 0, 4)->view());
}

struct RebindStreamColumnTest : public cudf::test::BaseFixture {};

TEST_F(RebindStreamColumnTest, RebindStreamPreservesNestedStructData)
{
  rmm::cuda_stream stream_a{};
  rmm::cuda_stream stream_b{};

  constexpr cudf::size_type num_rows{4};
  std::vector<int32_t> h_ints(static_cast<std::size_t>(num_rows));
  std::iota(h_ints.begin(), h_ints.end(), int32_t{0});

  auto d_ints = cudf::detail::make_device_uvector_async(
    h_ints, stream_a, cudf::get_current_device_resource_ref());
  auto null_mask = cudf::create_null_mask(
    num_rows, cudf::mask_state::ALL_VALID, stream_a, cudf::get_current_device_resource_ref());

  stream_a.synchronize();

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::make_unique<cudf::column>(std::move(d_ints), std::move(null_mask), 0));

  cudf::test::strings_column_wrapper strings_child{"", "a", "bb", "ccc"};
  children.push_back(strings_child.release());

  cudf::test::lists_column_wrapper<int32_t> lists_child{{1}, {2, 3}, {}, {4, 5}};
  children.push_back(lists_child.release());

  cudf::test::structs_column_wrapper struct_wrapper{std::move(children)};
  auto col = struct_wrapper.release();

  auto const expected = std::make_unique<cudf::column>(col->view(), cudf::get_default_stream());

  auto rebound = cudf::rebind_stream(std::move(*col), stream_b);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), rebound->view());

  cudf::test::fixed_width_column_wrapper<int32_t> idempotent_src{1, 2, 3, 4, 5};
  auto idempotent_col = idempotent_src.release();
  auto const idempotent_expected =
    std::make_unique<cudf::column>(idempotent_col->view(), cudf::get_default_stream());
  auto idempotent_rebound =
    cudf::rebind_stream(std::move(*idempotent_col), cudf::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(idempotent_expected->view(), idempotent_rebound->view());
}

CUDF_TEST_PROGRAM_MAIN()
