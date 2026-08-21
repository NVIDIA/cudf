/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/callback_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace {

class host_func_gate {
 public:
  ~host_func_gate() { release(); }

  void wait()
  {
    std::unique_lock<std::mutex> lock{mutex_};
    EXPECT_TRUE(condition_.wait_for(lock, std::chrono::seconds{10}, [this] { return released_; }));
    complete_.store(true);
  }

  void release()
  {
    {
      std::lock_guard<std::mutex> lock{mutex_};
      released_ = true;
    }
    condition_.notify_one();
  }

  bool complete() const { return complete_.load(); }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool released_{false};
  std::atomic<bool> complete_{false};
};

class lifetime_test_scalar : public cudf::numeric_scalar<int32_t> {
 public:
  using numeric_scalar::numeric_scalar;

  void set_data_async(int32_t const& value, cuda::stream_ref stream)
  {
    this->set_value(value, stream);
  }
};

}  // namespace

template <typename T>
struct TypedScalarTest : public cudf::test::BaseFixture {};

template <typename T>
struct TypedScalarTestWithoutFixedPoint : public cudf::test::BaseFixture {};

struct ScalarTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedScalarTest, cudf::test::FixedWidthTypes);
TYPED_TEST_SUITE(TypedScalarTestWithoutFixedPoint, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(TypedScalarTest, DefaultValidity)
{
  using Type = cudf::device_storage_type_t<TypeParam>;
  Type value = static_cast<Type>(cudf::test::make_type_param_scalar<TypeParam>(7));
  cudf::scalar_type_t<TypeParam> s(value);

  EXPECT_TRUE(s.is_valid());
  EXPECT_EQ(value, s.value());
}

TYPED_TEST(TypedScalarTest, ConstructNull)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(5);
  cudf::scalar_type_t<TypeParam> s(value, false);

  EXPECT_FALSE(s.is_valid());
}

TYPED_TEST(TypedScalarTestWithoutFixedPoint, SetValue)
{
  TypeParam init  = cudf::test::make_type_param_scalar<TypeParam>(0);
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(9);
  cudf::scalar_type_t<TypeParam> s(init, true);
  s.set_value(value);

  EXPECT_TRUE(s.is_valid());
  EXPECT_EQ(value, s.value());
}

TEST_F(ScalarTest, AsyncSetValueOwnsHostSource)
{
  rmm::cuda_stream stream;
  auto const stream_ref = cuda::stream_ref{stream.value()};
  int32_t source        = 42;
  lifetime_test_scalar scalar{0, true, stream_ref};
  host_func_gate gate;
  CUDF_CUDA_TRY(cudaLaunchHostFunc(
    stream.value(), [](void* data) { static_cast<host_func_gate*>(data)->wait(); }, &gate));

  scalar.set_data_async(source, stream_ref);
  source = -1;
  EXPECT_FALSE(gate.complete());

  gate.release();
  EXPECT_EQ(42, scalar.value(stream_ref));
}

TEST_F(ScalarTest, AsyncStringConstructionOwnsHostSource)
{
  rmm::cuda_stream stream;
  auto const stream_ref = cuda::stream_ref{stream.value()};
  host_func_gate gate;
  auto upstream = cudf::get_current_device_resource_ref();
  int allocations{0};
  rmm::mr::callback_memory_resource mr{
    [upstream, &gate, &allocations](
      std::size_t bytes, rmm::cuda_stream_view stream, void*) mutable {
      auto* ptr = upstream.allocate(stream, bytes, cuda::mr::default_cuda_malloc_alignment);
      if (allocations++ == 1) {
        CUDF_CUDA_TRY(cudaLaunchHostFunc(
          stream.value(), [](void* data) { static_cast<host_func_gate*>(data)->wait(); }, &gate));
      }
      return ptr;
    },
    [upstream](void* ptr, std::size_t bytes, rmm::cuda_stream_view stream, void*) mutable {
      upstream.deallocate(stream, ptr, bytes, cuda::mr::default_cuda_malloc_alignment);
    }};
  std::string expected{"expected"};
  auto source = cudf::detail::make_pinned_vector<char>(expected.size(), stream_ref);
  std::copy(expected.begin(), expected.end(), source.begin());

  cudf::string_scalar scalar{std::string_view{source.data(), source.size()},
                             true,
                             stream_ref,
                             rmm::device_async_resource_ref{mr}};
  std::fill(source.begin(), source.end(), 'x');
  EXPECT_FALSE(gate.complete());

  gate.release();
  EXPECT_EQ(expected, scalar.to_string(stream_ref));
}

TYPED_TEST(TypedScalarTestWithoutFixedPoint, SetNull)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(6);
  cudf::scalar_type_t<TypeParam> s(value, true);
  s.set_valid_async(false);

  EXPECT_FALSE(s.is_valid());
}

TYPED_TEST(TypedScalarTest, CopyConstructor)
{
  using Type = cudf::device_storage_type_t<TypeParam>;
  Type value = static_cast<Type>(cudf::test::make_type_param_scalar<TypeParam>(8));
  cudf::scalar_type_t<TypeParam> s(value);
  auto s2 = s;

  EXPECT_TRUE(s2.is_valid());
  EXPECT_EQ(value, s2.value());
}

TYPED_TEST(TypedScalarTest, MoveConstructor)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(8);
  cudf::scalar_type_t<TypeParam> s(value);
  auto data_ptr = s.data();
  auto mask_ptr = s.validity_data();
  decltype(s) s2(std::move(s));

  EXPECT_EQ(mask_ptr, s2.validity_data());
  EXPECT_EQ(data_ptr, s2.data());
}

struct StringScalarTest : public cudf::test::BaseFixture {};

TEST_F(StringScalarTest, DefaultValidity)
{
  std::string value = "test string";
  auto s            = cudf::string_scalar(value);

  EXPECT_TRUE(s.is_valid());
  EXPECT_EQ(value, s.to_string());
}

TEST_F(StringScalarTest, CopyConstructor)
{
  std::string value = "test_string";
  auto s            = cudf::string_scalar(value);
  auto s2           = s;

  EXPECT_TRUE(s2.is_valid());
  EXPECT_EQ(value, s2.to_string());
}

TEST_F(StringScalarTest, MoveConstructor)
{
  std::string value = "another test string";
  auto s            = cudf::string_scalar(value);
  auto data_ptr     = s.data();
  auto mask_ptr     = s.validity_data();
  decltype(s) s2(std::move(s));

  EXPECT_EQ(mask_ptr, s2.validity_data());
  EXPECT_EQ(data_ptr, s2.data());
}

TEST_F(StringScalarTest, OverflowCheckedBeforeAllocation)
{
  char source{};
  auto const oversized = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()) + 1;
  EXPECT_THROW(cudf::string_scalar(std::string_view{&source, oversized}), std::overflow_error);
}

struct ListScalarTest : public cudf::test::BaseFixture {};

TEST_F(ListScalarTest, DefaultValidityNonNested)
{
  auto data = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto s    = cudf::list_scalar(data);

  EXPECT_TRUE(s.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s.view());
}

TEST_F(ListScalarTest, DefaultValidityNested)
{
  auto data = cudf::test::lists_column_wrapper<int32_t>{{1, 2}, {2}, {}, {4, 5}};
  auto s    = cudf::list_scalar(data);

  EXPECT_TRUE(s.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s.view());
}

TEST_F(ScalarTest, OneRowColumnLayoutBaseline)
{
  auto const numeric      = cudf::numeric_scalar<int32_t>{42};
  auto const null_numeric = cudf::numeric_scalar<int32_t>{42, false};
  auto const decimal = cudf::fixed_point_scalar<numeric::decimal64>{1234, numeric::scale_type{-2}};
  auto const decimal128 =
    cudf::fixed_point_scalar<numeric::decimal128>{__int128_t{5678}, numeric::scale_type{-7}};
  auto const raw_int128      = cudf::numeric_scalar<__int128_t>{__int128_t{9012}};
  auto const string          = cudf::string_scalar{"scalar"};
  auto const null_string     = cudf::string_scalar{"null scalar", false};
  auto const list_elements   = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const list            = cudf::list_scalar{list_elements};
  auto const null_list       = cudf::list_scalar{list_elements, false};
  auto const nested_elements = cudf::test::lists_column_wrapper<int32_t>{{1, 2}, {}, {3}};
  auto const nested_list     = cudf::list_scalar{nested_elements};

  auto const numeric_column      = numeric.as_column_view();
  auto const null_numeric_column = null_numeric.as_column_view();
  auto const decimal_column      = decimal.as_column_view();
  auto const decimal128_column   = decimal128.as_column_view();
  auto const raw_int128_column   = raw_int128.as_column_view();
  auto const string_column       = string.as_column_view();
  auto const null_string_column  = null_string.as_column_view();
  auto const list_column         = list.as_column_view();
  auto const null_list_column    = null_list.as_column_view();
  auto const nested_list_column  = nested_list.as_column_view();

  auto const expected_numeric = cudf::test::fixed_width_column_wrapper<int32_t>{42};
  auto const expected_decimal =
    cudf::test::fixed_point_column_wrapper<int64_t>({1234}, numeric::scale_type{-2});
  auto const expected_decimal128 =
    cudf::test::fixed_point_column_wrapper<__int128_t>({5678}, numeric::scale_type{-7});
  auto const expected_raw_int128 =
    cudf::test::fixed_point_column_wrapper<__int128_t>({9012}, numeric::scale_type{0});
  auto const expected_string = cudf::test::strings_column_wrapper{"scalar"};
  auto const expected_list   = cudf::test::lists_column_wrapper<int32_t>{{1, 2, 3}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_numeric, numeric_column.as_column_view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_decimal, decimal_column.as_column_view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_decimal128, decimal128_column.as_column_view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_raw_int128, raw_int128_column.as_column_view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_string, string_column.as_column_view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_list, list_column.as_column_view());

  EXPECT_EQ(1, null_numeric_column.null_count());
  EXPECT_TRUE(null_numeric_column.nullable());
  EXPECT_EQ(cudf::data_type(cudf::type_id::DECIMAL128, 0), raw_int128_column.type());
  EXPECT_EQ(1, null_string_column.null_count());
  EXPECT_EQ(1, null_list_column.null_count());
  EXPECT_EQ(cudf::type_id::STRING, null_string_column.type().id());
  EXPECT_EQ(cudf::type_id::LIST, null_list_column.type().id());
  EXPECT_EQ(2, null_string_column.as_column_view().child(0).size());
  EXPECT_EQ(3, null_list_column.as_column_view().child(1).size());
  EXPECT_EQ(cudf::type_id::LIST, nested_list_column.as_column_view().child(1).type().id());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(nested_elements, nested_list_column.as_column_view().child(1));
  EXPECT_EQ(numeric.data(), numeric_column.data<int32_t>());
  EXPECT_EQ(string.data(), string_column.data<char>());
  EXPECT_EQ(list.view().data<int32_t>(), list_column.as_column_view().child(1).data<int32_t>());
}

TEST_F(ListScalarTest, MoveColumnConstructor)
{
  auto data = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto col  = cudf::column(data);
  auto ptr  = col.view().data<int32_t>();
  auto s    = cudf::list_scalar(std::move(col));

  EXPECT_TRUE(s.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s.view());
  EXPECT_EQ(ptr, s.view().data<int32_t>());
}

TEST_F(ListScalarTest, CopyConstructorNonNested)
{
  auto data = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto s    = cudf::list_scalar(data);
  auto s2   = s;

  EXPECT_TRUE(s2.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s2.view());
  EXPECT_NE(s.view().data<int32_t>(), s2.view().data<int32_t>());
}

TEST_F(ListScalarTest, CopyConstructorNested)
{
  auto data = cudf::test::lists_column_wrapper<int32_t>{{1, 2}, {2}, {}, {4, 5}};
  auto s    = cudf::list_scalar(data);
  auto s2   = s;

  EXPECT_TRUE(s2.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s2.view());
  EXPECT_NE(s.view().child(0).data<int32_t>(), s2.view().child(0).data<int32_t>());
  EXPECT_NE(s.view().child(1).data<int32_t>(), s2.view().child(1).data<int32_t>());
}

TEST_F(ListScalarTest, MoveConstructorNonNested)
{
  auto data     = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto s        = cudf::list_scalar(data);
  auto data_ptr = s.view().data<int32_t>();
  auto mask_ptr = s.validity_data();
  decltype(s) s2(std::move(s));

  EXPECT_EQ(mask_ptr, s2.validity_data());
  EXPECT_EQ(data_ptr, s2.view().data<int32_t>());
  EXPECT_EQ(s.view().data<int32_t>(), nullptr);  // NOLINT
}

TEST_F(ListScalarTest, MoveConstructorNested)
{
  auto data       = cudf::test::lists_column_wrapper<int32_t>{{1, 2}, {2}, {}, {4, 5}};
  auto s          = cudf::list_scalar(data);
  auto offset_ptr = s.view().child(0).data<cudf::size_type>();
  auto data_ptr   = s.view().child(1).data<int32_t>();
  auto mask_ptr   = s.validity_data();
  decltype(s) s2(std::move(s));

  EXPECT_EQ(mask_ptr, s2.validity_data());
  EXPECT_EQ(offset_ptr, s2.view().child(0).data<cudf::size_type>());
  EXPECT_EQ(data_ptr, s2.view().child(1).data<int32_t>());
  EXPECT_EQ(s.view().data<int32_t>(), nullptr);  // NOLINT
  EXPECT_EQ(s.view().num_children(), 0);         // NOLINT
}

struct StructScalarTest : public cudf::test::BaseFixture {};

TEST_F(StructScalarTest, Basic)
{
  cudf::test::fixed_width_column_wrapper<int> col0{1};
  cudf::test::strings_column_wrapper col1{"abc"};
  cudf::test::lists_column_wrapper<int> col2{{1, 2, 3}};
  cudf::test::structs_column_wrapper struct_col({col0, col1, col2});
  cudf::column_view cv = static_cast<cudf::column_view>(struct_col);
  std::vector<cudf::column_view> children(cv.child_begin(), cv.child_end());

  // table_view constructor
  {
    auto s = cudf::struct_scalar(children, true);
    EXPECT_TRUE(s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{children}, s.view());
  }

  // host_span constructor
  {
    auto s = cudf::struct_scalar(cudf::host_span<cudf::column_view const>{children}, true);
    EXPECT_TRUE(s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{children}, s.view());
  }
}

TEST_F(StructScalarTest, BasicNulls)
{
  cudf::test::fixed_width_column_wrapper<int> col0{1};
  cudf::test::strings_column_wrapper col1{"abc"};
  cudf::test::lists_column_wrapper<int> col2{{1, 2, 3}};
  std::vector<cudf::column_view> src_children({col0, col1, col2});

  std::vector<std::unique_ptr<cudf::column>> src_columns;

  // structs_column_wrapper takes ownership of the incoming columns, so make a copy
  src_columns.push_back(std::make_unique<cudf::column>(src_children[0]));
  src_columns.push_back(std::make_unique<cudf::column>(src_children[1]));
  src_columns.push_back(std::make_unique<cudf::column>(src_children[2]));
  cudf::test::structs_column_wrapper valid_struct_col(std::move(src_columns), {true});
  cudf::column_view vcv = static_cast<cudf::column_view>(valid_struct_col);
  std::vector<cudf::column_view> valid_children(vcv.child_begin(), vcv.child_end());

  // structs_column_wrapper takes ownership of the incoming columns, so make a copy
  src_columns.push_back(std::make_unique<cudf::column>(src_children[0]));
  src_columns.push_back(std::make_unique<cudf::column>(src_children[1]));
  src_columns.push_back(std::make_unique<cudf::column>(src_children[2]));
  cudf::test::structs_column_wrapper invalid_struct_col(std::move(src_columns), {false});
  cudf::column_view icv = static_cast<cudf::column_view>(invalid_struct_col);
  std::vector<cudf::column_view> invalid_children(icv.child_begin(), icv.child_end());

  // table_view constructor
  {
    auto s = cudf::struct_scalar(cudf::table_view{src_children}, true);
    EXPECT_TRUE(s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{valid_children}, s.view());
  }
  // host_span constructor
  {
    auto s = cudf::struct_scalar(cudf::host_span<cudf::column_view const>{src_children}, true);
    EXPECT_TRUE(s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{valid_children}, s.view());
  }

  // with nulls, we expect the incoming children to get nullified by passing false to
  // the scalar constructor itself. so we use the unmodified `children` as the input, but
  // we compare against the modified `invalid_children` produced by the source column as
  // proof that the scalar did the validity pushdown.

  // table_view constructor
  {
    auto s = cudf::struct_scalar(cudf::table_view{src_children}, false);
    EXPECT_TRUE(!s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{invalid_children}, s.view());
  }

  // host_span constructor
  {
    auto s = cudf::struct_scalar(cudf::host_span<cudf::column_view const>{src_children}, false);
    EXPECT_TRUE(!s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{invalid_children}, s.view());
  }
}

CUDF_TEST_PROGRAM_MAIN()
