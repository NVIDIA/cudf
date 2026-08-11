/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/interop.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/host_vector.h>

#include <dlpack/dlpack.h>

namespace {
struct dlpack_deleter {
  void operator()(DLManagedTensor* tensor) { tensor->deleter(tensor); }
};

using unique_managed_tensor = std::unique_ptr<DLManagedTensor, dlpack_deleter>;

template <typename T>
DLDataType get_dtype()
{
  uint8_t const bits{sizeof(T) * 8};
  uint16_t const lanes{1};
  if (std::is_floating_point_v<T>) {
    return DLDataType{kDLFloat, bits, lanes};
  } else if (std::is_signed_v<T>) {
    return DLDataType{kDLInt, bits, lanes};
  } else if (std::is_unsigned_v<T>) {
    return DLDataType{kDLUInt, bits, lanes};
  } else {
    static_assert(true, "unsupported type");
  }
}

template <typename T>
void validate_dtype(DLDataType const& dtype)
{
  switch (dtype.code) {
    case kDLInt: EXPECT_TRUE(std::is_integral_v<T> && std::is_signed_v<T>); break;
    case kDLUInt: EXPECT_TRUE(std::is_integral_v<T> && std::is_unsigned_v<T>); break;
    case kDLFloat: EXPECT_TRUE(std::is_floating_point_v<T>); break;
    default: FAIL();
  }
  EXPECT_EQ(1, dtype.lanes);
  EXPECT_EQ(sizeof(T) * 8, dtype.bits);
}
}  // namespace

class DLPackUntypedTests : public cudf::test::BaseFixture {};

TEST_F(DLPackUntypedTests, EmptyTableToDlpack)
{
  cudf::table_view empty(std::vector<cudf::column_view>{});
  // No type information to construct a correct empty dlpack object
  EXPECT_EQ(nullptr, cudf::to_dlpack(empty));
}

TEST_F(DLPackUntypedTests, EmptyColsToDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1({});
  cudf::test::fixed_width_column_wrapper<int32_t> col2({});
  cudf::table_view input({col1, col2});
  unique_managed_tensor tensor(cudf::to_dlpack(input));
  validate_dtype<int32_t>(tensor->dl_tensor.dtype);
  EXPECT_NE(nullptr, tensor);
  EXPECT_EQ(nullptr, tensor->dl_tensor.data);
  EXPECT_EQ(2, tensor->dl_tensor.ndim);
  EXPECT_EQ(0, tensor->dl_tensor.strides[0]);
  EXPECT_EQ(0, tensor->dl_tensor.strides[1]);
  EXPECT_EQ(0, tensor->dl_tensor.shape[0]);
  EXPECT_EQ(2, tensor->dl_tensor.shape[1]);
  EXPECT_EQ(kDLCUDA, tensor->dl_tensor.device.device_type);
  auto result = cudf::from_dlpack(tensor.get());
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, result->view());
}

TEST_F(DLPackUntypedTests, NullTensorFromDlpack)
{
  EXPECT_THROW(cudf::from_dlpack(nullptr), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, MultipleTypesToDlpack)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> col2({1, 2, 3, 4});
  cudf::table_view input({col1, col2});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::data_type_error);
}

TEST_F(DLPackUntypedTests, InvalidNullsToDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> col2({1, 2, 3, 4}, {true, false, true, true});
  cudf::table_view input({col1, col2});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, StringTypeToDlpack)
{
  cudf::test::strings_column_wrapper col({"foo", "bar", "baz"});
  cudf::table_view input({col});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedDeviceTypeFromDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported device type
  tensor->dl_tensor.device.device_type = kDLOpenCL;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, InvalidDeviceIdFromDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof the wrong device ID
  tensor->dl_tensor.device.device_id += 1;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedDimsFromDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported number of dims
  tensor->dl_tensor.ndim = 3;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, TooManyRowsFromDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof too many rows
  constexpr int64_t max_size_type{std::numeric_limits<int32_t>::max()};
  tensor->dl_tensor.shape[0] = max_size_type + 1;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), std::overflow_error);
}

TEST_F(DLPackUntypedTests, TooManyColsFromDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> col2({5, 6, 7, 8});
  cudf::table_view input({col1, col2});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof too many cols
  constexpr int64_t max_size_type{std::numeric_limits<int32_t>::max()};
  tensor->dl_tensor.shape[1] = max_size_type + 1;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), std::overflow_error);
}

TEST_F(DLPackUntypedTests, InvalidTypeFromDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an invalid data type
  tensor->dl_tensor.dtype.code = 3;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedIntBitsizeFromDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported bitsize
  tensor->dl_tensor.dtype.bits = 7;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedFloatBitsizeFromDlpack)
{
  cudf::test::fixed_width_column_wrapper<float> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported bitsize
  tensor->dl_tensor.dtype.bits = 7;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedLanesFromDlpack)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported number of lanes
  tensor->dl_tensor.dtype.lanes = 2;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedBroadcast1DTensorFromDlpack)
{
  using T            = float;
  constexpr int ndim = 1;
  // Broadcasted (stride-0) 1D tensor
  auto const data       = cudf::test::make_type_param_vector<T>({1});
  int64_t shape[ndim]   = {5};  // NOLINT
  int64_t strides[ndim] = {0};  // NOLINT

  DLManagedTensor tensor{};
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.dtype              = get_dtype<T>();
  tensor.dl_tensor.ndim               = ndim;
  tensor.dl_tensor.byte_offset        = 0;
  tensor.dl_tensor.shape              = shape;
  tensor.dl_tensor.strides            = strides;

  thrust::host_vector<T> host_vector(data.begin(), data.end());
  tensor.dl_tensor.data = host_vector.data();

  EXPECT_THROW(cudf::from_dlpack(&tensor), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedStrided1DTensorFromDlpack)
{
  using T            = float;
  constexpr int ndim = 1;
  // Strided 1D tensor
  auto const data       = cudf::test::make_type_param_vector<T>({1, 2, 3, 4});
  int64_t shape[ndim]   = {2};  // NOLINT
  int64_t strides[ndim] = {2};  // NOLINT

  DLManagedTensor tensor{};
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.dtype              = get_dtype<T>();
  tensor.dl_tensor.ndim               = ndim;
  tensor.dl_tensor.byte_offset        = 0;
  tensor.dl_tensor.shape              = shape;
  tensor.dl_tensor.strides            = strides;

  thrust::host_vector<T> host_vector(data.begin(), data.end());
  tensor.dl_tensor.data = host_vector.data();

  EXPECT_THROW(cudf::from_dlpack(&tensor), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedImplicitRowMajor2DTensorFromDlpack)
{
  using T            = float;
  constexpr int ndim = 2;
  // Row major 2D tensor
  auto const data     = cudf::test::make_type_param_vector<T>({1, 2, 3, 4});
  int64_t shape[ndim] = {2, 2};  // NOLINT

  DLManagedTensor tensor{};
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.dtype              = get_dtype<T>();
  tensor.dl_tensor.ndim               = ndim;
  tensor.dl_tensor.byte_offset        = 0;
  tensor.dl_tensor.shape              = shape;
  tensor.dl_tensor.strides            = nullptr;

  thrust::host_vector<T> host_vector(data.begin(), data.end());
  tensor.dl_tensor.data = host_vector.data();

  EXPECT_THROW(cudf::from_dlpack(&tensor), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedExplicitRowMajor2DTensorFromDlpack)
{
  using T            = float;
  constexpr int ndim = 2;
  // Row major 2D tensor with explicit strides
  auto const data       = cudf::test::make_type_param_vector<T>({1, 2, 3, 4});
  int64_t shape[ndim]   = {2, 2};  // NOLINT
  int64_t strides[ndim] = {2, 1};  // NOLINT

  DLManagedTensor tensor{};
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.dtype              = get_dtype<T>();
  tensor.dl_tensor.ndim               = ndim;
  tensor.dl_tensor.byte_offset        = 0;
  tensor.dl_tensor.shape              = shape;
  tensor.dl_tensor.strides            = strides;

  thrust::host_vector<T> host_vector(data.begin(), data.end());
  tensor.dl_tensor.data = host_vector.data();

  EXPECT_THROW(cudf::from_dlpack(&tensor), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedStridedColMajor2DTensorFromDlpack)
{
  using T            = float;
  constexpr int ndim = 2;
  // Column major, but strided in fastest dimension
  auto const data       = cudf::test::make_type_param_vector<T>({1, 2, 3, 4, 5, 6, 7, 8});
  int64_t shape[ndim]   = {2, 2};  // NOLINT
  int64_t strides[ndim] = {2, 4};  // NOLINT

  DLManagedTensor tensor{};
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.dtype              = get_dtype<T>();
  tensor.dl_tensor.ndim               = ndim;
  tensor.dl_tensor.byte_offset        = 0;
  tensor.dl_tensor.shape              = shape;
  tensor.dl_tensor.strides            = strides;

  thrust::host_vector<T> host_vector(data.begin(), data.end());
  tensor.dl_tensor.data = host_vector.data();

  EXPECT_THROW(cudf::from_dlpack(&tensor), cudf::logic_error);
}

template <typename T>
class DLPackTimestampTests : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(DLPackTimestampTests, cudf::test::ChronoTypes);

TYPED_TEST(DLPackTimestampTests, ChronoTypesToDlpack)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::logic_error);
}

template <typename T>
class DLPackNumericTests : public cudf::test::BaseFixture {};

// The list of supported types comes from DLDataType_to_data_type() in cpp/src/dlpack/dlpack.cpp
// TODO: Replace with `NumericTypes` when unsigned support is added. Issue #5353
using SupportedTypes =
  cudf::test::RemoveIf<cudf::test::ContainedIn<cudf::test::Types<bool>>, cudf::test::NumericTypes>;
TYPED_TEST_SUITE(DLPackNumericTests, SupportedTypes);

TYPED_TEST(DLPackNumericTests, ToDlpack1D)
{
  // Test nullable column with no nulls
  cudf::test::fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4}, {1, 1, 1, 1});
  auto const col_view = static_cast<cudf::column_view>(col);
  EXPECT_FALSE(col_view.has_nulls());
  EXPECT_TRUE(col_view.nullable());

  cudf::table_view input({col});
  unique_managed_tensor result(cudf::to_dlpack(input));

  auto const& tensor = result->dl_tensor;
  validate_dtype<TypeParam>(tensor.dtype);
  EXPECT_EQ(kDLCUDA, tensor.device.device_type);
  EXPECT_EQ(1, tensor.ndim);
  EXPECT_EQ(uint64_t{0}, tensor.byte_offset);
  EXPECT_EQ(nullptr, tensor.strides);

  EXPECT_NE(nullptr, tensor.data);
  EXPECT_NE(nullptr, tensor.shape);

  // Verify that data matches input column
  constexpr cudf::data_type type{cudf::type_to_id<TypeParam>()};
  cudf::column_view const result_view(
    type, tensor.shape[0], tensor.data, col_view.null_mask(), col_view.null_count());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_view, result_view);
}

TYPED_TEST(DLPackNumericTests, ToDlpack2D)
{
  using T             = TypeParam;
  auto const col1_tmp = cudf::test::make_type_param_vector<T>({1, 2, 3, 4});
  auto const col2_tmp = cudf::test::make_type_param_vector<T>({4, 5, 6, 7});
  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> cols;
  cols.push_back(
    cudf::test::fixed_width_column_wrapper<TypeParam>(col1_tmp.cbegin(), col1_tmp.cend()));
  cols.push_back(
    cudf::test::fixed_width_column_wrapper<TypeParam>(col2_tmp.cbegin(), col2_tmp.cend()));

  std::vector<cudf::column_view> col_views;
  std::transform(cols.begin(), cols.end(), std::back_inserter(col_views), [](auto const& col) {
    return static_cast<cudf::column_view>(col);
  });

  cudf::table_view input(col_views);
  unique_managed_tensor result(cudf::to_dlpack(input));

  auto const& tensor = result->dl_tensor;
  validate_dtype<TypeParam>(tensor.dtype);
  EXPECT_EQ(kDLCUDA, tensor.device.device_type);
  EXPECT_EQ(2, tensor.ndim);
  EXPECT_EQ(uint64_t{0}, tensor.byte_offset);

  EXPECT_NE(nullptr, tensor.data);
  EXPECT_NE(nullptr, tensor.shape);
  EXPECT_NE(nullptr, tensor.strides);

  EXPECT_EQ(1, tensor.strides[0]);
  EXPECT_EQ(tensor.shape[0], tensor.strides[1]);

  // Verify that data matches input columns
  cudf::size_type offset{0};
  for (auto const& col : input) {
    constexpr cudf::data_type type{cudf::type_to_id<TypeParam>()};
    cudf::column_view const result_view(type, tensor.shape[0], tensor.data, nullptr, 0, offset);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col, result_view);
    offset += tensor.strides[1];
  }
}

TYPED_TEST(DLPackNumericTests, FromDlpack1D)
{
  // Use to_dlpack to generate an input tensor
  cudf::test::fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Verify that from_dlpack(to_dlpack(input)) == input
  auto result = cudf::from_dlpack(tensor.get());
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, result->view());
}

TYPED_TEST(DLPackNumericTests, FromDlpack2D)
{
  // Use to_dlpack to generate an input tensor
  using T         = TypeParam;
  auto const col1 = cudf::test::make_type_param_vector<T>({1, 2, 3, 4});
  auto const col2 = cudf::test::make_type_param_vector<T>({4, 5, 6, 7});
  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> cols;
  cols.push_back(cudf::test::fixed_width_column_wrapper<T>(col1.cbegin(), col1.cend()));
  cols.push_back(cudf::test::fixed_width_column_wrapper<T>(col2.cbegin(), col2.cend()));

  std::vector<cudf::column_view> col_views;
  std::transform(cols.begin(), cols.end(), std::back_inserter(col_views), [](auto const& col) {
    return static_cast<cudf::column_view>(col);
  });

  cudf::table_view input(col_views);
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Verify that from_dlpack(to_dlpack(input)) == input
  auto result = cudf::from_dlpack(tensor.get());
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, result->view());
}

TYPED_TEST(DLPackNumericTests, FromDlpackCpu)
{
  // Host buffer with stride > rows and byte_offset > 0
  using T         = TypeParam;
  auto const data = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0});
  uint64_t const offset{sizeof(T)};
  int64_t shape[2]   = {4, 2};  // NOLINT
  int64_t strides[2] = {1, 5};  // NOLINT

  DLManagedTensor tensor{};
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.dtype              = get_dtype<T>();
  tensor.dl_tensor.ndim               = 2;
  tensor.dl_tensor.byte_offset        = offset;
  tensor.dl_tensor.shape              = shape;
  tensor.dl_tensor.strides            = strides;

  thrust::host_vector<T> host_vector(data.begin(), data.end());
  tensor.dl_tensor.data = host_vector.data();

  cudf::test::fixed_width_column_wrapper<TypeParam> col1({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<TypeParam> col2({5, 6, 7, 8});
  cudf::table_view expected({col1, col2});

  auto result = cudf::from_dlpack(&tensor);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}

TYPED_TEST(DLPackNumericTests, FromDlpackEmpty1D)
{
  // Use to_dlpack to generate an input tensor
  cudf::table_view input(std::vector<cudf::column_view>{});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  EXPECT_EQ(nullptr, tensor.get());
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}
