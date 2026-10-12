/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <vector>

struct CompoundColumnTest : public cudf::test::BaseFixture {};

namespace {
template <typename ColumnDeviceView>
struct checker_for_level1 {
  ColumnDeviceView d_column;
  bool __device__ operator()(int32_t idx)
  {
    int32_t val1 = d_column.child(0).template element<int32_t>(idx);
    int32_t val2 = d_column.child(1).template element<int32_t>(idx);
    int32_t val3 = d_column.child(2).template element<int32_t>(idx);
    return ((val1 + 100) == val2) && ((val2 + 100) == val3);
  }
};

template <typename ColumnDeviceView>
struct checker_for_level2 {
  ColumnDeviceView d_column;
  bool __device__ operator()(int32_t idx)
  {
    bool bcheck = true;
    for (int i = 0; i < 2 && bcheck; ++i) {
      auto child   = d_column.child(i);
      int32_t val1 = child.child(0).template element<int32_t>(idx);
      int32_t val2 = child.child(1).template element<int32_t>(idx);
      int32_t val3 = child.child(2).template element<int32_t>(idx);
      bcheck       = ((val1 + 100) == val2) && ((val2 + 100) == val3);
    }
    return bcheck;
  }
};
}  // namespace

TEST_F(CompoundColumnTest, ChildrenLevel1)
{
  rmm::device_uvector<int32_t> data(1000, cudf::get_default_stream());
  thrust::sequence(
    rmm::exec_policy_nosync(cudf::get_default_stream()), data.begin(), data.end(), 1);

  auto null_mask = cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED);
  rmm::device_buffer data1{data.data() + 100, 100 * sizeof(int32_t), cudf::get_default_stream()};
  rmm::device_buffer data2{data.data() + 200, 100 * sizeof(int32_t), cudf::get_default_stream()};
  rmm::device_buffer data3{data.data() + 300, 100 * sizeof(int32_t), cudf::get_default_stream()};
  auto child1 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   100,
                                   std::move(data1),
                                   cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED),
                                   0);
  auto child2 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   200,
                                   std::move(data2),
                                   cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED),
                                   0);
  auto child3 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   300,
                                   std::move(data3),
                                   cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED),
                                   0);

  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(std::move(child1));
  children.emplace_back(std::move(child2));
  children.emplace_back(std::move(child3));

  auto parent =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                   100,
                                   rmm::device_buffer{},
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                   0,
                                   std::move(children));

  {
    auto column = cudf::column_device_view::create(parent->view());
    EXPECT_TRUE(thrust::any_of(rmm::exec_policy_nosync(cudf::get_default_stream()),
                               cuda::counting_iterator<int32_t>{0},
                               cuda::counting_iterator<int32_t>{100},
                               checker_for_level1<cudf::column_device_view>{*column}));
  }
  {
    auto column = cudf::mutable_column_device_view::create(parent->mutable_view());
    EXPECT_TRUE(thrust::any_of(rmm::exec_policy_nosync(cudf::get_default_stream()),
                               cuda::counting_iterator<int32_t>{0},
                               cuda::counting_iterator<int32_t>{100},
                               checker_for_level1<cudf::mutable_column_device_view>{*column}));
  }
}

TEST_F(CompoundColumnTest, ChildrenLevel2)
{
  rmm::device_uvector<int32_t> data(1000, cudf::get_default_stream());
  thrust::sequence(
    rmm::exec_policy_nosync(cudf::get_default_stream()), data.begin(), data.end(), 1);

  auto null_mask = cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED);
  rmm::device_buffer data11{data.data() + 100, 100 * sizeof(int32_t), cudf::get_default_stream()};
  rmm::device_buffer data12{data.data() + 200, 100 * sizeof(int32_t), cudf::get_default_stream()};
  rmm::device_buffer data13{data.data() + 300, 100 * sizeof(int32_t), cudf::get_default_stream()};
  rmm::device_buffer data21{data.data() + 400, 100 * sizeof(int32_t), cudf::get_default_stream()};
  rmm::device_buffer data22{data.data() + 500, 100 * sizeof(int32_t), cudf::get_default_stream()};
  rmm::device_buffer data23{data.data() + 600, 100 * sizeof(int32_t), cudf::get_default_stream()};
  auto gchild11 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   100,
                                   std::move(data11),
                                   cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED),
                                   0);
  auto gchild12 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   200,
                                   std::move(data12),
                                   cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED),
                                   0);
  auto gchild13 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   300,
                                   std::move(data13),
                                   cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED),
                                   0);
  auto gchild21 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   400,
                                   std::move(data21),
                                   cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED),
                                   0);
  auto gchild22 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   500,
                                   std::move(data22),
                                   cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED),
                                   0);
  auto gchild23 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   600,
                                   std::move(data23),
                                   cudf::create_null_mask(100, cudf::mask_state::UNALLOCATED),
                                   0);

  std::vector<std::unique_ptr<cudf::column>> gchildren1;
  gchildren1.emplace_back(std::move(gchild11));
  gchildren1.emplace_back(std::move(gchild12));
  gchildren1.emplace_back(std::move(gchild13));
  std::vector<std::unique_ptr<cudf::column>> gchildren2;
  gchildren2.emplace_back(std::move(gchild21));
  gchildren2.emplace_back(std::move(gchild22));
  gchildren2.emplace_back(std::move(gchild23));

  auto children1 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                   100,
                                   rmm::device_buffer{},
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                   0,
                                   std::move(gchildren1));
  auto children2 =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                   100,
                                   rmm::device_buffer{},
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                   0,
                                   std::move(gchildren2));

  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(std::move(children1));
  children.emplace_back(std::move(children2));
  auto parent =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                   100,
                                   rmm::device_buffer{},
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                   0,
                                   std::move(children));

  {
    auto column = cudf::column_device_view::create(parent->view());
    EXPECT_TRUE(thrust::any_of(rmm::exec_policy_nosync(cudf::get_default_stream()),
                               cuda::counting_iterator<int32_t>{0},
                               cuda::counting_iterator<int32_t>{100},
                               checker_for_level2<cudf::column_device_view>{*column}));
  }
  {
    auto column = cudf::mutable_column_device_view::create(parent->mutable_view());
    EXPECT_TRUE(thrust::any_of(rmm::exec_policy_nosync(cudf::get_default_stream()),
                               cuda::counting_iterator<int32_t>{0},
                               cuda::counting_iterator<int32_t>{100},
                               checker_for_level2<cudf::mutable_column_device_view>{*column}));
  }
}
