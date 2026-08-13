/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cudf_streaming/channel_metadata.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>

#include <memory>
#include <vector>

using namespace cudf_streaming;

class StreamingChannelMetadata : public ::testing::Test {};

TEST_F(StreamingChannelMetadata, HashScheme)
{
  hash_scheme h{{0, 1}, 16};
  EXPECT_EQ(h.column_indices.size(), 2);
  EXPECT_EQ(h.column_indices[0], 0);
  EXPECT_EQ(h.column_indices[1], 1);
  EXPECT_EQ(h.modulus, 16);

  // Equality
  EXPECT_EQ(h, (hash_scheme{{0, 1}, 16}));
  EXPECT_NE(h, (hash_scheme{{0, 1}, 32}));
  EXPECT_NE(h, (hash_scheme{{2}, 16}));
}

TEST_F(StreamingChannelMetadata, OrderSchemeCtorRejectsEmptyKeys)
{
  EXPECT_THROW(static_cast<void>(order_scheme({}, nullptr)), std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, OrderSchemeCtorRejectsNullBoundaries)
{
  EXPECT_THROW(static_cast<void>(
                 order_scheme({{0, cudf::order::ASCENDING, cudf::null_order::BEFORE}}, nullptr)),
               std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, OrderSchemeCtorRejectsEmptyOrderings)
{
  EXPECT_THROW(static_cast<void>(order_scheme(std::vector<ordering>{})), std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, PartitioningSpec)
{
  // None
  auto spec_none = partitioning_spec::none();
  EXPECT_EQ(spec_none.type, partitioning_spec::type::NONE);

  // Inherit
  auto spec_inherit = partitioning_spec::inherit();
  EXPECT_EQ(spec_inherit.type, partitioning_spec::type::INHERIT);

  // Hash
  auto spec_hash = partitioning_spec::from_hash(hash_scheme{{0}, 16});
  EXPECT_EQ(spec_hash.type, partitioning_spec::type::HASH);
  EXPECT_EQ(spec_hash.hash->column_indices[0], 0);
  EXPECT_EQ(spec_hash.hash->modulus, 16);

  // Type checks (operator== removed; use field comparisons)
  EXPECT_EQ(spec_none.type, partitioning_spec::type::NONE);
  EXPECT_EQ(spec_inherit.type, partitioning_spec::type::INHERIT);
  EXPECT_EQ(spec_hash.type, partitioning_spec::type::HASH);
  EXPECT_NE(spec_none.type, spec_inherit.type);
  EXPECT_EQ(spec_hash.hash->modulus, 16);
  EXPECT_NE((partitioning_spec::from_hash(hash_scheme{{0}, 32}).hash->modulus), 16);
}

TEST_F(StreamingChannelMetadata, PartitioningScenarios)
{
  // Default construction
  partitioning p_default{};
  EXPECT_EQ(p_default.inter_rank.type, partitioning_spec::type::NONE);
  EXPECT_EQ(p_default.local.type, partitioning_spec::type::NONE);

  // Direct global shuffle: inter_rank=Hash, local=Inherit
  partitioning p_global{partitioning_spec::from_hash(hash_scheme{{0}, 16}),
                        partitioning_spec::inherit()};
  EXPECT_EQ(p_global.inter_rank.type, partitioning_spec::type::HASH);
  EXPECT_EQ(p_global.local.type, partitioning_spec::type::INHERIT);
  EXPECT_EQ(p_global.inter_rank.hash->modulus, 16);

  // Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
  partitioning p_twostage{partitioning_spec::from_hash(hash_scheme{{0}, 4}),
                          partitioning_spec::from_hash(hash_scheme{{0}, 8})};
  EXPECT_EQ(p_twostage.inter_rank.hash->modulus, 4);
  EXPECT_EQ(p_twostage.local.hash->modulus, 8);

  // Field comparisons (partitioning::operator== removed)
  {
    partitioning p_same{partitioning_spec::from_hash(hash_scheme{{0}, 16}),
                        partitioning_spec::inherit()};
    EXPECT_EQ(p_global.inter_rank.type, p_same.inter_rank.type);
    EXPECT_EQ(p_global.inter_rank.hash->modulus, p_same.inter_rank.hash->modulus);
    EXPECT_EQ(p_global.local.type, p_same.local.type);
  }
  EXPECT_NE(p_global.inter_rank.hash->modulus, p_twostage.inter_rank.hash->modulus);
}

TEST_F(StreamingChannelMetadata, ChannelMetadata)
{
  // Full construction - use std::move to avoid GCC false positive on vector copy
  partitioning p{partitioning_spec::from_hash(hash_scheme{{0}, 16}), partitioning_spec::inherit()};
  channel_metadata m{4, std::move(p), true};
  EXPECT_EQ(m.local_count, 4);
  EXPECT_EQ(m.partitioning.inter_rank.type, partitioning_spec::type::HASH);
  EXPECT_EQ(m.partitioning.local.type, partitioning_spec::type::INHERIT);
  EXPECT_TRUE(m.duplicated);

  // Minimal construction
  channel_metadata m_minimal{4};
  EXPECT_EQ(m_minimal.local_count, 4);
  EXPECT_FALSE(m_minimal.duplicated);

  // Equality - create fresh partitionings and move them
  channel_metadata m_same{
    4,
    partitioning{partitioning_spec::from_hash(hash_scheme{{0}, 16}), partitioning_spec::inherit()},
    true};
  channel_metadata m_diff{
    8,
    partitioning{partitioning_spec::from_hash(hash_scheme{{0}, 16}), partitioning_spec::inherit()},
    true};
  // Field comparisons (channel_metadata::operator== removed)
  EXPECT_EQ(m.local_count, m_same.local_count);
  EXPECT_EQ(m.duplicated, m_same.duplicated);
  EXPECT_EQ(m.partitioning.inter_rank.hash->modulus, m_same.partitioning.inter_rank.hash->modulus);
  EXPECT_NE(m.local_count, m_diff.local_count);
}

TEST_F(StreamingChannelMetadata, MessageRoundTrip)
{
  // channel_metadata round-trip
  partitioning part{partitioning_spec::from_hash(hash_scheme{{0}, 16}),
                    partitioning_spec::inherit()};
  auto m     = std::make_unique<channel_metadata>(4, std::move(part), false);
  auto msg_m = to_message(99, std::move(m));
  EXPECT_EQ(msg_m.sequence_number(), 99);
  EXPECT_TRUE(msg_m.holds<channel_metadata>());
  auto released = msg_m.release<channel_metadata>();
  EXPECT_EQ(released.local_count, 4);
  EXPECT_FALSE(released.duplicated);
  EXPECT_EQ(released.partitioning.inter_rank.hash->modulus, 16);
  EXPECT_TRUE(msg_m.empty());
}

class StreamingChannelMetadataGPU : public ::testing::Test {
 protected:
  rmm::cuda_stream_view stream{cudf::get_default_stream()};
  std::shared_ptr<rapidsmpf::BufferResource> br =
    rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());

  std::shared_ptr<table_chunk> make_chunk(std::vector<int32_t> vals)
  {
    rmm::device_buffer buf(vals.data(), vals.size() * sizeof(int32_t), stream);
    auto col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                              static_cast<cudf::size_type>(vals.size()),
                                              std::move(buf),
                                              rmm::device_buffer{},
                                              0);
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(col));
    return std::make_shared<table_chunk>(std::make_unique<cudf::table>(std::move(cols)), stream);
  }
};

TEST_F(StreamingChannelMetadataGPU, OrderingReplaceKeys)
{
  order_key k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  order_key k5{5, cudf::order::DESCENDING, cudf::null_order::AFTER};

  auto b      = make_chunk({100, 200});
  ordering o1 = ordering({k0}, b);
  auto o2     = o1.with_keys({k5});

  EXPECT_EQ(o2.keys[0].column_index, 5);
  EXPECT_EQ(o2.keys[0].order, cudf::order::DESCENDING);
  EXPECT_EQ(o2.strict_boundaries, o1.strict_boundaries);
  EXPECT_EQ(o2.boundaries->shape(), o1.boundaries->shape());
  EXPECT_EQ(o2.boundaries.get(), b.get());
  EXPECT_NE(o1.keys[0].column_index, o2.keys[0].column_index);

  EXPECT_THROW(static_cast<void>(o1.with_keys({k0, k5})), std::invalid_argument);
}

TEST_F(StreamingChannelMetadataGPU, OrderSchemeMultipleOrderings)
{
  order_key k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  order_key k2{2, cudf::order::DESCENDING, cudf::null_order::AFTER};

  auto b0 = make_chunk({100, 200});
  auto b1 = make_chunk({300, 400});
  order_scheme o({ordering{{k0}, b0, true}, ordering{{k2}, b1, false}});

  ASSERT_EQ(o.orderings.size(), 2);
  EXPECT_EQ(o.orderings[0].keys[0], k0);
  EXPECT_EQ(o.orderings[0].boundaries.get(), b0.get());
  EXPECT_TRUE(o.orderings[0].strict_boundaries);
  EXPECT_EQ(o.orderings[1].keys[0], k2);
  EXPECT_EQ(o.orderings[1].boundaries.get(), b1.get());
  EXPECT_FALSE(o.orderings[1].strict_boundaries);
}

TEST_F(StreamingChannelMetadataGPU, OrderingBoundariesAlignedWith)
{
  order_key k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  order_key k3{3, cudf::order::ASCENDING, cudf::null_order::BEFORE};

  ordering o1({k0}, make_chunk({100, 200}));
  ordering o2({k0}, make_chunk({100, 200}));
  EXPECT_TRUE(o1.boundaries_aligned_with(o2, *br));

  ordering o_shifted({k3}, make_chunk({100, 200}));
  EXPECT_TRUE(o1.boundaries_aligned_with(o_shifted, *br));

  ordering o_strict({k0}, make_chunk({100, 200}), /*strict_boundaries=*/true);
  EXPECT_FALSE(o1.boundaries_aligned_with(o_strict, *br));

  ordering o_diff({k0}, make_chunk({100, 300}));
  EXPECT_FALSE(o1.boundaries_aligned_with(o_diff, *br));
}

TEST_F(StreamingChannelMetadataGPU, PartitioningSpecOrder)
{
  order_key k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  order_scheme o({k0}, make_chunk({100, 200}));

  auto spec = partitioning_spec::from_order(o);
  EXPECT_EQ(spec.type, partitioning_spec::type::ORDER);
  EXPECT_TRUE(spec.order.has_value());
  EXPECT_EQ(spec.order->orderings[0].keys[0].column_index, 0);

  // Type checks only (partitioning_spec::operator== removed; ORDER value comparison
  // requires boundaries_aligned_with on the ordering directly)
  EXPECT_EQ(spec.type, partitioning_spec::type::ORDER);
  EXPECT_NE(spec.type, partitioning_spec::from_hash(hash_scheme{{0}, 16}).type);
  EXPECT_NE(spec.type, partitioning_spec::none().type);
}
