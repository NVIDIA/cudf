/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <cudf_streaming/channel_metadata.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cudf_streaming {
namespace {

void validate_ordering(ordering const& ordering)
{
  RAPIDSMPF_EXPECTS(
    !ordering.keys.empty(), "ordering: keys must not be empty", std::invalid_argument);
  RAPIDSMPF_EXPECTS(
    ordering.boundaries != nullptr, "ordering: boundaries must not be null", std::invalid_argument);
  RAPIDSMPF_EXPECTS(ordering.boundaries->is_available(),
                    "ordering: boundaries must be device-resident",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(
    ordering.keys.size() == static_cast<std::size_t>(ordering.boundaries->shape().second),
    "ordering: number of keys must match number of boundary columns",
    std::invalid_argument);
}

}  // namespace

ordering::ordering(std::vector<order_key> keys,
                   std::shared_ptr<table_chunk> boundaries,
                   bool strict_boundaries)
  : keys{std::move(keys)}, boundaries{std::move(boundaries)}, strict_boundaries{strict_boundaries}
{
  validate_ordering(*this);
}

ordering ordering::with_keys(std::vector<order_key> new_keys) const
{
  return ordering{std::move(new_keys), boundaries, strict_boundaries};
}

bool ordering::boundaries_aligned_with(ordering const& other, rapidsmpf::BufferResource& br) const
{
  if (strict_boundaries != other.strict_boundaries ||
      boundaries->shape() != other.boundaries->shape()) {
    return false;
  }
  if (!std::equal(
        keys.begin(), keys.end(), other.keys.begin(), [](order_key const& a, order_key const& b) {
          return a.order == b.order && a.null_order == b.null_order;
        })) {
    return false;
  }
  if (boundaries->shape().first == 0) { return true; }
  auto const lhs    = boundaries->table_view();
  auto const rhs    = other.boundaries->table_view();
  auto const stream = boundaries->stream();
  rapidsmpf::cuda_stream_join(stream, other.boundaries->stream());
  for (cudf::size_type i = 0; i < lhs.num_columns(); ++i) {
    auto eq      = cudf::binary_operation(lhs.column(i),
                                     rhs.column(i),
                                     cudf::binary_operator::NULL_EQUALS,
                                     cudf::data_type{cudf::type_id::BOOL8},
                                     stream,
                                     br.device_mr());
    auto result  = cudf::reduce(eq->view(),
                               *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               stream,
                               br.device_mr());
    auto& scalar = static_cast<cudf::numeric_scalar<bool>&>(*result);
    if (!scalar.value(stream)) { return false; }
  }
  return true;
}

order_scheme::order_scheme(std::vector<order_key> keys,
                           std::shared_ptr<table_chunk> boundaries,
                           bool strict_boundaries)
  : order_scheme(
      std::vector<ordering>{ordering{std::move(keys), std::move(boundaries), strict_boundaries}})
{
}

order_scheme::order_scheme(std::vector<ordering> orderings) : orderings{std::move(orderings)}
{
  RAPIDSMPF_EXPECTS(
    !this->orderings.empty(), "order_scheme: orderings must not be empty", std::invalid_argument);
  for (auto const& ordering : this->orderings) {
    RAPIDSMPF_EXPECTS(!ordering.keys.empty(),
                      "order_scheme: ordering entries must not be empty",
                      std::invalid_argument);
  }
}

partitioning_spec partitioning_spec::from_order(order_scheme o)
{
  return {.type = type::ORDER, .hash = std::nullopt, .order = std::move(o)};
}

rapidsmpf::streaming::Message to_message(std::uint64_t sequence_number,
                                         std::unique_ptr<channel_metadata> m)
{
  return rapidsmpf::streaming::Message{
    sequence_number,
    std::move(m),
    {},
    [](rapidsmpf::streaming::Message const& msg,
       rapidsmpf::MemoryReservation& /* reservation */) -> rapidsmpf::streaming::Message {
      auto copy = std::make_unique<channel_metadata>(msg.get<channel_metadata>());
      return rapidsmpf::streaming::Message{
        msg.sequence_number(), std::move(copy), {}, msg.copy_cb()};
    }};
}

}  // namespace cudf_streaming
