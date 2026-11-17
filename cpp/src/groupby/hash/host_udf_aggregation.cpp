/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {

namespace detail {

host_udf_aggregation::host_udf_aggregation(std::unique_ptr<host_udf_base> udf_ptr_)
  : aggregation{HOST_UDF}, udf_ptr{std::move(udf_ptr_)}
{
  CUDF_EXPECTS(udf_ptr != nullptr, "Invalid host_udf_base instance.");
}

host_udf_aggregation::~host_udf_aggregation() = default;

bool host_udf_aggregation::is_equal(aggregation const& _other) const
{
  if (!this->aggregation::is_equal(_other)) { return false; }
  auto const& other = dynamic_cast<host_udf_aggregation const&>(_other);
  return udf_ptr->is_equal(*other.udf_ptr);
}

size_t host_udf_aggregation::do_hash() const
{
  return this->aggregation::do_hash() ^ udf_ptr->do_hash();
}

std::unique_ptr<aggregation> host_udf_aggregation::clone() const
{
  return std::make_unique<host_udf_aggregation>(udf_ptr->clone());
}

}  // namespace detail

template <typename Base>
std::unique_ptr<Base> make_host_udf_aggregation(std::unique_ptr<host_udf_base> udf_ptr_)
{
  return std::make_unique<detail::host_udf_aggregation>(std::move(udf_ptr_));
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_host_udf_aggregation<aggregation>(
  std::unique_ptr<host_udf_base>);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
  make_host_udf_aggregation<groupby_aggregation>(std::unique_ptr<host_udf_base>);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
  make_host_udf_aggregation<reduce_aggregation>(std::unique_ptr<host_udf_base>);
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
  make_host_udf_aggregation<segmented_reduce_aggregation>(std::unique_ptr<host_udf_base>);

}  // namespace cudf
