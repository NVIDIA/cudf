/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

namespace tpc_reduction {

// Codes identify the three demographic/address branches after the dimension joins.
__host__ __device__ inline bool q13_selected(double sales_price,
                                             double net_profit,
                                             int32_t demographic_branch,
                                             int32_t address_branch,
                                             int32_t year)
{
  bool const demographic = (demographic_branch == 1 && sales_price >= 100 && sales_price <= 150) ||
                           (demographic_branch == 2 && sales_price >= 50 && sales_price <= 100) ||
                           (demographic_branch == 3 && sales_price >= 150 && sales_price <= 200);
  bool const address = (address_branch == 1 && net_profit >= 100 && net_profit <= 200) ||
                       (address_branch == 2 && net_profit >= 150 && net_profit <= 300) ||
                       (address_branch == 3 && net_profit >= 50 && net_profit <= 250);
  return year == 2001 && demographic && address;
}

extern "C" {

__device__ cudf::errc tpcds_q13_init(optional_double* quantity,
                                     optional_double* sales,
                                     optional_double* wholesale,
                                     optional_int64* quantity_count,
                                     optional_int64* sales_count,
                                     optional_int64* wholesale_count)
{
  static_cast<void>(init_sum_count(quantity, quantity_count));
  static_cast<void>(init_sum_count(sales, sales_count));
  return init_sum_count(wholesale, wholesale_count);
}

__device__ cudf::errc tpcds_q13_update(cudf::size_type,
                                       optional_double* quantity_sum,
                                       optional_double* sales_sum,
                                       optional_double* wholesale_sum,
                                       optional_int64* quantity_count,
                                       optional_int64* sales_count,
                                       optional_int64* wholesale_count,
                                       optional_int32 quantity,
                                       optional_double sales,
                                       optional_double wholesale)
{
  if (quantity) { add_value(quantity_sum, quantity_count, static_cast<double>(*quantity)); }
  add_value(sales_sum, sales_count, sales);
  add_value(wholesale_sum, wholesale_count, wholesale);
  return cudf::errc::SUCCESS;
}

__device__ cudf::errc tpcds_q13_row_update(cudf::size_type row,
                                           optional_double* quantity_sum,
                                           optional_double* sales_sum,
                                           optional_double* wholesale_sum,
                                           optional_int64* quantity_count,
                                           optional_int64* sales_count,
                                           optional_int64* wholesale_count,
                                           optional_int32 quantity,
                                           optional_double sales,
                                           optional_double wholesale,
                                           optional_double sales_price,
                                           optional_double net_profit,
                                           optional_int32 demographic_branch,
                                           optional_int32 address_branch,
                                           optional_int32 year)
{
  return tpcds_q13_update(row,
                          quantity_sum,
                          sales_sum,
                          wholesale_sum,
                          quantity_count,
                          sales_count,
                          wholesale_count,
                          quantity,
                          sales,
                          wholesale);
}

__device__ cudf::errc tpcds_q13_filter(cudf::size_type,
                                       bool* selected,
                                       optional_int32,
                                       optional_double,
                                       optional_double,
                                       optional_double sales_price,
                                       optional_double net_profit,
                                       optional_int32 demographic_branch,
                                       optional_int32 address_branch,
                                       optional_int32 year)
{
  *selected = sales_price && net_profit && demographic_branch && address_branch && year &&
              q13_selected(*sales_price, *net_profit, *demographic_branch, *address_branch, *year);
  return cudf::errc::SUCCESS;
}

__device__ cudf::errc tpcds_q13_merge(optional_double* quantity_sum,
                                      optional_double* sales_sum,
                                      optional_double* wholesale_sum,
                                      optional_int64* quantity_count,
                                      optional_int64* sales_count,
                                      optional_int64* wholesale_count,
                                      optional_double quantity,
                                      optional_double sales,
                                      optional_double wholesale,
                                      optional_int64 other_quantity_count,
                                      optional_int64 other_sales_count,
                                      optional_int64 other_wholesale_count)
{
  static_cast<void>(merge_sum_count(quantity_sum, quantity_count, quantity, other_quantity_count));
  static_cast<void>(merge_sum_count(sales_sum, sales_count, sales, other_sales_count));
  return merge_sum_count(wholesale_sum, wholesale_count, wholesale, other_wholesale_count);
}

__device__ cudf::errc tpcds_q13_finalize(optional_double* quantity_mean,
                                         optional_double* sales_mean,
                                         optional_double* wholesale_mean,
                                         optional_double* wholesale_sum,
                                         optional_double quantity,
                                         optional_double sales,
                                         optional_double wholesale,
                                         optional_int64 quantity_count,
                                         optional_int64 sales_count,
                                         optional_int64 wholesale_count)
{
  static_cast<void>(finalize_mean(quantity_mean, quantity, quantity_count));
  static_cast<void>(finalize_mean(sales_mean, sales, sales_count));
  static_cast<void>(finalize_mean(wholesale_mean, wholesale, wholesale_count));
  return finalize_sum(wholesale_sum, wholesale, wholesale_count);
}

}  // extern "C"
}  // namespace tpc_reduction
