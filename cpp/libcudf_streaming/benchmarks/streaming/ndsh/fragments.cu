/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/udf_includes.hpp>

extern "C" __device__ cudf::errc calculate_disc_price(double* disc_price,
                                                      double extprice,
                                                      double discount)
{
  *disc_price = extprice * (1 - discount);

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc calculate_charge(double* charge, double discprice, double tax)
{
  *charge = discprice * (1 + tax);

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc calculate_amount(
  double* amount, double discount, double extprice, double supplycost, double quantity)
{
  *amount = extprice * (1 - discount) - supplycost * quantity;

  return cudf::errc::SUCCESS;
}
