/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ndsh_data_generator.hpp"

#include <cudf_test/base_fixture.hpp>

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <gtest/gtest.h>

struct NDSHDataGeneratorTest : public cudf::test::BaseFixture {};

TEST_F(NDSHDataGeneratorTest, ScaleFactorPointZeroOne)
{
  constexpr double scale_factor = 0.01;

  auto [orders, lineitem, part] = cudf::datagen::generate_orders_lineitem_part(scale_factor);
  auto partsupp                 = cudf::datagen::generate_partsupp(scale_factor);
  auto supplier                 = cudf::datagen::generate_supplier(scale_factor);
  auto customer                 = cudf::datagen::generate_customer(scale_factor);
  auto nation                   = cudf::datagen::generate_nation();
  auto region                   = cudf::datagen::generate_region();

  auto const expect_cardinality =
    [](cudf::table const& table, cudf::size_type rows, cudf::size_type columns) {
      EXPECT_EQ(table.num_rows(), rows);
      EXPECT_EQ(table.num_columns(), columns);
    };

  expect_cardinality(*orders, 15'000, 9);
  EXPECT_GE(lineitem->num_rows(), 15'000);
  EXPECT_LE(lineitem->num_rows(), 105'000);
  EXPECT_EQ(lineitem->num_columns(), 16);
  expect_cardinality(*part, 2'000, 9);
  expect_cardinality(*partsupp, 8'000, 5);
  expect_cardinality(*supplier, 100, 7);
  expect_cardinality(*customer, 1'500, 8);
  expect_cardinality(*nation, 25, 4);
  expect_cardinality(*region, 5, 3);

  auto const expect_supplier_key_range = [](cudf::column_view const& keys,
                                            cudf::size_type supplier_rows) {
    EXPECT_EQ(keys.null_count(), 0);
    auto const [minimum, maximum] = cudf::minmax(keys);
    auto const min_key = static_cast<cudf::numeric_scalar<cudf::size_type> const*>(minimum.get());
    auto const max_key = static_cast<cudf::numeric_scalar<cudf::size_type> const*>(maximum.get());
    EXPECT_GE(min_key->value(), 1);
    EXPECT_LE(max_key->value(), supplier_rows);
  };

  expect_supplier_key_range(lineitem->view().column(2), supplier->num_rows());
  expect_supplier_key_range(partsupp->view().column(1), supplier->num_rows());
}
