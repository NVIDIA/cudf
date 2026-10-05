/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/optional_iterator_test.cuh>

using TestingTypes = cudf::test::ChronoTypes;

template <typename T>
struct ChronoOptionalIteratorTest : public IteratorTest<T> {};

TYPED_TEST_SUITE(ChronoOptionalIteratorTest, TestingTypes);
TYPED_TEST(ChronoOptionalIteratorTest, nonull_optional_iterator)
{ nonull_optional_iterator(*this); }
TYPED_TEST(ChronoOptionalIteratorTest, null_optional_iterator) { null_optional_iterator(*this); }
