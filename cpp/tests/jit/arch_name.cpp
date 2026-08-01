/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/arch_name.hpp"

#include <gtest/gtest.h>

TEST(JitArchName, BaselineOnly) { EXPECT_EQ(cudf::jit_arch_name_for(90, "90-real"), "sm_90"); }

TEST(JitArchName, ArchSpecificOnly)
{
  EXPECT_EQ(cudf::jit_arch_name_for(120, "120a-real"), "sm_120a");
  EXPECT_EQ(cudf::jit_arch_name_for(90, "90a-virtual"), "sm_90a");
}

TEST(JitArchName, FamilySpecificOnly)
{
  EXPECT_EQ(cudf::jit_arch_name_for(100, "100f-real"), "sm_100f");
}

TEST(JitArchName, BaselineWins)
{
  EXPECT_EQ(cudf::jit_arch_name_for(120, "120-real,120a-real"), "sm_120");
  EXPECT_EQ(cudf::jit_arch_name_for(120, "120a-real,120"), "sm_120");
}

TEST(JitArchName, ArchSpecificWinsOverFamily)
{
  EXPECT_EQ(cudf::jit_arch_name_for(100, "100a-real,100f-real"), "sm_100a");
}

TEST(JitArchName, RapidsDefaults)
{
  auto const defaults = "75-real,80-real,86-real,90a-real,100f-real,120a-real,120";
  EXPECT_EQ(cudf::jit_arch_name_for(80, defaults), "sm_80");
  EXPECT_EQ(cudf::jit_arch_name_for(90, defaults), "sm_90a");
  EXPECT_EQ(cudf::jit_arch_name_for(100, defaults), "sm_100f");
  EXPECT_EQ(cudf::jit_arch_name_for(120, defaults), "sm_120");
}

TEST(JitArchName, UnlistedCapabilityFallsBack)
{
  EXPECT_EQ(cudf::jit_arch_name_for(121, "120a-real,120"), "sm_121");
  EXPECT_EQ(cudf::jit_arch_name_for(121, ""), "sm_121");
}

TEST(JitArchName, PrefixDoesNotMatchLongerCapability)
{
  // "120a" must not be taken as a feature-set entry for compute capability 12.
  EXPECT_EQ(cudf::jit_arch_name_for(12, "120a-real"), "sm_12");
}
