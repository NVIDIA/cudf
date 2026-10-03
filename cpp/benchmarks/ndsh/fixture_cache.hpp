/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Reuse one scale factor's NDS-H fixtures across serial benchmark states.
 * Replacing a scale releases its files and reference data before generating the next fixture.
 */

#pragma once

#include <map>

namespace ndsh {

// Retain one scale per fixture type across serial benchmark states. The returned reference is
// invalidated by the next request for a different scale, even if replacement construction fails.
template <typename Files>
Files const& local_fixture(double scale_factor)
{
  static std::map<double, Files> fixtures;
  if (!fixtures.contains(scale_factor)) {
    fixtures.clear();
    fixtures.try_emplace(scale_factor, scale_factor);
  }
  return fixtures.at(scale_factor);
}

}  // namespace ndsh
