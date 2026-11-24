// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sim_backend.hpp"

namespace fixture {
// Deliberately incomplete evaluation starter. Repair this adapter only.
inline Answer execute(Backend& backend, Input input, std::int64_t cutoff) {
  auto staged = backend.allocate(input.resource);
  backend.upload(*input.producer, input.rows, staged);
  auto output = backend.allocate(input.resource);
  backend.select_sum(*input.producer, staged, output, cutoff);
  backend.wrapper_checkpoint();
  input.producer->drain();
  return backend.read(output);
}
}  // namespace fixture
