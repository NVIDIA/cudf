// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sim_backend.hpp"

namespace fixture {
// Deliberately incomplete evaluation starter. Repair this adapter only.
struct QueryResult {
  Backend* backend;
  std::weak_ptr<Stream> consumer;
  std::weak_ptr<Buffer> output;
  Answer fallback;
  bool use_backend;
  Answer get() {
    if (!use_backend) return fallback;
    auto stream = consumer.lock();
    if (!stream) throw std::runtime_error("consumer stream expired");
    stream->drain();
    return backend->read(output.lock());
  }
};
inline std::unique_ptr<QueryResult> submit(Backend& backend, Input input,
                                         std::shared_ptr<Stream> consumer,
                                         std::int64_t cutoff, bool use_backend) {
  auto result = std::make_unique<QueryResult>();
  result->backend = &backend; result->use_backend = use_backend;
  if (!use_backend) { result->fallback = backend.cpu(input.rows->rows, cutoff); return result; }
  auto staged = backend.allocate(input.resource);
  backend.upload(*input.producer, input.rows, staged);
  auto output = backend.allocate(input.resource);
  backend.select_sum(*consumer, staged, output, cutoff);
  result->consumer = consumer; result->output = output;
  return result;
}
}  // namespace fixture
