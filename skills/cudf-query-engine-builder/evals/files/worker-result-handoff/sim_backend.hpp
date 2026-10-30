// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// CPU simulation of an engine/backend boundary. No CUDA or cuDF executes here.
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fixture {
using Rows = std::vector<std::optional<std::int64_t>>;
enum class Origin { backend, cpu };
struct Answer { std::int64_t sum = 0; std::int64_t count = 0; Origin origin = Origin::cpu; };
struct Audit {
  int early_release = 0, allocator_lifetime_error = 0, stream_abandoned = 0;
  int live_buffers = 0, pending_uses = 0, backend_operations = 0, cpu_operations = 0;
  int asynchronous_errors = 0;
};
struct Resource { explicit Resource(std::shared_ptr<Audit> a) : audit(std::move(a)) {} std::shared_ptr<Audit> audit; };
struct Buffer {
  Buffer(std::shared_ptr<Audit> a, std::weak_ptr<Resource> r, bool device)
      : audit(std::move(a)), resource(std::move(r)), device(device) { ++audit->live_buffers; }
  ~Buffer() {
    if (pending != 0) ++audit->early_release;
    if (device && resource.expired()) ++audit->allocator_lifetime_error;
    --audit->live_buffers;
  }
  std::shared_ptr<Audit> audit;
  std::weak_ptr<Resource> resource;
  bool device, ready = false;
  int pending = 0;
  Rows rows;
  Answer answer;
};
using Owner = std::shared_ptr<Buffer>;
struct Stream;
struct Event { std::weak_ptr<Stream> stream; std::uint64_t sequence = 0; };
struct Stream : std::enable_shared_from_this<Stream> {
  explicit Stream(std::shared_ptr<Audit> a) : audit(std::move(a)) {}
  ~Stream() { if (!queue.empty()) ++audit->stream_abandoned; }
  std::uint64_t submit(std::function<void()> work) {
    queue.emplace_back(++submitted, std::move(work));
    return submitted;
  }
  Event event() { return {weak_from_this(), submitted}; }
  void wait(Event dependency) {
    submit([dependency] {
      auto producer = dependency.stream.lock();
      if (!producer) throw std::runtime_error("producer stream owner expired");
      producer->drain_until(dependency.sequence);
    });
  }
  void drain_until(std::uint64_t through) {
    std::exception_ptr first;
    while (!queue.empty() && queue.front().first <= through) {
      auto item = std::move(queue.front()); queue.pop_front();
      try { item.second(); } catch (...) { if (!first) first = std::current_exception(); }
      completed = item.first;
    }
    if (first) { ++audit->asynchronous_errors; std::rethrow_exception(first); }
  }
  void drain() { drain_until(submitted); }
  std::shared_ptr<Audit> audit;
  std::deque<std::pair<std::uint64_t, std::function<void()>>> queue;
  std::uint64_t submitted = 0, completed = 0;
};
struct Input {
  // Reverse destruction order keeps resource alive after host/stream destruction.
  std::shared_ptr<Resource> resource;
  std::shared_ptr<Stream> producer;
  Owner rows;
};
struct Fault { int allocation_number = 0; bool wrapper = false; };
class Backend {
 public:
  std::shared_ptr<Audit> audit = std::make_shared<Audit>();
  Fault fault;
  int allocations = 0;
  Input input(Rows rows) {
    Input in{std::make_shared<Resource>(audit), std::make_shared<Stream>(audit), {}};
    in.rows = std::make_shared<Buffer>(audit, in.resource, false);
    in.rows->rows = std::move(rows); in.rows->ready = true;
    return in;
  }
  std::shared_ptr<Stream> stream() { return std::make_shared<Stream>(audit); }
  Owner allocate(std::shared_ptr<Resource> const& resource) {
    if (++allocations == fault.allocation_number) throw std::bad_alloc();
    return std::make_shared<Buffer>(audit, resource, true);
  }
  void wrapper_checkpoint() { if (fault.wrapper) throw std::bad_alloc(); }
  void upload(Stream& stream, Owner const& host, Owner const& device) {
    queue_operation(stream, host, device, [this](Buffer& src, Buffer& dst) {
      dst.rows = src.rows; dst.ready = true; ++audit->backend_operations;
    });
  }
  void select_sum(Stream& stream, Owner const& input, Owner const& output, std::int64_t cutoff) {
    queue_operation(stream, input, output, [this, cutoff](Buffer& src, Buffer& dst) {
      if (!src.ready) throw std::runtime_error("consumer ran before upload completed");
      dst.answer = {}; dst.answer.origin = Origin::backend;
      for (auto const& value : src.rows) if (value && *value > cutoff) {
        dst.answer.sum += *value; ++dst.answer.count;
      }
      dst.ready = true; ++audit->backend_operations;
    });
  }
  Answer read(Owner const& output) const {
    if (!output || !output->ready) throw std::runtime_error("output not ready or not owned");
    return output->answer;
  }
  Answer cpu(Rows const& rows, std::int64_t cutoff) {
    ++audit->cpu_operations;
    Answer answer;
    for (auto const& value : rows) if (value && *value > cutoff) { answer.sum += *value; ++answer.count; }
    return answer;
  }
 private:
  template<class Work> void queue_operation(Stream& stream, Owner const& first, Owner const& second, Work work) {
    if (!first || !second) throw std::invalid_argument("missing buffer");
    // Queue entries retain borrowed handles only. The adapter owns the leases.
    auto a = std::weak_ptr<Buffer>(first), b = std::weak_ptr<Buffer>(second);
    stream.submit([a,b,work,audit = audit] {
      auto first = a.lock(), second = b.lock();
      struct ReleasePending {
        Owner first, second; std::shared_ptr<Audit> audit;
        ~ReleasePending() { if(first) --first->pending; if(second) --second->pending; audit->pending_uses -= 2; }
      } release{first, second, audit};
      if (!first || !second) throw std::runtime_error("queued operation lost a buffer owner");
      if ((first->device && first->resource.expired()) || (second->device && second->resource.expired()))
        throw std::runtime_error("queued operation lost allocator owner");
      work(*first,*second);
    });
    ++first->pending; ++second->pending; audit->pending_uses += 2;
  }
};
inline bool clean(Audit const& a) {
  return a.early_release == 0 && a.allocator_lifetime_error == 0 && a.stream_abandoned == 0 &&
         a.live_buffers == 0 && a.pending_uses == 0 && a.asynchronous_errors == 0;
}
}  // namespace fixture
