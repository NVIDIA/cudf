/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include <cupti.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define CAPACITY (1u << 18)
#define WRITING  (UINT64_C(1) << 63)

struct record {
  _Atomic uint64_t seq;
  uint64_t ns, stream, object, aux, context;
  uint32_t tid, correlation;
  int32_t phase, result;
  char api[40];
};
_Static_assert(sizeof(struct record) == 104, "trace format");

static struct record* trace_records;
static _Atomic uint64_t next_seq;
static _Atomic uint64_t write_failures;
static _Thread_local uint32_t thread_id;

uint64_t stream_trace_position(void)
{
  return atomic_load_explicit(&next_seq, memory_order_relaxed);
}
uint64_t stream_trace_write_failures(void)
{
  return atomic_load_explicit(&write_failures, memory_order_relaxed);
}

static void callback(void* userdata,
                     CUpti_CallbackDomain domain,
                     CUpti_CallbackId id,
                     void const* data)
{
  (void)userdata;
  (void)domain;
  CUpti_CallbackData const* cb = data;
  struct record line           = {
              .context     = (uintptr_t)cb->context,
              .correlation = cb->correlationId,
              .phase       = cb->callbackSite,
              .result =
      cb->callbackSite == CUPTI_API_EXIT ? (int32_t)*(CUresult const*)cb->functionReturnValue : -1,
  };
  int const output = line.phase == CUPTI_API_EXIT && line.result == CUDA_SUCCESS;
  (void)output;
  switch (id) {
#define API(name, stream_value, object_value, aux_value) \
  case CUPTI_DRIVER_TRACE_CBID_##name: {                 \
    const name##_params* p = cb->functionParams;         \
    line.stream            = (uintptr_t)(stream_value);  \
    line.object            = (uintptr_t)(object_value);  \
    line.aux               = (uintptr_t)(aux_value);     \
    __builtin_memcpy(line.api, #name, sizeof(#name));    \
    break;                                               \
  }
#define APIS(name, stream_value, object_value, aux_value) \
  API(name, stream_value, object_value, aux_value)        \
  API(name##_ptsz, stream_value, object_value, aux_value)
#define NOARGS(name) \
  case CUPTI_DRIVER_TRACE_CBID_##name: __builtin_memcpy(line.api, #name, sizeof(#name)); break;
#include "stream_callbacks.inc"
#undef NOARGS
#undef APIS
#undef API
    default: return;
  }
  struct timespec stamp;
  if (clock_gettime(CLOCK_MONOTONIC, &stamp) == 0) {
    line.ns = (uint64_t)stamp.tv_sec * 1000000000u + stamp.tv_nsec;
  }
  if (!thread_id) { thread_id = (uint32_t)syscall(SYS_gettid); }
  line.tid                  = thread_id;
  uint64_t const sequence   = atomic_fetch_add_explicit(&next_seq, 1, memory_order_relaxed) + 1;
  struct record* const slot = &trace_records[(sequence - 1) % CAPACITY];
  uint64_t previous         = atomic_load_explicit(&slot->seq, memory_order_relaxed);
  // A stalled callback must not overwrite a newer record after the ring wraps.
  if ((previous & WRITING) || previous >= sequence ||
      !atomic_compare_exchange_strong_explicit(
        &slot->seq, &previous, sequence | WRITING, memory_order_acquire, memory_order_relaxed)) {
    atomic_fetch_add_explicit(&write_failures, 1, memory_order_relaxed);
    return;
  }

  // Publish only complete records. The file-backed ring survives a fatal signal
  // without a write syscall for every CUDA call.
  __builtin_memcpy(&slot->ns, &line.ns, sizeof(line) - sizeof(line.seq));
  atomic_store_explicit(&slot->seq, sequence, memory_order_release);
}

int install_stream_trace(int fd, char const* library)
{
  void* cupti = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!cupti) { return -1; }
  __typeof__(&cuptiSubscribe) subscribe   = dlsym(cupti, "cuptiSubscribe");
  __typeof__(&cuptiEnableCallback) enable = dlsym(cupti, "cuptiEnableCallback");
  if (!subscribe || !enable) { return -2; }
  size_t const bytes = CAPACITY * sizeof(struct record);
  if (ftruncate(fd, (off_t)bytes) != 0) { return -3; }
  trace_records = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (trace_records == MAP_FAILED) { return -4; }
  CUpti_SubscriberHandle subscriber;
  CUptiResult result = subscribe(&subscriber, callback, NULL);
  if (result != CUPTI_SUCCESS) { return result; }
#define API(name, ...)                                                                        \
  result = enable(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_##name); \
  if (result != CUPTI_SUCCESS) { return result; }
#define APIS(name, ...) API(name) API(name##_ptsz)
#define NOARGS(name)    API(name)
#include "stream_callbacks.inc"
#undef NOARGS
#undef APIS
#undef API
  return 0;
}
