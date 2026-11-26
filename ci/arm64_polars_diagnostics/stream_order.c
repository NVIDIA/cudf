/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include <cupti.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define CAPACITY (1u << 18)

struct record {
  uint64_t seq, ns, stream, object, aux, context;
  uint32_t tid, correlation;
  int32_t phase, result;
  char api[40];
};
_Static_assert(sizeof(struct record) == 104, "trace format");

static int trace_fd;
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
  line.tid = thread_id;
  // One bounded file per process survives a fatal signal without a flush or a lock.
  line.seq           = atomic_fetch_add_explicit(&next_seq, 1, memory_order_relaxed) + 1;
  off_t const offset = (off_t)((line.seq - 1) % CAPACITY) * sizeof(line);
  if (pwrite(trace_fd, &line, sizeof(line), offset) != sizeof(line)) {
    atomic_fetch_add_explicit(&write_failures, 1, memory_order_relaxed);
  }
}

int install_stream_trace(int fd, char const* library)
{
  void* cupti = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!cupti) { return -1; }
  __typeof__(&cuptiSubscribe) subscribe   = dlsym(cupti, "cuptiSubscribe");
  __typeof__(&cuptiEnableCallback) enable = dlsym(cupti, "cuptiEnableCallback");
  if (!subscribe || !enable) { return -2; }
  trace_fd = fd;
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
