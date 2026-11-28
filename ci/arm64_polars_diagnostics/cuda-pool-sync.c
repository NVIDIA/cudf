/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include <cuda.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DRIVER_APIS(X)         \
  X(cuInit)                    \
  X(cuDriverGetVersion)        \
  X(cuDeviceGet)               \
  X(cuDeviceGetName)           \
  X(cuDevicePrimaryCtxRetain)  \
  X(cuDevicePrimaryCtxRelease) \
  X(cuCtxSetCurrent)           \
  X(cuMemPoolCreate)           \
  X(cuMemPoolDestroy)          \
  X(cuMemAllocFromPoolAsync)   \
  X(cuMemFreeAsync)            \
  X(cuStreamCreate)            \
  X(cuStreamDestroy)           \
  X(cuStreamSynchronize)       \
  X(cuStreamWaitEvent)         \
  X(cuEventCreate)             \
  X(cuEventRecord)             \
  X(cuEventDestroy)            \
  X(cuMemsetD8Async)           \
  X(cuMemcpyDtoDAsync)

#define DECLARE(api) static __typeof__(&api) driver_##api;
DRIVER_APIS(DECLARE)
#undef DECLARE

#define CHECK(api, ...)                                 \
  do {                                                  \
    CUresult const status = driver_##api(__VA_ARGS__);  \
    if (status != CUDA_SUCCESS) {                       \
      fprintf(stderr, "%s failed: %d\n", #api, status); \
      _Exit(EXIT_FAILURE);                              \
    }                                                   \
  } while (0)
#define STRINGIFY_(value) #value
#define STRINGIFY(value)  STRINGIFY_(value)

int install_fault_report(int fd);

struct worker {
  CUcontext context;
  CUmemoryPool pool;
  pthread_barrier_t* start;
  size_t rounds;
  unsigned id;
  int event_work;
  int handoff;
  atomic_bool* allocating;
};

static CUdeviceptr allocate(struct worker const* worker, size_t bytes, CUstream stream)
{
  CUdeviceptr pointer;
  atomic_store_explicit(&worker->allocating[worker->id], true, memory_order_relaxed);
  CHECK(cuMemAllocFromPoolAsync, &pointer, bytes, worker->pool, stream);
  atomic_store_explicit(&worker->allocating[worker->id], false, memory_order_relaxed);
  return pointer;
}

static void finish_streams(CUstream stream, CUstream producer, CUevent ready)
{
  CHECK(cuStreamSynchronize, stream);
  if (ready) {
    CHECK(cuStreamSynchronize, producer);
    CHECK(cuEventDestroy, ready);
    CHECK(cuStreamDestroy, producer);
  }
  CHECK(cuStreamDestroy, stream);
}

static void* replay(void* argument)
{
  struct worker const* worker = argument;
  CUstream stream, producer = NULL;
  CUevent ready   = NULL;
  size_t overlaps = 0;
  CHECK(cuCtxSetCurrent, worker->context);
  pthread_barrier_wait(worker->start);
  for (size_t round = 0; round < worker->rounds; ++round) {
    // The original stream was 37 ms old. Rotate only the handoff variant
    // so the simpler cases also exercise long-lived streams.
    if (round == 0 || (worker->handoff && round % 256 == 0)) {
      if (round) { finish_streams(stream, producer, ready); }
      CHECK(cuStreamCreate, &stream, CU_STREAM_NON_BLOCKING);
      if (worker->event_work) {
        CHECK(cuStreamCreate, &producer, CU_STREAM_NON_BLOCKING);
        CHECK(cuEventCreate, &ready, CU_EVENT_DISABLE_TIMING);
      }
    }

    CUdeviceptr blocks[7];
    for (size_t i = 0; i < sizeof(blocks) / sizeof(*blocks); ++i) {
      blocks[i] = allocate(worker, 512 * (1 + (i + round) % 4), stream);
    }
    if (ready) {
      CUdeviceptr const produced = allocate(worker, 512, producer);
      CHECK(cuMemsetD8Async, produced, 0x7f, 512, producer);
      if (!worker->handoff) { CHECK(cuMemFreeAsync, produced, producer); }
      CHECK(cuEventRecord, ready, producer);
      CHECK(cuStreamWaitEvent, stream, ready, 0);
      if (worker->handoff) {
        CHECK(cuMemcpyDtoDAsync, blocks[0], produced, 512, stream);
        CHECK(cuMemFreeAsync, produced, stream);
      } else {
        CHECK(cuMemsetD8Async, blocks[0], 0x5a, 512, stream);
      }
    }
    for (size_t i = 0; i < sizeof(blocks) / sizeof(*blocks); ++i) {
      CHECK(cuMemFreeAsync, blocks[i], stream);
    }
    // The captured stream completed one sync after freeing scratch allocations.
    // Its next allocation succeeded, and the following sync faulted.
    CHECK(cuStreamSynchronize, stream);
    CUdeviceptr const reused = allocate(worker, 512, stream);
    overlaps += atomic_load_explicit(&worker->allocating[1 - worker->id], memory_order_relaxed);
    CHECK(cuStreamSynchronize, stream);
    CHECK(cuMemFreeAsync, reused, stream);
  }
  finish_streams(stream, producer, ready);
  CHECK(cuCtxSetCurrent, NULL);
  printf("worker %u completed %zu rounds; observed %zu peer allocations before final sync\n",
         worker->id,
         worker->rounds,
         overlaps);
  return NULL;
}

int main(int argc, char** argv)
{
  if (argc != 4 ||
      (strcmp(argv[1], "shared") && strcmp(argv[1], "separate") && strcmp(argv[1], "host")) ||
      (strcmp(argv[2], "alloc") && strcmp(argv[2], "events") && strcmp(argv[2], "handoff"))) {
    fprintf(stderr, "Usage: %s {shared|separate|host} {alloc|events|handoff} ROUNDS\n", argv[0]);
    return EXIT_FAILURE;
  }
  char* end;
  uintmax_t const rounds = strtoumax(argv[3], &end, 10);
  if (!rounds || *end || argv[3][0] == '-' || rounds > SIZE_MAX) { return EXIT_FAILURE; }

  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    fprintf(stderr, "%s\n", dlerror());
    return EXIT_FAILURE;
  }
#define LOAD(api)                                           \
  driver_##api = dlsym(library, STRINGIFY(api));            \
  if (!driver_##api) {                                      \
    fprintf(stderr, "%s: %s\n", STRINGIFY(api), dlerror()); \
    return EXIT_FAILURE;                                    \
  }
  DRIVER_APIS(LOAD)
#undef LOAD
  int const fault = open("cuda-pool-sync.fault.log", O_CREAT | O_TRUNC | O_WRONLY, 0600);
  if (fault < 0 || install_fault_report(fault) != 0) {
    perror("fault reporter");
    return EXIT_FAILURE;
  }

  int version;
  CUdevice device;
  CUcontext context;
  char name[128];
  CHECK(cuInit, 0);
  CHECK(cuDriverGetVersion, &version);
  CHECK(cuDeviceGet, &device, 0);
  CHECK(cuDeviceGetName, name, sizeof(name), device);
  CHECK(cuDevicePrimaryCtxRetain, &context, device);
  CHECK(cuCtxSetCurrent, context);
  Dl_info driver;
  if (!dladdr((void*)driver_cuInit, &driver)) { return EXIT_FAILURE; }
  printf("GPU %s, CUDA driver API %d, library %s, %s/%s, %" PRIuMAX " rounds per thread\n",
         name,
         version,
         driver.dli_fname,
         argv[1],
         argv[2],
         rounds);
  fflush(stdout);

  CUmemPoolProps props = {.allocType = CU_MEM_ALLOCATION_TYPE_PINNED,
                          .location  = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = device}};
  if (!strcmp(argv[1], "host")) { props.location.type = CU_MEM_LOCATION_TYPE_HOST; }
  CUmemoryPool pools[2];
  unsigned const pool_count = !strcmp(argv[1], "separate") ? 2 : 1;
  for (unsigned i = 0; i < pool_count; ++i) {
    CHECK(cuMemPoolCreate, &pools[i], &props);
  }
  pthread_barrier_t start;
  if (pthread_barrier_init(&start, NULL, 2) != 0) { return EXIT_FAILURE; }
  pthread_t threads[2];
  struct worker workers[2];
  atomic_bool allocating[2] = {false, false};
  for (unsigned i = 0; i < 2; ++i) {
    workers[i] = (struct worker){context,
                                 pools[i % pool_count],
                                 &start,
                                 rounds,
                                 i,
                                 strcmp(argv[2], "alloc") != 0,
                                 !strcmp(argv[2], "handoff"),
                                 allocating};
    if (pthread_create(&threads[i], NULL, replay, &workers[i]) != 0) { _Exit(EXIT_FAILURE); }
  }
  for (unsigned i = 0; i < 2; ++i) {
    if (pthread_join(threads[i], NULL) != 0) { _Exit(EXIT_FAILURE); }
  }
  pthread_barrier_destroy(&start);
  for (unsigned i = 0; i < pool_count; ++i) {
    CHECK(cuMemPoolDestroy, pools[i]);
  }
  CHECK(cuCtxSetCurrent, NULL);
  CHECK(cuDevicePrimaryCtxRelease, device);
  close(fault);
  return EXIT_SUCCESS;
}
