/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include <cuda.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
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
  X(cuMemsetD8Async)

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
};

static void* replay(void* argument)
{
  struct worker const* worker = argument;
  CUstream stream, producer = NULL;
  CUevent ready = NULL;
  CHECK(cuCtxSetCurrent, worker->context);
  CHECK(cuStreamCreate, &stream, CU_STREAM_NON_BLOCKING);
  if (worker->event_work) {
    CHECK(cuStreamCreate, &producer, CU_STREAM_NON_BLOCKING);
    CHECK(cuEventCreate, &ready, CU_EVENT_DISABLE_TIMING);
  }
  pthread_barrier_wait(worker->start);
  for (size_t round = 0; round < worker->rounds; ++round) {
    CUdeviceptr blocks[7], reused;
    for (size_t i = 0; i < sizeof(blocks) / sizeof(*blocks); ++i) {
      CHECK(cuMemAllocFromPoolAsync, &blocks[i], 512 * (1 + (i + round) % 4), worker->pool, stream);
    }
    if (ready) {
      CUdeviceptr produced;
      CHECK(cuMemAllocFromPoolAsync, &produced, 512, worker->pool, producer);
      CHECK(cuMemsetD8Async, produced, 0x7f, 512, producer);
      CHECK(cuMemFreeAsync, produced, producer);
      CHECK(cuEventRecord, ready, producer);
      CHECK(cuStreamWaitEvent, stream, ready, 0);
      CHECK(cuMemsetD8Async, blocks[0], 0x5a, 512, stream);
    }
    for (size_t i = 0; i < sizeof(blocks) / sizeof(*blocks); ++i) {
      CHECK(cuMemFreeAsync, blocks[i], stream);
    }
    // The captured stream completed one sync after freeing scratch allocations.
    // Its next allocation succeeded, and the following sync faulted.
    CHECK(cuStreamSynchronize, stream);
    CHECK(cuMemAllocFromPoolAsync, &reused, 512, worker->pool, stream);
    CHECK(cuStreamSynchronize, stream);
    CHECK(cuMemFreeAsync, reused, stream);
  }
  CHECK(cuStreamSynchronize, stream);
  if (ready) {
    CHECK(cuStreamSynchronize, producer);
    CHECK(cuEventDestroy, ready);
    CHECK(cuStreamDestroy, producer);
  }
  CHECK(cuStreamDestroy, stream);
  CHECK(cuCtxSetCurrent, NULL);
  printf("worker %u completed %zu rounds\n", worker->id, worker->rounds);
  return NULL;
}

int main(int argc, char** argv)
{
  if (argc != 4 ||
      (strcmp(argv[1], "shared") && strcmp(argv[1], "separate") && strcmp(argv[1], "host")) ||
      (strcmp(argv[2], "alloc") && strcmp(argv[2], "events"))) {
    fprintf(stderr, "Usage: %s {shared|separate|host} {alloc|events} ROUNDS\n", argv[0]);
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
  for (unsigned i = 0; i < 2; ++i) {
    workers[i] = (struct worker){
      context, pools[i % pool_count], &start, rounds, i, !strcmp(argv[2], "events")};
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
