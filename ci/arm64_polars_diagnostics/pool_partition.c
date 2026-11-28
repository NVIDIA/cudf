/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include <cuda.h>

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ALLOCATE \
  "_ZN3rmm9_RMM_27_22mr31cuda_async_view_memory_resource8allocateEN4cuda3__410stream_refEmm"
#define POOL_HANDLE "_ZNK3rmm9_RMM_27_22mr31cuda_async_view_memory_resource11pool_handleEv"
#define VIEW_CONSTRUCTOR \
  "_ZN3rmm9_RMM_27_22mr31cuda_async_view_memory_resourceC1EP18CUmemPoolHandle_st"

static void* (*allocate)(void*, CUstream, size_t, size_t);
static CUmemoryPool (*pool_handle)(void const*);
static void (*construct_view)(void*, CUmemoryPool);
static __typeof__(&cuCtxGetDevice) get_device;
static __typeof__(&cuMemPoolCreate) pool_create;
static __typeof__(&cuMemPoolGetAttribute) pool_get;
static __typeof__(&cuMemPoolSetAttribute) pool_set;
static pthread_once_t init_once   = PTHREAD_ONCE_INIT;
static pthread_mutex_t pools_lock = PTHREAD_MUTEX_INITIALIZER;
static int separate;
static FILE* journal;

struct partition {
  CUmemoryPool original, selected;
  CUstream stream;
  struct partition* next;
};

static struct partition* partitions;
static _Thread_local struct partition* last;

static void fail(char const* operation, int code)
{
  fprintf(stderr, "pool experiment setup failed: %s (%d)\n", operation, code);
  abort();
}

static void* symbol(void* library, char const* name)
{
  void* result = library ? dlsym(library, name) : NULL;
  if (!result) { fail(name, 0); }
  return result;
}

static void check(CUresult result, char const* operation)
{
  if (result != CUDA_SUCCESS) { fail(operation, result); }
}

static void initialize(void)
{
  char const* mode = getenv("CUDF_POLARS_POOL_MODE");
  if (!mode || (strcmp(mode, "shared") && strcmp(mode, "separate"))) {
    fail("invalid CUDF_POLARS_POOL_MODE", 0);
  }
  separate  = strcmp(mode, "separate") == 0;
  void* rmm = dlopen("librmm.so", RTLD_LAZY | RTLD_NOLOAD);
  allocate  = symbol(rmm, ALLOCATE);
  if (separate) {
    pool_handle    = symbol(rmm, POOL_HANDLE);
    construct_view = symbol(rmm, VIEW_CONSTRUCTOR);
    void* cuda     = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    get_device     = symbol(cuda, "cuCtxGetDevice");
    pool_create    = symbol(cuda, "cuMemPoolCreate");
    pool_get       = symbol(cuda, "cuMemPoolGetAttribute");
    pool_set       = symbol(cuda, "cuMemPoolSetAttribute");
  }

  char path[4096];
  char const* dir = getenv("CUDF_POLARS_DIAGNOSTICS_DIR");
  if (!dir ||
      snprintf(path, sizeof(path), "%s/pools-%ld.tsv", dir, (long)getpid()) >= (int)sizeof(path)) {
    fail("pool journal path", 0);
  }
  journal = fopen(path, "w");
  if (!journal) { fail("open pool journal", 0); }
  setvbuf(journal, NULL, _IOLBF, 0);
  fprintf(journal, "%s\n", mode);
}

static struct partition* partition_for(CUmemoryPool original, CUstream stream)
{
  if (last && last->original == original && last->stream == stream) { return last; }

  pthread_mutex_lock(&pools_lock);
  struct partition* entry = partitions;
  while (entry && (entry->original != original || entry->stream != stream)) {
    entry = entry->next;
  }
  if (!entry) {
    entry = calloc(1, sizeof(*entry));
    if (!entry) { fail("allocate pool mapping", 0); }
    entry->original = original;
    entry->stream   = stream;
    CUdevice device;
    check(get_device(&device), "cuCtxGetDevice");
    CUmemPoolProps props = {.allocType = CU_MEM_ALLOCATION_TYPE_PINNED,
                            .location  = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = device}};
    check(pool_create(&entry->selected, &props), "cuMemPoolCreate");
    cuuint64_t threshold;
    check(pool_get(original, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold),
          "get release threshold");
    check(pool_set(entry->selected, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold),
          "set release threshold");
    CUmemPool_attribute const attrs[] = {CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                                         CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                                         CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES};
    for (size_t i = 0; i < sizeof(attrs) / sizeof(*attrs); ++i) {
      int value;
      check(pool_get(original, attrs[i], &value), "get reuse policy");
      check(pool_set(entry->selected, attrs[i], &value), "set reuse policy");
    }
    // Keep replacement pools alive through outstanding frees on other streams.
    // Each replay worker exits with its CUDA context after one pytest round.
    entry->next = partitions;
    partitions  = entry;
    fprintf(journal, "%p\t%p\t%p\n", (void*)original, (void*)stream, (void*)entry->selected);
  }
  pthread_mutex_unlock(&pools_lock);
  last = entry;
  return entry;
}

// These are the exported AArch64 RMM member-function ABIs in the pinned
// replay packages. stream_ref and cuda_async_view_memory_resource each hold one
// pointer. The real RMM constructor and allocator keep its validation, alignment
// and error paths in both conditions.
void* partitioned_allocate(void*, CUstream, size_t, size_t) __asm__(ALLOCATE);
void* partitioned_allocate(void* resource, CUstream stream, size_t bytes, size_t alignment)
{
  pthread_once(&init_once, initialize);
  // Legacy and per-thread default stream sentinels are not explicit stream handles.
  if (!separate || (uintptr_t)stream <= 2 || !bytes) {
    return allocate(resource, stream, bytes, alignment);
  }
  struct partition* entry = partition_for(pool_handle(resource), stream);
  CUmemoryPool view;
  construct_view(&view, entry->selected);
  return allocate(&view, stream, bytes, alignment);
}
