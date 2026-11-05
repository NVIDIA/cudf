"""TEMP: time pinned host memory pool creation on WSL."""

import os
import time

from cuda.bindings import driver as cu


def check(r):
    err = r[0] if isinstance(r, tuple) else r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(cu.cuGetErrorName(err)[1])
    return r[1] if isinstance(r, tuple) and len(r) == 2 else r


check(cu.cuInit(0))
dev = check(cu.cuDeviceGet(0))
ctx = check(cu.cuDevicePrimaryCtxRetain(dev))
check(cu.cuCtxSetCurrent(ctx))
host_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
print(f"host RAM {host_bytes / 2**30:.1f} GiB", flush=True)


def make_pool(max_size):
    props = cu.CUmemPoolProps()
    props.allocType = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    props.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    props.location.id = 0
    props.maxSize = max_size
    t = time.perf_counter()
    pool = check(cu.cuMemPoolCreate(props))
    return pool, time.perf_counter() - t


def alloc_free(pool, nbytes=1 << 20):
    stream = check(cu.cuStreamCreate(0))
    t = time.perf_counter()
    ptr = check(cu.cuMemAllocFromPoolAsync(nbytes, pool, stream))
    check(cu.cuMemFreeAsync(ptr, stream))
    check(cu.cuStreamSynchronize(stream))
    check(cu.cuStreamDestroy(stream))
    return time.perf_counter() - t


for label, max_size in [
    ("max=0", 0),
    ("max=1GiB", 1 << 30),
    ("max=80%host", int(0.8 * host_bytes)),
]:
    for keep in (False, True):
        pools, times = [], []
        deadline = time.perf_counter() + 120
        for i in range(40):
            pool, dt = make_pool(max_size)
            times.append(dt)
            alloc_free(pool)
            if keep:
                pools.append(pool)
            else:
                check(cu.cuMemPoolDestroy(pool))
            if time.perf_counter() > deadline:
                break
        for p in pools:
            check(cu.cuMemPoolDestroy(p))
        print(
            f"{label} keep={keep}: n={len(times)} first={times[0] * 1e3:.1f}ms "
            f"max={max(times) * 1e3:.1f}ms last={times[-1] * 1e3:.1f}ms "
            f"total={sum(times):.2f}s",
            flush=True,
        )
