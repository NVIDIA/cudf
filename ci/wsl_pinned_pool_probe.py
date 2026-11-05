"""TEMP: time CUDA memory pool creation vs number of live pools on WSL."""

import subprocess
import sys
import time

from cuda.bindings import driver as cu


def check(r):
    err = r[0] if isinstance(r, tuple) else r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(cu.cuGetErrorName(err)[1])
    return r[1] if isinstance(r, tuple) and len(r) == 2 else r


def init():
    check(cu.cuInit(0))
    dev = check(cu.cuDeviceGet(0))
    ctx = check(cu.cuDevicePrimaryCtxRetain(dev))
    check(cu.cuCtxSetCurrent(ctx))
    return dev


def run(kind, max_gib, keep, n):
    init()
    stream = check(cu.cuStreamCreate(0))
    pools = []
    t0 = time.perf_counter()
    for i in range(n):
        props = cu.CUmemPoolProps()
        props.allocType = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        if kind == "device":
            props.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        else:
            props.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        props.location.id = 0
        props.maxSize = int(max_gib * 2**30)
        t = time.perf_counter()
        pool = check(cu.cuMemPoolCreate(props))
        dt_create = time.perf_counter() - t
        t = time.perf_counter()
        ptr = check(cu.cuMemAllocFromPoolAsync(1 << 20, pool, stream))
        check(cu.cuMemFreeAsync(ptr, stream))
        check(cu.cuStreamSynchronize(stream))
        dt_alloc = time.perf_counter() - t
        if keep:
            pools.append(pool)
        else:
            check(cu.cuMemPoolDestroy(pool))
        print(
            f"  i={i} create={dt_create * 1e3:.1f}ms alloc={dt_alloc * 1e3:.1f}ms "
            f"t={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    t = time.perf_counter()
    for p in pools:
        check(cu.cuMemPoolDestroy(p))
    print(f"  destroyed {len(pools)} in {time.perf_counter() - t:.2f}s", flush=True)


def vareserve(gib, n):
    init()
    ptrs = []
    t0 = time.perf_counter()
    for i in range(n):
        t = time.perf_counter()
        ptr = check(cu.cuMemAddressReserve(int(gib * 2**30), 0, 0, 0))
        ptrs.append(ptr)
        print(
            f"  i={i} reserve={(time.perf_counter() - t) * 1e3:.1f}ms "
            f"t={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    for p in ptrs:
        check(cu.cuMemAddressFree(p, int(gib * 2**30)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "va":
            vareserve(float(sys.argv[2]), int(sys.argv[3]))
        else:
            run(sys.argv[1], float(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4]))
        sys.exit(0)
    dev = init()
    total = check(cu.cuDeviceTotalMem(dev))
    print(f"device total {total / 2**30:.1f} GiB", flush=True)
    configs = [
        ("device", 0, 0, 20),
        ("device", 0, 1, 40),
        ("device", 4, 1, 40),
        ("host", 0, 1, 40),
        ("host", 1, 1, 40),
        ("va", 64, None, 40),
        ("va", 1, None, 40),
    ]
    for kind, gib, keep, n in configs:
        if kind == "va":
            args, label = ["va", str(gib), str(n)], f"va reserve {gib}GiB x{n}"
        else:
            args = [kind, str(gib), str(keep), str(n)]
            label = f"{kind} maxSize={gib}GiB keep={keep} x{n}"
        print(label, flush=True)
        t = time.perf_counter()
        try:
            subprocess.run([sys.executable, __file__, *args], timeout=90, check=False)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT after 90s", flush=True)
        print(f"  wall {time.perf_counter() - t:.1f}s", flush=True)
