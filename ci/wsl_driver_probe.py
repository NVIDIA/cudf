# TEMP: time CUDA driver memory-pool calls on WSL
import ctypes
import time

cu = ctypes.CDLL("libcuda.so.1")


def check(r, what):
    if r != 0:
        raise RuntimeError(f"{what} -> CUresult {r}")


check(cu.cuInit(0), "cuInit")
dev = ctypes.c_int()
check(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = ctypes.c_void_p()
t = time.perf_counter()
check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "ctx")
check(cu.cuCtxSetCurrent(ctx), "set")
print(f"context: {time.perf_counter() - t:.3f}s")


class Props(ctypes.Structure):
    _fields_ = [
        ("allocType", ctypes.c_int),
        ("handleTypes", ctypes.c_int),
        ("locType", ctypes.c_int),
        ("locId", ctypes.c_int),
        ("win32", ctypes.c_void_p),
        ("maxSize", ctypes.c_size_t),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 54),
    ]


for i in range(5):
    p = Props(allocType=1, handleTypes=0, locType=1, locId=0)
    pool = ctypes.c_void_p()
    t = time.perf_counter()
    check(cu.cuMemPoolCreate(ctypes.byref(pool), ctypes.byref(p)), "poolcreate")
    t1 = time.perf_counter()
    check(cu.cuMemPoolDestroy(pool), "pooldestroy")
    print(f"cuMemPoolCreate {t1 - t:.3f}s destroy {time.perf_counter() - t1:.3f}s")

stream = ctypes.c_void_p(0)
for size in [1 << 20, 1 << 26, 1 << 30, 8 << 30]:
    ptr = ctypes.c_uint64()
    t = time.perf_counter()
    check(cu.cuMemAllocAsync(ctypes.byref(ptr), ctypes.c_size_t(size), stream), "allocasync")
    check(cu.cuStreamSynchronize(stream), "sync")
    t1 = time.perf_counter()
    check(cu.cuMemFreeAsync(ptr, stream), "freeasync")
    check(cu.cuStreamSynchronize(stream), "sync")
    t2 = time.perf_counter()
    check(cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(size)), "alloc")
    t3 = time.perf_counter()
    check(cu.cuMemFree_v2(ptr), "free")
    t4 = time.perf_counter()
    print(
        f"size {size >> 20} MiB: allocAsync {t1 - t:.4f}s freeAsync {t2 - t1:.4f}s "
        f"cuMemAlloc {t3 - t2:.4f}s cuMemFree {t4 - t3:.4f}s"
    )
