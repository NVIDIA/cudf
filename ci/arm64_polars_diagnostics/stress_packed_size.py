# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise packed-size and allocator churn without Polars or distributed transports."""

import argparse
import faulthandler
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context


def run_worker(mode, worker, pools, calls):
    import pyarrow as pa
    import pylibcudf as plc
    import rmm.mr
    from cudf_streaming.partition_utils import split_and_pack_cost
    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.memory.buffer_resource import BufferResource
    from repro_diagnostics import _install_fault_report
    from rmm.pylibrmm.stream import Stream

    faulthandler.enable()
    _reporter = _install_fault_report(f"stress-{mode}-{worker}")
    resource_type = {
        "async": rmm.mr.CudaAsyncMemoryResource,
        "synchronous": rmm.mr.CudaMemoryResource,
    }[mode]
    inputs = (
        pa.table(
            {
                "key": pa.array(
                    [None, 4, 5, None, 6, 8, 9, 4, 4, 5, 2, 3, 7],
                    type=pa.int8(),
                )
            }
        ),
        pa.table({"key": pa.array([9], type=pa.int8())}),
    )
    previous = rmm.mr.get_current_device_resource()
    print(f"{mode} worker {worker} starting {pools} pools", flush=True)
    for _ in range(pools):
        mr = resource_type()
        rmm.mr.set_current_device_resource(mr)
        try:
            br = BufferResource(mr)
            stream = Stream()
            for i in range(calls):
                table = plc.Table.from_arrow(
                    inputs[i % len(inputs)], stream=stream
                )
                chunk = TableChunk.from_pylibcudf_table(
                    table, stream, exclusive_view=True, br=br
                )
                size = split_and_pack_cost(table, stream, br)
                if size <= 0 or size != chunk.data_alloc_size():
                    raise RuntimeError(
                        f"Incorrect packed size for {mode}, {worker}, {i}: {size}"
                    )
                del chunk, table
            del br, stream
        finally:
            rmm.mr.set_current_device_resource(previous)
        del mr
    return (
        f"{mode} worker {worker} completed {pools * calls} packing iterations"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pools", type=int, default=30)
    parser.add_argument("--calls", type=int, default=32)
    args = parser.parse_args()
    failed = False
    for mode in ("async", "synchronous"):
        with ProcessPoolExecutor(
            args.workers, mp_context=get_context("spawn")
        ) as pool:
            jobs = [
                pool.submit(run_worker, mode, i, args.pools, args.calls)
                for i in range(args.workers)
            ]
            for job in as_completed(jobs):
                try:
                    print(job.result(), flush=True)
                except Exception as error:
                    failed = True
                    print(mode, repr(error), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
