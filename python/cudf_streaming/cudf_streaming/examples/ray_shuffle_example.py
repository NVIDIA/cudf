# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example running a RapidsMPF Shuffle operation using Ray and UCXX communication."""

from __future__ import annotations

import argparse
import math

import numpy as np
import pylibcudf as plc
import ray

import rmm
from cudf_streaming.partition_utils import (
    partition_and_pack,
    unpack_and_concat,
)
from cudf_streaming.testing import assert_eq
from rapidsmpf.integrations.ray import RapidsMPFActor, setup_ray_ucxx_cluster
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.spill import unspill_partitions
from rapidsmpf.shuffler import Shuffler


class ShufflingActor(RapidsMPFActor):
    """
    An example of a Ray actor that performs a shuffle operation.

    Parameters
    ----------
    nranks
        Number of ranks.
    num_rows
        Number of rows in the input dataframe.
    batch_size
        Batch size (rows) of the input. The input dataframe will be split into batches of this size.
    total_nparts
        Total number of partitions into which the input dataframe will be partitioned.
    """

    def __init__(
        self,
        nranks: int,
        num_rows: int = 100,
        batch_size: int = -1,
        total_nparts: int = -1,
    ):
        super().__init__(nranks, statistics=None)
        self._num_rows: int = num_rows
        self._batch_size: int = batch_size
        self._total_nparts: int = total_nparts if total_nparts > 0 else nranks

    def _gen_table(self) -> plc.Table:
        """
        Generate a dummy table with three columns ("a", "b", "c").

        Returns
        -------
        plc.Table
            The input table.
        """
        # Every rank creates the full input table and all the expected partitions
        # (also partitions this rank might not get after the shuffle).

        rng = np.random.default_rng(
            42
        )  # Make sure all ranks create the same input table.

        return plc.Table(
            [
                plc.Column.from_iterable_of_py(
                    range(self._num_rows), plc.DataType(plc.TypeId.INT64)
                ),
                plc.Column.from_array(rng.integers(0, 1000, self._num_rows)),
                plc.Column.from_iterable_of_py(
                    (["cat", "dog"] * ((self._num_rows + 1) // 2))[
                        : self._num_rows
                    ],
                    plc.DataType(plc.TypeId.STRING),
                ),
            ]
        )

    def run(self) -> None:
        """Run the shuffle operation, and this will be called remotely from the client."""
        # If DEFAULT_STREAM was imported outside of this context, it will be pickled,
        # and it is not serializable. Therefore, we need to import it here.
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        df = self._gen_table()
        columns_to_hash = (1,)

        mr = rmm.mr.get_current_device_resource()
        br = BufferResource(mr)
        stream = DEFAULT_STREAM  # use the default stream

        # Calculate the expected output partitions on all ranks
        expected = {
            partition_id: unpack_and_concat(
                [packed],
                br=br,
                stream=stream,
            )
            for partition_id, packed in partition_and_pack(
                df,
                columns_to_hash=columns_to_hash,
                num_partitions=self._total_nparts,
                br=br,
                stream=stream,
            ).items()
        }

        shuffler = Shuffler(
            self.comm,
            0,
            total_num_partitions=self._total_nparts,
            br=br,
        )

        # Slice df and submit local slices to shuffler
        stride = math.ceil(self._num_rows / self.comm.nranks)
        local_df = plc.copying.slice(
            df,
            [
                self.comm.rank * stride,
                min((self.comm.rank + 1) * stride, self._num_rows),
            ],
        )[0]
        num_rows_local = local_df.num_rows()
        self._batch_size = (
            num_rows_local if self._batch_size < 0 else self._batch_size
        )
        for i in range(0, num_rows_local, self._batch_size):
            batch = plc.copying.slice(
                local_df, [i, min(i + self._batch_size, num_rows_local)]
            )[0]
            packed_inputs = partition_and_pack(
                batch,
                columns_to_hash=columns_to_hash,
                num_partitions=self._total_nparts,
                br=br,
                stream=stream,
            )
            shuffler.insert_chunks(packed_inputs)

        # Tell shuffler we are done adding data
        shuffler.insert_finished()

        # Extract and check shuffled partitions
        shuffler.wait()
        for partition_id in shuffler.local_partitions():
            packed_chunks = shuffler.extract(partition_id)
            partition = unpack_and_concat(
                unspill_partitions(
                    packed_chunks, br=br, allow_overbooking=True
                ),
                br=br,
                stream=stream,
            )
            assert_eq(
                partition,
                expected[partition_id],
                sort_rows=0,
            )

        shuffler.shutdown()


@ray.remote(num_gpus=1)
class GpuShufflingActor(ShufflingActor):
    """Shuffle example class with 1 GPU resource."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RapidsMPF Ray Shuffling Actor example ",
    )
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument("--num_rows", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--total_nparts", type=int, default=-1)
    args = parser.parse_args()

    ray.init()  # init ray with all resources

    # Create shufflling actors
    gpu_actors = setup_ray_ucxx_cluster(
        GpuShufflingActor,
        args.nranks,
        args.num_rows,
        args.batch_size,
        args.total_nparts,
    )

    try:
        # run the ShufflingActor.run method remotely
        ray.get([actor.run.remote() for actor in gpu_actors])  # type: ignore

    finally:
        for actor in gpu_actors:
            ray.kill(actor)
