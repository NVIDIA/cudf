# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example running a Bulk RapidsMPF Shuffle operation using Ray and UCXX communication."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pylibcudf as plc
import ray

import rmm.mr
from cudf_streaming.partition_utils import (
    partition_and_pack,
    unpack_and_concat,
)
from rapidsmpf.integrations.ray import RapidsMPFActor, setup_ray_ucxx_cluster
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.spill import unspill_partitions
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.statistics import Statistics
from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe
from rapidsmpf.utils.string import format_bytes, parse_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator


@ray.remote(num_gpus=1, num_cpus=4)
class BulkRayShufflerActor(RapidsMPFActor):
    """
    Actor that performs a bulk shuffle operation using Ray.

    Parameters
    ----------
    nranks
        Number of ranks in the communication group.
    total_nparts
        Total number of output partitions.
    shuffle_on
        List of column names to shuffle on.
    batchsize
        Number of files to process in a batch.
    output_path
        Path to write output files.
    rmm_pool_size
        Size of the RMM memory pool in bytes.
    spill_device
        Device memory limit for spilling to host in bytes.
    enable_statistics
        Whether to collect statistics.
    """

    def __init__(
        self,
        nranks: int,
        total_nparts: int,
        shuffle_on: list[str],
        batchsize: int = 1,
        output_path: str = "./",
        rmm_pool_size: int = 1024 * 1024 * 1024,
        spill_device: int | None = None,
        *,
        enable_statistics: bool = False,
    ):
        self.batchsize = batchsize
        self.shuffle_on = shuffle_on
        self.output_path = output_path
        self.total_nparts = total_nparts
        self.rmm_pool_size = rmm_pool_size
        self.spill_device = spill_device

        # Initialize actor-local resources (statistics, memory resource).
        # `BufferResource` wraps the base device MR in an internal tracking
        # `RmmResourceAdaptor`, exposed via `device_mr_adaptor()`.
        base_mr = rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=self.rmm_pool_size,
            maximum_pool_size=self.rmm_pool_size,
        )
        # Create a buffer resource that limits device memory if `--spill-device`
        memory_limits = (
            None
            if self.spill_device is None
            else {MemoryType.DEVICE: self.spill_device}
        )
        br = BufferResource(base_mr, memory_limits=memory_limits)
        self.mr = br.device_mr_adaptor()
        # Install the tracking adaptor as the current device resource so
        # libcudf temp allocations are tracked too.
        rmm.mr.set_current_device_resource(self.mr)
        self.br = br
        super().__init__(nranks, Statistics(enable=enable_statistics))

    def setup_worker(self, root_address_bytes: bytes) -> None:
        """
        Setup the UCXX communication and a shuffle operation.

        Parameters
        ----------
        root_address_bytes
            Address of the root worker for UCXX initialization.
        """
        super().setup_worker(root_address_bytes)
        self.shuffler: Shuffler = Shuffler(
            self.comm,
            0,
            total_num_partitions=self.total_nparts,
            br=self.br,
        )

    def cleanup(self) -> None:
        """Cleanup the UCXX communication and the shuffle operation."""
        self.comm.logger.info(self.statistics.report())
        if self.shuffler is not None:
            self.shuffler.shutdown()

    def read_batch(self, paths: list[str]) -> tuple[plc.Table, list[str]]:
        """
        Read a single batch of Parquet files.

        Parameters
        ----------
        paths
            List of file paths to the Parquet files.

        Returns
        -------
            A tuple containing the read in table and the column names.
        """
        options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(paths)
        ).build()
        tbl_w_meta = plc.io.parquet.read_parquet(options)
        return (
            tbl_w_meta.tbl,
            tbl_w_meta.column_names(include_children=False),
        )

    def write_table(
        self,
        table: plc.Table,
        output_path: str,
        id: int | str,
        column_names: list[str],
    ) -> None:
        """
        Write a pylibcudf Table to a Parquet file.

        Parameters
        ----------
        table
            The table to write.
        output_path
            The path to write the table to.
        id
            Partition id used for naming the output file.
        column_names
            The column names of the table.
        """
        path = f"{output_path}/part.{id}.parquet"
        pylibcudf_to_cudf_dataframe(
            table,
            column_names=column_names,
        ).to_parquet(path)

    def insert_chunk(self, table: plc.Table, column_names: list[str]) -> None:
        """
        Insert a pylibcudf Table into the shuffler.

        Parameters
        ----------
        table
            The table to insert.
        column_names
            The column names of the table.
        """
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        columns_to_hash = tuple(
            column_names.index(val) for val in self.shuffle_on
        )
        packed_inputs = partition_and_pack(
            table,
            columns_to_hash=columns_to_hash,
            num_partitions=self.total_nparts,
            br=self.br,
            stream=DEFAULT_STREAM,
        )
        self.shuffler.insert_chunks(packed_inputs)

    def read_and_insert(self, paths: list[str]) -> list[str]:
        """
        Read the list of parquet files every batchsize and insert the partitions into the shuffler.

        Parameters
        ----------
        paths
            List of file paths to the Parquet files.

        Returns
        -------
            The column names of the table.
        """
        for i in range(0, len(paths), self.batchsize):
            tbl, column_names = self.read_batch(paths[i : i + self.batchsize])
            self.insert_chunk(tbl, column_names)
        self.insert_finished()
        return column_names

    def insert_finished(self) -> None:
        """Tell the shuffler that we are done inserting data."""
        self.shuffler.insert_finished()
        self.comm.logger.info("Insert finished")

    def extract(self) -> Iterator[tuple[int, plc.Table]]:
        """
        Extract shuffled partitions.

        Returns
        -------
            An iterator over the shuffled partitions.
        """
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        self.shuffler.wait()
        for partition_id in self.shuffler.local_partitions():
            packed_chunks = self.shuffler.extract(partition_id)
            partition = unpack_and_concat(
                unspill_partitions(
                    packed_chunks,
                    br=self.br,
                    allow_overbooking=True,
                ),
                br=self.br,
                stream=DEFAULT_STREAM,
            )
            yield partition_id, partition

    def extract_and_write(self, column_names: list[str]) -> None:
        """
        Extract and write shuffled partitions.

        Parameters
        ----------
        column_names
            The column names of the table.
        """
        for partition_id, partition in self.extract():
            self.write_table(
                partition, self.output_path, partition_id, column_names
            )


def bulk_ray_shuffle(
    paths: list[str],
    shuffle_on: list[str],
    output_path: str,
    num_workers: int = 2,
    batchsize: int = 1,
    num_output_files: int | None = None,
    rmm_pool_size: int = 1024 * 1024 * 1024,
    spill_device: int | None = None,
    *,
    enable_statistics: bool = False,
) -> None:
    """
    Perform a bulk shuffle operation using Ray and UCXX communication.

    Parameters
    ----------
    paths
        The list of paths to the input files.
    shuffle_on
        The list of column names to shuffle on.
    output_path
        The directory to write the shuffled data.
    num_workers
        The number of workers to use.
    batchsize
        The number of files to read on each rank at once.
    num_output_files
        The number of output files to write.
    rmm_pool_size
        The size of the RMM pool.
    spill_device
        Device memory limit for spilling to host.
    enable_statistics
        Whether to collect statistics.
    """
    # Initialize the UCXX cluster
    num_input_files = len(paths)
    num_output_files = num_output_files or num_input_files
    total_num_partitions = num_output_files
    files_per_rank = math.ceil(num_input_files / num_workers)

    actors = setup_ray_ucxx_cluster(
        BulkRayShufflerActor,
        num_workers=num_workers,
        total_nparts=total_num_partitions,
        shuffle_on=shuffle_on,
        batchsize=batchsize,
        output_path=output_path,
        enable_statistics=enable_statistics,
        rmm_pool_size=rmm_pool_size,
        spill_device=spill_device,
    )
    start_time = time.time()
    insert_tasks = []
    for i, actor in enumerate(actors):
        # Calculate the start and end indices for this actor's files
        start = i * files_per_rank
        # Use min to ensure we don't go beyond the end of the paths list
        end = min(start + files_per_rank, num_input_files)
        insert_tasks.append(actor.read_and_insert.remote(paths[start:end]))
    column_names = ray.get(insert_tasks)
    ray.get(
        [
            actor.extract_and_write.remote(column_name)
            for actor, column_name in zip(actors, column_names, strict=False)
        ]
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")  # noqa: T201
    ray.get([actor.cleanup.remote() for actor in actors])


def dir_path(path: str) -> Path:
    """
    Validate that the given path is a directory and return a Path object.

    Parameters
    ----------
    path
        The path to check.

    Returns
    -------
    Path
        A Path object representing the directory.

    Raises
    ------
    ValueError
        If the path is not a directory.
    """
    ret = Path(path)
    if not ret.is_dir():
        raise ValueError(f"{path} path is not a directory")
    return ret


def setup_and_run(args: argparse.Namespace) -> None:
    """
    Setup and run the bulk shuffle operation.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    if args.ray_address or os.environ.get("RAY_ADDRESS") is not None:
        ray.init(address="auto")  # connect to existing cluster
    else:
        ray.init(num_gpus=args.num_workers, dashboard_host="0.0.0.0")

    bulk_ray_shuffle(
        paths=sorted(map(str, args.input.glob("**/*"))),
        shuffle_on=args.on.split(","),
        output_path=args.output,
        num_workers=args.num_workers,
        batchsize=args.batchsize,
        num_output_files=args.n_output_files,
        enable_statistics=args.statistics,
        rmm_pool_size=args.rmm_pool_size,
        spill_device=args.spill_device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Bulk-synchronous Ray shuffle",
        description="Shuffle a dataset at rest (on disk) on both ends.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers to use.",
    )
    parser.add_argument(
        "input",
        type=dir_path,
        metavar="INPUT_DIR_PATH",
        help="Input directory path.",
    )
    parser.add_argument(
        "output",
        type=dir_path,
        metavar="OUTPUT_DIR_PATH",
        help="Output directory path.",
    )
    parser.add_argument(
        "on",
        metavar="COLUMN_LIST",
        type=str,
        help="Comma-separated list of column names to shuffle on.",
    )
    parser.add_argument(
        "--n-output-files",
        type=int,
        default=None,
        help="Number of output files. Default preserves input file count.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Number of files to read on each MPI rank at once.",
    )
    parser.add_argument(
        "--rmm-pool-size",
        type=parse_bytes,
        default=format_bytes(int(rmm.mr.available_device_memory()[1] * 0.8)),
        help=(
            "The size of the RMM pool as a string with unit such as '2MiB' and '4KiB'. "
            "Default to 80%% of the total device memory, which is %(default)s."
        ),
    )
    parser.add_argument(
        "--spill-device",
        type=lambda x: None if x is None else parse_bytes(x),
        default=None,
        help=(
            "Spilling device-to-host threshold as a string with unit such as '2MiB' "
            "and '4KiB'. Default is no spilling"
        ),
    )
    parser.add_argument(
        "--statistics",
        default=False,
        action="store_true",
        help="Enable statistics.",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Connect to an existing Ray cluster.",
    )
    args = parser.parse_args()
    args.rmm_pool_size = (
        args.rmm_pool_size // 256
    ) * 256  # Align to 256 bytes
    setup_and_run(args)
