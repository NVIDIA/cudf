# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bulk-synchronous MPI shuffle."""

from __future__ import annotations

import argparse
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pylibcudf as plc
from mpi4py import MPI

import rapidsmpf.bootstrap
import rapidsmpf.communicator.mpi
import rmm.mr
from cudf_streaming.partition_utils import (
    partition_and_pack,
    unpack_and_concat,
)
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.spill import unspill_partitions
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.statistics import Statistics
from rapidsmpf.utils.string import format_bytes, parse_bytes
from rmm.pylibrmm.stream import DEFAULT_STREAM

try:
    from rapidsmpf.cupti import CuptiMonitor

    CUPTI_AVAILABLE = True
except ImportError:
    CUPTI_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapidsmpf.communicator.communicator import Communicator


def barrier(comm: Communicator) -> None:
    """
    Blocks until all processes in the communicator have reached this point.

    Parameters
    ----------
    comm
        The communicator to barrier.
    """
    if rapidsmpf.bootstrap.is_running_with_rrun():
        from rapidsmpf.communicator.ucxx import barrier as ucxx_barrier

        ucxx_barrier(comm)
    else:
        MPI.COMM_WORLD.barrier()


def read_batch(paths: list[str]) -> tuple[plc.Table, list[str]]:
    """
    Read a single batch of Parquet files.

    Parameters
    ----------
    paths
        List of file paths to the Parquet files.

    Returns
    -------
    plc.Table
        The table containing the data read from the Parquet files.
    list of str
        Column names from the Parquet files, excluding nested children.
    """
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(paths)
    ).build()
    tbl_w_meta = plc.io.parquet.read_parquet(options)
    return (tbl_w_meta.tbl, tbl_w_meta.column_names(include_children=False))


def write_table(
    table: plc.Table,
    output_path: str,
    id: int | str,
    column_names: list[str] | None,
) -> None:
    """
    Write a pylibcudf Table to a Parquet file.

    Parameters
    ----------
    table
        The table to be written to the Parquet file.
    output_path : str
        Directory where the Parquet file will be written.
    id
        Unique identifier used to generate the filename using `part.{id}.parque`.
    column_names
        List of column names.
    """
    path = f"{output_path}/part.{id}.parquet"
    builder = plc.io.parquet.ParquetWriterOptions.builder(
        plc.io.SinkInfo([path]), table
    )
    if column_names is not None:
        metadata = plc.io.types.TableInputMetadata(table)
        for col_meta, name in zip(
            metadata.column_metadata, column_names, strict=True
        ):
            col_meta.set_name(name)
        builder = builder.metadata(metadata)
    plc.io.parquet.write_parquet(builder.build())


def bulk_mpi_shuffle(
    paths: list[str],
    shuffle_on: list[str],
    output_path: str,
    comm: Communicator,
    br: BufferResource,
    *,
    num_output_files: int | None = None,
    batchsize: int = 1,
    read_func: Callable = read_batch,
    write_func: Callable = write_table,
    baseline: bool = False,
    statistics: Statistics | None = None,
) -> None:
    """
    Perform a bulk-synchronous dataset shuffle.

    Parameters
    ----------
    paths
        List of file paths to shuffle. This list contains all files in the
        dataset (not just the files that will be processed by the local rank).
    shuffle_on
        List of column names to shuffle on.
    output_path
        Path of the output directory where the data will be written. This
        directory does not need to be on a shared filesystem.
    comm
        The communicator to use.
    br
        Buffer resource to use.
    num_output_files
        Number of output files to produce. Default will preserve the
        input file count.
    batchsize
        Number of files to read at once on each rank.
    read_func
        Call-back function to read the input data. This function must accept a
        list of file paths, and return a pylibcudf Table and the list of column
        names in the table. Default logic will use `pylibcudf.read_parquet`.
    write_func
        Call-back function to write shuffled data to disk. This function must
        accept `table`, `output_path`, `id`, and `column_names` arguments.
        Default logic will write the pylibcudf table to a parquet file
        (e.g. `f"{output_path}/part.{id}.parquet"`).
    baseline
        Whether to skip the shuffle and run a simple IO baseline.
    statistics
        The statistics instance to use. If None, statistics is disabled.

    Notes
    -----
    This function is executed on each rank of the MPI communicator in a
    bulk-synchronous fashion. This means all ranks are expected to call
    this same function with the same arguments.
    """
    # Create output directory if necessary
    Path(output_path).mkdir(exist_ok=True)

    # Determine which files to process on this rank
    num_input_files = len(paths)
    num_output_files = num_output_files or num_input_files
    total_num_partitions = num_output_files
    files_per_rank = math.ceil(num_input_files / comm.nranks)
    start = files_per_rank * comm.rank
    finish = start + files_per_rank
    local_files = paths[start:finish]
    num_local_files = len(local_files)
    num_batches = math.ceil(num_local_files / batchsize)

    if baseline:
        # Skip the shuffle - Run IO baseline
        for batch_id in range(num_batches):
            batch = local_files[
                batch_id * batchsize : (batch_id + 1) * batchsize
            ]
            table, columns = read_func(batch)
            write_func(
                table,
                output_path,
                str(uuid.uuid4()),
                columns,
            )
    else:
        br = BufferResource(rmm.mr.get_current_device_resource())
        shuffler = Shuffler(
            comm,
            op_id=0,
            total_num_partitions=total_num_partitions,
            br=br,
        )

        # Read batches and submit them to the shuffler
        column_names = None
        for batch_id in range(num_batches):
            batch = local_files[
                batch_id * batchsize : (batch_id + 1) * batchsize
            ]
            table, columns = read_func(batch)
            if column_names is None:
                column_names = columns
            columns_to_hash = tuple(columns.index(val) for val in shuffle_on)
            packed_inputs = partition_and_pack(
                table,
                columns_to_hash=columns_to_hash,
                num_partitions=total_num_partitions,
                br=br,
                stream=DEFAULT_STREAM,
            )
            shuffler.insert_chunks(packed_inputs)

        # Tell the shuffler we are done adding local data
        shuffler.insert_finished()

        # Write shuffled partitions to disk
        shuffler.wait()
        for partition_id in shuffler.local_partitions():
            table = unpack_and_concat(
                unspill_partitions(
                    shuffler.extract(partition_id),
                    br=br,
                    allow_overbooking=True,
                ),
                br=br,
                stream=DEFAULT_STREAM,
            )
            write_func(
                table,
                output_path,
                partition_id,
                column_names,
            )
        shuffler.shutdown()


def ucxx_mpi_setup(
    options: Options, progress_thread: ProgressThread
) -> Communicator:
    """
    Bootstrap UCXX cluster using MPI.

    Parameters
    ----------
    options
        Configuration options.
    progress_thread
        Progress thread for the initialized communicator.

    Returns
    -------
    Communicator
        A new ucxx communicator.
    """
    import ucxx._lib.libucxx as ucx_api

    from rapidsmpf.communicator.ucxx import (
        barrier,
        get_root_ucxx_address,
        new_communicator,
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        comm = new_communicator(
            MPI.COMM_WORLD.size, None, None, options, progress_thread
        )
        root_address_bytes = get_root_ucxx_address(comm)
    else:
        root_address_bytes = None

    root_address_bytes = MPI.COMM_WORLD.bcast(root_address_bytes, root=0)

    if MPI.COMM_WORLD.Get_rank() != 0:
        root_address = ucx_api.UCXAddress.create_from_buffer(
            root_address_bytes
        )
        comm = new_communicator(
            MPI.COMM_WORLD.size, None, root_address, options, progress_thread
        )

    assert comm.nranks == MPI.COMM_WORLD.size
    barrier(comm)
    return comm


def setup_and_run(args: argparse.Namespace) -> None:
    """
    Set up the environment and run the shuffle example.

    Parameters
    ----------
    args
        Command-line arguments containing the configuration for the shuffle example.
    """
    options = Options(get_environment_variables())

    # Create a device pool memory resource. `BufferResource` wraps it in an
    # internal tracking `RmmResourceAdaptor` exposed via `device_mr_adaptor()`.
    base_mr = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(),
        initial_pool_size=args.rmm_pool_size,
        maximum_pool_size=args.rmm_pool_size,
    )

    # Create a buffer resource that limits device memory if `--spill-device`
    # is not None.
    memory_limits = (
        None
        if args.spill_device is None
        else {MemoryType.DEVICE: args.spill_device}
    )
    br = BufferResource(base_mr, memory_limits=memory_limits)
    mr = br.device_mr_adaptor()
    # Install the tracking adaptor as the current device resource so libcudf
    # temporary allocations are also tracked.
    rmm.mr.set_current_device_resource(mr)

    stats = Statistics(enable=args.statistics)

    progress_thread = ProgressThread(stats)
    if args.cluster_type == "mpi":
        comm = rapidsmpf.communicator.mpi.new_communicator(
            MPI.COMM_WORLD, options, progress_thread
        )
    elif args.cluster_type == "ucxx":
        if rapidsmpf.bootstrap.is_running_with_rrun():
            comm = rapidsmpf.bootstrap.create_ucxx_comm(
                progress_thread,
                type=rapidsmpf.bootstrap.BackendType.AUTO,
                options=options,
            )
        else:
            comm = ucxx_mpi_setup(options, progress_thread)
    cupti_monitor = None
    if args.monitor_memory is not None:
        if not CUPTI_AVAILABLE:
            if comm.rank == 0:
                comm.logger.print(
                    "WARNING: --memory-monitor specified but CUPTI support not available. "
                    "CUPTI monitoring disabled."
                )
        else:
            cupti_monitor = CuptiMonitor(enable_periodic_sampling=False)
            if comm.rank == 0:
                comm.logger.print("CUPTI memory monitoring enabled")

    if comm.rank == 0:
        spill_device = (
            "disabled"
            if args.spill_device is None
            else format_bytes(args.spill_device)
        )
        comm.logger.print(
            f"""\
Shuffle:
    input: {args.input}
    output: {args.output}
    on: {args.on}
  --cluster-type: {args.cluster_type}
  --n-output-files: {args.n_output_files}
  --batchsize: {args.batchsize}
  --baseline: {args.baseline}
  --rmm-pool-size: {format_bytes(args.rmm_pool_size)}
  --spill-device: {spill_device}"""
        )

    barrier(comm)

    if cupti_monitor is not None:
        cupti_monitor.start_monitoring()

    start_time = MPI.Wtime()
    bulk_mpi_shuffle(
        paths=sorted(map(str, args.input.glob("**/*"))),
        shuffle_on=args.on.split(","),
        output_path=args.output,
        comm=comm,
        br=br,
        num_output_files=args.n_output_files,
        batchsize=args.batchsize,
        baseline=args.baseline,
        statistics=stats,
    )
    elapsed_time = MPI.Wtime() - start_time
    barrier(comm)

    if cupti_monitor is not None:
        cupti_monitor.stop_monitoring()

        csv_filename = f"{args.monitor_memory}_{comm.rank}.csv"
        try:
            # Write CSV files
            cupti_monitor.write_csv(csv_filename)
            comm.logger.print(
                f"CUPTI memory data written to {csv_filename} "
                f"({cupti_monitor.get_sample_count()} samples, "
                f"{cupti_monitor.get_total_callback_count()} callbacks)"
            )

            # Print callback summary for rank 0
            if comm.rank == 0:
                comm.logger.print(
                    f"CUPTI Callback Summary:\n{cupti_monitor.get_callback_summary()}"
                )
        except Exception as e:
            comm.logger.print(f"Failed to write CUPTI CSV file: {e}")

    mem_peak = format_bytes(mr.get_main_record().peak())
    comm.logger.print(
        f"elapsed: {elapsed_time:.2f} sec | rmm device memory peak: {mem_peak}"
    )
    if stats.enabled:
        comm.logger.print(stats.report(mr=mr))


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
    """
    ret = Path(path)
    if not ret.is_dir():
        raise ValueError()
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Bulk-synchronous MPI shuffle",
        description="Shuffle a dataset at rest (on disk) on both ends.",
    )
    parser.add_argument(
        "input",
        type=dir_path,
        metavar="INPUT_DIR_PATH",
        help="Input directory path.",
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT_DIR_PATH",
        type=Path,
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
        "--baseline",
        default=False,
        action="store_true",
        help="Run an IO baseline without any shuffling.",
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
        "--cluster-type",
        type=str,
        default="mpi",
        choices=("mpi", "ucxx"),
        help=(
            "Cluster type to setup. Regardless of the cluster type selected it must "
            "be launched with 'mpirun'."
        ),
    )
    parser.add_argument(
        "--monitor-memory",
        type=str,
        default=None,
        help=(
            "Enable memory monitoring with CUPTI and save CSV files with given path "
            "prefix. For example, /tmp/test will write files to /tmp/test_<rank>.csv. "
            "Requires CUPTI support to be compiled in."
        ),
    )
    args = parser.parse_args()
    args.rmm_pool_size = (
        args.rmm_pool_size // 256
    ) * 256  # Align to 256 bytes
    setup_and_run(args)
