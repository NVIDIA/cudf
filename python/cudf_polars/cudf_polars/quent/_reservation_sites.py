# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable names for memory-reservation attribution sites."""

from __future__ import annotations

from enum import StrEnum


class ReservationSite(StrEnum):
    """
    An identifier for a reservation.

    These are used to disambiguate memory reservations sites *within*
    an Actor, which might make multiple reservations.
    """

    ALLGATHER_AGGREGATE = "allgather.aggregate"
    ALLGATHER_PARTITIONS = "allgather.partitions"
    ALLGATHER_REDUCE = "allgather.reduce"
    BOUNDARIES_ALLGATHER = "boundaries.allgather"
    BROADCAST_ALLGATHER = "broadcast.allgather"
    BROADCAST_COLLECT = "broadcast.collect"
    BROADCAST_EVALUATE = "broadcast.evaluate"
    BROADCAST_LARGE = "broadcast.large"
    CALLBACK_CONSUME = "callback.consume"
    CALLBACK_EXCHANGE = "callback.exchange"
    CHUNK_COLLECTION = "chunk.collection"
    CONCATENATE_BATCH = "concatenate.batch"
    CONCATENATE_INPUTS = "concatenate.inputs"
    DATA_ALLGATHER = "data.allgather"
    DATA_EMIT = "data.emit"
    DATA_EXCHANGE = "data.exchange"
    DATA_FORWARD = "data.forward"
    ENDPOINTS_ALLGATHER = "endpoints.allgather"
    ENDPOINTS_LOCAL = "endpoints.local"
    EVALUATE_CHUNK = "evaluate.chunk"
    EVALUATE_MAIN = "evaluate.main"
    EVALUATE_TRIM_COPY = "evaluate.trim.copy"
    FANOUT_BOUNDED = "fanout.bounded"
    FANOUT_UNBOUNDED_BUFFER = "fanout.unbounded.buffer"
    FANOUT_UNBOUNDED_COPY = "fanout.unbounded.copy"
    GLOBAL_SHUFFLE = "global.shuffle"
    GLOBAL_SHUFFLE_INPUT = "global.shuffle.input"
    MULTI_EVALUATE = "multi.evaluate"
    ORDERED_GROUPBY_BOUNDARIES = "ordered.groupby.boundaries"
    ORDERED_GROUPBY_EXCHANGE = "ordered.groupby.exchange"
    ORDERED_GROUPBY_FIRST_KEYS = "ordered.groupby.first.keys"
    ORDERED_GROUPBY_SPLIT = "ordered.groupby.split"
    ORDERING_ALLGATHER = "ordering.allgather"
    OVERLAP_RESERVE = "overlap.reserve"
    PARTITION_OUTPUT = "partition.output"
    PARTITION_REMOTE = "partition.remote"
    PLANNING_COLLECTIVE = "planning.collective"
    PREFILTER_BLOOM = "prefilter.bloom"
    PREFILTER_PROJECT_KEYS = "prefilter.project.keys"
    PREFILTER_SAMPLE_INPUTS = "prefilter.sample.inputs"
    PYTHON_SCAN = "python.scan"
    SAMPLE_STATISTICS = "sample.statistics"
    SCAN_READ = "scan.read"
    SHUFFLE_DATA = "shuffle.data"
    SHUFFLE_FORWARD = "shuffle.forward"
    SHUFFLE_INPUT = "shuffle.input"
    SHUFFLE_PREPARE = "shuffle.prepare"
    SHUFFLE_RETURN = "shuffle.return"
    SHUFFLE_STATISTICS = "shuffle.statistics"
    SINK_WRITE = "sink.write"
    TOPK_ALLGATHER = "topk.allgather"


__all__ = ["ReservationSite"]
