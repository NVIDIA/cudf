# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RapidsMPF streaming-engine support."""

from __future__ import annotations

import cudf_polars.streaming.actor_graph.collectives.shuffle
import cudf_polars.streaming.actor_graph.collectives.sort

# Side-effect imports: each module registers
# ``@generate_ir_sub_network.register(...)`` handlers at import time so the
# dispatch table is populated before any query is evaluated.
import cudf_polars.streaming.actor_graph.groupby
import cudf_polars.streaming.actor_graph.io
import cudf_polars.streaming.actor_graph.join
import cudf_polars.streaming.actor_graph.over
import cudf_polars.streaming.actor_graph.prefilter_actor
import cudf_polars.streaming.actor_graph.repartition
import cudf_polars.streaming.actor_graph.union  # noqa: F401

__all__: list[str] = []
