# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracing and monitoring IR execution."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Concatenate, TypeVar

import nvtx

CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"

nvtx_annotate_cudf_polars = functools.partial(
    nvtx.annotate, domain=CUDF_POLARS_NVTX_DOMAIN
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import cudf_polars.containers
    from cudf_polars.dsl import ir
    from cudf_polars.dsl.ir import IRExecutionContext


IRType = TypeVar("IRType", bound="ir.IR")


def log_do_evaluate[IRType: ir.IR, **P](
    func: Callable[Concatenate[type[IRType], P], cudf_polars.containers.DataFrame],
) -> Callable[Concatenate[type[IRType], P], cudf_polars.containers.DataFrame]:
    """Emit a Quent lifecycle around an ``IR.do_evaluate`` method."""

    @functools.wraps(func)
    def wrapper(
        cls: type[IRType],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> (
        cudf_polars.containers.DataFrame
    ):  # pragma: no cover; requires cudf_polars_quent
        ir_execution_context: IRExecutionContext = kwargs["context"]  # type: ignore[assignment]
        if (
            quent_state := ir_execution_context.actor_bound_quent_ir_execution_state
        ) is None:
            return func(cls, *args, **kwargs)

        import cudf_polars_quent as _quent

        # By convention, all non-dataframe arguments (non-child) come first.
        # Anything remaining is a dataframe, except for 'context' kwarg.
        frames: list[cudf_polars.containers.DataFrame] = (
            list(args) + [value for key, value in kwargs.items() if key != "context"]
        )[cls._n_non_child_args :]  # type: ignore[assignment]
        quent_evaluate_id = _quent.now_v7()
        runtime = quent_state.query_worker_state.runtime
        runtime.emit_evaluate_begin(
            cls,
            quent_evaluate_id,
            (
                f"{cls.__name__}-{quent_state.operator_id.hex[:8]}-"
                f"{quent_evaluate_id.hex[:8]}"
            ),
            quent_state,
            frames,
        )
        if quent_state.scan_task_node_id is not None:
            kwargs["context"] = dataclasses.replace(
                ir_execution_context,
                actor_bound_quent_ir_execution_state=dataclasses.replace(
                    quent_state,
                    scan_task_node_id=None,
                    scan_task_node_type=None,
                    io_bytes=None,
                ),
            )
        try:
            result = func(cls, *args, **kwargs)
        except BaseException as error:
            runtime.emit_evaluate_end(quent_evaluate_id, None, error)
            raise
        else:
            runtime.emit_evaluate_end(quent_evaluate_id, result, None)
            return result

    return wrapper
