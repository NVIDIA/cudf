# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document StructFunction to remove noqa
# ruff: noqa: D101
"""Struct DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from io import StringIO
from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl  # noqa: TC002 (used at runtime for pl.Struct, pl.Series etc.)

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from typing import Self

    from polars import polars  # type: ignore[attr-defined]

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["StructFunction"]


class StructFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `StructFunction`."""

        FieldByName = auto()
        RenameFields = auto()
        PrefixFields = auto()
        SuffixFields = auto()
        JsonEncode = auto()
        WithFields = auto()  # TODO: https://github.com/rapidsai/cudf/issues/19284
        MapFieldNames = auto()  # TODO: https://github.com/rapidsai/cudf/issues/19285
        FieldByIndex = auto()
        MultipleFields = (
            auto()
        )  # https://github.com/pola-rs/polars/pull/23022#issuecomment-2933910958

        @classmethod
        def from_polars(cls, obj: polars._expr_nodes.StructFunction) -> Self:
            """Convert from polars' `StructFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "StructFunction":
                raise ValueError("StructFunction required")
            return getattr(cls, name)

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    _supported_ops: ClassVar[set[Name]] = {
        Name.FieldByName,
        Name.RenameFields,
        Name.PrefixFields,
        Name.SuffixFields,
        Name.JsonEncode,
    }

    def __init__(
        self,
        dtype: DataType,
        name: StructFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = True
        if self.name not in self._supported_ops:
            raise NotImplementedError(
                f"Struct function {self.name}"
            )  # pragma: no cover

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = [child.evaluate(df, context=context) for child in self.children]
        (column,) = columns
        # Type checker doesn't know polars only calls StructFunction with struct types
        if self.name == StructFunction.Name.FieldByName:
            field_index = next(
                (
                    i
                    for i, field in enumerate(
                        cast("pl.Struct", self.children[0].dtype.polars_type).fields
                    )
                    if field.name == self.options[0]
                ),
                None,
            )
            assert field_index is not None
            return Column(
                column.obj.children()[field_index],
                dtype=self.dtype,
            )
        elif self.name == StructFunction.Name.JsonEncode:
            # Once https://github.com/rapidsai/cudf/issues/19338 is implemented,
            # we can use do this conversion on host.
            if column.size == 0:
                # write_json emits no lines for an empty input, which makes
                # from_iterable_of_py unable to infer a dtype. Skip the
                # round-trip and return a typed empty column.
                return Column(
                    make_empty_column(self.dtype, df.stream),
                    dtype=self.dtype,
                )
            buff = StringIO()
            target = plc.io.SinkInfo([buff])
            table = plc.Table(column.obj.children())
            metadata = plc.io.TableWithMetadata(
                table,
                [
                    (field.name, [])
                    for field in cast(
                        "pl.Struct", self.children[0].dtype.polars_type
                    ).fields
                ],
            )
            options = (
                plc.io.json.JsonWriterOptions.builder(target, table)
                .lines(val=True)
                .na_rep("null")
                .include_nulls(val=True)
                .metadata(metadata)
                .utf8_escaped(val=False)
                .build()
            )
            plc.io.json.write_json(options, stream=df.stream)
            return Column(
                plc.Column.from_iterable_of_py(
                    buff.getvalue().split(), stream=df.stream
                ),
                dtype=self.dtype,
            )
        elif self.name in {
            StructFunction.Name.RenameFields,
            StructFunction.Name.PrefixFields,
            StructFunction.Name.SuffixFields,
        }:
            return column
        else:
            raise NotImplementedError(
                f"Struct function {self.name}"
            )  # pragma: no cover
