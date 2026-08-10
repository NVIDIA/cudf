# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document StringFunction to remove noqa
# ruff: noqa: D101
"""DSL nodes for string operations."""

from __future__ import annotations

import functools
import re
from datetime import datetime
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar, cast

from polars import Struct as pl_Struct  # noqa: TC002 (used in runtime cast())
from polars.exceptions import InvalidOperationError

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal, LiteralColumn
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.utils.dtypes import make_empty_column
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_136,
    POLARS_VERSION_LT_138,
)

if TYPE_CHECKING:
    from typing import Self

    from polars import polars  # type: ignore[attr-defined]

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["StringFunction"]

JsonDecodeType = list[tuple[str, plc.DataType, "JsonDecodeType"]]


def _dtypes_for_json_decode(dtype: DataType) -> JsonDecodeType:
    """Get the dtypes for json decode."""
    # Type checker doesn't narrow polars_type through dtype.id() check
    if dtype.id() == plc.TypeId.STRUCT:
        return [
            (field.name, child.plc_type, _dtypes_for_json_decode(child))
            for field, child in zip(
                cast("pl_Struct", dtype.polars_type).fields,
                dtype.children,
                strict=True,
            )
        ]
    else:
        return []


class StringFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `StringFunction`."""

        Base64Decode = auto()
        Base64Encode = auto()
        ConcatHorizontal = auto()
        ConcatVertical = auto()
        Contains = auto()
        ContainsAny = auto()
        CountMatches = auto()
        EndsWith = auto()
        EscapeRegex = auto()
        Extract = auto()
        ExtractAll = auto()
        ExtractGroups = auto()
        Find = auto()
        Head = auto()
        HexDecode = auto()
        HexEncode = auto()
        JsonDecode = auto()
        JsonPathMatch = auto()
        LenBytes = auto()
        LenChars = auto()
        Lowercase = auto()
        Normalize = auto()
        PadEnd = auto()
        PadStart = auto()
        Replace = auto()
        ReplaceMany = auto()
        Reverse = auto()
        Slice = auto()
        Split = auto()
        SplitExact = auto()
        SplitN = auto()
        SplitRegex = auto()
        StartsWith = auto()
        StripChars = auto()
        StripCharsEnd = auto()
        StripCharsStart = auto()
        StripPrefix = auto()
        StripSuffix = auto()
        Strptime = auto()
        Tail = auto()
        Titlecase = auto()
        ToDecimal = auto()
        ToInteger = auto()
        Uppercase = auto()
        ZFill = auto()

        @classmethod
        def from_polars(cls, obj: polars._expr_nodes.StringFunction) -> Self:
            """Convert from polars' `StringFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "StringFunction":
                raise ValueError("StringFunction required")
            return getattr(cls, name)

    _valid_ops: ClassVar[set[Name]] = {
        Name.ConcatHorizontal,
        Name.ConcatVertical,
        Name.ContainsAny,
        Name.Contains,
        Name.CountMatches,
        Name.EndsWith,
        Name.EscapeRegex,
        Name.Extract,
        Name.ExtractGroups,
        Name.Find,
        Name.Head,
        Name.JsonDecode,
        Name.JsonPathMatch,
        Name.LenBytes,
        Name.LenChars,
        Name.Lowercase,
        Name.PadEnd,
        Name.PadStart,
        Name.Replace,
        Name.ReplaceMany,
        Name.Slice,
        Name.SplitN,
        Name.SplitExact,
        Name.SplitRegex,
        Name.Strptime,
        Name.StartsWith,
        Name.StripChars,
        Name.StripCharsStart,
        Name.StripCharsEnd,
        Name.StripPrefix,
        Name.StripSuffix,
        Name.ToInteger,
        Name.Uppercase,
        Name.Reverse,
        Name.Tail,
        Name.Titlecase,
        Name.ZFill,
    }
    # Regex meta characters escaped by ``str.escape_regex`` (matching polars'
    # ``regex_syntax::escape``). Each matched character is prefixed with a
    # backslash via a back-reference replacement template.
    _ESCAPE_REGEX_PATTERN: ClassVar[str] = r"([#$&()*+\-.?\[\\\]\^{|}~])"
    _ESCAPE_REGEX_REPLACEMENT: ClassVar[str] = r"\\1"
    __slots__ = ("_regex_program", "name", "options")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self,
        dtype: DataType,
        name: StringFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = self.name != StringFunction.Name.ConcatVertical
        self._validate_input()

    def _validate_input(self) -> None:
        if self.name not in self._valid_ops:
            raise NotImplementedError(f"String function {self.name!r}")
        if self.name is StringFunction.Name.CountMatches:
            (literal,) = self.options
            if literal:
                raise NotImplementedError(
                    f"{literal=} is not supported for count_matches"
                )
            literal_expr = self.children[1]
            assert isinstance(literal_expr, Literal)
            pattern = literal_expr.value
            self._regex_program = self._create_regex_program(pattern)
        elif self.name is StringFunction.Name.Contains:
            literal, strict = self.options
            if not literal:
                if not strict:
                    raise NotImplementedError(
                        f"{strict=} is not supported for regex contains"
                    )
                if not isinstance(self.children[1], Literal):
                    raise NotImplementedError(
                        "Regex contains only supports a scalar pattern"
                    )
                pattern = self.children[1].value
                self._regex_program = self._create_regex_program(pattern)
        elif self.name is StringFunction.Name.EscapeRegex:
            self._regex_program = self._create_regex_program(self._ESCAPE_REGEX_PATTERN)
        elif self.name is StringFunction.Name.Extract:
            (group_index,) = self.options
            if group_index == 0:
                raise NotImplementedError(f"{group_index=} is not supported")
            literal_expr = self.children[1]
            assert isinstance(literal_expr, Literal)
            pattern = literal_expr.value
            self._regex_program = self._create_regex_program(pattern)
        elif self.name is StringFunction.Name.ExtractGroups:
            (_, pattern) = self.options
            self._regex_program = self._create_regex_program(pattern)
        elif self.name is StringFunction.Name.Find:
            literal, strict = self.options
            if not literal:
                if not strict:
                    raise NotImplementedError(
                        f"{strict=} is not supported for regex contains"
                    )
                if not isinstance(self.children[1], Literal):
                    raise NotImplementedError(
                        "Regex contains only supports a scalar pattern"
                    )
                pattern = self.children[1].value
                self._regex_program = self._create_regex_program(pattern)
        elif self.name is StringFunction.Name.Replace:
            _, literal = self.options
            if not literal:
                raise NotImplementedError("literal=False is not supported for replace")
            if not all(isinstance(expr, Literal) for expr in self.children[1:]):
                raise NotImplementedError("replace only supports scalar target")
            target = self.children[1]
            # Above, we raise NotImplementedError if the target is not a Literal,
            # so we can safely access .value here.
            if target.value == "":  # type: ignore[attr-defined]
                raise NotImplementedError(
                    "libcudf replace does not support empty strings"
                )
        elif self.name is StringFunction.Name.ReplaceMany:
            if POLARS_VERSION_LT_136:
                (ascii_case_insensitive,) = self.options  # pragma: no cover
            else:
                ascii_case_insensitive, leftmost = self.options
                if leftmost:
                    raise NotImplementedError(
                        "leftmost=True not implemented for replace_many"
                    )
            if ascii_case_insensitive:
                raise NotImplementedError(
                    "ascii_case_insensitive not implemented for replace_many"
                )
            if not all(
                isinstance(expr, (LiteralColumn, Literal)) for expr in self.children[1:]
            ):
                raise NotImplementedError("replace_many only supports literal inputs")
            target = self.children[1]
            # Above, we raise NotImplementedError if the target is not a Literal,
            # so we can safely access .value here.
            if (isinstance(target, Literal) and target.value == "") or (
                isinstance(target, LiteralColumn) and (target.value == "").any()
            ):
                raise NotImplementedError(
                    "libcudf replace_many is implemented differently from polars "
                    "for empty strings"
                )
        elif self.name is StringFunction.Name.Slice:
            if not all(isinstance(child, Literal) for child in self.children[1:]):
                raise NotImplementedError(
                    "Slice only supports literal start and stop values"
                )
        elif self.name is StringFunction.Name.SplitExact:
            (_, inclusive) = self.options
            if inclusive:
                raise NotImplementedError(f"{inclusive=} is not supported for split")
        elif not POLARS_VERSION_LT_138 and self.name is StringFunction.Name.SplitRegex:
            # See https://github.com/pola-rs/polars/pull/26060
            # SplitRegex introduced in polars>=1.38, but we don't support it yet
            raise NotImplementedError(
                "String split with regex (literal=False) is not supported."
            )
        elif self.name is StringFunction.Name.Strptime:
            format, strict, exact, cache = self.options
            if not format and not strict:
                raise NotImplementedError("format inference requires strict checking")
            if cache:
                raise NotImplementedError("Strptime cache is a CPU feature")
            if not exact:
                raise NotImplementedError("Strptime does not support exact=False")
        elif self.name in {
            StringFunction.Name.StripChars,
            StringFunction.Name.StripCharsStart,
            StringFunction.Name.StripCharsEnd,
        }:
            if not isinstance(self.children[1], Literal):
                raise NotImplementedError(
                    "strip operations only support scalar patterns"
                )
        elif self.name is StringFunction.Name.ToInteger:
            base = self.children[1]
            if not isinstance(base, Literal) or base.value != 10:
                raise NotImplementedError(
                    "str.to_integer only supports base 10 on the GPU engine"
                )

    @staticmethod
    def _create_regex_program(
        pattern: str,
        flags: plc.strings.regex_flags.RegexFlags = plc.strings.regex_flags.RegexFlags.DEFAULT,
    ) -> plc.strings.regex_program.RegexProgram:
        if pattern == "":
            raise NotImplementedError("Empty regex pattern is not yet supported")
        try:
            return plc.strings.regex_program.RegexProgram.create(
                pattern,
                flags=flags,
            )
        except RuntimeError as e:
            raise NotImplementedError(
                f"Unsupported regex {pattern} for GPU engine."
            ) from e

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is StringFunction.Name.ConcatHorizontal:
            columns = [
                Column(
                    child.evaluate(df, context=context).obj, dtype=child.dtype
                ).astype(self.dtype, stream=df.stream)
                for child in self.children
            ]
            if len(columns) == 1:
                return columns[0]

            non_unit_sizes = [c.size for c in columns if c.size != 1]
            broadcasted = broadcast(
                *columns,
                target_length=max(non_unit_sizes) if non_unit_sizes else None,
                stream=df.stream,
            )

            delimiter, ignore_nulls = self.options

            return Column(
                plc.strings.combine.concatenate(
                    plc.Table([col.obj for col in broadcasted]),
                    plc.Scalar.from_py(
                        delimiter, self.dtype.plc_type, stream=df.stream
                    ),
                    None
                    if ignore_nulls
                    else plc.Scalar.from_py(
                        None, self.dtype.plc_type, stream=df.stream
                    ),
                    None,
                    plc.strings.combine.SeparatorOnNulls.NO,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ConcatVertical:
            (child,) = self.children
            column = child.evaluate(df, context=context).astype(
                self.dtype, stream=df.stream
            )
            delimiter, ignore_nulls = self.options
            if column.null_count > 0 and not ignore_nulls:
                return Column(
                    plc.Column.all_null_like(column.obj, 1, stream=df.stream),
                    dtype=self.dtype,
                )
            return Column(
                plc.strings.combine.join_strings(
                    column.obj,
                    plc.Scalar.from_py(
                        delimiter, self.dtype.plc_type, stream=df.stream
                    ),
                    plc.Scalar.from_py(None, self.dtype.plc_type, stream=df.stream),
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ZFill:
            # TODO: expensive validation
            # polars pads based on bytes, libcudf by visual width
            # only pass chars if the visual width matches the byte length
            column = self.children[0].evaluate(df, context=context)
            col_len_bytes = plc.strings.attributes.count_bytes(
                column.obj, stream=df.stream
            )
            col_len_chars = plc.strings.attributes.count_characters(
                column.obj, stream=df.stream
            )
            equal = plc.binaryop.binary_operation(
                col_len_bytes,
                col_len_chars,
                plc.binaryop.BinaryOperator.NULL_EQUALS,
                plc.DataType(plc.TypeId.BOOL8),
                stream=df.stream,
            )
            if not plc.reduce.reduce(
                equal,
                plc.aggregation.all(),
                plc.DataType(plc.TypeId.BOOL8),
                stream=df.stream,
            ).to_py(stream=df.stream):
                raise InvalidOperationError(
                    "zfill only supports ascii strings with no unicode characters"
                )
            if isinstance(self.children[1], Literal):
                width = self.children[1]
                assert isinstance(width, Literal)
                if width.value is None:
                    return Column(
                        plc.Column.from_scalar(
                            plc.Scalar.from_py(
                                None, self.dtype.plc_type, stream=df.stream
                            ),
                            column.size,
                            stream=df.stream,
                        ),
                        self.dtype,
                    )
                return Column(
                    plc.strings.padding.zfill(
                        column.obj, width.value, stream=df.stream
                    ),
                    self.dtype,
                )
            else:
                col_width = self.children[1].evaluate(df, context=context)
                assert isinstance(col_width, Column)
                widths = col_width.obj
                return Column(
                    plc.strings.padding.zfill_by_widths(
                        column.obj, widths, stream=df.stream
                    ),
                    self.dtype,
                )

        elif self.name is StringFunction.Name.Contains:
            child, arg = self.children
            column = child.evaluate(df, context=context)

            literal, _ = self.options
            if literal:
                pat = arg.evaluate(df, context=context)
                pattern = (
                    pat.obj_scalar(stream=df.stream)
                    if pat.is_scalar and pat.size != column.size
                    else pat.obj
                )
                return Column(
                    plc.strings.find.contains(column.obj, pattern, stream=df.stream),
                    dtype=self.dtype,
                )
            else:
                return Column(
                    plc.strings.contains.contains_re(
                        column.obj, self._regex_program, stream=df.stream
                    ),
                    dtype=self.dtype,
                )
        elif self.name is StringFunction.Name.ContainsAny:
            (ascii_case_insensitive,) = self.options
            child, arg = self.children
            plc_column = child.evaluate(df, context=context).obj
            plc_targets = arg.evaluate(df, context=context).obj
            if plc_column.size() == 0:
                # contains_multiple launches a kernel with grid_1d sized
                # against the input rows, which asserts num_blocks > 0.
                # Skip the kernel call on empty input.
                return Column(
                    make_empty_column(self.dtype, df.stream),
                    dtype=self.dtype,
                )
            if ascii_case_insensitive:
                plc_column = plc.strings.case.to_lower(plc_column, stream=df.stream)
                plc_targets = plc.strings.case.to_lower(plc_targets, stream=df.stream)
            contains = plc.strings.find_multiple.contains_multiple(
                plc_column,
                plc_targets,
                stream=df.stream,
            )
            binary_or = functools.partial(
                plc.binaryop.binary_operation,
                op=plc.binaryop.BinaryOperator.BITWISE_OR,
                output_type=self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(
                functools.reduce(binary_or, contains.columns()),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.CountMatches:
            (child, _) = self.children
            plc_column = child.evaluate(df, context=context).obj
            return Column(
                plc.unary.cast(
                    plc.strings.contains.count_re(
                        plc_column, self._regex_program, stream=df.stream
                    ),
                    self.dtype.plc_type,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Extract:
            (group_index,) = self.options
            plc_column = self.children[0].evaluate(df, context=context).obj
            return Column(
                plc.strings.extract.extract_single(
                    plc_column, self._regex_program, group_index - 1, stream=df.stream
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ExtractGroups:
            plc_column = self.children[0].evaluate(df, context=context).obj
            plc_table = plc.strings.extract.extract(
                plc_column, self._regex_program, stream=df.stream
            )
            return Column(
                plc.Column.struct_from_children(plc_table.columns()),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Find:
            literal, _ = self.options
            (child, expr) = self.children
            plc_column = child.evaluate(df, context=context).obj
            if literal:
                assert isinstance(expr, Literal)
                plc_column = plc.strings.find.find(
                    plc_column,
                    plc.Scalar.from_py(
                        expr.value, expr.dtype.plc_type, stream=df.stream
                    ),
                    stream=df.stream,
                )
            else:
                plc_column = plc.strings.findall.find_re(
                    plc_column, self._regex_program, stream=df.stream
                )
            # Polars returns None for not found, libcudf returns -1
            new_mask, null_count = plc.transform.bools_to_mask(
                plc.binaryop.binary_operation(
                    plc_column,
                    plc.Scalar.from_py(-1, plc_column.type(), stream=df.stream),
                    plc.binaryop.BinaryOperator.NOT_EQUAL,
                    plc.DataType(plc.TypeId.BOOL8),
                    stream=df.stream,
                ),
                stream=df.stream,
            )
            plc_column = plc.unary.cast(
                plc_column.with_mask(new_mask, null_count),
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(plc_column, dtype=self.dtype)
        elif self.name is StringFunction.Name.JsonDecode:
            plc_column = self.children[0].evaluate(df, context=context).obj
            if plc_column.size() == 0:
                # read_json_from_string_column raises
                # "Generated token count exceeds the expected token count"
                # on empty input. Return a typed empty struct column directly.
                return Column(
                    make_empty_column(self.dtype, df.stream),
                    dtype=self.dtype,
                )
            plc_table_with_metadata = plc.io.json.read_json_from_string_column(
                plc_column,
                plc.Scalar.from_py("\n", stream=df.stream),
                plc.Scalar.from_py("NULL", stream=df.stream),
                _dtypes_for_json_decode(self.dtype),
                stream=df.stream,
            )
            return Column(
                plc.Column.struct_from_children(plc_table_with_metadata.columns),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.JsonPathMatch:
            (child, expr) = self.children
            plc_column = child.evaluate(df, context=context).obj
            assert isinstance(expr, Literal)
            json_path = plc.Scalar.from_py(
                expr.value, expr.dtype.plc_type, stream=df.stream
            )
            return Column(
                plc.json.get_json_object(plc_column, json_path, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ToInteger:
            (strict,) = self.options
            plc_column = self.children[0].evaluate(df, context=context).obj
            parse_ok = plc.strings.convert.convert_integers.is_integer(
                plc_column, self.dtype.plc_type, stream=df.stream
            )
            if parse_ok.null_count() > 0:
                # is_integer marks null inputs as null; treat them as
                # non-parseable so they map to null in the output.
                parse_ok = plc.replace.replace_nulls(
                    parse_ok,
                    plc.Scalar.from_py(
                        False,  # noqa: FBT003
                        plc.DataType(plc.TypeId.BOOL8),
                        stream=df.stream,
                    ),
                    stream=df.stream,
                )
            if strict:
                is_null = plc.unary.is_null(plc_column, stream=df.stream)
                ok_or_null = plc.binaryop.binary_operation(
                    parse_ok,
                    is_null,
                    plc.binaryop.BinaryOperator.LOGICAL_OR,
                    plc.DataType(plc.TypeId.BOOL8),
                    stream=df.stream,
                )
                if not plc.reduce.reduce(
                    ok_or_null,
                    plc.aggregation.all(),
                    plc.DataType(plc.TypeId.BOOL8),
                    stream=df.stream,
                ).to_py(stream=df.stream):
                    raise InvalidOperationError("conversion from `str` failed.")
            result = plc.strings.convert.convert_integers.to_integers(
                plc_column, self.dtype.plc_type, stream=df.stream
            )
            new_mask, null_count = plc.transform.bools_to_mask(
                parse_ok, stream=df.stream
            )
            return Column(
                result.with_mask(new_mask, null_count),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.LenBytes:
            plc_column = self.children[0].evaluate(df, context=context).obj
            return Column(
                plc.unary.cast(
                    plc.strings.attributes.count_bytes(plc_column, stream=df.stream),
                    self.dtype.plc_type,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.LenChars:
            plc_column = self.children[0].evaluate(df, context=context).obj
            return Column(
                plc.unary.cast(
                    plc.strings.attributes.count_characters(
                        plc_column, stream=df.stream
                    ),
                    self.dtype.plc_type,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Slice:
            child, expr_offset, expr_length = self.children
            assert isinstance(expr_offset, Literal)
            assert isinstance(expr_length, Literal)

            column = child.evaluate(df, context=context)
            # libcudf slices via [start,stop).
            # polars slices with offset + length where start == offset
            # stop = start + length. Negative values for start look backward
            # from the last element of the string. If the end index would be
            # below zero, an empty string is returned.
            # Do this maths on the host
            start = expr_offset.value
            length = expr_length.value

            if length == 0:
                stop = start
            else:
                # No length indicates a scan to the end
                # The libcudf equivalent is a null stop
                stop = start + length if length else None
                if length and start < 0 and length >= -start:
                    stop = None
            return Column(
                plc.strings.slice.slice_strings(
                    column.obj,
                    plc.Scalar.from_py(
                        start, plc.DataType(plc.TypeId.INT32), stream=df.stream
                    ),
                    plc.Scalar.from_py(
                        stop, plc.DataType(plc.TypeId.INT32), stream=df.stream
                    ),
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name in {
            StringFunction.Name.SplitExact,
            StringFunction.Name.SplitN,
        }:
            is_split_n = self.name is StringFunction.Name.SplitN
            n = self.options[0]
            child, expr = self.children
            column = child.evaluate(df, context=context)
            if n == 1 and self.name is StringFunction.Name.SplitN:
                plc_column = plc.Column(
                    self.dtype.plc_type,
                    column.obj.size(),
                    None,
                    None,
                    0,
                    column.obj.offset(),
                    [column.obj],
                )
            else:
                assert isinstance(expr, Literal)
                by = plc.Scalar.from_py(
                    expr.value, expr.dtype.plc_type, stream=df.stream
                )
                # See https://github.com/pola-rs/polars/issues/11640
                # for SplitN vs SplitExact edge case behaviors
                max_splits = n if is_split_n else 0
                plc_table = plc.strings.split.split.split(
                    column.obj,
                    by,
                    max_splits - 1,
                    stream=df.stream,
                )
                children = plc_table.release()
                ref_column = children[0]
                if (remainder := n + int(not is_split_n) - len(children)) > 0:
                    # Reach expected number of splits by padding with nulls
                    children.extend(
                        plc.Column.all_null_like(
                            ref_column, ref_column.size(), stream=df.stream
                        )
                        for _ in range(remainder)
                    )
                if not is_split_n:
                    children = children[: n + 1]
                # TODO: Use plc.Column.struct_from_children once it is generalized
                # to handle columns that don't share the same null_mask/null_count
                plc_column = plc.Column(
                    self.dtype.plc_type,
                    ref_column.size(),
                    None,
                    None,
                    0,
                    ref_column.offset(),
                    children,
                )
            return Column(plc_column, dtype=self.dtype)
        elif self.name in {
            StringFunction.Name.StripPrefix,
            StringFunction.Name.StripSuffix,
        }:
            child, expr = self.children
            plc_column = child.evaluate(df, context=context).obj
            assert isinstance(expr, Literal)
            target = plc.Scalar.from_py(
                expr.value, expr.dtype.plc_type, stream=df.stream
            )
            if self.name == StringFunction.Name.StripPrefix:
                find = plc.strings.find.starts_with
                start = len(expr.value)
                end: int | None = None
            else:
                find = plc.strings.find.ends_with
                start = 0
                end = -len(expr.value)

            mask = find(plc_column, target, stream=df.stream)
            sliced = plc.strings.slice.slice_strings(
                plc_column,
                plc.Scalar.from_py(
                    start, plc.DataType(plc.TypeId.INT32), stream=df.stream
                ),
                plc.Scalar.from_py(
                    end, plc.DataType(plc.TypeId.INT32), stream=df.stream
                ),
                stream=df.stream,
            )
            return Column(
                plc.copying.copy_if_else(
                    sliced,
                    plc_column,
                    mask,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name in {
            StringFunction.Name.StripChars,
            StringFunction.Name.StripCharsStart,
            StringFunction.Name.StripCharsEnd,
        }:
            column, chars = (c.evaluate(df, context=context) for c in self.children)
            if self.name is StringFunction.Name.StripCharsStart:
                side = plc.strings.SideType.LEFT
            elif self.name is StringFunction.Name.StripCharsEnd:
                side = plc.strings.SideType.RIGHT
            else:
                side = plc.strings.SideType.BOTH
            return Column(
                plc.strings.strip.strip(
                    column.obj,
                    side,
                    chars.obj_scalar(stream=df.stream),
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )

        elif self.name is StringFunction.Name.Tail:
            column = self.children[0].evaluate(df, context=context)

            assert isinstance(self.children[1], Literal)
            if self.children[1].value is None:
                return Column(
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(None, self.dtype.plc_type, stream=df.stream),
                        column.size,
                        stream=df.stream,
                    ),
                    self.dtype,
                )
            elif self.children[1].value == 0:
                result = plc.Column.from_scalar(
                    plc.Scalar.from_py("", self.dtype.plc_type, stream=df.stream),
                    column.size,
                    stream=df.stream,
                )
                if column.obj.null_mask():
                    result = result.with_mask(
                        column.obj.null_mask(), column.obj.null_count()
                    )
                return Column(result, self.dtype)

            else:
                start = -(self.children[1].value)
                end = 2**31 - 1
                return Column(
                    plc.strings.slice.slice_strings(
                        column.obj,
                        plc.Scalar.from_py(
                            start, plc.DataType(plc.TypeId.INT32), stream=df.stream
                        ),
                        plc.Scalar.from_py(
                            end, plc.DataType(plc.TypeId.INT32), stream=df.stream
                        ),
                        None,
                        stream=df.stream,
                    ),
                    self.dtype,
                )
        elif self.name is StringFunction.Name.Head:
            column = self.children[0].evaluate(df, context=context)

            assert isinstance(self.children[1], Literal)

            end = self.children[1].value
            if end is None:
                return Column(
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(None, self.dtype.plc_type, stream=df.stream),
                        column.size,
                        stream=df.stream,
                    ),
                    self.dtype,
                )
            return Column(
                plc.strings.slice.slice_strings(
                    column.obj,
                    plc.Scalar.from_py(
                        0, plc.DataType(plc.TypeId.INT32), stream=df.stream
                    ),
                    plc.Scalar.from_py(
                        end, plc.DataType(plc.TypeId.INT32), stream=df.stream
                    ),
                    stream=df.stream,
                ),
                self.dtype,
            )

        columns = [child.evaluate(df, context=context) for child in self.children]
        if self.name is StringFunction.Name.Lowercase:
            (column,) = columns
            return Column(
                plc.strings.case.to_lower(column.obj, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Uppercase:
            (column,) = columns
            return Column(
                plc.strings.case.to_upper(column.obj, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.EndsWith:
            column, suffix = columns
            return Column(
                plc.strings.find.ends_with(
                    column.obj,
                    suffix.obj_scalar(stream=df.stream)
                    if column.size != suffix.size and suffix.is_scalar
                    else suffix.obj,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.StartsWith:
            column, prefix = columns
            return Column(
                plc.strings.find.starts_with(
                    column.obj,
                    prefix.obj_scalar(stream=df.stream)
                    if column.size != prefix.size and prefix.is_scalar
                    else prefix.obj,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Strptime:
            # TODO: ignores ambiguous
            format, strict, _, _ = self.options
            col = self.children[0].evaluate(df, context=context)
            plc_col = col.obj
            if plc_col.null_count() == plc_col.size():
                return Column(
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(None, self.dtype.plc_type, stream=df.stream),
                        plc_col.size(),
                        stream=df.stream,
                    ),
                    self.dtype,
                )
            if format is None:
                # Polars begins inference with the first non null value
                if plc_col.null_mask() is not None:
                    boolmask = plc.unary.is_valid(plc_col, stream=df.stream)
                    table = plc.stream_compaction.apply_boolean_mask(
                        plc.Table([plc_col]), boolmask, stream=df.stream
                    )
                    filtered = table.columns()[0]
                    first_valid_data = plc.copying.get_element(
                        filtered, 0, stream=df.stream
                    ).to_py(stream=df.stream)
                else:
                    first_valid_data = plc.copying.get_element(
                        plc_col, 0, stream=df.stream
                    ).to_py(stream=df.stream)

                # See https://github.com/rapidsai/cudf/issues/20202 for we type ignore
                format = _infer_datetime_format(first_valid_data)  # type: ignore[arg-type]
                if not format:
                    raise InvalidOperationError(
                        "Unable to infer datetime format from data"
                    )

            is_timestamps = plc.strings.convert.convert_datetime.is_timestamp(
                plc_col, format, stream=df.stream
            )
            if strict:
                if not plc.reduce.reduce(
                    is_timestamps,
                    plc.aggregation.all(),
                    plc.DataType(plc.TypeId.BOOL8),
                    stream=df.stream,
                ).to_py(stream=df.stream):
                    raise InvalidOperationError("conversion from `str` failed.")
            else:
                not_timestamps = plc.unary.unary_operation(
                    is_timestamps, plc.unary.UnaryOperator.NOT, stream=df.stream
                )
                null = plc.Scalar.from_py(None, plc_col.type(), stream=df.stream)
                plc_col = plc.copying.boolean_mask_scatter(
                    [null], plc.Table([plc_col]), not_timestamps, stream=df.stream
                ).columns()[0]

            return Column(
                plc.strings.convert.convert_datetime.to_timestamps(
                    plc_col, self.dtype.plc_type, format, stream=df.stream
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Replace:
            col_column, col_target, col_repl = columns
            n, _ = self.options
            return Column(
                plc.strings.replace.replace(
                    col_column.obj,
                    col_target.obj_scalar(stream=df.stream),
                    col_repl.obj_scalar(stream=df.stream),
                    maxrepl=n,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ReplaceMany:
            col_column, col_target, col_repl = columns
            return Column(
                plc.strings.replace.replace_multiple(
                    col_column.obj, col_target.obj, col_repl.obj, stream=df.stream
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.PadStart:
            (column, width_col) = columns
            (char,) = self.options
            # TODO: Maybe accept a string scalar in
            # cudf::strings::pad to avoid DtoH transfer
            # See https://github.com/rapidsai/cudf/issues/20202
            width_py = width_col.obj.to_scalar(stream=df.stream).to_py(stream=df.stream)
            assert width_py is not None
            pad_width = int(width_py)

            return Column(
                plc.strings.padding.pad(
                    column.obj,
                    pad_width,
                    plc.strings.SideType.LEFT,
                    char,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.PadEnd:
            (column, width_col) = columns
            (char,) = self.options
            # TODO: Maybe accept a string scalar in
            # cudf::strings::pad to avoid DtoH transfer
            width_py = width_col.obj.to_scalar(stream=df.stream).to_py(stream=df.stream)
            assert width_py is not None
            pad_width = int(width_py)

            return Column(
                plc.strings.padding.pad(
                    column.obj,
                    pad_width,
                    plc.strings.SideType.RIGHT,
                    char,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Reverse:
            (column,) = columns
            return Column(
                plc.strings.reverse.reverse(column.obj, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Titlecase:
            (column,) = columns
            return Column(
                plc.strings.capitalize.title(column.obj, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.EscapeRegex:
            (column,) = columns
            return Column(
                plc.strings.replace_re.replace_with_backrefs(
                    column.obj,
                    self._regex_program,
                    self._ESCAPE_REGEX_REPLACEMENT,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        raise NotImplementedError(
            f"StringFunction {self.name}"
        )  # pragma: no cover; handled by init raising


def _infer_datetime_format(val: str) -> str | None:
    # port of parts of infer.rs and patterns.rs from polars rust
    DATETIME_DMY_RE = re.compile(
        r"""
        ^
        ['"]?
        (\d{1,2})
        [-/\.]
        (?P<month>[01]?\d{1})
        [-/\.]
        (\d{4,})
        (
            [T\ ]
            (\d{1,2})
            :?
            (\d{1,2})
            (
                :?
                (\d{1,2})
                (
                    \.(\d{1,9})
                )?
            )?
        )?
        ['"]?
        $
    """,
        re.VERBOSE,
    )

    DATETIME_YMD_RE = re.compile(
        r"""
        ^
        ['"]?
        (\d{4,})
        [-/\.]
        (?P<month>[01]?\d{1})
        [-/\.]
        (\d{1,2})
        (
            [T\ ]
            (\d{1,2})
            :?
            (\d{1,2})
            (
                :?
                (\d{1,2})
                (
                    \.(\d{1,9})
                )?
            )?
        )?
        ['"]?
        $
    """,
        re.VERBOSE,
    )

    DATETIME_YMDZ_RE = re.compile(
        r"""
        ^
        ['"]?
        (\d{4,})
        [-/\.]
        (?P<month>[01]?\d{1})
        [-/\.]
        (\d{1,2})
        [T\ ]
        (\d{2})
        :?
        (\d{2})
        (
            :?
            (\d{2})
            (
                \.(\d{1,9})
            )?
        )?
        (
            [+-](\d{2})(:?(\d{2}))? | Z
        )
        ['"]?
        $
    """,
        re.VERBOSE,
    )
    PATTERN_FORMATS = {
        "DATETIME_DMY": [
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d.%m.%Y",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%d.%m.%Y %H:%M:%S",
        ],
        "DATETIME_YMD": [
            "%Y/%m/%d",
            "%Y-%m-%d",
            "%Y.%m.%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y.%m.%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ],
        "DATETIME_YMDZ": [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
        ],
    }
    for pattern_name, regex in [
        ("DATETIME_DMY", DATETIME_DMY_RE),
        ("DATETIME_YMD", DATETIME_YMD_RE),
        ("DATETIME_YMDZ", DATETIME_YMDZ_RE),
    ]:
        m = regex.match(val)
        if m:
            month = int(m.group("month"))
            if not (1 <= month <= 12):
                continue
            for fmt in PATTERN_FORMATS[pattern_name]:
                try:
                    datetime.strptime(val, fmt)
                except ValueError:
                    continue
                else:
                    return fmt
    return None
