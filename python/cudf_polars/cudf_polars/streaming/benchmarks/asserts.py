# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assertions for validating the results cudf-polars benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import polars.testing

if TYPE_CHECKING:
    from typing import Any


class ValidationError(AssertionError):
    """
    Exception raised when validation fails.

    Parameters
    ----------
    message : str
        The message to display when the validation fails.
    details : dict[str, Any] | None, optional
        Additional details about the validation failure.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details


def assert_tpch_result_equal(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    check_row_order: bool = True,
    check_column_order: bool = True,
    check_dtypes: bool = True,
    check_exact: bool = True,
    rel_tol: float = 1e-05,
    abs_tol: float = 1e-08,
    categorical_as_str: bool = False,
    sort_by: list[tuple[str, bool]],
    nulls_last: bool = True,
    sort_keys: list[tuple[pl.Expr, bool]] | None = None,
    limit: int | None = None,
) -> None:
    """
    Validate the computed result against the expected answer.

    Parameters
    ----------
    left : pl.DataFrame
        The computed result to validate.
    right : pl.DataFrame
        The expected answer to validate against.
    check_row_order, check_column_order, check_dtypes, check_exact, categorical_as_str, rel_tol, abs_tol
        Same meaning as in polars.
    sort_by : list[tuple[str, bool]]
        The columns to sort by, and the sort order. This *must* be the same
        as the ``sort_by`` and ``descending`` required by the query
    nulls_last : bool, optional
        Whether NULLs should be placed last when checking sortedness.
        Must match the NULL placement used by the query's ``ORDER BY``.
        DuckDB defaults to NULLS LAST for ``ASC`` and NULLS FIRST for
        ``DESC``; some TPC-DS queries override this with explicit
        ``NULLS FIRST`` on all columns, which requires ``nulls_last=False``.
        Defaults to ``True`` (NULLS LAST), matching DuckDB's ``ASC`` default.
    sort_keys : list[tuple[pl.Expr, bool]] | None, optional
        Polars expressions for the sortedness check. Use this when the query
        sorts by a conditional expression that cannot be represented as a plain
        column name in ``sort_by`` (e.g. ROLLUP queries that sort by
        ``CASE WHEN lochierarchy = 0 THEN i_category END``). When provided,
        these expressions are evaluated and used only for the sortedness check;
        ``sort_by`` still drives the ties/limit boundary logic.
    limit : int | None, optional
        The limit (passed to ``.head``) used in the query, if any. This is
        used to break ties in the ``sort_by`` columns. See notes below.

    Returns
    -------
    validation_result

    Notes
    -----
    This validates that:

    1. The schema (column names and data types) match
    2. The values match, with some special handling
       - approximate comparison (for floating point values)
       - sorting stability / distributed execution

    Consider a set of ``(key, value)`` records like::

       ("a", 1)
       ("b", 1)
       ("c", 1)
       ("d", 1)

    Now suppose we run a query that sorts on ``value``. *Any* ordering of those
    records is as correct as any other, since the ``value`` is the same and they
    query says nothing about the sorting of the other columns.

    To handle this, this function sorts the result and expected dataframes, taking
    care to sort by the ``sort_by`` columns *first* (preserving the semantics of the
    query) and then by the remaining columns.

    After sorting by all the columns, any remaining differences are should be
    real, *unless* the query includes a ``limit`` / ``.head(n)`` component. Consider
    a query that includes a ``.sort_by("value").head(2)`` component. In our example,
    any result that returns exactly two rows is as good as any other.

    To handle this, this comparison function does the value comparison in two
    parts when there's a ``.sort_by(...).head(n)`` component:

    1. For all the values "before" the last value (defined by ``sort_by``), we
       compare the results directly using ``pl.testing.assert_frame_equal``.
    2. For the "ties", we make sure that the lengths of the two dataframes match,
       but we *don't* compare the values since, aside from the columns in ``sort_by``,
       the values may differ, and that's OK.

    Sorting by float columns introduces some additional fuzziness we need to account for.
    Here are a few examples of ``(key, value)`` records that ought to be considered equal,
    with an abs_tol of 0.01:

    First, a ``sort(by=["value"])``, but for some reason the ``value`` associated
    with the keys "a" and "b" are different, but within the tolerance::

       result: [(a, 0.99), (b, 1.00), (c, 1.01)]
       expected: [(b, 0.99), (a, 1.00), (c, 1.01)]

    Consider the first row of ``result``: ``(a, 0.99)``. We'd consider it equal
    to any row in ``expected`` where the ``value`` is within 0.01 of 0.99 (and
    where the keys match, i.e. ``key=="a"``). Searching through ``expected``, we
    see a match in the second row: ``(a, 0.99)``. Repeating that process for
    the other rows shows that the two ought to be considered equal.

    Here's an example where they are different::

       result: [(a, 0.99), (b, 1.00), (c, 1.01)]
       expected: [(c, 0.99), (b, 1.00), (a, 1.01)]

    Now when we consider matches for the first row of ``result`` we see that
    the only rows from ``expected`` that are within tolerance are the second and third rows::

       candidates: [(b, 1.00), (a, 1.01)]

    None of these rows could be considered equal to ``(a, 0.99)``; the first
    has the wrong key, and the second's value is outside the tolerance.
    """
    detail: dict[str, Any]

    polars_kwargs: dict[str, bool | float] = {
        "check_row_order": check_row_order,
        "check_column_order": check_column_order,
        "check_dtypes": check_dtypes,
        "check_exact": check_exact,
        "categorical_as_str": categorical_as_str,
    }

    tol_kwargs = {"rel_tol": rel_tol, "abs_tol": abs_tol}
    polars_kwargs.update(tol_kwargs)

    if left.columns != right.columns:
        extra = set(left.columns) - set(right.columns)
        missing = set(right.columns) - set(left.columns)
        detail = {
            "type": "column_names_mismatch",
            "expected_columns": right.columns,
            "result_columns": left.columns,
            "mismatched_columns": {
                "extra": sorted(extra),
                "missing": sorted(missing),
            },
        }
        raise ValidationError(message="Column names mismatch", details=detail)

    # Then, check the schema
    if left.schema != right.schema:
        detail = {
            "type": "schema_mismatch",
            "expected_schema": {k: str(v) for k, v in right.schema.items()},
            "result_schema": {k: str(v) for k, v in left.schema.items()},
            "mismatched_columns": [
                {
                    "name": col,
                    "expected_type": str(right.schema[col]),
                    "result_type": str(left.schema[col]),
                }
                for col in left.columns
                if left.schema[col] != right.schema[col]
            ],
        }
        raise ValidationError(message="Schema mismatch", details=detail)

    # For reasons... the polars / cudf-polars Decimal implementation differs
    # slightly from the DuckDB implementation, in ways that can result in *small*
    # but *real* differences in the results (off by 1%).
    float_casts = [
        pl.col(col).cast(pl.Float64())
        for col in left.columns
        if left.schema[col].is_decimal()
    ]
    right = right.with_columns(*float_casts)
    left = left.with_columns(*float_casts)

    non_float_columns = [
        col for col in left.columns if left.schema[col] not in (pl.Float32, pl.Float64)
    ]
    float_columns = [
        col for col in left.columns if left.schema[col] in (pl.Float32, pl.Float64)
    ]
    grouped_sort_columns = [*non_float_columns, *float_columns]

    def sort_for_comparison(df: pl.DataFrame) -> pl.DataFrame:
        # We know that each dataframe is sorted on `sort_by` according to itself.
        # Now we have some freedom to reorder the rows. We'll use this freedom to avoid
        # any kind of fuzziness from sorting on floating-point columns.
        #
        # As long as we sort by the non-float columns *first*, we'll avoid any
        # false positives / false negatives from comparing two tables that have the
        # same values but happen to be in a different order. Sorting by floating-point
        # columns *last* ensures that records that are close to each other appear in
        # (roughly) the same order, such that polar's approximate equality checks
        # will allow them to be considered equal (or not, if the aren't actually close).
        #
        # Sort keys are intersected with df.schema so callers can pre-project
        # the frame (e.g. to compare only `sort_by` columns) without payload
        # columns influencing the order.
        local_sort_columns = [c for c in grouped_sort_columns if c in df.schema]
        return (
            df.sort(by=local_sort_columns, nulls_last=nulls_last)
            if local_sort_columns
            else df
        )

    if sort_by:
        by, descending = list(zip(*sort_by, strict=True))

        # First, validate the sortedness. We can do this independently for each dataframe.
        # And we don't really need to worry about floating-point columns here.
        # If sort_keys is provided, use those expressions for the sortedness check
        # (e.g. for ROLLUP queries with conditional sort expressions like
        # CASE WHEN lochierarchy = 0 THEN i_category END). Otherwise fall back to sort_by.
        if sort_keys is not None:
            exprs, check_desc = zip(*sort_keys, strict=True)
            check_cols = [f"_sk{i}" for i in range(len(exprs))]
            check_select = [
                e.alias(col) for e, col in zip(exprs, check_cols, strict=True)
            ]
            check_msg = "sort_keys expressions"
        else:
            check_cols = list(by)
            check_desc = descending
            check_msg = "sort_by columns"

        for side, df in [("left", left), ("right", right)]:
            try:
                tmp = df.select(check_select) if sort_keys is not None else df
                polars.testing.assert_frame_equal(
                    tmp,
                    tmp.sort(
                        by=check_cols,
                        descending=check_desc,
                        maintain_order=True,
                        nulls_last=nulls_last,
                    ),
                )
            except AssertionError as e:
                raise ValidationError(
                    message=f"{side} dataframe is not sorted by {check_msg}",
                    details={"error": str(e)},
                ) from e

        left_sorted = sort_for_comparison(left)
        right_sorted = sort_for_comparison(right)

        if limit is None or left.is_empty():
            try:
                polars.testing.assert_frame_equal(
                    left_sorted,
                    right_sorted,
                    **polars_kwargs,  # type: ignore[arg-type]
                )
            except AssertionError as e:
                raise ValidationError(
                    message="Result mismatch", details={"error": str(e)}
                ) from e

        else:
            # Handle the .sort_by(...).head(n) case; First, split the data into two parts
            # "before" and "ties"
            # Problem: suppose we're splitting on a float column: small, floating-point precision
            # differences, which we'd ignore in assert_frame_equal, might cause us to
            # split the two dataframes at different, but not meaningfully different, points.
            # Suppose you have:
            #
            # result  : [a, b, c, d, d+epsilon]
            # expected: [a, b, c, d-epsilon, d]
            #
            # For epsilon less than our tolerance, we'd want to consider this valid.
            # Meaning that the split point should be `d+epsilon` and the
            # partitions should be
            # result:   [ [a, b, c], [d, d + epsilon] ]
            # expected: [ [a, b, c], [d - epsilon, d] ]

            (split_at,) = (
                left.select(by)
                .sort(by=by, descending=descending, nulls_last=nulls_last)
                .tail(1)
                .to_dicts()
            )
            # Note that we multiply abs_tol by 2; In our example above, our split point will
            # be d + epsilon; but we want to consider d - epsilon tied to the "real" split point
            # of 'd' as well.

            filter_exprs = []
            for (col, val), desc in zip(split_at.items(), descending, strict=True):
                if isinstance(val, float):
                    filter_exprs.append(
                        pl.col(col).lt(val - 2 * abs_tol)
                        | pl.col(col).gt(val + 2 * abs_tol)
                    )
                elif val is None:
                    filter_exprs.append(
                        pl.col(col).is_not_null() if nulls_last else pl.lit(value=False)
                    )
                else:
                    if desc:
                        # then "before" means "greater than"
                        op = pl.col(col).gt
                    else:
                        op = pl.col(col).lt
                    filter_exprs.append(op(val))

            expr = pl.Expr.or_(*filter_exprs)

            result_first = left.filter(expr)
            expected_first = right.filter(expr)
            result_ties = left.filter(~expr)
            expected_ties = right.filter(~expr)

            try:
                polars.testing.assert_frame_equal(
                    sort_for_comparison(result_first),
                    sort_for_comparison(expected_first),
                    **polars_kwargs,  # type: ignore[arg-type]
                )
            except AssertionError as e:
                raise ValidationError(
                    message="Result mismatch in non-ties part",
                    details={"error": str(e)},
                ) from e

            # We already know that the lengths match (we've validated that the
            # *total* lengths match and the non-ties lengths match, so this rump
            # must match too.). We already know that the schema matches.
            # We *do* need to validate that that `split_at`, computed just
            # on `left` above, actually matches. Because it's a potentially
            # a floating point value, we'll use approximate comparison.

            try:
                polars.testing.assert_frame_equal(
                    sort_for_comparison(result_ties.select(by)),
                    sort_for_comparison(expected_ties.select(by)),
                    **polars_kwargs,  # type: ignore[arg-type]
                )
            except AssertionError as e:
                raise ValidationError(
                    message="Result mismatch in ties part",
                    details={"error": str(e)},
                ) from e

    else:
        left_sorted = sort_for_comparison(left)
        right_sorted = sort_for_comparison(right)

        # no sort_by, compare after grouped sort to ignore nondeterministic row order.
        try:
            polars.testing.assert_frame_equal(
                left_sorted,
                right_sorted,
                **polars_kwargs,  # type: ignore[arg-type]
            )
        except AssertionError as e:
            raise ValidationError(
                message="Result mismatch", details={"error": str(e)}
            ) from e
    return None
