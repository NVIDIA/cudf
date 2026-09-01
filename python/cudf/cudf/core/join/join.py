# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
from pandas.errors import MergeError

import pylibcudf as plc

import cudf
from cudf.core._internals import sorting
from cudf.core.column import ColumnBase, access_columns
from cudf.core.copy_types import GatherMap
from cudf.core.dtype.validators import (
    is_dtype_obj_numeric,
    is_dtype_obj_string,
)
from cudf.core.join._join_helpers import (
    _coerce_to_tuple,
    _ColumnIndexer,
    _IndexIndexer,
    _match_join_keys,
)
from cudf.options import get_option
from cudf.utils.dtypes import SIZE_TYPE_DTYPE

if TYPE_CHECKING:
    from cudf.core.dataframe import DataFrame
    from cudf.core.index import Index


class Merge:
    @staticmethod
    def _joiner(
        lhs: list[ColumnBase],
        rhs: list[ColumnBase],
        how: str,
    ) -> tuple[ColumnBase, ColumnBase]:
        if how == "outer":
            how = "full"
        if (join_func := getattr(plc.join, f"{how}_join", None)) is None:
            raise ValueError(f"Invalid join type {how}")

        with access_columns(
            *lhs, *rhs, mode="read", scope="internal"
        ) as accessed:
            # Split accessed tuple back into lhs and rhs
            n_lhs = len(lhs)
            lhs = accessed[:n_lhs]  # type: ignore[assignment]
            rhs = accessed[n_lhs:]  # type: ignore[assignment]
            left_rows, right_rows = join_func(
                plc.Table([col.plc_column for col in lhs]),
                plc.Table([col.plc_column for col in rhs]),
                plc.types.NullEquality.EQUAL,
            )
            return (
                ColumnBase.create(left_rows, SIZE_TYPE_DTYPE),
                ColumnBase.create(right_rows, SIZE_TYPE_DTYPE),
            )

    def __init__(
        self,
        lhs,
        rhs,
        *,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        how,
        sort,
        indicator,
        suffixes,
    ):
        """
        Manage the merging of two Frames.

        Parameters
        ----------
        lhs : DataFrame
            The left operand of the merge
        rhs : DataFrame
            The right operand of the merge
        on : string or list like
            A set of key columns in the left and right operands
            elements must be common to both frames
        left_on : string or list like
            A set of key columns in the left operand. Must be
            specified with right_on or right_index concurrently
        right_on : string or list like
            A set of key columns in the right operand. Must be
            specified with left_on or left_index concurrently
        left_index : bool
            Boolean flag indicating the left index column or columns
            are to be used as join keys in order.
        right_index : bool
            Boolean flag indicating the right index column or columns
            are to be used as join keys in order.
        how : string
            The type of join. Possible values are
            'inner', 'outer', 'left', 'leftsemi' and 'leftanti'
        sort : bool
            Boolean flag indicating if the output Frame is to be
            sorted on the output's join keys, in left to right order.
        suffixes : list like
            Left and right suffixes specified together, unpacked into lsuffix
            and rsuffix.
        """
        self._validate_merge_params(
            lhs,
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            suffixes=suffixes,
        )

        self.lhs = lhs.copy(deep=False)
        self.rhs = rhs.copy(deep=False)
        self.how = how
        # Whether the join maps contain unmatched (nullifying) entries for
        # each side; computed in ``perform_merge``. Only nulls *introduced*
        # by the join trigger pandas' int -> float64 upcast (a column may
        # already hold cudf-native nulls before the merge).
        self._left_unmatched = False
        self._right_unmatched = False
        # If the user requests that the result is sorted or we're in
        # pandas-compatible mode we have various obligations on the
        # output order:
        #
        # compat-> | False                    | True
        # sort     |                          |
        # ---------+--------------------------+-------------------------------
        #     False| no obligation            | ordering as per pandas docs(*)
        #     True | sorted lexicographically | sorted lexicographically(*)
        #
        # (*) If two keys are equal, tiebreak is to use input table order.
        #
        # In pandas-compat mode, we have obligations on the order to
        # match pandas (even if sort=False), see
        # pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html.
        # The ordering requirements differ depending on which join
        # type is specified:
        #
        # - left: preserve key order (only keeping left keys)
        # - right: preserve key order (only keeping right keys)
        # - inner: preserve key order (of left keys)
        # - outer: sort keys lexicographically
        # - cross (not supported): preserve key order (of left keys)
        #
        # Moreover, in all cases, whenever there is a tiebreak
        # situation (for sorting or otherwise), the deciding order is
        # "input table order"
        self.sort = sort or (
            get_option("mode.pandas_compatible") and how == "outer"
        )
        self.preserve_key_order = get_option(
            "mode.pandas_compatible"
        ) and how in {
            "inner",
            "outer",
            "left",
            "right",
        }
        self.lsuffix, self.rsuffix = suffixes

        # Record whether the caller passed the ``left_index``/``right_index``
        # boolean flags. This is distinct from ``_using_{left,right}_index``
        # below, which is also True when ``on``/``left_on``/``right_on`` names
        # an index *level*. pandas takes the result index from a frame's index
        # (and coalesces the opposite key into it) only when that frame joined
        # via the ``*_index`` flag; an index level used as a key via ``on=`` is
        # treated like a column and yields a default RangeIndex.
        self._left_index_flag = left_index
        self._right_index_flag = right_index

        # At this point validation guarantees that if on is not None we
        # don't have any other args, so we can apply it directly to left_on and
        # right_on.
        self._using_left_index = bool(left_index)
        left_on = (
            lhs.index._column_names
            if left_index
            else left_on
            if left_on
            else on
        )
        self._using_right_index = bool(right_index)
        right_on = (
            rhs.index._column_names
            if right_index
            else right_on
            if right_on
            else on
        )

        if left_on or right_on:
            self._left_keys = [
                _ColumnIndexer(name=on)
                if not self._using_left_index and on in lhs._data
                else _IndexIndexer(name=on)
                for on in (_coerce_to_tuple(left_on) if left_on else [])
            ]
            self._right_keys = [
                _ColumnIndexer(name=on)
                if not self._using_right_index and on in rhs._data
                else _IndexIndexer(name=on)
                for on in (_coerce_to_tuple(right_on) if right_on else [])
            ]
            if len(self._left_keys) != len(self._right_keys):
                raise ValueError(
                    "Merge operands must have same number of join key columns"
                )
            self._using_left_index = any(
                isinstance(idx, _IndexIndexer) for idx in self._left_keys
            )
            self._using_right_index = any(
                isinstance(idx, _IndexIndexer) for idx in self._right_keys
            )
        else:
            # if `on` is not provided and we're not merging
            # index with column or on both indexes, then use
            # the intersection of columns in both frames, in left-frame
            # column order like pandas (a set here would make the key
            # order -- and hence the sorted output of an outer merge --
            # vary with the process hash seed).
            right_names = set(rhs._column_names)
            on_names = [
                name for name in lhs._column_names if name in right_names
            ]
            self._left_keys = [_ColumnIndexer(name=on) for on in on_names]
            self._right_keys = [_ColumnIndexer(name=on) for on in on_names]
            self._using_left_index = False
            self._using_right_index = False

        self._key_columns_with_same_name = (
            set(_coerce_to_tuple(on))
            if on
            else {
                lkey.name
                for lkey, rkey in zip(
                    self._left_keys, self._right_keys, strict=True
                )
                if lkey.name == rkey.name
                and not (
                    isinstance(lkey, _IndexIndexer)
                    or isinstance(rkey, _IndexIndexer)
                )
            }
        )

    def _gather_maps(self, left_cols, right_cols):
        # Produce gather maps for the join, optionally reordering to
        # match pandas-order in compat mode.
        maps = self._joiner(
            left_cols,
            right_cols,
            how=self.how,
        )
        if not self.preserve_key_order:
            return maps
        # We should only get here if we're in a join on which
        # pandas-compat places some ordering obligation (which
        # precludes a semi-join)
        # We must perform this reordering even if sort=True since the
        # obligation to ensure tiebreaks appear in input table order
        # means that the gather maps must be permuted into an original
        # order.
        assert self.how in {"inner", "outer", "left", "right"}
        # And hence both maps returned from the libcudf join should be
        # non-None.
        assert all(m is not None for m in maps)
        lengths = [len(left_cols[0]), len(right_cols[0])]
        # Only nullify those maps that need it.
        nullify = [
            self.how not in {"inner", "left"},
            self.how not in {"inner", "right"},
        ]
        # To reorder maps so that they are in order of the input
        # tables, we gather from iota on both right and left, and then
        # sort the gather maps with those two columns as key.
        key_order = [
            ColumnBase.from_range(range(n))
            .astype(SIZE_TYPE_DTYPE)
            .take(map_, nullify=null, check_bounds=False)
            for map_, n, null in zip(maps, lengths, nullify, strict=True)
        ]
        if self.how == "right":
            # If how is right, right map is primary sort key.
            key_order = reversed(key_order)
        return [
            ColumnBase.create(col, SIZE_TYPE_DTYPE)
            for col in sorting.sort_by_key(
                maps,
                key_order,
                itertools.repeat(True, times=len(key_order)),
                itertools.repeat("last", times=len(key_order)),
                stable=True,
            )
        ]

    def perform_merge(self) -> DataFrame:
        left_join_cols = []
        right_join_cols = []

        for left_key, right_key in zip(
            self._left_keys, self._right_keys, strict=True
        ):
            lcol = left_key.get(self.lhs)
            rcol = right_key.get(self.rhs)
            if len(lcol) and len(rcol):
                # pandas refuses to merge a numeric key against a string key
                # (a numeric-looking string is NOT silently parsed). Empty
                # keys are inferred as ``empty`` rather than ``string`` by
                # pandas and so are exempt from this check.
                l_num = (
                    is_dtype_obj_numeric(lcol.dtype)
                    and lcol.dtype.kind in "iuf"
                )
                r_num = (
                    is_dtype_obj_numeric(rcol.dtype)
                    and rcol.dtype.kind in "iuf"
                )
                l_str = is_dtype_obj_string(lcol.dtype)
                r_str = is_dtype_obj_string(rcol.dtype)
                if (l_str and r_num) or (r_str and l_num):
                    raise ValueError(
                        f"You are trying to merge on {lcol.dtype} and "
                        f"{rcol.dtype} columns for key '{left_key.name}'. "
                        "If you wish to proceed you should use pd.concat"
                    )
            lcol_casted, rcol_casted = _match_join_keys(lcol, rcol, self.how)
            # The common-typed columns are always used to compute the join
            # maps; the columns written into the output frame may differ.
            left_join_cols.append(lcol_casted)
            right_join_cols.append(rcol_casted)
            # ``_match_join_keys`` returns the keys unchanged (still
            # categorical) when both sides share the same categories, and
            # otherwise decategorizes them -- matching pandas, which keeps the
            # result categorical only when the category sets match.
            output_lcol, output_rcol = lcol_casted, rcol_casted

            # pandas keeps an *empty* key column at its original dtype even when
            # the other (empty) side has a different numeric/object dtype;
            # cudf's common-type cast would otherwise change it. The join maps
            # are unaffected (an empty side yields an empty gather map).
            if (len(lcol) == 0 or len(rcol) == 0) and lcol.dtype != rcol.dtype:
                l_numeric = is_dtype_obj_numeric(lcol.dtype)
                r_numeric = is_dtype_obj_numeric(rcol.dtype)
                l_objlike = is_dtype_obj_string(lcol.dtype) or (
                    isinstance(lcol.dtype, np.dtype) and lcol.dtype == object
                )
                r_objlike = is_dtype_obj_string(rcol.dtype) or (
                    isinstance(rcol.dtype, np.dtype) and rcol.dtype == object
                )
                if (l_numeric and r_objlike) or (l_objlike and r_numeric):
                    output_lcol, output_rcol = lcol, rcol

            left_key.set(self.lhs, output_lcol)
            right_key.set(self.rhs, output_rcol)

        from cudf.core.dataframe import DataFrame

        if self.how == "cross":
            lib_table = plc.join.cross_join(
                plc.Table(
                    [col.plc_column for col in self.lhs._columns],
                    num_rows=len(self.lhs),
                ),
                plc.Table(
                    [col.plc_column for col in self.rhs._columns],
                    num_rows=len(self.rhs),
                ),
            )
            columns = lib_table.columns()
            # A cross join of two column-less operands is a column-less table
            # whose row count is the product of the operands' row counts.
            # ``columns`` cannot carry that count, so remember it explicitly.
            cross_num_rows = lib_table.num_rows()
            num_left_cols = len(self.lhs._column_names)
            left_result = DataFrame._from_data(
                {
                    col: ColumnBase.create(lib_col, dtype)
                    for (col, dtype), lib_col in zip(
                        self.lhs._dtypes,
                        columns[:num_left_cols],
                        strict=True,
                    )
                }
            )
            right_result = DataFrame._from_data(
                {
                    col: ColumnBase.create(lib_col, dtype)
                    for (col, dtype), lib_col in zip(
                        self.rhs._dtypes,
                        columns[num_left_cols:],
                        strict=True,
                    )
                }
            )
            del columns, lib_table
        else:
            left_rows, right_rows = self._gather_maps(
                left_join_cols, right_join_cols
            )
            if self.how not in {"inner", "leftsemi", "leftanti"}:
                # libcudf marks unmatched rows in a gather map with a negative
                # out-of-bounds sentinel; gathering with such a map nullifies
                # that side's columns at those rows.
                self._left_unmatched = (
                    left_rows is not None
                    and len(left_rows) > 0
                    and left_rows.minmax()[0] < 0
                )
                self._right_unmatched = (
                    right_rows is not None
                    and len(right_rows) > 0
                    and right_rows.minmax()[0] < 0
                )
            gather_kwargs = {
                "keep_index": self._using_left_index
                or self._using_right_index,
            }
            left_result = (
                self.lhs._gather(
                    GatherMap.from_column_unchecked(
                        left_rows, len(self.lhs), nullify=True
                    ),
                    **gather_kwargs,
                )
                if left_rows is not None
                else DataFrame._from_data({})
            )
            del left_rows
            right_result = (
                self.rhs._gather(
                    GatherMap.from_column_unchecked(
                        right_rows, len(self.rhs), nullify=True
                    ),
                    **gather_kwargs,
                )
                if right_rows is not None
                else DataFrame._from_data({})
            )
            del right_rows
        result = DataFrame._from_data(
            *self._merge_results(left_result, right_result)
        )
        if self.how == "cross" and result._num_columns == 0:
            from cudf.core.index import RangeIndex

            # Both operands were column-less, so the cross product is a
            # column-less table whose row count an empty ColumnAccessor cannot
            # carry. Rebuild the result with the row count captured above.
            result = DataFrame._from_data({}, index=RangeIndex(cross_num_rows))

        if self.sort:
            result = self._sort_result(result)
        # Mirror pandas' merge `__finalize__` with `input_objs`: propagate
        # attrs only when both inputs have equal non-empty attrs, and AND
        # ``allows_duplicate_labels`` across inputs.
        lhs_attrs = self.lhs.attrs
        rhs_attrs = self.rhs.attrs
        if lhs_attrs and rhs_attrs and lhs_attrs == rhs_attrs:
            result._attrs = copy.deepcopy(lhs_attrs)
        result.flags["allows_duplicate_labels"] = (
            self.lhs.flags.allows_duplicate_labels
            and self.rhs.flags.allows_duplicate_labels
        )
        return result

    @staticmethod
    def _check_duplicate_output_labels(
        left_names, right_names, llabels, rlabels, to_rename
    ):
        # Mirror pandas' ``_items_overlap_with_suffix`` duplicate detection.
        # A duplicate is only an error when it is *introduced by suffixing*
        # (labels already duplicated in an input are allowed), or when a
        # suffixed label collides with a non-renamed label on the other side.
        def _within_frame_dups(labels, names):
            seen_labels: set = set()
            seen_names: set = set()
            dups = []
            for label, name in zip(labels, names, strict=True):
                if label in seen_labels and name not in seen_names:
                    # ``label`` repeats but ``name`` did not (i.e. the
                    # duplication came from suffixing, not the input).
                    dups.append(label)
                seen_labels.add(label)
                seen_names.add(name)
            return dups

        dups = _within_frame_dups(llabels, left_names)
        dups.extend(_within_frame_dups(rlabels, right_names))
        # A renamed label colliding with a non-renamed label on the other side.
        right_not_renamed = set(right_names) - set(to_rename)
        left_not_renamed = set(left_names) - set(to_rename)
        dups.extend(
            label for label in set(llabels) if label in right_not_renamed
        )
        dups.extend(
            label for label in set(rlabels) if label in left_not_renamed
        )
        if dups:
            raise MergeError(
                f"Passing 'suffixes' which cause duplicate columns "
                f"{set(dups)} is not allowed."
            )

    @staticmethod
    def _promote_column_with_nulls(col: ColumnBase) -> ColumnBase:
        # pandas upcasts a numpy integer column that has acquired missing
        # values (from unmatched rows) to float64. cudf represents nulls
        # natively via a mask; upcast here to match pandas' output dtype.
        if (
            isinstance(col.dtype, np.dtype)
            and col.dtype.kind in "iu"
            and col.null_count
        ):
            return col.astype(np.dtype(np.float64))
        return col

    @staticmethod
    def _promote_index_with_nulls(index: Index) -> Index:
        # A gathered index carrying unmatched rows has nulls; pandas drops the
        # index name(s) and upcasts a numpy integer index to float64.
        if isinstance(index, cudf.MultiIndex):
            return index
        col = index._column
        if col.null_count == 0:
            return index
        col = Merge._promote_column_with_nulls(col)
        return cudf.Index._from_column(col, name=None)

    def _merge_results(self, left_result: DataFrame, right_result: DataFrame):
        # Merge the DataFrames `left_result` and `right_result` into a single
        # `DataFrame`, suffixing column names if necessary.

        # If two key columns have the same name, a single output column appears
        # in the result. For all non-outer join types, the key column from the
        # rhs is simply dropped. For outer joins, the two key columns are
        # combined by filling nulls in the left key column with corresponding
        # values from the right key column:
        if self.how == "outer":
            for lkey, rkey in zip(
                self._left_keys, self._right_keys, strict=True
            ):
                if lkey.name == rkey.name:
                    # fill nulls in lhs from values in the rhs
                    lkey.set(
                        left_result,
                        lkey.get(left_result).fillna(rkey.get(right_result)),
                    )

        # For a single-flag mixed merge (left_on + right_index, or
        # left_index + right_on), pandas fills the surviving column key with
        # the opposite frame's index values for the unmatched rows.
        if self._left_index_flag != self._right_index_flag:
            for lkey, rkey in zip(
                self._left_keys, self._right_keys, strict=True
            ):
                if self._right_index_flag and isinstance(lkey, _ColumnIndexer):
                    lkey.set(
                        left_result,
                        lkey.get(left_result).fillna(rkey.get(right_result)),
                    )
                elif self._left_index_flag and isinstance(
                    rkey, _ColumnIndexer
                ):
                    rkey.set(
                        right_result,
                        rkey.get(right_result).fillna(lkey.get(left_result)),
                    )

        # All columns from the left table make it into the output. Non-key
        # columns that share a name with a column in the right table are
        # suffixed with the provided suffix.
        common_names = set(left_result._column_names) & set(
            right_result._column_names
        )

        cols_to_suffix = (
            common_names
            if self.how == "cross"
            else common_names - self._key_columns_with_same_name
        )

        def _suffixed(name, suffix):
            # Mirror pandas: a ``None`` suffix leaves the label (and its type,
            # e.g. an integer column name) unchanged, while a string suffix is
            # appended (coercing the label to a string).
            if name in cols_to_suffix and suffix is not None:
                return f"{name}{suffix}"
            return name

        # All left columns appear in the output; right key columns that share a
        # name with a left key *column* are dropped (their values come from the
        # left key column). A right key column whose left counterpart is an
        # index level is not a duplicate (the level is not emitted as a
        # column), so it is kept. The remaining labels are suffixed per the
        # rule above.
        left_items = list(left_result._column_labels_and_values)
        right_items = [
            (name, col)
            for name, col in right_result._column_labels_and_values
            if name not in common_names
            or self.how == "cross"
            or name not in self._key_columns_with_same_name
        ]
        left_names = [name for name, _ in left_items]
        right_names = [name for name, _ in right_items]
        llabels = [_suffixed(name, self.lsuffix) for name in left_names]
        rlabels = [_suffixed(name, self.rsuffix) for name in right_names]

        # pandas raises MergeError when suffixing would introduce a duplicate
        # column label that was not already duplicated in the inputs.
        self._check_duplicate_output_labels(
            left_names, right_names, llabels, rlabels, cols_to_suffix
        )

        # pandas permits some duplicate output labels (e.g. two ``B_x`` columns
        # from ``suffixes=("_x", "_x")``); cudf represents columns as a dict
        # keyed by label and cannot hold duplicates, so surface these as
        # NotImplementedError (cudf.pandas falls back to pandas transparently).
        output_labels = llabels + rlabels
        if len(set(output_labels)) != len(output_labels):
            raise NotImplementedError(
                f"suffixes={(self.lsuffix, self.rsuffix)} would introduce a "
                "duplicate column label, which is not supported."
            )

        # pandas has no integer missing-value sentinel, so a merge that
        # introduces nulls into a numpy integer column upcasts it to float64.
        # Only unmatched rows introduce nulls -- a column that merely carried
        # cudf-native nulls into the merge keeps its dtype.
        promotions = itertools.chain(
            itertools.repeat(self._left_unmatched, len(left_items)),
            itertools.repeat(self._right_unmatched, len(right_items)),
        )
        data = {
            label: self._promote_column_with_nulls(col) if promote else col
            for label, (_, col), promote in zip(
                llabels + rlabels,
                left_items + right_items,
                promotions,
                strict=True,
            )
        }

        # determine if the result has multiindex columns.  The result
        # of a join has a MultiIndex as its columns if:
        # - both the `lhs` and `rhs` have a MultiIndex columns
        # OR
        # - either one of `lhs` or `rhs` have a MultiIndex columns,
        #   and the other is empty (i.e., no columns)
        if self.lhs._data and self.rhs._data:
            multiindex_columns = (
                self.lhs._data.multiindex and self.rhs._data.multiindex
            )
            rangeindex_columns = (
                self.lhs._data.rangeindex and self.rhs._data.rangeindex
            )
        elif self.lhs._data:
            multiindex_columns = self.lhs._data.multiindex
            rangeindex_columns = self.lhs._data.rangeindex
        elif self.rhs._data:
            multiindex_columns = self.rhs._data.multiindex
            rangeindex_columns = self.rhs._data.rangeindex
        else:
            multiindex_columns = False
            rangeindex_columns = (
                self.lhs._data.rangeindex and self.rhs._data.rangeindex
            )

        # The result keeps a frame's index when that frame joined via the
        # ``left_index``/``right_index`` flag, or when the *same* index level
        # is used as the key on both sides (e.g. ``on=<index level>``). An
        # index level used as a key on only one side (via ``left_on``/
        # ``right_on``), or with differing names, is treated like a column and
        # yields a default RangeIndex.
        both_on_index = (
            self._using_left_index
            and self._using_right_index
            and all(
                lkey.name == rkey.name
                for lkey, rkey in zip(
                    self._left_keys, self._right_keys, strict=True
                )
            )
        )
        index: Index | None
        if (self._left_index_flag and self._right_index_flag) or both_on_index:
            index = left_result.index
        elif self._right_index_flag:
            # right_index (+ left_on): result index is the mapped left index.
            index = left_result.index
        elif self._left_index_flag:
            # left_index (+ right_on): result index is the mapped right index.
            index = right_result.index
        else:
            index = None

        # For a single-flag mixed merge, unmatched rows introduce nulls into
        # the mapped index; pandas drops the index name and upcasts a numpy
        # integer index to float64. The mapped index was gathered from the
        # left frame when ``right_index`` was passed (and vice versa), so
        # consult that side's unmatched flag.
        if index is not None and (
            self._left_index_flag != self._right_index_flag
        ):
            index_side_unmatched = (
                self._left_unmatched
                if self._right_index_flag
                else self._right_unmatched
            )
            if index_side_unmatched:
                index = self._promote_index_with_nulls(index)

        # Construct result from data and index:
        return (
            left_result._data.__class__(
                data=data,
                multiindex=multiindex_columns,
                rangeindex=rangeindex_columns,
            ),
            index,
        )

    def _sort_result(self, result: DataFrame) -> DataFrame:
        # Pandas sorts on the key columns in the
        # same order as given in 'on'. If the indices are used as
        # keys, the index will be sorted. If one index is specified,
        # the key columns on the other side will be used to sort.
        # In pandas-compatible mode, tie-breaking for multiple equal
        # sort keys is to produce output in input dataframe order.
        # This is taken care of by using a stable sort here, and (in
        # pandas-compat mode) reordering the gather maps before
        # producing the input result.
        by: list[Any] = []
        if self._using_left_index and self._using_right_index:
            by.extend(result.index._columns)
        if not self._using_left_index:
            by.extend([result._data[col.name] for col in self._left_keys])
        if not self._using_right_index:
            by.extend([result._data[col.name] for col in self._right_keys])
        if by:
            keep_index = self._using_left_index or self._using_right_index
            if keep_index:
                to_sort = list(
                    itertools.chain(result.index._columns, result._columns)
                )
                index_names = result.index.names
            else:
                to_sort = list(result._columns)
                index_names = None
            result_columns = sorting.sort_by_key(
                to_sort,
                by,
                itertools.repeat(True, times=len(by)),
                itertools.repeat("last", times=len(by)),
                stable=True,
            )
            result = result._from_columns_like_self(
                [
                    ColumnBase.create(col, original.dtype)
                    for col, original in zip(
                        result_columns, to_sort, strict=True
                    )
                ],
                result._column_names,
                index_names,
            )
        return result

    @staticmethod
    def _validate_merge_params(
        lhs,
        rhs,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        how,
        suffixes,
    ):
        # Error for various invalid combinations of merge input parameters
        from cudf.core.dataframe import DataFrame
        from cudf.core.series import Series

        if not isinstance(lhs, (Series, DataFrame)):
            raise TypeError("left must be a Series or DataFrame")
        if not isinstance(rhs, (Series, DataFrame)):
            raise TypeError("right must be a Series or DataFrame")
        if not isinstance(left_index, bool):
            raise ValueError(
                f"left_index parameter must be of type bool, not "
                f"{type(left_index)}"
            )
        if not isinstance(right_index, bool):
            raise ValueError(
                f"right_index parameter must be of type bool, not "
                f"{type(right_index)}"
            )
        # ``suffixes`` must be an ordered pair; pandas rejects unordered/mapping
        # containers such as sets and dicts.
        if not isinstance(suffixes, (list, tuple)):
            raise TypeError(
                f"Passing 'suffixes' as a {type(suffixes)}, is not supported. "
                "Provide 'suffixes' as a tuple instead."
            )
        # We must actually support the requested merge type
        if how not in {
            "left",
            "inner",
            "outer",
            "leftanti",
            "leftsemi",
            "cross",
        }:
            raise NotImplementedError(f"{how} merge not supported yet")

        if on:
            if left_on or right_on:
                # Passing 'on' with 'left_on' or 'right_on' is ambiguous
                raise ValueError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )
            elif left_index or right_index:
                # Passing 'on' with 'left_index' or 'right_index' is ambiguous
                raise ValueError(
                    'Can only pass argument "on" OR "left_index" '
                    'and "right_index", not a combination of both.'
                )
            else:
                # the validity of 'on' being checked by _Indexer
                return
        elif left_on and left_index:
            raise ValueError(
                'Can only pass argument "left_on" OR "left_index" not both.'
            )
        elif right_on and right_index:
            raise ValueError(
                'Can only pass argument "right_on" OR "right_index" not both.'
            )

        # Can't merge on a column name that is present in both a frame and its
        # indexes.
        if on:
            for key in on:
                if (key in lhs._data and key in lhs.index._data) or (
                    key in rhs._data and key in rhs.index._data
                ):
                    raise ValueError(
                        f"{key} is both an index level and a "
                        "column label, which is ambiguous."
                    )
        if left_on:
            for key in left_on:
                if key in lhs._data and key in lhs.index._data:
                    raise ValueError(
                        f"{key} is both an index level and a "
                        "column label, which is ambiguous."
                    )
        if right_on:
            for key in right_on:
                if key in rhs._data and key in rhs.index._data:
                    raise ValueError(
                        f"{key} is both an index level and a "
                        "column label, which is ambiguous."
                    )

        # Can't merge on unnamed Series
        if (isinstance(lhs, Series) and not lhs.name) or (
            isinstance(rhs, Series) and not rhs.name
        ):
            raise ValueError("Cannot merge on unnamed Series")

        # If nothing specified, must have common cols to use implicitly
        same_named_columns = set(lhs._data) & set(rhs._data)
        if how != "cross" and (
            not (left_index or right_index)
            and not (left_on or right_on)
            and len(same_named_columns) == 0
        ):
            raise ValueError("No common columns to perform merge on")

        lsuffix, rsuffix = suffixes
        for name in same_named_columns:
            if name == left_on == right_on:
                continue
            elif left_on and right_on:
                if (name in left_on and name in right_on) and (
                    left_on.index(name) == right_on.index(name)
                ):
                    continue
            else:
                if not (lsuffix or rsuffix):
                    raise ValueError(
                        "there are overlapping columns but "
                        "lsuffix and rsuffix are not defined"
                    )

        if (
            isinstance(lhs, DataFrame)
            and isinstance(rhs, DataFrame)
            # An empty column is considered to have 1 level by pandas (can be
            # seen by using lhs.columns.nlevels, but we don't want to use
            # columns internally because it's expensive).
            # TODO: Investigate whether ColumnAccessor.nlevels should be
            # modified in the size 0 case.
            and max(lhs._data.nlevels, 1) != max(rhs._data.nlevels, 1)
        ):
            raise ValueError(
                "Not allowed to merge between different levels. "
                f"({lhs._data.nlevels} levels on "
                f"the left, {rhs._data.nlevels} on the right)"
            )


class MergeSemi(Merge):
    @staticmethod
    def _joiner(  # type: ignore[override]
        lhs: list[ColumnBase],
        rhs: list[ColumnBase],
        how: str,
    ) -> tuple[ColumnBase, None]:
        if (
            join_func := getattr(
                plc.join, f"{how.replace('left', 'left_')}_join", None
            )
        ) is None:
            raise ValueError(f"Invalid join type {how}")

        with access_columns(
            *lhs, *rhs, mode="read", scope="internal"
        ) as accessed:
            # Split accessed tuple back into lhs and rhs
            n_lhs = len(lhs)
            lhs = accessed[:n_lhs]  # type: ignore[assignment]
            rhs = accessed[n_lhs:]  # type: ignore[assignment]
            return ColumnBase.create(
                join_func(
                    plc.Table([col.plc_column for col in lhs]),
                    plc.Table([col.plc_column for col in rhs]),
                    plc.types.NullEquality.EQUAL,
                ),
                SIZE_TYPE_DTYPE,
            ), None

    def _merge_results(self, lhs: DataFrame, rhs: DataFrame):
        # semi-join result includes only lhs columns
        return lhs._data, lhs.index
