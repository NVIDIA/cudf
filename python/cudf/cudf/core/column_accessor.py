# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

import cudf
from cudf.api.types import infer_dtype, is_scalar
from cudf.core import column
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import DEFAULT_STRING_DTYPE, is_mixed_with_object_dtype

if TYPE_CHECKING:
    from typing import Self

    from cudf._typing import DtypeObj
    from cudf.core.column import ColumnBase


def _is_bool(val: Any) -> bool:
    return isinstance(val, (bool, np.bool_))


def _is_nan_scalar(val: Any) -> bool:
    return isinstance(val, (float, np.floating)) and val != val


def _label_contains_nan(label: Any) -> bool:
    if isinstance(label, tuple):
        return any(_is_nan_scalar(lv) for lv in label)
    return _is_nan_scalar(label)


def _canonicalize_nan_label(label: Any) -> Any:
    """Map float NaN elements of a label to the np.nan singleton.

    No NaN object ever compares equal to any NaN (including itself), so
    equality alone can never match a NaN-containing label. dict lookups and
    tuple comparison, however, short-circuit on object *identity* before
    trying ``==`` — and that shortcut is exactly what this canonicalization
    targets: with every NaN mapped to the one ``np.nan`` object, two
    canonicalized labels holding NaN in the same position match by identity.
    This restores pandas' all-NaNs-are-equal label semantics for labels
    round-tripped through a pandas Index, which materializes fresh NaN
    objects on iteration.
    """
    if isinstance(label, tuple):
        return tuple(np.nan if _is_nan_scalar(lv) else lv for lv in label)
    if _is_nan_scalar(label):
        return np.nan
    return label


class _NestedGetItemDict(dict):
    """A dictionary whose __getitem__ method accesses nested dicts.

    This class directly subclasses dict for performance, so there are a number
    of gotchas: 1) the only safe accessor for nested elements is
    `__getitem__` (all other accessors will fail to perform nested lookups), 2)
    nested mappings will not exhibit the same behavior (they will be raw
    dictionaries unless explicitly created to be of this class), and 3) to
    construct this class you _must_ use `from_zip` to get appropriate treatment
    of tuple keys.
    """

    @classmethod
    def from_zip(cls, data: Iterator):
        """Create from zip, specialized factory for nesting."""
        obj = cls()
        for key, value in data:
            d = obj
            for k in key[:-1]:
                d = d.setdefault(k, {})
            d[key[-1]] = value
        return obj

    def __getitem__(self, key):
        """Recursively apply dict.__getitem__ for nested elements."""
        # As described in the pandas docs
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-indexing-with-hierarchical-index
        # accessing nested elements of a multiindex must be done using a tuple.
        # Lists and other sequences are treated as accessing multiple elements
        # at the top level of the index.
        if isinstance(key, tuple):
            return reduce(dict.__getitem__, key, self)
        return super().__getitem__(key)


def _to_flat_dict_inner(d: dict, parents: tuple = ()):
    for k, v in d.items():
        if not isinstance(v, d.__class__):
            if parents:
                k = (*parents, k)
            yield (k, v)
        else:
            yield from _to_flat_dict_inner(d=v, parents=(*parents, k))


class ColumnAccessor(MutableMapping):
    """
    Parameters
    ----------
    data : mapping
        Mapping of keys to column values.
    multiindex : bool, optional
        Whether tuple keys represent a hierarchical
        index with multiple "levels" (default=False).
    level_names : tuple, optional
        Tuple containing names for each of the levels.
        For a non-hierarchical index, a tuple of size 1
        may be passe.
    rangeindex : bool, optional
        Whether the keys should be returned as a RangeIndex
        in `to_pandas_index` (default=False).
    label_dtype : DtypeObj, optional
        What dtype should be returned in `to_pandas_index`
        (default=None).
    verify : bool, optional
        For non ColumnAccessor inputs, whether to verify
        column length and data.values() are all Columns
    pandas_index : pd.Index, optional
        The source pandas index the keys were taken from, if any.
        A matching pd.MultiIndex primes the ``to_pandas_index`` cache
        (see ``_prime_to_pandas_index``).
    """

    _data: dict[Hashable, ColumnBase]
    _level_names: tuple[Hashable, ...]
    _level_dtypes: tuple[DtypeObj, ...] | None

    def __init__(
        self,
        data: MutableMapping[Hashable, ColumnBase] | Self,
        multiindex: bool = False,
        level_names=None,
        rangeindex: bool = False,
        label_dtype: DtypeObj | None = None,
        verify: bool = True,
        level_dtypes: tuple[DtypeObj, ...] | None = None,
        pandas_index: pd.Index | None = None,
    ) -> None:
        if isinstance(data, ColumnAccessor):
            self._data = data._data
            self._level_names = data.level_names
            self.multiindex: bool = data.multiindex
            self.rangeindex: bool = data.rangeindex
            self.label_dtype: DtypeObj | None = data.label_dtype
            self._level_dtypes = data._level_dtypes
            if "to_pandas_index" in data.__dict__:
                # carry over the primed/cached pandas index: it holds
                # fidelity (e.g. explicit unsorted level order) that a
                # rebuild from tuples would lose
                self.to_pandas_index = data.__dict__["to_pandas_index"]
        elif isinstance(data, MutableMapping):
            # This code path is performance-critical for copies and should be
            # modified with care.
            if data and verify:
                # Faster than next(iter(data.values()))
                column_length = len(data[next(iter(data))])
                # TODO: we should validate the keys of `data`
                for col in data.values():
                    if not isinstance(col, column.ColumnBase):
                        raise ValueError(
                            f"All data.values() must be Column, not {type(col).__name__}"
                        )
                    if len(col) != column_length:
                        raise ValueError("All columns must be of equal length")

            if not isinstance(data, dict):
                data = dict(data)
            self._data = data

            if rangeindex and multiindex:
                raise ValueError(
                    f"{rangeindex=} and {multiindex=} cannot both be True."
                )
            self.rangeindex = rangeindex
            self.multiindex = multiindex
            self.label_dtype = label_dtype
            self._level_names = level_names
            self._level_dtypes = level_dtypes
        else:
            raise ValueError(
                f"data must be a ColumnAccessor or MutableMapping, not {type(data).__name__}"
            )
        if pandas_index is not None:
            self._prime_to_pandas_index(pandas_index)

    def _prime_to_pandas_index(self, index: pd.Index) -> None:
        """Prime the cached ``to_pandas_index`` with the exact source index.

        Rebuilding a pandas MultiIndex from the stored tuple labels re-sorts
        its levels, losing an explicit unsorted level layout (the level
        order affects pandas operations that work on level codes, e.g.
        legacy ``stack(sort=True)``). Keeping the source MultiIndex itself
        preserves that fidelity. Only a hierarchical columns axis whose
        length matches the data is primed; anything else is ignored.
        """
        if (
            self.multiindex
            and isinstance(index, pd.MultiIndex)
            and len(self._data) == len(index)
        ):
            self.to_pandas_index = index

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __getitem__(self, key: Hashable) -> ColumnBase:
        try:
            return self._data[key]
        except KeyError:
            if _label_contains_nan(key):
                # NaN labels lose object identity when round-tripped through
                # a pandas Index; retry with NaNs canonicalized so all NaNs
                # match by identity, as pandas label semantics require.
                canon = _canonicalize_nan_label(key)
                for existing in self._data:
                    c = _canonicalize_nan_label(existing)
                    try:
                        match = c is canon or bool(c == canon)
                    except TypeError:
                        # e.g. a pd.NA label: its comparisons return pd.NA,
                        # whose truthiness raises. Ambiguity is not a match;
                        # keep scanning for a genuine NaN label.
                        match = False
                    if match:
                        return self._data[existing]
            raise

    def __setitem__(self, key: Hashable, value: ColumnBase) -> None:
        self.set_by_label(key, value)

    def __delitem__(self, key: Hashable) -> None:
        old_ncols = len(self)
        del self._data[key]
        new_ncols = len(self)
        self._clear_cache(old_ncols, new_ncols)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        type_info = (
            f"{self.__class__.__name__}("
            f"multiindex={self.multiindex}, "
            f"level_names={self.level_names}, "
            f"rangeindex={self.rangeindex}, "
            f"label_dtype={self.label_dtype})"
        )
        column_info = "\n".join(
            [f"{name}: {col.dtype}" for name, col in self.items()]
        )
        return f"{type_info}\n{column_info}"

    def _from_columns_like_self(
        self, columns: Iterable[ColumnBase], verify: bool = True
    ) -> Self:
        """
        Return a new ColumnAccessor with columns and the properties of self.

        Parameters
        ----------
        columns : iterable of Columns
            New columns for the ColumnAccessor.
        verify : bool, optional
            Whether to verify column length and type.
        """
        return type(self)(
            data=dict(zip(self.names, columns, strict=True)),
            multiindex=self.multiindex,
            level_names=self.level_names,
            rangeindex=self.rangeindex,
            label_dtype=self.label_dtype,
            verify=verify,
            level_dtypes=self._level_dtypes,
        )

    @property
    def level_names(self) -> tuple[Hashable, ...]:
        if self.is_cached("to_pandas_index"):
            return self.to_pandas_index.names
        if self._level_names is None or len(self._level_names) == 0:
            return tuple((None,) * max(1, self.nlevels))
        else:
            return self._level_names

    @property
    def level_dtypes(self) -> tuple[DtypeObj, ...] | None:
        """Per-level dtypes used to rebuild an empty MultiIndex column axis."""
        return self._level_dtypes

    def is_cached(self, attr_name: str) -> bool:
        return attr_name in self.__dict__

    @property
    def nlevels(self) -> int:
        if len(self) == 0:
            return 0
        if not self.multiindex:
            return 1
        else:
            return len(next(iter(self.keys())))

    @property
    def name(self) -> Hashable:
        return self.level_names[-1]

    @cached_property
    def nrows(self) -> int:
        if len(self) == 0:
            return 0
        else:
            return len(next(iter(self.values())))

    @cached_property
    def names(self) -> tuple[Hashable, ...]:
        return tuple(self.keys())

    @cached_property
    def columns(self) -> tuple[ColumnBase, ...]:
        return tuple(self.values())

    @cached_property
    def _grouped_data(self) -> MutableMapping:
        """
        If self.multiindex is True,
        return the underlying mapping as a nested mapping.
        """
        if self.multiindex:
            return _NestedGetItemDict.from_zip(
                zip(self.names, self.columns, strict=True)
            )
        else:
            return self._data

    def _clear_cache(self, old_ncols: int, new_ncols: int) -> None:
        """
        Clear cached attributes.

        Parameters
        ----------
        old_ncols: int
            len(self) before self._data was modified
        new_ncols: int
            len(self) after self._data was modified
        """
        cached_properties = (
            "columns",
            "names",
            "_grouped_data",
            "to_pandas_index",
        )
        for attr in cached_properties:
            try:
                self.__delattr__(attr)
            except AttributeError:
                pass

        # nrows should only be cleared if empty before/after the op.
        if (old_ncols == 0) ^ (new_ncols == 0):
            try:
                del self.nrows
            except AttributeError:
                pass

    @cached_property
    def to_pandas_index(self) -> pd.Index:
        """Convert the keys of the ColumnAccessor to a Pandas Index object."""
        if self.multiindex and len(self.level_names) > 0:
            if len(self.names) == 0 and self._level_dtypes is not None:
                # An empty MultiIndex cannot have its per-level dtypes inferred
                # from (zero) tuples by ``from_tuples`` (every level would
                # default to object). Rebuild from the preserved per-level
                # dtypes so e.g. an empty integer level stays integer rather
                # than becoming object.
                result = pd.MultiIndex.from_arrays(
                    [
                        pd.Index([], dtype=level_dtype)
                        for level_dtype in self._level_dtypes
                    ],
                    names=self.level_names,
                )
            else:
                result = pd.MultiIndex.from_tuples(
                    self.names,
                    names=self.level_names,
                )
                if (
                    self._level_dtypes is not None
                    and len(self._level_dtypes) == result.nlevels
                ):
                    # ``from_tuples`` re-infers every level dtype from the
                    # materialized labels, degrading e.g. categorical levels
                    # to str, object levels to str once mixed-type labels are
                    # selected away, and int64 levels with missing entries to
                    # float64. Restore each preserved level dtype when the
                    # cast is lossless (round-trips to the inferred values).
                    new_levels = []
                    changed = False
                    for lvl, level_dtype in zip(
                        result.levels, self._level_dtypes, strict=True
                    ):
                        if lvl.dtype != level_dtype:
                            try:
                                cast_lvl = lvl.astype(level_dtype)
                            except (TypeError, ValueError):
                                pass
                            else:
                                # missing-aware equality: ``==`` would treat
                                # NaN entries as unequal to themselves
                                if cast_lvl.astype(lvl.dtype).equals(lvl):
                                    lvl = cast_lvl
                                    changed = True
                        new_levels.append(lvl)
                    if changed:
                        result = result.set_levels(new_levels)
        else:
            # Determine if we can return a RangeIndex
            if self.rangeindex:
                if not self.names:
                    return pd.RangeIndex(
                        start=0, stop=0, step=1, name=self.name
                    )
                elif infer_dtype(self.names) == "integer":
                    if len(self.names) == 1:
                        start = cast("int", self.names[0])
                        return pd.RangeIndex(
                            start=start, stop=start + 1, step=1, name=self.name
                        )
                    uniques = np.unique(np.diff(np.array(self.names)))
                    if len(uniques) == 1 and uniques[0] != 0:
                        diff = uniques[0]
                        new_range = range(
                            cast("int", self.names[0]),
                            cast("int", self.names[-1]) + diff,
                            diff,
                        )
                        return pd.RangeIndex(new_range, name=self.name)
            # Avoid pandas returning Index[object]
            dtype = (
                self.label_dtype
                if self.label_dtype is not None or len(self.names) > 0
                else DEFAULT_STRING_DTYPE
            )
            result = pd.Index(
                self.names,
                name=self.name,
                tupleize_cols=False,
                dtype=dtype,
            )
        return result

    def insert(self, name: Hashable, value: ColumnBase, loc: int = -1) -> None:
        """
        Insert column into the ColumnAccessor at the specified location.

        Parameters
        ----------
        name : Name corresponding to the new column
        value : ColumnBase
        loc : int, optional
            The location to insert the new value at.
            Must be (0 <= loc <= ncols). By default, the column is added
            to the end.

        Returns
        -------
        None, this function operates in-place.
        """
        name = self._pad_key(name)
        if name in self._data:
            raise ValueError(f"Cannot insert '{name}', already exists")

        old_ncols = len(self)
        if loc == -1:
            loc = old_ncols
        elif not (0 <= loc <= old_ncols):
            raise ValueError(
                f"insert: loc out of bounds: must be  0 <= loc <= {old_ncols}"
            )

        if not isinstance(value, column.ColumnBase):
            raise ValueError("value must be a Column")
        elif old_ncols > 0 and len(value) != self.nrows:
            raise ValueError("All columns must be of equal length")

        if cudf.get_option("mode.pandas_compatible"):
            try:
                pd_idx1 = pd.Index(
                    [*list(self.names), name], dtype=self.label_dtype
                )
                pd_idx2 = pd.Index([*list(self.names), name])
                if (
                    pd_idx1.dtype != pd_idx2.dtype
                    and is_mixed_with_object_dtype(pd_idx1, pd_idx2)
                    and pd_idx1.inferred_type != pd_idx2.inferred_type
                ):
                    raise MixedTypeError(
                        "Cannot insert column with mixed types when label_dtype is set"
                    )
            except Exception as e:
                raise e
        else:
            self.label_dtype = None
        # TODO: we should move all insert logic here
        if loc == old_ncols:
            self._data[name] = value
        else:
            new_keys = (*self.names[:loc], name, *self.names[loc:])
            new_values = (*self.columns[:loc], value, *self.columns[loc:])
            self._data = dict(zip(new_keys, new_values, strict=True))
        self._clear_cache(old_ncols, old_ncols + 1)
        # The type(name) may no longer match the prior label_dtype

    def copy(self, deep: bool = False) -> Self:
        """
        Make a copy of this ColumnAccessor.
        """
        data = {k: v.copy(deep=deep) for k, v in self._data.items()}
        return self.__class__(
            data=data,
            multiindex=self.multiindex,
            level_names=self.level_names,
            rangeindex=self.rangeindex,
            label_dtype=self.label_dtype,
            verify=False,
            level_dtypes=self._level_dtypes,
        )

    def select_by_label(self, key: Any) -> Self:
        """
        Return a subset of this column accessor,
        composed of the keys specified by `key`.

        Parameters
        ----------
        key : slice, list-like, tuple or scalar

        Returns
        -------
        ColumnAccessor
        """
        if isinstance(key, slice):
            return self._select_by_label_slice(key)
        elif not (isinstance(key, tuple) or is_scalar(key)):
            return self._select_by_label_list_like(tuple(key))
        else:
            if isinstance(key, tuple):
                if any(isinstance(k, slice) for k in key):
                    return self._select_by_label_with_wildcard(key)
            return self._select_by_label_grouped(key)

    def get_labels_by_index(
        self, index: slice | int | Iterable[int | bool]
    ) -> tuple:
        """Get the labels corresponding to the provided column indices.

        Parameters
        ----------
        index : integer, integer slice, boolean mask,
            or list-like of integers
            The column indexes.

        Returns
        -------
        tuple
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return self.names[start:stop:step]
        elif isinstance(index, int):
            return (self.names[index],)
        elif (bn := len(index)) > 0 and all(map(_is_bool, index)):  # type: ignore[arg-type]
            if bn != (n := len(self)):
                raise IndexError(
                    f"Boolean mask has wrong length: {bn} not {n}"
                )
            if isinstance(index, (pd.Series, cudf.Series)):
                # Don't allow iloc indexing with series
                raise NotImplementedError(
                    "Cannot use Series object for mask iloc indexing"
                )
            # TODO: Doesn't handle on-device columns
            return tuple(
                n for n, keep in zip(self.names, index, strict=True) if keep
            )
        else:
            if len(set(index)) != len(index):  # type: ignore[arg-type]
                raise NotImplementedError(
                    "Selecting duplicate column labels is not supported."
                )
            return tuple(self.names[i] for i in index)

    def select_by_index(self, index: Any) -> Self:
        """
        Return a ColumnAccessor composed of the columns
        specified by index.

        Parameters
        ----------
        key : integer, integer slice, boolean mask,
            or list-like of integers

        Returns
        -------
        ColumnAccessor
        """
        keys = self.get_labels_by_index(index)
        data = {k: self._data[k] for k in keys}
        return type(self)(
            data,
            multiindex=self.multiindex,
            level_names=self.level_names,
            label_dtype=self.label_dtype,
            level_dtypes=self._level_dtypes,
            verify=False,
        )

    def swaplevel(self, i: Hashable = -2, j: Hashable = -1) -> Self:
        """
        Swap level i with level j.
        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int or str, default -2
            First level of index to be swapped.
        j : int or str, default -1
            Second level of index to be swapped.

        Returns
        -------
        ColumnAccessor
        """
        if not self.multiindex:
            raise ValueError(
                "swaplevel is only valid for self.multiindex=True"
            )

        i = _get_level(i, self.nlevels, self.level_names)
        j = _get_level(j, self.nlevels, self.level_names)

        new_keys = [list(row) for row in self]
        new_dict = {}

        # swap old keys for i and j
        for n, row in enumerate(self.names):
            new_keys[n][i], new_keys[n][j] = row[j], row[i]  # type: ignore[call-overload, index]
            new_dict.update({row: tuple(new_keys[n])})

        new_data = {new_dict[k]: v.copy(deep=False) for k, v in self.items()}

        # swap level_names for i and j
        new_names = list(self.level_names)
        new_names[i], new_names[j] = new_names[j], new_names[i]  # type: ignore[call-overload]

        new_level_dtypes = self._level_dtypes
        if new_level_dtypes is not None:
            level_dtypes = list(new_level_dtypes)
            level_dtypes[i], level_dtypes[j] = level_dtypes[j], level_dtypes[i]  # type: ignore[call-overload]
            new_level_dtypes = tuple(level_dtypes)

        return type(self)(
            new_data,  # type: ignore[arg-type]
            multiindex=self.multiindex,
            level_names=new_names,
            rangeindex=self.rangeindex,
            label_dtype=self.label_dtype,
            verify=False,
            level_dtypes=new_level_dtypes,
        )

    def set_by_label(self, key: Hashable, value: ColumnBase) -> None:
        """
        Add (or modify) column by name.

        Parameters
        ----------
        key
            name of the column
        value : Column
            The value to insert into the column.
        """
        key = self._pad_key(key)
        if not isinstance(value, column.ColumnBase):
            raise ValueError("value must be a Column")
        if len(self) > 0 and len(value) != self.nrows:
            raise ValueError("All columns must be of equal length")

        old_ncols = len(self)
        self._data[key] = value
        new_ncols = len(self)
        self._clear_cache(old_ncols, new_ncols)

    def _select_by_label_list_like(self, key: tuple) -> Self:
        # Special-casing for boolean mask
        if (bn := len(key)) > 0 and all(map(_is_bool, key)):
            if bn != (n := len(self)):
                raise IndexError(
                    f"Boolean mask has wrong length: {bn} not {n}"
                )
            data = dict(
                item
                for item, keep in zip(
                    self._grouped_data.items(), key, strict=True
                )
                if keep
            )
        else:
            data = {k: self._grouped_data[k] for k in key}
            if len(data) != len(key):
                raise ValueError(
                    "Selecting duplicate column labels is not supported."
                )
        if self.multiindex:
            data = dict(_to_flat_dict_inner(data))
        return type(self)(
            data,
            multiindex=self.multiindex,
            level_names=self.level_names,
            label_dtype=self.label_dtype,
            level_dtypes=self._level_dtypes,
            verify=False,
        )

    def _select_by_label_grouped(self, key: Hashable) -> Self:
        result = self._grouped_data[key]
        if isinstance(result, column.ColumnBase):
            # self._grouped_data[key] = self._data[key] so skip validation
            return type(self)(
                data={key: result},
                multiindex=self.multiindex,
                label_dtype=self.label_dtype,
                verify=False,
            )
        else:
            if self.multiindex:
                result = dict(_to_flat_dict_inner(result))
            if not isinstance(key, tuple):
                key = (key,)
            return self.__class__(
                result,
                multiindex=self.nlevels - len(key) > 1,
                level_names=self.level_names[len(key) :],
                level_dtypes=(
                    None
                    if self._level_dtypes is None
                    else self._level_dtypes[len(key) :]
                ),
                verify=False,
            )

    def _select_by_label_slice(self, key: slice) -> Self:
        start, stop = key.start, key.stop

        if len(self) == 0:
            # https://github.com/rapidsai/cudf/issues/18376
            # Any slice is valid when we have no columns
            return self._from_columns_like_self([], verify=False)

        if key.step is not None:
            raise TypeError("Label slicing with step is not supported")

        if start is None:
            start = self.names[0]
        if stop is None:
            stop = self.names[-1]
        start = self._pad_key(start, slice(None))
        stop = self._pad_key(stop, slice(None))
        for idx, name in enumerate(self.names):
            if _keys_equal(name, start):  # type: ignore[arg-type]  # (padded key matches label shape)
                start_idx = idx
                break
        for idx, name in enumerate(reversed(self.names)):
            if _keys_equal(name, stop):  # type: ignore[arg-type]  # (padded key matches label shape)
                stop_idx = len(self) - idx
                break
        keys = self.names[start_idx:stop_idx]
        return type(self)(
            {k: self._data[k] for k in keys},
            multiindex=self.multiindex,
            level_names=self.level_names,
            label_dtype=self.label_dtype,
            level_dtypes=self._level_dtypes,
            verify=False,
        )

    def _select_by_label_with_wildcard(self, key: tuple) -> Self:
        pad_key = self._pad_key(key, slice(None))
        data = {
            k: self._data[k]
            for k in self.names
            if _keys_equal(k, pad_key)  # type: ignore[arg-type]
        }
        return type(self)(
            data,
            multiindex=self.multiindex,
            level_names=self.level_names,
            label_dtype=self.label_dtype,
            level_dtypes=self._level_dtypes,
            verify=False,
        )

    def _pad_key(self, key: Hashable, pad_value: str | slice = "") -> Hashable:
        """
        Pad the provided key to a length equal to the number
        of levels.
        """
        if not self.multiindex:
            return key
        if not isinstance(key, tuple):
            key = (key,)
        return key + (pad_value,) * (self.nlevels - len(key))

    def rename_levels(
        self,
        mapper: Mapping[Hashable, Hashable] | Callable,
        level: int | None = None,
    ) -> Self:
        """
        Rename the specified levels of the given ColumnAccessor

        Parameters
        ----------
        self : ColumnAccessor of a given dataframe

        mapper : dict-like or function transformations to apply to
            the column label values depending on selected ``level``.

            If dict-like, only replace the specified level of the
            ColumnAccessor's keys (that match the mapper's keys) with
            mapper's values

            If callable, the function is applied only to the specified level
            of the ColumnAccessor's keys.

        level : int
            In case of RangeIndex, only supported level is [0, None].
            In case of a MultiColumn, only the column labels in the specified
            level of the ColumnAccessor's keys will be transformed.

        Returns
        -------
        A new ColumnAccessor with values in the keys replaced according
        to the given mapper and level.

        """
        new_col_names: Iterable
        if self.multiindex:

            def rename_column(x):
                x = list(x)
                if isinstance(mapper, Mapping):
                    x[level] = mapper.get(x[level], x[level])
                else:
                    x[level] = mapper(x[level])
                x = tuple(x)
                return x

            if level is None:
                level = 0
            new_col_names = (rename_column(k) for k in self.keys())

        else:
            if level is None:
                level = 0
            if level != 0:
                raise IndexError(
                    f"Too many levels: Index has only 1 level, not {level + 1}"
                )

            if isinstance(mapper, Mapping):
                new_col_names = [
                    mapper.get(col_name, col_name) for col_name in self.keys()
                ]
            else:
                new_col_names = [mapper(col_name) for col_name in self.keys()]

            if len(new_col_names) != len(set(new_col_names)):
                raise ValueError("Duplicate column names are not allowed")

        label_dtype = self.label_dtype
        if len(self) > 0 and label_dtype is not None:
            old_type = type(next(iter(self.keys())))
            if not all(isinstance(label, old_type) for label in new_col_names):
                label_dtype = None

        data = dict(zip(new_col_names, self.values(), strict=True))
        return type(self)(
            data=data,
            level_names=self.level_names,
            multiindex=self.multiindex,
            label_dtype=label_dtype,
            verify=False,
            level_dtypes=self._level_dtypes,
        )

    def droplevel(self, level: int) -> None:
        # drop the nth level
        if level < 0:
            level += self.nlevels

        old_ncols = len(self)
        self._data = {
            _remove_key_level(key, level): value  # type: ignore[arg-type]
            for key, value in self._data.items()
        }
        new_ncols = len(self)
        self._level_names = (
            self.level_names[:level] + self.level_names[level + 1 :]
        )
        if self._level_dtypes is not None:
            self._level_dtypes = (
                self._level_dtypes[:level] + self._level_dtypes[level + 1 :]
            )

        if len(self.level_names) == 1:
            # can't use nlevels, as it depends on multiindex
            self.multiindex = False
        self._clear_cache(old_ncols, new_ncols)


def _keys_equal(target: Hashable, key: Iterable) -> bool:
    """
    Compare `key` to `target`.

    Return True if each value in `key` == corresponding value in `target`.
    If any value in `key` is slice(None), it is considered equal
    to the corresponding value in `target`.
    """
    if not isinstance(target, tuple):
        return target == key
    for k1, k2 in itertools.zip_longest(target, key, fillvalue=None):
        if k2 == slice(None):
            continue
        if k1 != k2:
            return False
    return True


def _remove_key_level(key: tuple, level: int) -> Hashable:
    """
    Remove a level from key. If detupleize is True, and if only a
    single level remains, convert the tuple to a scalar.
    """
    result = key[:level] + key[level + 1 :]
    if len(result) == 1:
        return result[0]
    return result


def _get_level(
    x: Hashable, nlevels: int, level_names: tuple[Hashable, ...]
) -> Hashable:
    """Get the level index from a level number or name.

    If given an integer, this function will handle wraparound for
    negative values. If given a string (the level name), this function
    will extract the index of that level from `level_names`.

    Parameters
    ----------
    x
        The level number to validate
    nlevels
        The total available levels in the MultiIndex
    level_names
        The names of the levels.
    """
    if isinstance(x, int):
        if x < 0:
            x += nlevels
        if x >= nlevels:
            raise IndexError(
                f"Level {x} out of bounds. Index has {nlevels} levels."
            )
        return x
    else:
        x = level_names.index(x)
        return x
