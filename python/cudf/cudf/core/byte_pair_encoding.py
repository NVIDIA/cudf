# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pylibcudf as plc

from cudf.core.series import Series


class BytePairEncoder:
    """
    Byte pair encoder using a table of ranked merge pairs.

    Each string is encoded by repeatedly merging the lowest ranked adjacent
    pair of tokens found in the merge pairs table, starting from the
    individual characters. The remaining tokens are joined with a separator.

    Parameters
    ----------
    merges_pair : cudf.Series
        Strings series of unique merge pairs. Each row contains the two
        halves of a pair separated by a single space. Pairs are ranked by
        their position in the series; earlier rows have higher priority.

    Raises
    ------
    ValueError
        If the merge pairs contain duplicates.
    """

    def __init__(self, merges_pair: Series) -> None:
        self.merge_pairs = plc.nvtext.byte_pair_encode.BPEMergePairs(
            merges_pair._column.plc_column
        )

    def __call__(self, text: Series, separator: str = " ") -> Series:
        """
        Encode the strings using the merge pairs.

        Parameters
        ----------
        text : cudf.Series
            Strings to be encoded.
        separator : str, default " "
            Single-byte string inserted between the encoded tokens.

        Returns
        -------
        cudf.Series
            Encoded strings. Characters that are not part of any merge pair
            (such as spaces) remain individual tokens.

        Examples
        --------
        >>> import cudf
        >>> from cudf.core.byte_pair_encoding import BytePairEncoder
        >>> mps = cudf.Series(["e n", "i t", "i s", "e s", "en t",
        ...                    "c e", "es t", "en ce", "t h", "h i",
        ...                    "th is", "t est", "s ent"])
        >>> bpe = BytePairEncoder(mps)
        >>> str_series = cudf.Series(["thisisit", "this is a test"])
        >>> bpe(str_series)
        0              this is it
        1    this   is   a   test
        dtype: str
        >>> bpe(str_series, separator="_")
        0              this_is_it
        1    this_ _is_ _a_ _test
        dtype: str
        """
        return Series._from_column(
            text._column.byte_pair_encoding(self.merge_pairs, separator)
        )
