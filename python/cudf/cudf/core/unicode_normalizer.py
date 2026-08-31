# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unicode TR15 normalization support for cudf strings columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf.core.column.column import ColumnBase
from cudf.core.series import Series

if TYPE_CHECKING:
    from cudf.core.dataframe import DataFrame


class UnicodeNormalizer:
    """
    A normalizer object for Unicode TR15 normalization (NFD, NFC, NFKD, NFKC).

    The normalizer is constructed from the contents of the Unicode Character
    Database ``UnicodeData.txt`` file, which must be loaded by the caller as a
    :class:`cudf.DataFrame` (e.g. via :func:`cudf.read_csv`).

    The file is published by the Unicode Consortium and can be downloaded from:
    https://unicode.org/Public/15.1.0/ucd/UnicodeData.txt

    It is a semicolon-delimited file with 15 fields per row.  The three fields
    required by this API are field 0 (code point hex), field 3 (CCC), and
    field 5 (decomposition mapping).  Load only those three columns:

    .. code-block:: python

        unicode_data = cudf.read_csv(
            "UnicodeData.txt",
            sep=";",
            header=None,
            usecols=[0, 3, 5],
            dtype={0: "str", 3: "int32", 5: "str"},
        )
        normalizer = UnicodeNormalizer(unicode_data, form="NFKC")
        result = normalizer.normalize(series)

    The ``unicode_data`` DataFrame must contain exactly three columns in the
    following order:

    - column[0]: ``str``   Code point values as uppercase hex strings (e.g. ``"00C9"``)
    - column[1]: ``int32`` Canonical_Combining_Class (CCC) values in range [0, 254]
    - column[2]: ``str``   Decomposition_Mapping field; empty string for identity
      mappings, optionally prefixed with a compatibility tag such as
      ``<compat>``, ``<font>``, ``<wide>``, etc.

    Decomposition of Hangul syllables (U+AC00..U+D7A3) is performed
    algorithmically per the Unicode standard and does not require entries in
    the provided table.

    Parameters
    ----------
    unicode_data : cudf.DataFrame
        Three-column DataFrame loaded from ``UnicodeData.txt`` as described above.
    form : str
        Normalization form: one of ``"NFD"``, ``"NFC"``, ``"NFKD"``,
        or ``"NFKC"``.
    """

    _FORM_MAP = {
        "NFD": plc.nvtext.unicode_normalize.UnicodeNormalizationForm.NFD,
        "NFC": plc.nvtext.unicode_normalize.UnicodeNormalizationForm.NFC,
        "NFKD": plc.nvtext.unicode_normalize.UnicodeNormalizationForm.NFKD,
        "NFKC": plc.nvtext.unicode_normalize.UnicodeNormalizationForm.NFKC,
    }

    def __init__(
        self,
        unicode_data: DataFrame,
        form: str = "NFC",
    ) -> None:
        if form not in self._FORM_MAP:
            raise ValueError(
                f"Invalid normalization form {form!r}. "
                f"Expected one of: {list(self._FORM_MAP)}"
            )
        tbl = plc.Table.from_arrow(unicode_data.to_arrow())
        self._normalizer = plc.nvtext.unicode_normalize.UnicodeNormalizer(
            tbl, self._FORM_MAP[form]
        )

    def normalize(self, text: Series) -> Series:
        """
        Normalize a strings Series using Unicode TR15 normalization.

        Parameters
        ----------
        text : cudf.Series
            The UTF-8 strings to normalize.  Null entries produce null output.

        Returns
        -------
        cudf.Series
            New Series of normalized UTF-8 strings.
        """
        plc_column = plc.nvtext.unicode_normalize.normalize_unicode(
            text._column.plc_column, self._normalizer
        )
        return Series._from_column(
            ColumnBase.create(plc_column, text._column.dtype)
        )
