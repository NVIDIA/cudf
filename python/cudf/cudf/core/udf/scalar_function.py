# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from functools import cache

from numba import cuda
from numba.np import numpy_support

from cudf.core.dtype.validators import is_dtype_obj_string
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.masked_typing import MaskedType
from cudf.core.udf.strings_typing import string_view
from cudf.core.udf.templates import (
    masked_input_initializer_template,
    scalar_kernel_template,
    unmasked_input_initializer_template,
)
from cudf.core.udf.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.utils import (
    _mask_get,
)


class SeriesApplyKernel(ApplyKernelBase):
    """
    Class representing a kernel that computes the result of
    a Series.apply operation. Expects that the user passed
    a function that operates on an single element of the Series,
    for example

    def f(x):
        return x + 1
    """

    @property
    def kernel_type(self):
        return "series_apply"

    def _get_frame_type(self):
        return MaskedType(
            string_view
            if is_dtype_obj_string(self.frame.dtype)
            else numpy_support.from_dtype(self.frame.dtype)
        )

    def _get_kernel_string(self):
        extra_args = ", ".join(
            [f"extra_arg_{i}" for i in range(len(self.args))]
        )

        masked_initializer = (
            masked_input_initializer_template
            if self.frame._column.mask
            else unmasked_input_initializer_template
        ).format(idx=0)

        return scalar_kernel_template.format(
            extra_args=extra_args, masked_initializer=masked_initializer
        )

    @cache
    def _get_kernel_string_exec_context(self):
        # This is the global execution context that will be used
        # to compile the kernel. It contains the function being
        # compiled and the cuda module.
        return {
            "cuda": cuda,
            "Masked": Masked,
            "_mask_get": _mask_get,
            "pack_return": pack_return,
        }
