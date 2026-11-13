# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR (numba_cuda_mlir) kernel string templates (_row_slot / _group_slot allocation)."""

row_kernel_template = """\
def _kernel(retval, size, {input_columns}, {input_offsets}, {extra_args}):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval
    if i < size:
        row = _row_slot()

{masked_input_initializers}
{row_initializers}

        ret = f_(row, {extra_args})

        ret_masked = pack_return(ret)
        ret_data_arr[i] = ret_masked.value
        ret_mask_arr[i] = ret_masked.valid
"""

groupby_apply_kernel_template = """
def _kernel(offset, out, index, {input_columns}, {extra_args}):
    tid = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    tb_size = cuda.blockDim.x

    dataframe_group = _group_slot()

    if block_id < (len(offset) - 1):

        size = offset[block_id+1] - offset[block_id]

{group_initializers}

        result = f_(dataframe_group, {extra_args})
        if cuda.threadIdx.x == 0:
                out[block_id] = result
"""
