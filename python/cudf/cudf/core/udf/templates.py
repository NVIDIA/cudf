# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel string templates shared by both backends."""

unmasked_input_initializer_template = """\
        d_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i + offset_{idx}], True)
"""

masked_input_initializer_template = """\
        d_{idx}, m_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i + offset_{idx}], _mask_get(m_{idx}, i + offset_{idx}))
"""

row_initializer_template = """\
        row[_col_names[{idx}]] = masked_{idx}
"""

group_initializer_template = """\
        arr_{idx} = input_col_{idx}[offset[block_id]:offset[block_id+1]]
        idx_{idx} = index[offset[block_id]:offset[block_id+1]]
        dataframe_group[_col_names[{idx}]] = CooperativeArray(arr_{idx}, size, idx_{idx})
"""

scalar_kernel_template = """
def _kernel(retval, size, input_col_0, offset_0, {extra_args}):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval

    if i < size:

{masked_initializer}

        ret = f_(masked_0, {extra_args})

        ret_masked = pack_return(ret)
        ret_data_arr[i] = ret_masked.value
        ret_mask_arr[i] = ret_masked.valid
"""
