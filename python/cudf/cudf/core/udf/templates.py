# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
        dataframe_group[_col_names[{idx}]] = Group(arr_{idx}, size, arr_index)
"""

row_kernel_template = """\
def _kernel(retval, size, {input_columns}, {input_offsets}, {extra_args}):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval
    if i < size:
        # Create a structured array with the desired fields
        rows = cuda.local.array(1, dtype=row_type)

        # one element of that array
        row = rows[0]

{masked_input_initializers}
{row_initializers}

        # pass the assembled row into the udf
        ret = f_(row, {extra_args})

        # pack up the return values and set them
        ret_masked = pack_return(ret)
        ret_data_arr[i] = ret_masked.value
        ret_mask_arr[i] = ret_masked.valid
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

groupby_apply_kernel_template = """
def _kernel(offset, out, index, {input_columns}, {extra_args}):
    tid = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    tb_size = cuda.blockDim.x

    recarray = cuda.local.array(1, dtype=dataframe_group_type)
    dataframe_group = recarray[0]

    if block_id < (len(offset) - 1):

        size = offset[block_id+1] - offset[block_id]
        arr_index = index[offset[block_id]:offset[block_id+1]]

{group_initializers}

        result = f_(dataframe_group, {extra_args})
        if cuda.threadIdx.x == 0:
                out[block_id] = result
"""
