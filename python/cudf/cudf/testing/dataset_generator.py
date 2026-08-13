# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This module is for generating "synthetic" datasets. It was originally
# designed for testing filtered reading. Generally, it should be useful
# if you want to generate data where certain phenomena (e.g., cardinality)
# are exaggerated.

import copy
import string
import uuid
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

import cudf
from cudf.utils.dtypes import cudf_dtype_to_pa_type


class ColumnParameters:
    """Parameters for generating column of data

    Attributes
    ----------
    cardinality : int or None
        Size of a random set of values that generated data is sampled from.
        The values in the random set are derived from the given generator.
        If cardinality is None, the Iterable returned by the given generator
        is invoked for each value to be generated.
    null_frequency : 0.1
        Probability of a generated value being null
    generator : Callable
        Function for generating random data.
    is_sorted : bool
        Sort this column. Columns are sorted in same order as ColumnParameters
        instances stored in column_params of Parameters. If there are one or
        more columns marked as sorted, the generated PyArrow Table will be
        converted to a Pandas DataFrame to do the sorting. This may implicitly
        convert numbers to floats in the presence of nulls.
    dtype : optional
        a numpy dtype to control the format of the data
    """

    def __init__(
        self,
        cardinality=100,
        null_frequency=0.1,
        generator=None,
        is_sorted=True,
        dtype=None,
    ):
        self.cardinality = cardinality
        self.null_frequency = null_frequency
        if generator is None:
            rng = np.random.default_rng(seed=0)
            self.generator = lambda: [
                _generate_string(
                    string.ascii_letters, rng, rng.integers(4, 8).item()
                )
                for _ in range(100)
            ]
        else:
            self.generator = generator
        self.is_sorted = is_sorted
        self.dtype = dtype


class Parameters:
    """Parameters for random dataset generation

    Attributes
    ----------
    num_rows : int
        Number of rows to generate
    column_parameters : List[ColumnParams]
        ColumnParams for each column
    seed : int or None, default None
        Seed for random data generation
    """

    def __init__(
        self,
        num_rows=2048,
        column_parameters=None,
        seed=None,
    ):
        self.num_rows = num_rows
        if column_parameters is None:
            column_parameters = []
        self.column_parameters = column_parameters
        self.seed = seed


def _write(tbl, path, format):
    if format["name"] == "parquet":
        if isinstance(tbl, pa.Table):
            pq.write_table(tbl, path, row_group_size=format["row_group_size"])
        elif isinstance(tbl, pd.DataFrame):
            tbl.to_parquet(path, row_group_size=format["row_group_size"])


def _generate_column(column_params, num_rows, rng):
    # If cardinality is specified, we create a set to sample from.
    # Otherwise, we simply use the given generator to generate each value.

    if column_params.cardinality is not None:
        # Construct set of values to sample from where
        # set size = cardinality

        if (
            isinstance(column_params.dtype, str)
            and column_params.dtype == "category"
        ):
            vals = pa.array(
                column_params.generator,
                size=column_params.cardinality,
                safe=False,
            )
            return pa.DictionaryArray.from_arrays(
                dictionary=vals,
                indices=rng.integers(low=0, high=len(vals), size=num_rows),
                mask=rng.choice(
                    [True, False],
                    size=num_rows,
                    p=[
                        column_params.null_frequency,
                        1 - column_params.null_frequency,
                    ],
                )
                if column_params.null_frequency > 0.0
                else None,
            )

        if hasattr(column_params.dtype, "to_arrow"):
            arrow_type = column_params.dtype.to_arrow()
        elif column_params.dtype is not None:
            arrow_type = cudf_dtype_to_pa_type(cudf.dtype(column_params.dtype))
        else:
            arrow_type = None

        is_arrow_type_decimal = isinstance(
            arrow_type, (pa.Decimal128Type, pa.Decimal64Type, pa.Decimal32Type)
        )

        if isinstance(column_params.dtype, cudf.StructDtype):
            vals = pa.StructArray.from_arrays(
                column_params.generator,
                names=column_params.dtype.fields.keys(),
                mask=pa.array(
                    rng.choice(
                        [True, False],
                        size=num_rows,
                        p=[
                            column_params.null_frequency,
                            1 - column_params.null_frequency,
                        ],
                    )
                )
                if column_params.null_frequency > 0.0
                else None,
            )
            return vals
        elif not is_arrow_type_decimal:
            vals = pa.array(
                column_params.generator,
                size=column_params.cardinality,
                safe=False,
                type=arrow_type,
            )
        vals = pa.array(
            rng.choice(column_params.generator, size=num_rows)
            if is_arrow_type_decimal
            else rng.choice(vals, size=num_rows),
            mask=rng.choice(
                [True, False],
                size=num_rows,
                p=[
                    column_params.null_frequency,
                    1 - column_params.null_frequency,
                ],
            )
            if column_params.null_frequency > 0.0
            else None,
            size=num_rows,
            safe=False,
            type=None if is_arrow_type_decimal else arrow_type,
        )
        if is_arrow_type_decimal:
            vals = vals.cast(arrow_type, safe=False)
        return vals
    else:
        # Generate data for current column
        return pa.array(
            column_params.generator,
            mask=rng.choice(
                [True, False],
                size=num_rows,
                p=[
                    column_params.null_frequency,
                    1 - column_params.null_frequency,
                ],
            )
            if column_params.null_frequency > 0.0
            else None,
            size=num_rows,
            safe=False,
        )


def generate(
    path,
    parameters,
    format=None,
    use_threads=True,
):
    """
    Generate dataset using given parameters and write to given format

    Parameters
    ----------
    path : str or file-like object
        Path to write to
    parameters : Parameters
        Parameters specifying how to randomly generate data
    format : Dict
        Format to write
    """
    if format is None:
        format = {"name": "parquet", "row_group_size": 64}
    df = get_dataframe(parameters, use_threads)

    # Write
    _write(df, path, format)


def get_dataframe(parameters, use_threads):
    # Initialize seeds
    if parameters.seed is not None:
        rng = np.random.default_rng(seed=parameters.seed)
    else:
        rng = np.random.default_rng(seed=0)

    # For each column, invoke the data generator
    for column_params in parameters.column_parameters:
        column_params.generator = column_params.generator()

    # Get schema for each column
    table_fields = []
    for i, column_params in enumerate(parameters.column_parameters):
        if (
            isinstance(column_params.dtype, str)
            and column_params.dtype == "category"
        ):
            arrow_type = pa.dictionary(
                index_type=pa.int64(),
                value_type=cudf_dtype_to_pa_type(
                    cudf.dtype(type(next(iter(column_params.generator))))
                ),
            )
        elif hasattr(column_params.dtype, "to_arrow"):
            arrow_type = column_params.dtype.to_arrow()
        else:
            arrow_type = cudf_dtype_to_pa_type(
                cudf.dtype(type(next(iter(column_params.generator))))
                if column_params.dtype is None
                else column_params.dtype
            )
        table_fields.append(
            pa.field(
                name=str(i),
                type=arrow_type,
                nullable=column_params.null_frequency > 0,
            )
        )

    schema = pa.schema(table_fields)

    # Initialize column data and which columns should be sorted
    column_data = [None] * len(parameters.column_parameters)
    columns_to_sort = [
        str(i)
        for i, column_params in enumerate(parameters.column_parameters)
        if column_params.is_sorted
    ]
    # Generate data
    if not use_threads:
        for i, column_params in enumerate(parameters.column_parameters):
            column_data[i] = _generate_column(
                column_params,
                parameters.num_rows,
                rng,
            )
    else:
        pool = Pool(pa.cpu_count())
        column_data = pool.starmap(
            _generate_column,
            [
                (column_params, parameters.num_rows, rng)
                for i, column_params in enumerate(parameters.column_parameters)
            ],
        )
        pool.close()
        pool.join()
    # Convert to Pandas DataFrame and sort columns appropriately
    tbl = pa.Table.from_arrays(
        column_data,
        schema=schema,
    )
    if columns_to_sort:
        tbl = tbl.to_pandas()
        tbl = tbl.sort_values(columns_to_sort)
        tbl = pa.Table.from_pandas(tbl, schema)
    return tbl


def rand_dataframe(dtypes_meta, rows, seed=0, use_threads=True):
    """
    Generates a random table.

    Parameters
    ----------
    dtypes_meta : List of dict
        Specifies list of dtype meta data. dtype meta data should
        be a dictionary of the form example:
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10}
        `"str"` dtype can contain an extra key `max_string_length` to
        control the maximum size of the strings being generated in each row.
        If not specified, it will default to 1000.
    rows : int
        Specifies the number of rows to be generated.
    seed : int
        Specifies the `seed` value to be utilized by all downstream
        random data generation APIs.
    use_threads : bool
        Indicates whether to use threads pools to build the columns

    Returns
    -------
    PyArrow Table
        A Table with columns of corresponding dtypes mentioned in `dtypes_meta`
    """
    # Apply seed
    rng = np.random.default_rng(seed=seed)

    column_params = []
    for meta in dtypes_meta:
        dtype = copy.deepcopy(meta["dtype"])
        null_frequency = copy.deepcopy(meta["null_frequency"])
        cardinality = copy.deepcopy(meta["cardinality"])

        if dtype == "list":
            lists_max_length = meta["lists_max_length"]
            nesting_max_depth = meta["nesting_max_depth"]
            value_type = meta["value_type"]
            nesting_depth = rng.integers(1, nesting_max_depth)

            dtype = cudf.core.dtypes.ListDtype(value_type)

            # Determining the `dtype` from the `value_type`
            # and the nesting_depth
            i = 1
            while i < nesting_depth:
                dtype = cudf.core.dtypes.ListDtype(dtype)
                i += 1

            column_params.append(
                ColumnParameters(
                    cardinality=cardinality,
                    null_frequency=null_frequency,
                    generator=list_generator(
                        dtype=value_type,
                        size=cardinality,
                        nesting_depth=nesting_depth,
                        lists_max_length=lists_max_length,
                        rng=rng,
                    ),
                    is_sorted=False,
                    dtype=dtype,
                )
            )
        elif dtype == "struct":
            nesting_max_depth = meta["nesting_max_depth"]
            max_types_at_each_level = meta["max_types_at_each_level"]
            max_null_frequency = meta["max_null_frequency"]
            nesting_depth = rng.integers(1, nesting_max_depth)
            structDtype = create_nested_struct_type(
                max_types_at_each_level=max_types_at_each_level,
                nesting_level=nesting_depth,
                rng=rng,
            )

            column_params.append(
                ColumnParameters(
                    cardinality=cardinality,
                    null_frequency=null_frequency,
                    generator=struct_generator(
                        dtype=structDtype,
                        cardinality=cardinality,
                        size=rows,
                        max_null_frequency=max_null_frequency,
                        rng=rng,
                    ),
                    is_sorted=False,
                    dtype=structDtype,
                )
            )
        elif dtype == "decimal64":
            max_precision = meta.get(
                "max_precision", cudf.Decimal64Dtype.MAX_PRECISION
            )
            precision = rng.integers(1, max_precision)
            scale = rng.integers(0, precision)
            dtype = cudf.Decimal64Dtype(precision=precision, scale=scale)
            column_params.append(
                ColumnParameters(
                    cardinality=cardinality,
                    null_frequency=null_frequency,
                    generator=decimal_generator(
                        dtype=dtype, size=cardinality, rng=rng
                    ),
                    is_sorted=False,
                    dtype=dtype,
                )
            )
        elif dtype == "decimal32":
            max_precision = meta.get(
                "max_precision", cudf.Decimal32Dtype.MAX_PRECISION
            )
            precision = rng.integers(1, max_precision)
            scale = rng.integers(0, precision)
            dtype = cudf.Decimal32Dtype(precision=precision, scale=scale)
            column_params.append(
                ColumnParameters(
                    cardinality=cardinality,
                    null_frequency=null_frequency,
                    generator=decimal_generator(
                        dtype=dtype, size=cardinality, rng=rng
                    ),
                    is_sorted=False,
                    dtype=dtype,
                )
            )
        elif dtype == "decimal128":
            max_precision = meta.get(
                "max_precision", cudf.Decimal128Dtype.MAX_PRECISION
            )
            precision = rng.integers(1, max_precision)
            scale = rng.integers(0, precision)
            dtype = cudf.Decimal128Dtype(precision=precision, scale=scale)
            column_params.append(
                ColumnParameters(
                    cardinality=cardinality,
                    null_frequency=null_frequency,
                    generator=decimal_generator(
                        dtype=dtype, size=cardinality, rng=rng
                    ),
                    is_sorted=False,
                    dtype=dtype,
                )
            )
        elif dtype == "category":
            column_params.append(
                ColumnParameters(
                    cardinality=cardinality,
                    null_frequency=null_frequency,
                    generator=lambda cardinality=cardinality: [
                        _unique_string() for _ in range(cardinality)
                    ],
                    is_sorted=False,
                    dtype="category",
                )
            )
        else:
            dtype = cudf.dtype(dtype)
            if dtype.kind in ("i", "u"):
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=int_generator(
                            dtype=dtype,
                            size=cardinality,
                            min_bound=meta.get("min_bound", None),
                            max_bound=meta.get("max_bound", None),
                            rng=rng,
                        ),
                        is_sorted=False,
                        dtype=dtype,
                    )
                )
            elif dtype.kind == "f":
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=float_generator(
                            dtype=dtype,
                            size=cardinality,
                            min_bound=meta.get("min_bound", None),
                            max_bound=meta.get("max_bound", None),
                            rng=rng,
                        ),
                        is_sorted=False,
                        dtype=dtype,
                    )
                )
            elif dtype.kind in ("U", "O"):
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=lambda cardinality=cardinality: [
                            _generate_string(
                                string.printable,
                                rng,
                                rng.integers(
                                    low=0,
                                    high=meta.get("max_string_length", 1000),
                                    size=1,
                                )[0],
                            )
                            for _ in range(cardinality)
                        ],
                        is_sorted=False,
                        dtype=dtype,
                    )
                )
            elif dtype.kind == "M":
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=datetime_generator(
                            dtype=dtype,
                            size=cardinality,
                            min_bound=meta.get("min_bound", None),
                            max_bound=meta.get("max_bound", None),
                            rng=rng,
                        ),
                        is_sorted=False,
                        dtype=cudf.dtype(dtype),
                    )
                )
            elif dtype.kind == "m":
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=timedelta_generator(
                            dtype=dtype,
                            size=cardinality,
                            min_bound=meta.get("min_bound", None),
                            max_bound=meta.get("max_bound", None),
                            rng=rng,
                        ),
                        is_sorted=False,
                        dtype=cudf.dtype(dtype),
                    )
                )
            elif dtype.kind == "b":
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=boolean_generator(cardinality, rng),
                        is_sorted=False,
                        dtype=cudf.dtype(dtype),
                    )
                )
            else:
                raise TypeError(f"Unsupported dtype: {dtype}")
            # TODO: Add List column support once
            # https://github.com/rapidsai/cudf/pull/6075
            # is merged.

    df = get_dataframe(
        Parameters(
            num_rows=rows,
            column_parameters=column_params,
            seed=seed,
        ),
        use_threads=use_threads,
    )

    return df


def int_generator(dtype, size, rng, min_bound=None, max_bound=None):
    """
    Generator for int data
    """
    if min_bound is not None and max_bound is not None:
        low, high = min_bound, max_bound
    else:
        iinfo = np.iinfo(dtype)
        low, high = iinfo.min, iinfo.max

    return lambda: rng.integers(
        low=low,
        high=high,
        size=size,
        dtype=dtype,
    )


def float_generator(dtype, size, rng, min_bound=None, max_bound=None):
    """
    Generator for float data
    """
    if min_bound is not None and max_bound is not None:
        low, high = min_bound, max_bound
        return lambda: rng.uniform(
            low=low,
            high=high,
            size=size,
        )
    else:
        finfo = np.finfo(dtype)
        return lambda: (
            rng.uniform(
                low=finfo.min / 2,
                high=finfo.max / 2,
                size=size,
            )
            * 2
        )


def datetime_generator(dtype, size, rng, min_bound=None, max_bound=None):
    """
    Generator for datetime data
    """
    if min_bound is not None and max_bound is not None:
        low, high = min_bound, max_bound
    else:
        iinfo = np.iinfo("int64")
        low, high = iinfo.min + 1, iinfo.max

    return lambda: rng.integers(
        low=np.datetime64(low, "ns").astype(dtype).astype("int"),
        high=np.datetime64(high, "ns").astype(dtype).astype("int"),
        size=size,
    )


def timedelta_generator(dtype, size, rng, min_bound=None, max_bound=None):
    """
    Generator for timedelta data
    """
    if min_bound is not None and max_bound is not None:
        low, high = min_bound, max_bound
    else:
        iinfo = np.iinfo("int64")
        low, high = iinfo.min + 1, iinfo.max

    return lambda: rng.integers(
        low=np.timedelta64(low, "ns").astype(dtype).astype("int"),
        high=np.timedelta64(high, "ns").astype(dtype).astype("int"),
        size=size,
    )


def boolean_generator(size, rng):
    """
    Generator for bool data
    """
    return lambda: rng.choice(a=[False, True], size=size)


def decimal_generator(dtype, size, rng):
    max_integral = 10 ** (dtype.precision - dtype.scale) - 1
    max_float = (10**dtype.scale - 1) if dtype.scale != 0 else 0
    return lambda: rng.uniform(
        low=-max_integral,
        high=max_integral + (max_float / 10**dtype.scale),
        size=size,
    )


def get_values_for_nested_data(dtype, rng, lists_max_length=None, size=None):
    """
    Returns list of values based on dtype.
    """
    if size is None:
        cardinality = rng.integers(0, lists_max_length)
    else:
        cardinality = size

    dtype = cudf.dtype(dtype)
    if dtype.kind in ("i", "u"):
        values = int_generator(dtype=dtype, size=cardinality, rng=rng)()
    elif dtype.kind == "f":
        values = float_generator(dtype=dtype, size=cardinality, rng=rng)()
    elif dtype.kind in ("U", "O"):
        values = [
            _generate_string(
                string.printable,
                rng,
                100,
            )
            for _ in range(cardinality)
        ]
    elif dtype.kind == "M":
        values = datetime_generator(
            dtype=dtype, size=cardinality, rng=rng
        )().astype(dtype)
    elif dtype.kind == "m":
        values = timedelta_generator(dtype=dtype, size=cardinality)().astype(
            dtype
        )
    elif dtype.kind == "b":
        values = boolean_generator(cardinality)().astype(dtype)
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

    return values


def make_lists(dtype, lists_max_length, nesting_depth, top_level_list, rng):
    """
    Helper to create random list of lists with `nesting_depth` and
    specified value type `dtype`.
    """
    nesting_depth -= 1
    if nesting_depth >= 0:
        L = rng.integers(1, lists_max_length)
        for i in range(L):
            top_level_list.append(
                make_lists(
                    dtype=dtype,
                    lists_max_length=lists_max_length,
                    nesting_depth=nesting_depth,
                    top_level_list=[],
                    rng=rng,
                )
            )
    else:
        top_level_list = get_values_for_nested_data(
            dtype=dtype,
            lists_max_length=lists_max_length,
            rng=rng,
        )
        # To ensure numpy arrays are not passed as input to
        # list constructor, returning a python list object here.
        if isinstance(top_level_list, np.ndarray):
            top_level_list = top_level_list.tolist()

    return top_level_list


def make_array_for_struct(dtype, cardinality, size, max_null_frequency, rng):
    """
    Helper to create a pa.array with `size` and `dtype`
    for a `StructArray`.
    """

    null_frequency = rng.uniform(low=0, high=max_null_frequency)
    local_cardinality = max(rng.integers(low=0, high=cardinality), 1)
    data = get_values_for_nested_data(
        dtype=dtype.type.to_pandas_dtype(), size=local_cardinality, rng=rng
    )
    vals = rng.choice(data, size=size)

    return pa.array(
        vals,
        mask=rng.choice(
            [True, False],
            size=size,
            p=[null_frequency, 1 - null_frequency],
        )
        if null_frequency > 0.0
        else None,
        size=size,
        safe=False,
        type=dtype.type,
    )


def get_nested_lists(dtype, size, nesting_depth, lists_max_length, rng):
    """
    Returns a list of nested lists with random nesting
    depth and random nested lists length.
    """
    list_of_lists = []

    while len(list_of_lists) <= size:
        list_of_lists.extend(
            make_lists(
                dtype=dtype,
                lists_max_length=lists_max_length,
                nesting_depth=nesting_depth,
                top_level_list=[],
                rng=rng,
            )
        )

    return list_of_lists


def get_nested_structs(dtype, cardinality, size, max_null_frequency, rng):
    """
    Returns a list of arrays with random data
    corresponding to the dtype provided.
    ``dtype`` here should be a ``cudf.StructDtype``
    """
    list_of_arrays = []

    for name, col_dtype in dtype.fields.items():
        if isinstance(col_dtype, cudf.StructDtype):
            result_arrays = get_nested_structs(
                col_dtype, cardinality, size, max_null_frequency, rng
            )
            result_arrays = pa.StructArray.from_arrays(
                result_arrays, names=col_dtype.fields.keys()
            )
        else:
            result_arrays = make_array_for_struct(
                dtype=dtype._typ[name],
                cardinality=cardinality,
                size=size,
                max_null_frequency=max_null_frequency,
                rng=rng,
            )
        list_of_arrays.append(result_arrays)

    return list_of_arrays


def list_generator(dtype, size, nesting_depth, lists_max_length, rng):
    """
    Generator for list data
    """
    return lambda: get_nested_lists(
        dtype=dtype,
        size=size,
        nesting_depth=nesting_depth,
        lists_max_length=lists_max_length,
        rng=rng,
    )


def struct_generator(dtype, cardinality, size, max_null_frequency, rng):
    """
    Generator for struct data
    """
    return lambda: get_nested_structs(
        dtype=dtype,
        cardinality=cardinality,
        size=size,
        max_null_frequency=max_null_frequency,
        rng=rng,
    )


def create_nested_struct_type(max_types_at_each_level, nesting_level, rng):
    dtypes_list = cudf.utils.dtypes.ALL_TYPES
    picked_types = rng.choice(list(dtypes_list), max_types_at_each_level)
    type_dict = {}
    for name, type_ in enumerate(picked_types):
        if type_ == "struct":
            type_dict[str(name)] = create_nested_struct_type(
                max_types_at_each_level, nesting_level - 1, rng
            )
        else:
            type_dict[str(name)] = cudf.dtype(type_)
    return cudf.StructDtype(type_dict)


def _generate_string(str_seq: str, rng, length: int = 10) -> str:
    return "".join(rng.choice(list(str_seq), size=length))


def _unique_string() -> str:
    return str(uuid.uuid4()).replace("-", "")
