# cudf.pandas
The use of the cuDF pandas accelerator mode (`cudf.pandas`) is explained [in the user guide](index.rst).
The purpose of this document is to explain how the fast-slow proxy mechanism works and document internal environment variables that can be used to debug `cudf.pandas` itself.

## fast-slow proxy mechanism
The core of `cudf.pandas` is implemented through proxy types defined in [`fast_slow_proxy.py`](https://github.com/NVIDIA/cudf/blob/5f45803b2a68b49d330d94e2f701791a7590612a/python/cudf/cudf/pandas/fast_slow_proxy.py), which link a pair of "fast" and "slow" libraries.
`cudf.pandas` works by wrapping each "slow" type and its corresponding "fast" type in a new proxy type, also known as a fast-slow proxy type.
The purpose of these proxy types is so we can first attempt computations on the fast object, and then fall back to the slow object if the fast version fails.
While the core wrapping functionality is generic, the current usage mainly involves providing a proxy pair using cuDF and Pandas.
In the rest of this document, to maintain a concrete pair of libraries in mind, we use cuDF and Pandas interchangeably as names for the "fast" and "slow" libraries, respectively, with the understanding that any pair of API-matching libraries could be used.
For example, future support could include pairs such as CuPy (as the "fast" library) and NumPy (as the "slow" library).

```{note}
1. We currently do not wrap the entire NumPy library because it exposes a C API. But we do wrap NumPy's `numpy.ndarray` and CuPy's `cupy.ndarray` in a proxy type.
2. There is a `custom_iter` method defined to always utilize slow objects `iter` method, that way we don't move the objects to GPU and trigger an error and again move the object to CPU to execute the iteration successfully.
```

### Types:
#### Wrapped Types and Proxy Types
The "wrapped" types/classes are the Pandas and cuDF specific types that have been wrapped into proxy types.
Wrapped objects and proxy objects are instances of wrapped types and proxy types, respectively.
In the snippet below `s1` and `s2` are wrapped objects and `s3` is a fast-slow proxy object.
Also note that the module `xpd` is a wrapped module and contains cuDF and Pandas modules as attributes.
To check if an object is a proxy type, we can use `cudf.pandas.is_proxy_object`.
  ```python
  import cudf.pandas
  cudf.pandas.install()
  import pandas as xpd

  cudf = xpd._fsproxy_fast
  pd = xpd._fsproxy_slow

  s1 = cudf.Series([1,2])
  s2 = pd.Series([1,2])
  s3 = xpd.Series([1,2])

  from cudf.pandas import is_proxy_object

  is_proxy_object(s1) # returns False

  is_proxy_object(s2) # returns False

  is_proxy_object(s3) # returns True
  ```

```{note}
Note that users should never have to interact with the wrapped objects directly in this way.
This code is purely for demonstrative purposes.
```

#### The Different Kinds of Proxy Types
In `cudf.pandas`, there are two main kinds of proxy types: final types and intermediate types.

##### Final and Intermediate Proxy Types
Final types are types for which known operations exist for converting an object of a "fast" type to a "slow" type and vice versa.
For example, `cudf.DataFrame` can be converted to Pandas using the method `to_pandas`, and `pd.DataFrame` can be converted to cuDF using the function `cudf.from_pandas`.
Intermediate types are the types of the results of operations invoked on final types.
For example, `xpd.DataFrameGroupBy` is an intermediate type that will be created during a groupby operation on the final type `xpd.DataFrame`.

##### Attributes and Callable Proxy Types
Final proxy types are typically classes or modules, both of which have attributes.
Classes also have methods.
These attributes and methods must be wrapped as well to support the fast-slow proxy scheme.

#### Creating New Proxy Types
`_FinalProxy` and `_IntermediateProxy` types are created using the functions `make_final_proxy_type` and `make_intermediate_proxy` type, respectively.
Creating a new final type looks like this.

```python
DataFrame = make_final_proxy_type(
    "DataFrame",
    cudf.DataFrame,
    pd.DataFrame,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
)
```

### The Fallback Mechanism
Proxied calls are implemented with fallback via [`_fast_slow_function_call`](https://github.com/NVIDIA/cudf/blob/57aeeb78d85e169ac18b82f51d2b1cbd01b0608d/python/cudf/cudf/pandas/fast_slow_proxy.py#L869). This implements the mechanism by which we attempt operations the fast way (using cuDF) and then fall back to the slow way (using Pandas) on failure.
The function looks like this:
```python
def _fast_slow_function_call(func: Callable, *args, **kwargs):
    try:
        ...
        fast_args, fast_kwargs = _fast_arg(args), _fast_arg(kwargs)
        result = func(*fast_args, **fast_kwargs)
        ...
    except Exception:
        ...
        slow_args, slow_kwargs = _slow_arg(args), _slow_arg(kwargs)
        result = func(*slow_args, **slow_kwargs)
        ...
    return _maybe_wrap_result(result, func, *args, **kwargs), fast
```
As we can see the function attempts to call `func` the fast way using cuDF and if any `Exception` occurs, it calls the function using Pandas.
In essence, this `try-except` is what allows `cudf.pandas` to support the bulk of the Pandas API.

At the end, the function wraps the result from either path in a fast-slow proxy object, if necessary.

#### Converting Proxy Objects
Note that before the `func` is called, the proxy object and its attributes need to be converted to either their cuDF or Pandas implementations.
This conversion is handled in the function `_transform_arg` which both `_fast_arg` and `_slow_arg` call.

`_transform_arg` is a recursive function that will call itself depending on the type or argument passed to it (eg. `_transform_arg` is called for each element in a list of arguments).

### Using Metaclasses
`cudf.pandas` uses a [metaclass](https://docs.python.org/3/glossary.html#term-metaclass) called (`_FastSlowProxyMeta`) to find class attributes and classmethods of fast-slow proxy types.
For example, in the snippet below, the `xpd.Series` type is an instance of `_FastSlowProxyMeta`.
Therefore we can access the property `_fsproxy_fast` defined in the metaclass.
```python
import cudf.pandas
cudf.pandas.install()
import pandas as xpd

print(xpd.Series._fsproxy_fast) # output is cudf.core.series.Series
```

## debugging `cudf.pandas`
Several environment variables are available for debugging purposes.

Setting the environment variable `CUDF_PANDAS_DEBUGGING` produces a warning when the results from cuDF and Pandas differ from one another.
For example, the snippet below produces the warning below.
```python
import cudf.pandas
cudf.pandas.install()
import pandas as pd
import numpy as np

setattr(pd.Series.mean, "_fsproxy_slow", lambda self, *args, **kwargs: np.float64(1))
s = pd.Series([1,2,3])
s.mean()
```
```
UserWarning: The results from cudf and pandas were different. The exception was
Arrays are not almost equal to 7 decimals
 ACTUAL: 1.0
 DESIRED: 2.0.
```

Setting the environment variable `CUDF_PANDAS_FAIL_ON_FALLBACK` causes `cudf.pandas` to fail when falling back from cuDF to Pandas.
For example,
```python
import cudf.pandas
cudf.pandas.install()
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'complex_col': [1 + 2j, 3 + 4j, 5 + 6j]
})

print(df)
```
```
ProxyFallbackError: The operation failed with cuDF, the reason was <class 'NotImplementedError'>: Series with Complex128DType is not supported.
```
