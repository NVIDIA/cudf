(memory-profiling-user-doc)=

# Memory Profiling

Peak memory usage is a common concern in GPU programming because GPU memory is typically smaller than available CPU memory. To easily identify memory hotspots, cuDF provides a memory profiler. It comes with an overhead so avoid using it in performance-sensitive code.

## Enabling Memory Profiling

First, enable memory profiling in RMM by calling {py:func}`rmm.statistics.enable_statistics()`. This adds a statistics resource adaptor to the current RMM memory resource, which enables cuDF to access memory profiling information. See the [RMM documentation](https://docs.rapids.ai/api/rmm/stable/user_guide/guide/#memory-statistics-and-profiling) for more details.

Second, enable memory profiling in cuDF by setting the `memory_profiling` option to `True`. Use {py:func}`cudf.set_option` or set the environment variable ``CUDF_MEMORY_PROFILING=1`` prior to the launch of the Python interpreter.

To get the result of the profiling, use {py:func}`cudf.utils.performance_tracking.print_memory_report` or access the raw profiling data by using: {py:func}`cudf.utils.performance_tracking.get_memory_records`.

### Example
In the following, we enable profiling, do some work, and then print the profiling results:

```python
>>> import cudf
>>> from cudf.utils.performance_tracking import print_memory_report
>>> from rmm.statistics import enable_statistics
>>> enable_statistics()
>>> cudf.set_option("memory_profiling", True)
>>> cudf.DataFrame({"a": [1, 2, 3]})  # Some work
   a
0  1
1  2
2  3
>>> print_memory_report()  # Pretty print the result of the profiling
Memory Profiling
================

Legends:
ncalls       - number of times the function or code block was called
memory_peak  - peak memory allocated in function or code block (in bytes)
memory_total - total memory allocated in function or code block (in bytes)

Ordered by: memory_peak

ncalls memory_peak memory_total filename:lineno(function)
     1          32           32 cudf/core/dataframe.py:690(DataFrame.__init__)
     2           0            0 cudf/core/index.py:214(RangeIndex.__init__)
     6           0            0 cudf/core/index.py:424(RangeIndex.__len__)
```
