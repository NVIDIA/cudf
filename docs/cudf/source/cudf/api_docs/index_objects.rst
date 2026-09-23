=============
Index objects
=============

Index
-----
.. currentmodule:: cudf

**Many of these methods or variants thereof are available on the objects
that contain an index (Series/DataFrame) and those should most likely be
used before calling these methods directly.**

.. autosummary::
   :toctree: api/

   Index

Properties
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.dtype
   Index.duplicated
   Index.empty
   Index.has_duplicates
   Index.hasnans
   Index.inferred_type
   Index.is_monotonic_increasing
   Index.is_monotonic_decreasing
   Index.is_unique
   Index.name
   Index.names
   Index.ndim
   Index.nlevels
   Index.shape
   Index.size
   Index.transpose
   Index.T
   Index.values

Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.all
   Index.any
   Index.copy
   Index.drop_duplicates
   Index.equals
   Index.factorize
   Index.min
   Index.max
   Index.rename
   Index.repeat
   Index.where
   Index.take
   Index.unique
   Index.nunique

Compatibility with MultiIndex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.set_names

Missing values
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.fillna
   Index.dropna
   Index.isna
   Index.isnull
   Index.notna
   Index.notnull

Memory usage
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.memory_usage

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.astype
   Index.deserialize
   Index.device_deserialize
   Index.device_serialize
   Index.host_deserialize
   Index.host_serialize
   Index.serialize
   Index.tolist
   Index.to_arrow
   Index.to_cupy
   Index.to_list
   Index.to_numpy
   Index.to_series
   Index.to_frame
   Index.to_pandas
   Index.to_dlpack
   Index.to_pylibcudf
   Index.from_pylibcudf
   Index.from_arrow

Sorting
~~~~~~~
.. autosummary::
   :toctree: api/

   Index.argsort
   Index.find_label_range
   Index.searchsorted
   Index.sort_values

Time-specific operations
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.shift

Combining / joining / set operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.append
   Index.union
   Index.intersection
   Index.join
   Index.difference

Selecting
~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.get_indexer
   Index.get_level_values
   Index.get_loc
   Index.get_slice_bound
   Index.isin

String Operations
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.str

.. _api.numericindex:

Numeric Index
-------------
.. autosummary::
   :toctree: api/

   RangeIndex
   RangeIndex.start
   RangeIndex.stop
   RangeIndex.step
   RangeIndex.to_numpy
   RangeIndex.to_arrow

.. _api.categoricalindex:

CategoricalIndex
----------------
.. autosummary::
   :toctree: api/

   CategoricalIndex

Categorical components
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CategoricalIndex.codes
   CategoricalIndex.categories

Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CategoricalIndex.equals

.. _api.intervalindex:

IntervalIndex
-------------
.. autosummary::
   :toctree: api/

   IntervalIndex

IntervalIndex components
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   IntervalIndex.from_breaks
   IntervalIndex.values
   IntervalIndex.get_indexer
   IntervalIndex.get_loc

.. _api.multiindex:

MultiIndex
----------
.. autosummary::
   :toctree: api/

   MultiIndex

MultiIndex constructors
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.from_arrays
   MultiIndex.from_tuples
   MultiIndex.from_product
   MultiIndex.from_frame
   MultiIndex.from_arrow

MultiIndex properties
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.names
   MultiIndex.levels
   MultiIndex.codes
   MultiIndex.dtypes
   MultiIndex.nlevels

MultiIndex components
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.to_flat_index
   MultiIndex.to_frame
   MultiIndex.droplevel
   MultiIndex.swaplevel

MultiIndex selecting
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.get_indexer
   MultiIndex.get_loc
   MultiIndex.get_level_values

.. _api.datetimeindex:

DatetimeIndex
-------------
.. autosummary::
   :toctree: api/

   DatetimeIndex

Time/date components
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DatetimeIndex.year
   DatetimeIndex.month
   DatetimeIndex.day
   DatetimeIndex.hour
   DatetimeIndex.minute
   DatetimeIndex.second
   DatetimeIndex.microsecond
   DatetimeIndex.nanosecond
   DatetimeIndex.day_of_year
   DatetimeIndex.dayofyear
   DatetimeIndex.dayofweek
   DatetimeIndex.weekday
   DatetimeIndex.quarter
   DatetimeIndex.is_leap_year

   DatetimeIndex.isocalendar

Time-specific operations
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DatetimeIndex.round
   DatetimeIndex.ceil
   DatetimeIndex.floor
   DatetimeIndex.tz_convert
   DatetimeIndex.tz_localize
   DatetimeIndex.indexer_between_time

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DatetimeIndex.to_series
   DatetimeIndex.to_frame

TimedeltaIndex
--------------
.. autosummary::
   :toctree: api/

   TimedeltaIndex

Components
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   TimedeltaIndex.days
   TimedeltaIndex.seconds
   TimedeltaIndex.microseconds
   TimedeltaIndex.nanoseconds
   TimedeltaIndex.components
   TimedeltaIndex.inferred_freq

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   TimedeltaIndex.to_series
   TimedeltaIndex.to_frame
