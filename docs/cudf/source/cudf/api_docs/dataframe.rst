=========
DataFrame
=========
.. currentmodule:: cudf

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Axes**

.. autosummary::
   :toctree: api/

   DataFrame.axes
   DataFrame.index
   DataFrame.columns

.. autosummary::
   :toctree: api/

   DataFrame.dtypes
   DataFrame.info
   DataFrame.select_dtypes
   DataFrame.values
   DataFrame.ndim
   DataFrame.size
   DataFrame.shape
   DataFrame.memory_usage
   DataFrame.empty

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.astype
   DataFrame.convert_dtypes
   DataFrame.copy

Indexing, iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.head
   DataFrame.at
   DataFrame.iat
   DataFrame.loc
   DataFrame.iloc
   DataFrame.insert
   DataFrame.__iter__
   DataFrame.items
   DataFrame.keys
   DataFrame.iterrows
   DataFrame.itertuples
   DataFrame.pop
   DataFrame.tail
   DataFrame.isin
   DataFrame.squeeze
   DataFrame.where
   DataFrame.mask
   DataFrame.query

Binary operator functions
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.add
   DataFrame.sub
   DataFrame.subtract
   DataFrame.mul
   DataFrame.multiply
   DataFrame.truediv
   DataFrame.div
   DataFrame.divide
   DataFrame.floordiv
   DataFrame.mod
   DataFrame.pow
   DataFrame.dot
   DataFrame.radd
   DataFrame.rsub
   DataFrame.rmul
   DataFrame.rdiv
   DataFrame.rtruediv
   DataFrame.rfloordiv
   DataFrame.rmod
   DataFrame.rpow
   DataFrame.lt
   DataFrame.gt
   DataFrame.le
   DataFrame.ge
   DataFrame.ne
   DataFrame.eq

Function application, GroupBy & window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.agg
   DataFrame.apply
   DataFrame.groupby
   DataFrame.map
   DataFrame.pipe
   DataFrame.rolling

.. _api.dataframe.stats:

Computations / descriptive stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.abs
   DataFrame.all
   DataFrame.any
   DataFrame.clip
   DataFrame.corr
   DataFrame.count
   DataFrame.cov
   DataFrame.cummax
   DataFrame.cummin
   DataFrame.cumprod
   DataFrame.cumsum
   DataFrame.describe
   DataFrame.diff
   DataFrame.eval
   DataFrame.ewm
   DataFrame.kurt
   DataFrame.kurtosis
   DataFrame.max
   DataFrame.mean
   DataFrame.median
   DataFrame.min
   DataFrame.mode
   DataFrame.pct_change
   DataFrame.prod
   DataFrame.product
   DataFrame.quantile
   DataFrame.rank
   DataFrame.round
   DataFrame.scale
   DataFrame.skew
   DataFrame.sum
   DataFrame.std
   DataFrame.var
   DataFrame.nunique
   DataFrame.value_counts

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.add_prefix
   DataFrame.add_suffix
   DataFrame.drop
   DataFrame.drop_duplicates
   DataFrame.duplicated
   DataFrame.equals
   DataFrame.head
   DataFrame.reindex
   DataFrame.rename
   DataFrame.reset_index
   DataFrame.sample
   DataFrame.searchsorted
   DataFrame.set_index
   DataFrame.repeat
   DataFrame.tail
   DataFrame.take
   DataFrame.tile
   DataFrame.truncate

.. _api.dataframe.missing:

Missing data handling
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.bfill
   DataFrame.dropna
   DataFrame.ffill
   DataFrame.fillna
   DataFrame.interpolate
   DataFrame.isna
   DataFrame.isnull
   DataFrame.nans_to_nulls
   DataFrame.notna
   DataFrame.notnull
   DataFrame.replace

Reshaping, sorting, transposing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.argsort
   DataFrame.interleave_columns
   DataFrame.partition_by_hash
   DataFrame.pivot
   DataFrame.pivot_table
   DataFrame.scatter_by_map
   DataFrame.sort_values
   DataFrame.sort_index
   DataFrame.nlargest
   DataFrame.nsmallest
   DataFrame.swaplevel
   DataFrame.stack
   DataFrame.unstack
   DataFrame.melt
   DataFrame.explode
   DataFrame.to_struct
   DataFrame.T
   DataFrame.transpose

Combining / comparing / joining / merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.assign
   DataFrame.join
   DataFrame.merge
   DataFrame.update

Time Series-related
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.shift
   DataFrame.resample


Metadata
~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.attrs
   DataFrame.flags
   DataFrame.set_flags

Serialization / IO / conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.deserialize
   DataFrame.device_deserialize
   DataFrame.device_serialize
   DataFrame.from_arrow
   DataFrame.from_dict
   DataFrame.from_records
   DataFrame.hash_values
   DataFrame.host_deserialize
   DataFrame.host_serialize
   DataFrame.serialize
   DataFrame.to_arrow
   DataFrame.to_dict
   DataFrame.to_dlpack
   DataFrame.to_parquet
   DataFrame.to_csv
   DataFrame.to_cupy
   DataFrame.to_hdf
   DataFrame.to_json
   DataFrame.to_numpy
   DataFrame.to_pandas
   DataFrame.to_feather
   DataFrame.to_records
   DataFrame.to_string
   DataFrame.to_pylibcudf
   DataFrame.from_pylibcudf
