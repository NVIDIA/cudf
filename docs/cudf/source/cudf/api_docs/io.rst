.. _api.io:

============
Input/output
============
.. currentmodule:: cudf

CSV
~~~
.. autosummary::
   :toctree: api/

   read_csv
   DataFrame.to_csv

Text
~~~~
.. autosummary::
   :toctree: api/

   read_text

JSON
~~~~
.. autosummary::
   :toctree: api/

   read_json
   DataFrame.to_json

Parquet
~~~~~~~
.. autosummary::
   :toctree: api/

   read_parquet
   DataFrame.to_parquet
   io.parquet.read_parquet_metadata
   io.parquet.ParquetDatasetWriter
   io.parquet.ParquetDatasetWriter.close
   io.parquet.ParquetDatasetWriter.write_table


ORC
~~~
.. autosummary::
   :toctree: api/

   read_orc
   DataFrame.to_orc

HDFStore: PyTables (HDF5)
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_hdf
   DataFrame.to_hdf

.. warning::

   HDF reader and writers are not GPU accelerated. These currently use CPU via Pandas.
   This may be GPU accelerated in the future.

Feather
~~~~~~~
.. autosummary::
   :toctree: api/

   read_feather
   DataFrame.to_feather

.. warning::

   Feather reader and writers are not GPU accelerated. These currently use CPU via Pandas.
   This may be GPU accelerated in the future.

Avro
~~~~
.. autosummary::
   :toctree: api/

   read_avro
