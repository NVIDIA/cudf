================
Extension Dtypes
================
.. currentmodule:: cudf.core.dtypes

cuDF supports a number of extension dtypes that build on top of the types that pandas supports. These dtypes are not directly available in pandas, which instead relies on object dtype arrays that run at Python rather than native speeds. The following dtypes are supported:


cudf.CategoricalDtype
=====================
.. autosummary::
   :toctree: api/

   CategoricalDtype


Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

    CategoricalDtype.categories
    CategoricalDtype.construct_from_string
    CategoricalDtype.deserialize
    CategoricalDtype.device_deserialize
    CategoricalDtype.device_serialize
    CategoricalDtype.host_deserialize
    CategoricalDtype.host_serialize
    CategoricalDtype.is_dtype
    CategoricalDtype.name
    CategoricalDtype.ordered
    CategoricalDtype.serialize
    CategoricalDtype.str
    CategoricalDtype.to_pandas
    CategoricalDtype.type


cudf.Decimal32Dtype
===================
.. autosummary::
   :toctree: api/

   Decimal32Dtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   Decimal32Dtype.ITEMSIZE
   Decimal32Dtype.MAX_PRECISION
   Decimal32Dtype.deserialize
   Decimal32Dtype.device_deserialize
   Decimal32Dtype.device_serialize
   Decimal32Dtype.from_arrow
   Decimal32Dtype.host_deserialize
   Decimal32Dtype.host_serialize
   Decimal32Dtype.is_dtype
   Decimal32Dtype.itemsize
   Decimal32Dtype.precision
   Decimal32Dtype.scale
   Decimal32Dtype.serialize
   Decimal32Dtype.str
   Decimal32Dtype.to_arrow

cudf.Decimal64Dtype
===================
.. autosummary::
   :toctree: api/

   Decimal64Dtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   Decimal64Dtype.ITEMSIZE
   Decimal64Dtype.MAX_PRECISION
   Decimal64Dtype.deserialize
   Decimal64Dtype.device_deserialize
   Decimal64Dtype.device_serialize
   Decimal64Dtype.from_arrow
   Decimal64Dtype.host_deserialize
   Decimal64Dtype.host_serialize
   Decimal64Dtype.is_dtype
   Decimal64Dtype.itemsize
   Decimal64Dtype.precision
   Decimal64Dtype.scale
   Decimal64Dtype.serialize
   Decimal64Dtype.str
   Decimal64Dtype.to_arrow

cudf.Decimal128Dtype
====================
.. autosummary::
   :toctree: api/

   Decimal128Dtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   Decimal128Dtype.ITEMSIZE
   Decimal128Dtype.MAX_PRECISION
   Decimal128Dtype.deserialize
   Decimal128Dtype.device_deserialize
   Decimal128Dtype.device_serialize
   Decimal128Dtype.from_arrow
   Decimal128Dtype.host_deserialize
   Decimal128Dtype.host_serialize
   Decimal128Dtype.is_dtype
   Decimal128Dtype.itemsize
   Decimal128Dtype.precision
   Decimal128Dtype.scale
   Decimal128Dtype.serialize
   Decimal128Dtype.str
   Decimal128Dtype.to_arrow

cudf.ListDtype
==============
.. autosummary::
   :toctree: api/

   ListDtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   ListDtype.deserialize
   ListDtype.device_deserialize
   ListDtype.device_serialize
   ListDtype.element_type
   ListDtype.from_arrow
   ListDtype.host_deserialize
   ListDtype.host_serialize
   ListDtype.is_dtype
   ListDtype.leaf_type
   ListDtype.serialize
   ListDtype.to_arrow
   ListDtype.type

cudf.StructDtype
================
.. autosummary::
   :toctree: api/

   StructDtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   StructDtype.deserialize
   StructDtype.device_deserialize
   StructDtype.device_serialize
   StructDtype.fields
   StructDtype.from_arrow
   StructDtype.host_deserialize
   StructDtype.host_serialize
   StructDtype.is_dtype
   StructDtype.serialize
   StructDtype.to_arrow
   StructDtype.type


cudf.IntervalDtype
==================
.. autosummary::
   :toctree: api/

   IntervalDtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   IntervalDtype.deserialize
   IntervalDtype.device_deserialize
   IntervalDtype.device_serialize
   IntervalDtype.fields
   IntervalDtype.from_arrow
   IntervalDtype.host_deserialize
   IntervalDtype.host_serialize
   IntervalDtype.is_dtype
   IntervalDtype.serialize
   IntervalDtype.to_arrow
   IntervalDtype.type
