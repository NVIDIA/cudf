.. _api.window:

======
Window
======

Rolling objects are returned by ``.rolling`` calls: :func:`cudf.DataFrame.rolling`, :func:`cudf.Series.rolling`, etc.

.. _api.functions_rolling:

Rolling window functions
------------------------
.. currentmodule:: cudf.core.window.rolling

.. autosummary::
   :toctree: api/

   Rolling.count
   Rolling.sum
   Rolling.mean
   Rolling.var
   Rolling.std
   Rolling.min
   Rolling.max
   Rolling.apply
