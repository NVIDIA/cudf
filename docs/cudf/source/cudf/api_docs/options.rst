.. _api.options:

====================
Options and settings
====================

.. autosummary::
   :toctree: api/

   cudf.get_option
   cudf.set_option
   cudf.describe_option
   cudf.option_context

Display options are controlled by pandas
----------------------------------------

Options for display are inherited from pandas. This includes commonly accessed options such as:

- ``display.max_columns``
- ``display.max_info_rows``
- ``display.max_rows``
- ``display.max_seq_items``

For example, to show all rows of a DataFrame or Series in a Jupyter notebook, call ``pandas.set_option("display.max_rows", None)``.

See also the :ref:`full list of pandas display options <pandas:options.available>`.

Available options
-----------------

You can get a list of available options and their descriptions with :func:`~cudf.describe_option`. When called
with no argument :func:`~cudf.describe_option` will print out the descriptions for all available options.

.. ipython:: python

   import cudf
   cudf.describe_option()
