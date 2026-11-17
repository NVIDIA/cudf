====
join
====

.. versionchanged:: 27.02
   ``FilteredJoin`` retains its build table and memory resource, changing its
   Cython extension type layout. Downstream extensions that cimport this type
   must be rebuilt against the matching pylibcudf version.

.. automodule:: pylibcudf.join
   :members:
