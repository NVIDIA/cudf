cuDF Java
=========

cuDF Java provides Java bindings for :doc:`libcudf <../libcudf/index>`, for
processing large amounts of tabular data on the GPU from the JVM. It is
published to Maven Central as ``ai.rapids:cudf``:

.. code-block:: xml

   <dependency>
       <groupId>ai.rapids</groupId>
       <artifactId>cudf</artifactId>
       <version>${cudf.version}</version>
   </dependency>

Replace ``${cudf.version}`` with a published cuDF version, or define the
``cudf.version`` property in your POM.

The jar bundles native libraries, so it only runs on the platforms it was
built for. See the `cuDF Java README
<https://github.com/NVIDIA/cudf/blob/main/java/README.md>`_ for details on
building from source and using cuDF Java on systems with multiple GPUs.

.. toctree::
   :maxdepth: 2

   api/index
