cuDF for Java
=============

cuDF for Java provides Java bindings for NVIDIA cuDF, a library for
GPU-accelerated structured data processing. It is published to Maven Central
as ``ai.rapids:cudf``:

.. code-block:: xml

   <dependency>
       <groupId>ai.rapids</groupId>
       <artifactId>cudf</artifactId>
       <version>${cudf.version}</version>
   </dependency>

Replace ``${cudf.version}`` with a published cuDF version, or define the
``cudf.version`` property in your POM.

The jar bundles native libraries, so it only runs on the platforms it was
built for. See the `cuDF for Java README
<https://github.com/NVIDIA/cudf/blob/main/java/README.md>`_ for details on
building from source and using the Java bindings on systems with multiple GPUs.

.. toctree::
   :maxdepth: 2

   api/index
