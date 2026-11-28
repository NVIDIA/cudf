Row IR
======

Row IR is libcudf's internal semantic lowering layer for AST-driven JIT execution. It translates expressions into a row-wise program before emitting a device function. Kernel and UDF authors working directly with :doc:`LTO fragments <kernel_authoring>` do not need to construct Row IR.

Lowering separates semantic analysis from source generation. Before emitting code, libcudf must resolve input and result types, determine whether null-aware execution is needed, and select how each operation handles errors.

Row IR resolves expression semantics and performs CSE. It is not a general-purpose GPU IR. CUDA's compiler and linker perform lower-level optimization after Row IR has emitted the device function. Keeping construction, semantic resolution, and emission separate makes each phase inspectable without coupling AST traversal to CUDA source generation.

This system also allows cuDF's developers to use the same operators for host and pre-compiled code.

The types below are internal implementation details, rather than a public authoring API.

AST & Row IR
------------

The public AST describes the requested expression: column references, literals, operators, and UDF calls. Row IR is the internal intermediate representation to make its row-level value dependencies and resolved semantics explicit. ``row_ir::ast_converter`` translates AST nodes into Row IR nodes, then resolves their types, null/error behavior, and common subexpressions using the input metadata.

Row IR produces a CUDA C++ UDF and the kernel configuration. libcudf then invokes the kernels needed to execute it.

.. mermaid::

   flowchart TD
       AST["AST expressions"] --> RowIR["Row IR lowering"]
       RowIR --> UDF["CUDA C++ UDF & Kernel Configuration"]
       UDF --> Execute["Invoke the required cuDF kernels"]

Strict SSA
----------

Row IR is strictly **static single-assignment (SSA)**: each intermediate value has exactly one definition, and its dependencies are explicit references to other values.

This restriction keeps Row IR small, easy to inspect, and optimize. Transformations and analyses can follow the explicit value dependencies without interpreting arbitrary C++, recovering hidden control flow, or reconstructing the effects of reassignment. Operator implementation details stay outside the representation, where CUDA's compiler can lower and optimize them.

1. Construct the Program
------------------------

The implementation uses ``node``, with three forms identified by its opcode and references:

- A ``node`` constructed from ``input_reference`` uses ``opcode::GET_INPUT``. The reference identifies an input registered with the context. Its column or scalar metadata is resolved through that context, rather than stored in the reference.
- An operation ``node`` specifies an ``opcode``, an optional target scale, an ``error_policy``, and child nodes. Operator properties such as null dependence and fallibility are derived from the opcode and policy; they are not all independent fields stored on every node.
- A ``node`` constructed from ``output_reference`` and a child uses ``opcode::SET_OUTPUT``. It writes the child's value to a registered output.

Example: Vector Magnitude
~~~~~~~~~~~~~~~~~~~~~~~~~

For two input columns, the program computes ``sqrt(x * x + y * y)``. In conceptual SSA notation:

.. code-block:: text

   %x = GET_INPUT 0
   %y = GET_INPUT 1
   %x_square = MUL %x, %x
   %y_square = MUL %y, %y
   %dot_product = ADD %x_square, %y_square
   %magnitude = SQRT %dot_product
   SET_OUTPUT 0 %magnitude

The corresponding C++ construction is:

.. code-block:: cpp

   #include "jit/row_ir.hpp"

   #include <cudf/types.hpp>

   #include <optional>
   #include <utility>

   cudf::detail::row_ir::node make_magnitude_program()
   {
     using namespace cudf::detail::row_ir;
     auto constexpr policy = cudf::error_policy::PROPAGATE;

     node x_square{
       opcode::MUL, std::nullopt, policy, node{input_reference{0}}, node{input_reference{0}}};
     node y_square{
       opcode::MUL, std::nullopt, policy, node{input_reference{1}}, node{input_reference{1}}};
     node dot_product{opcode::ADD, std::nullopt, policy, std::move(x_square), std::move(y_square)};
     node magnitude{opcode::SQRT, std::nullopt, policy, std::move(dot_product)};
     return node{output_reference{0}, std::move(magnitude)};
   }

Row IR constructs and resolves these nodes on the caller's behalf.

2. Instantiate and Resolve
--------------------------

Row IR determines the operation result types from the opcode and argument types. Operator metadata and child properties also determine null propagation, whether a result is always valid, and whether a UDF is fallible. The converter uses input nullability and these expression properties to choose null-aware execution and output nullability.

Instantiation also performs common subexpression elimination (CSE): the context records structurally equivalent nodes, and later nodes can alias an existing node's identifier and type.

For two non-nullable ``FLOAT32`` inputs, the vector-magnitude program resolves to ``float`` arithmetic without optional values. These particular ``MUL``, ``ADD``, and ``SQRT`` operations are classified as nonfallible by Row IR. Nullable inputs or other opcodes can change the required handling.

The following diagram shows the resolved attributes for this case.

.. mermaid::

   graph TD
       A["%x = GET_INPUT 0<br/>type: float<br/>nullable: false<br/>fallible: false"] --> C
       B["%y = GET_INPUT 1<br/>type: float<br/>nullable: false<br/>fallible: false"] --> D
       C["%x_squared = MUL %x, %x<br/>type: float<br/>nullable: false<br/>fallible: false"] --> E
       D["%y_squared = MUL %y, %y<br/>type: float<br/>nullable: false<br/>fallible: false"] --> E
       E["%dot_product = ADD<br/>type: float<br/>nullable: false<br/>fallible: false"] --> F
       F["%magnitude = SQRT<br/>type: float<br/>nullable: false<br/>fallible: false"] --> G
       G["SET_OUTPUT 0"]

3. Emit a UDF
-------------------------

Row IR uses the resolved SSA to emit CUDA C++. Operations call ``cudf::detail::row_ir::evaluate``, parameterized by opcode and error policy; their implementations remain outside Row IR. The following illustrates the resulting function:

.. code-block:: cpp

   #include <cudf/detail/row_ir/opcode.hpp>
   #include <cudf/errc.hpp>
   #include <cudf/types.hpp>

   __device__ cudf::errc compute_operation(float* out, float x, float y)
   {
     using namespace cudf::detail::row_ir;
     auto constexpr policy = cudf::error_policy::PROPAGATE;

     float x_square    = evaluate<opcode::MUL, policy>(x, x);
     float y_square    = evaluate<opcode::MUL, policy>(y, y);
     float dot_product = evaluate<opcode::ADD, policy>(x_square, y_square);
     float magnitude   = evaluate<opcode::SQRT, policy>(dot_product);
     *out              = magnitude;
     return cudf::errc::SUCCESS;
   }

For nullable execution, the generated types use ``cuda::std::optional``. For a fallible operation, the emitter supplies standardized result checks to propagate its error or nullify it according to the policy. The generated function is then supplied to the JIT execution path; Row IR itself does not emit the column-loading loop or decide the launch dimensions.
