# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import functools
from decimal import Decimal

from pylibcudf.libcudf.expressions import \
    ast_operator as ASTOperator  # no-cython-lint
from pylibcudf.libcudf.expressions import \
    table_reference as TableReference  # no-cython-lint

from cython.operator cimport dereference
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.libcudf cimport expressions as libcudf_exp
from pylibcudf.libcudf.scalar.scalar cimport (
    duration_scalar,
    numeric_scalar,
    string_scalar,
    timestamp_scalar,
    fixed_point_scalar,
)
from pylibcudf.libcudf.types cimport size_type, type_id
from pylibcudf.libcudf.wrappers.durations cimport (
    duration_D,
    duration_ms,
    duration_ns,
    duration_s,
    duration_us,
)
from pylibcudf.libcudf.wrappers.timestamps cimport (
    timestamp_D,
    timestamp_ms,
    timestamp_ns,
    timestamp_s,
    timestamp_us,
)
from pylibcudf.libcudf.fixed_point.fixed_point cimport decimal128, decimal64, decimal32

from .scalar cimport Scalar
from .traits cimport is_chrono, is_numeric, is_fixed_point
from .types cimport DataType


# Aliases for simplicity
ctypedef unique_ptr[libcudf_exp.expression] expression_ptr

__all__ = [
    "ASTOperator",
    "ColumnNameReference",
    "ColumnReference",
    "Expression",
    "Literal",
    "Operation",
    "TableReference",
    "to_expression"
]

# Define this class just to have a docstring for it
cdef class Expression:
    """
    The base class for all expression types.
    This class cannot be instantiated directly, please
    instantiate one of its child classes instead.

    For details, see :cpp:class:`cudf::ast::expression`.
    """
    __hash__ = None

cdef class Literal(Expression):
    """
    A literal value used in an abstract syntax tree.

    For details, see :cpp:class:`cudf::ast::literal`.

    Parameters
    ----------
    value : Scalar
        The Scalar value of the Literal.
        Must be either numeric, string, or a timestamp/duration scalar.
    """
    def __cinit__(self, Scalar value):
        self.scalar = value
        cdef DataType typ = value.type()
        cdef type_id tid = value.type().id()
        if not (
            is_numeric(typ)
            or is_chrono(typ)
            or is_fixed_point(typ)
            or tid == type_id.STRING
        ):
            raise ValueError(
                "Only numeric, string, decimal or timestamp/duration \
                scalars are accepted"
            )
        # TODO: Accept type-erased scalar in AST C++ code
        # Then a lot of this code can be deleted
        if tid == type_id.INT64:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[int64_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.INT32:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[int32_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.INT16:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[int16_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.INT8:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[int8_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.UINT64:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[uint64_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.UINT32:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[uint32_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.UINT16:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[uint16_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.UINT8:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[uint8_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.BOOL8:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[bool] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.FLOAT64:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[double] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.FLOAT32:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[float] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.STRING:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <string_scalar &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DECIMAL32:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <fixed_point_scalar[decimal32] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DECIMAL64:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <fixed_point_scalar[decimal64] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DECIMAL128:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <fixed_point_scalar[decimal128] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_NANOSECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_ns] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_MICROSECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_us] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_MILLISECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_ms] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_MILLISECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_ms] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_SECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_s] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_DAYS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_D] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_NANOSECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_ns] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_MICROSECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_us] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_MILLISECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_ms] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_MILLISECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_ms] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_SECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_s] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_DAYS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_D] &>dereference(self.scalar.c_obj)
            ))
        else:
            raise NotImplementedError(
                f"Don't know how to make literal with type id {tid}"
            )

cdef class ColumnReference(Expression):
    """
    An expression referring to data from a column in a table.

    For details, see :cpp:class:`cudf::ast::column_reference`.

    Parameters
    ----------
    index : size_type
        The index of this column in the table
        (provided when the expression is evaluated).
    table_source : TableReference, default TableReferenece.LEFT
        Which table to use in cases with two tables (e.g. joins)
    """
    def __cinit__(
        self,
        size_type index,
        table_reference table_source=table_reference.LEFT
    ):
        self.c_obj = <expression_ptr>move(make_unique[libcudf_exp.column_reference](
            index, table_source
        ))


cdef class Operation(Expression):
    """
    An operation expression holds an operator and zero or more operands.

    For details, see :cpp:class:`cudf::ast::operation`.

    Parameters
    ----------
    op : Operator
    left : Expression
        Left input expression (left operand)
    right: Expression, default None
        Right input expression (right operand).
        You should only pass this if the input expression is a binary operation.
    """
    def __cinit__(self, ast_operator op, Expression left, Expression right=None):
        self.left = left
        self.right = right
        if right is None:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.operation](
                op, dereference(left.c_obj)
            ))
        else:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.operation](
                op, dereference(left.c_obj), dereference(right.c_obj)
            ))

cdef class ColumnNameReference(Expression):
    """
    An expression referring to data from a column in a table.

    For details, see :cpp:class:`cudf::ast::column_name_reference`.

    Parameters
    ----------
    column_name : str
        Name of this column in the table metadata
        (provided when the expression is evaluated).
    """
    def __cinit__(self, str name):
        self.c_obj = <expression_ptr> \
            move(make_unique[libcudf_exp.column_name_reference](
                <string>(name.encode("utf-8"))
            ))


# This dictionary encodes the mapping from Python AST operators to their cudf
# counterparts.
_python_cudf_operator_map = {
    # Binary operators
    ast.Add: ASTOperator.ADD,
    ast.Sub: ASTOperator.SUB,
    ast.Mult: ASTOperator.MUL,
    ast.Div: ASTOperator.DIV,
    ast.FloorDiv: ASTOperator.FLOOR_DIV,
    ast.Mod: ASTOperator.PYMOD,
    ast.Pow: ASTOperator.POW,
    ast.Eq: ASTOperator.EQUAL,
    ast.NotEq: ASTOperator.NOT_EQUAL,
    ast.Lt: ASTOperator.LESS,
    ast.Gt: ASTOperator.GREATER,
    ast.LtE: ASTOperator.LESS_EQUAL,
    ast.GtE: ASTOperator.GREATER_EQUAL,
    ast.BitXor: ASTOperator.BITWISE_XOR,
    # TODO: The mapping of logical/bitwise operators here is inconsistent with
    # pandas. In pandas, Both `BitAnd` and `And` map to
    # `ASTOperator.LOGICAL_AND` for booleans, while they map to
    # `ASTOperator.BITWISE_AND` for integers. However, there is no good way to
    # encode this at present because expressions can be arbitrarily nested so
    # we won't know the dtype of the input without inserting a much more
    # complex traversal of the expression tree to determine the output types at
    # each node. For now, we'll rely on users to use the appropriate operator.
    ast.BitAnd: ASTOperator.BITWISE_AND,
    ast.BitOr: ASTOperator.BITWISE_OR,
    ast.And: ASTOperator.LOGICAL_AND,
    ast.Or: ASTOperator.LOGICAL_OR,
    # Unary operators
    ast.Invert: ASTOperator.BIT_INVERT,
    ast.Not: ASTOperator.NOT,
    # TODO: Missing USub, possibility other unary ops?
}


# Mapping between Python function names encode in an ast.Call node and the
# corresponding libcudf C++ AST operators.
_python_cudf_function_map = {
    # TODO: Operators listed on
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#expression-evaluation-via-eval  # noqa: E501
    # that we don't support yet:
    # expm1, log1p, arctan2 and log10.
    "isnull": ASTOperator.IS_NULL,
    "isna": ASTOperator.IS_NULL,
    "sin": ASTOperator.SIN,
    "cos": ASTOperator.COS,
    "tan": ASTOperator.TAN,
    "arcsin": ASTOperator.ARCSIN,
    "arccos": ASTOperator.ARCCOS,
    "arctan": ASTOperator.ARCTAN,
    "sinh": ASTOperator.SINH,
    "cosh": ASTOperator.COSH,
    "tanh": ASTOperator.TANH,
    "arcsinh": ASTOperator.ARCSINH,
    "arccosh": ASTOperator.ARCCOSH,
    "arctanh": ASTOperator.ARCTANH,
    "exp": ASTOperator.EXP,
    "log": ASTOperator.LOG,
    "sqrt": ASTOperator.SQRT,
    "abs": ASTOperator.ABS,
    "ceil": ASTOperator.CEIL,
    "floor": ASTOperator.FLOOR,
    # TODO: Operators supported by libcudf with no Python function analog.
    # ast.rint: ASTOperator.RINT,
    # ast.cbrt: ASTOperator.CBRT,
}


class ExpressionTransformer(ast.NodeVisitor):
    """A NodeVisitor specialized for constructing a libcudf expression tree.

    This visitor is designed to handle AST nodes that have libcudf equivalents.
    It constructs column references from names and literals from constants,
    then builds up operations. The resulting expression is returned by the
    `visit` method

    Parameters
    ----------
    column_mapping : dict[str, ColumnNameReference | ColumnReference]
        Mapping from names to column references or column name references.
        The former can be used for `compute_column` the latter in IO filters.
    """

    def __init__(self, dict column_mapping):
        self.column_mapping = column_mapping

    def generic_visit(self, node):
        raise ValueError(
            f"Not expecting expression to have node of type {node.__class__.__name__}"
        )

    def visit_Module(self, node):
        try:
            expr, = node.body
        except ValueError:
            raise ValueError(
                f"Expecting exactly one expression, not {len(node.body)}"
            )
        return self.visit(expr)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        try:
            return self.column_mapping[node.id]
        except KeyError:
            raise ValueError(f"Unknown column name {node.id}")

    def visit_Constant(self, node):
        if not isinstance(node.value, (float, int, str, complex, Decimal)):
            raise ValueError(
                f"Unsupported literal {repr(node.value)} of type "
                "{type(node.value).__name__}"
            )
        return Literal(Scalar.from_py(node.value))

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            # TODO: Except for leaf nodes, we won't know the type of the
            # operand, so there's no way to know whether this should be a float
            # or an int. We should maybe see what Spark does, and this will
            # probably require casting.
            minus_one = Literal(Scalar.from_py(-1))
            return Operation(ASTOperator.MUL, minus_one, operand)
        elif isinstance(node.op, ast.UAdd):
            return operand
        else:
            op = _python_cudf_operator_map[type(node.op)]
            return Operation(op, operand)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = _python_cudf_operator_map[type(node.op)]
        return Operation(op, left, right)

    def visit_BoolOp(self, node):
        return functools.reduce(
            functools.partial(Operation, ASTOperator.LOGICAL_AND),
            (
                Operation(
                    _python_cudf_operator_map[type(node.op)],
                    self.visit(left),
                    self.visit(right),
                )
                for left, right in zip(
                    node.values[:-1], node.values[1:], strict=True
                )
            )
        )

    def visit_Compare(self, node):
        operands = [node.left, *node.comparators]
        return functools.reduce(
            functools.partial(Operation, ASTOperator.LOGICAL_AND),
            (
                Operation(
                    _python_cudf_operator_map[type(op)],
                    self.visit(left),
                    self.visit(right),
                )
                for op, left, right in zip(
                    node.ops, operands[:-1], operands[1:], strict=True
                )
            )
        )

    def visit_Call(self, node):
        try:
            op = _python_cudf_function_map[node.func.id]
        except KeyError:
            raise ValueError(f"Unsupported function {node.func}.")
        # Assuming only unary functions are supported, which is checked above.
        if len(node.args) != 1 or node.keywords:
            raise ValueError(
                f"Function {node.func} only accepts one positional "
                "argument."
            )
        return Operation(op, self.visit(node.args[0]))


@functools.lru_cache(256)
def to_expression(str expr, tuple column_names: tuple[str, ...]):
    """
    Create an expression for `pylibcudf.transform.compute_column`.

    Parameters
    ----------
    expr : str
        The expression to evaluate. In (restricted) Python syntax.
    column_names : tuple[str]
        Ordered tuple of names. When calling `compute_column` on the resulting
        expression, the provided table must have columns in the same order
        as given here.

    Notes
    -----
    This function keeps a small cache of recently used expressions.

    Returns
    -------
    Expression
        Expression for the given expr and col_names
    """
    visitor = ExpressionTransformer(
        {name: ColumnReference(i) for i, name in enumerate(column_names)}
    )
    return visitor.visit(ast.parse(expr))


ASTOperator.__str__ = ASTOperator.__repr__
TableReference.__str__ = TableReference.__repr__
