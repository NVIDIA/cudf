# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Traversal and visitor utilities for nodes."""

from __future__ import annotations

from collections import Counter, deque
from typing import TYPE_CHECKING, Generic

from cudf_polars.typing import (
    StateT_co,
    U_contra,
    V_co,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, MutableMapping, Sequence

    from cudf_polars.typing import GenericTransformer, NodeT


__all__: list[str] = [
    "CachingVisitor",
    "collect_refcount",
    "make_recursive",
    "post_traversal",
    "reuse_if_unchanged",
    "traversal",
]


def collect_refcount(nodes: Sequence[NodeT]) -> Mapping[NodeT, int]:
    """
    Determine reference counts of all nodes in a DAG.

    Parameters
    ----------
    nodes
         Sequence of root nodes

    Returns
    -------
    Mapping from nodes to frequency of occurrence in the DAG.
    """
    refcount = Counter(nodes)
    for node in traversal(nodes):
        refcount.update(node.children)
    return refcount


def traversal(nodes: Sequence[NodeT]) -> Generator[NodeT, None, None]:
    """
    Pre-order traversal of nodes in an expression.

    Parameters
    ----------
    nodes
        Roots of expressions to traverse.

    Yields
    ------
    Unique nodes in the expressions, parent before child, children
    in-order from left to right.
    """
    seen: set[NodeT] = set()
    lifo: deque[NodeT] = deque()

    for node in nodes:
        if node not in seen:
            lifo.append(node)
            seen.add(node)

    while lifo:
        node = lifo.pop()
        yield node
        for child in reversed(node.children):
            if child not in seen:
                seen.add(child)
                lifo.append(child)


def post_traversal(nodes: Sequence[NodeT]) -> Generator[NodeT, None, None]:
    """
    Post-order traversal of nodes in an expression.

    Parameters
    ----------
    nodes
        Roots of expressions to traverse.

    Yields
    ------
    Unique nodes in the expressions, child before parent, children
    in-order from left to right.
    """
    seen: set[NodeT] = set()
    lifo: deque[NodeT] = deque()

    for node in nodes:
        if node not in seen:
            lifo.append(node)
            seen.add(node)

    while lifo:
        node = lifo[-1]
        for child in node.children:
            if child not in seen:
                lifo.append(child)
                seen.add(child)
                break
        else:
            yield node
            lifo.pop()


def reuse_if_unchanged(
    node: NodeT, fn: GenericTransformer[NodeT, NodeT, StateT_co]
) -> NodeT:
    """
    Recipe for transforming nodes that returns the old object if unchanged.

    Parameters
    ----------
    node
         Node to recurse on
    fn
         Function to transform children

    Notes
    -----
    This can be used as a generic "base case" handler when
    writing transforms that take nodes and produce new nodes.

    Returns
    -------
    Existing node `e` if transformed children are unchanged, otherwise
    reconstructed node with new children.
    """
    new_children = [fn(c) for c in node.children]
    if all(new == old for new, old in zip(new_children, node.children, strict=True)):
        return node
    return node.reconstruct(new_children)


def make_recursive(
    fn: Callable[[U_contra, GenericTransformer[U_contra, V_co, StateT_co]], V_co],
    *,
    # make_recursive is a type constructor with covariant state parameter
    # not a normal function for which the parameter would be contravariant
    # hence the type ignore
    state: StateT_co,  # type: ignore[misc]
) -> GenericTransformer[U_contra, V_co, StateT_co]:
    """
    No-op wrapper for recursive visitors.

    Facilitates using visitors that don't need caching but are written
    in the same style.

    Parameters
    ----------
    fn
        Function to transform inputs to outputs. Should take as its
        second argument a callable from input to output.
    state
        Arbitrary *immutable* state that should be accessible to the
        visitor through the `state` property.

    Notes
    -----
    All transformation functions *must* be free of side-effects.

    Usually, prefer a :class:`CachingVisitor`, but if we know that we
    don't need caching in a transformation and then this no-op
    approach is slightly cheaper.

    Returns
    -------
    Recursive function without caching.

    See Also
    --------
    CachingVisitor
    """

    def rec(node: U_contra) -> V_co:
        return fn(node, rec)  # type: ignore[arg-type]

    rec.state = state  # type: ignore[attr-defined]
    return rec  # type: ignore[return-value]


class CachingVisitor(Generic[U_contra, V_co, StateT_co]):
    """
    Caching wrapper for recursive visitors.

    Facilitates writing visitors where already computed results should
    be cached and reused. The cache is managed automatically, and is
    tied to the lifetime of the wrapper.

    Parameters
    ----------
    fn
        Function to transform inputs to outputs. Should take as its
        second argument the recursive cache manager.
    state
        Arbitrary *immutable* state that should be accessible to the
        visitor through the `state` property.

    Notes
    -----
    All transformation functions *must* be free of side-effects.

    Returns
    -------
    Recursive function with caching.
    """

    def __init__(
        self,
        fn: Callable[[U_contra, GenericTransformer[U_contra, V_co, StateT_co]], V_co],
        *,
        state: StateT_co,
    ) -> None:
        self.fn = fn
        self.cache: MutableMapping[U_contra, V_co] = {}
        self.state = state

    def __call__(self, value: U_contra) -> V_co:
        """
        Apply the function to a value.

        Parameters
        ----------
        value
            The value to transform.

        Returns
        -------
        A transformed value.
        """
        try:
            return self.cache[value]
        except KeyError:
            return self.cache.setdefault(value, self.fn(value, self))
