# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for IR nodes, and utilities."""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from collections.abc import Generator, Hashable, Sequence
    from typing import Self


__all__: list[str] = ["Node"]

T = TypeVar("T", bound="Node[Any]")


def _update_stable_hasher(hasher: Any, obj: Hashable) -> None:
    """
    Feed ``hasher`` a bottom-up structural digest of ``obj``.

    Nested :class:`Node` instances contribute their full cached digest
    rather than a truncated id or a full subtree expansion, so hashing an
    entire DAG is linear in total local payload size while distinct child
    subtrees remain distinguishable when composing parent digests.
    """
    if isinstance(obj, Node):
        hasher.update(b"N")
        hasher.update(obj._get_stable_digest())
    elif isinstance(obj, tuple):
        hasher.update(b"T")
        hasher.update(len(obj).to_bytes(4, "big"))
        for x in obj:
            _update_stable_hasher(hasher, x)
    else:
        hasher.update(b"L")
        hasher.update(repr(obj).encode("utf-8"))


class Node(Generic[T]):
    """
    An abstract node type.

    Nodes are immutable!

    This contains a (potentially empty) tuple of child nodes,
    along with non-child data. For uniform reconstruction and
    implementation of hashing and equality schemes, child classes need
    to provide a certain amount of metadata when they are defined.
    Specifically, the ``_non_child`` attribute must list, in-order,
    the names of the slots that are passed to the constructor. The
    constructor must take arguments in the order ``(*_non_child,
    *children).``
    """

    __slots__ = (
        "_hash_value",
        "_repr_value",
        "_stable_digest",
        "_stable_hash_value",
        "_stable_plan_id",
        "children",
    )
    _hash_value: int
    _stable_digest: bytes
    _stable_hash_value: int
    _stable_plan_id: uuid.UUID
    _repr_value: str
    children: tuple[T, ...]
    _non_child: ClassVar[tuple[str, ...]] = ()

    def _ctor_arguments(self, children: Sequence[T]) -> Sequence[Any | T]:
        return (*(getattr(self, attr) for attr in self._non_child), *children)

    def reconstruct(self, children: Sequence[T]) -> Self:
        """
        Rebuild this node with new children.

        Parameters
        ----------
        children
            New children

        Returns
        -------
        New node with new children. Non-child data is shared with the input.
        """
        return type(self)(*self._ctor_arguments(children))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle a Node object."""
        return (
            type(self),
            self._ctor_arguments(self.children),
        )

    def get_hashable(self) -> Hashable:
        """
        Return a hashable object for the node.

        Returns
        -------
        Hashable object.

        Notes
        -----
        This method is used by the :meth:`__hash__` implementation
        (which does caching). If your node type needs special-case
        handling for some of its attributes, override this method, not
        :meth:`__hash__`.
        """
        return (type(self), self._ctor_arguments(self.children))

    def _get_stable_digest(self) -> bytes:
        """Return the full MD5 digest for this node, computing it if needed."""
        try:
            return self._stable_digest
        except AttributeError:
            h = hashlib.md5(usedforsecurity=False)
            _update_stable_hasher(h, self.get_hashable())
            self._stable_digest = h.digest()
            return self._stable_digest

    def get_stable_id(self) -> int:
        """
        Compute a stable identifier for Node.

        Digests :meth:`get_hashable` bottom-up with MD5 so the result is
        deterministic across process boundaries (Python's ``hash()`` uses
        ``PYTHONHASHSEED``).

        Returns
        -------
        int
            A stable 32-bit identifier for this node.
        """
        try:
            return self._stable_hash_value
        except AttributeError:
            # First 4 digest bytes == int(hexdigest()[:8], 16).
            self._stable_hash_value = int.from_bytes(
                self._get_stable_digest()[:4], "big"
            )
            return self._stable_hash_value

    def get_stable_plan_id(self) -> uuid.UUID:
        """Return a stable ID for the full IR plan rooted at this node."""
        try:
            return self._stable_plan_id
        except AttributeError:
            ids = tuple(node.get_stable_id() for node in traversal([self]))
            self._stable_plan_id = uuid.uuid5(uuid.NAMESPACE_URL, str(ids))
            return self._stable_plan_id

    def __hash__(self) -> int:
        """
        Hash of an expression with caching.

        See Also
        --------
        get_hashable
        """
        try:
            return self._hash_value
        except AttributeError:
            self._hash_value = hash(self.get_hashable())
            return self._hash_value

    def is_equal(self, other: Self) -> bool:
        """
        Equality of two nodes of equal type.

        Override this in subclasses, rather than :meth:`__eq__`.

        Parameters
        ----------
        other
            object of same type to compare to.

        Notes
        -----
        Since nodes are immutable, this does common subexpression
        elimination when two nodes are determined to be equal.

        :meth:`__eq__` handles the case where the objects being
        compared are not of the same type, so in this method, we only
        need to implement equality of equal types.

        Returns
        -------
        True if the two nodes are equal, false otherwise.
        """
        if self is other:
            return True
        result = self._ctor_arguments(self.children) == other._ctor_arguments(
            other.children
        )
        # Eager CSE for nodes that match.
        if result:
            self.children = other.children
        return result

    def __eq__(self, other: Any) -> bool:
        """
        Equality of expressions.

        See Also
        --------
        is_equal
        """
        if type(self) is not type(other) or hash(self) != hash(other):
            return False
        else:
            return self.is_equal(other)

    def __ne__(self, other: Any) -> bool:
        """Inequality of expressions."""
        return not self.__eq__(other)

    def __repr__(self) -> str:
        """String representation of an expression with caching."""
        try:
            return self._repr_value
        except AttributeError:
            args = ", ".join(f"{arg!r}" for arg in self._ctor_arguments(self.children))
            self._repr_value = f"{type(self).__name__}({args})"
            return self._repr_value

    def __rich_repr__(self) -> Generator[Any, None, None]:
        """Formatting for rich.pretty.pprint."""
        for attr in self._non_child:
            yield attr, getattr(self, attr)

        yield from self.children
