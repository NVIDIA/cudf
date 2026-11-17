# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


class NoNewAttributesMixin:
    """
    Mixin which prevents adding new attributes.

    Vendored from ``pandas.core.base.NoNewAttributesMixin``.

    Prevents additional attributes via ``xxx.attribute = "something"`` after
    a call to ``self._freeze()``. Mainly used to prevent the user from using
    wrong attributes on an accessor (`Series.cat/.str/.dt`).

    If you really want to add a new attribute at a later time, you need to
    use ``object.__setattr__(self, key, value)``.
    """

    def _freeze(self) -> None:
        """
        Prevents setting additional attributes.
        """
        object.__setattr__(self, "__frozen", True)

    # prevent adding any attribute via s.xxx.new_attribute = ...
    def __setattr__(self, key: str, value: Any) -> None:
        # _cache is used by a decorator
        # We need to check both 1.) cls.__dict__ and 2.) getattr(self, key)
        # because
        # 1.) getattr is false for attributes that raise errors
        # 2.) cls.__dict__ doesn't traverse into base classes
        if getattr(self, "__frozen", False) and not (
            key == "_cache"
            or key in type(self).__dict__
            or getattr(self, key, None) is not None
        ):
            raise AttributeError(f"You cannot add any new attribute '{key}'")
        object.__setattr__(self, key, value)
