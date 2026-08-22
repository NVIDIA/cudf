# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Define and manage cuDF's global configuration options."""

from __future__ import annotations

import os
import textwrap
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Container


@dataclass
class Option:
    """Store the definition and current state of a single cuDF option."""

    default: Any
    value: Any
    description: str
    validator: Callable


_OPTIONS: dict[str, Option] = {}


def _env_get_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (ValueError, TypeError):
        return default


def _env_get_bool(name, default):
    env = os.getenv(name)
    if env is None:
        return default
    as_a_int = _env_get_int(name, None)
    env = env.lower().strip()
    if env == "true" or env == "on" or as_a_int:
        return True
    if env == "false" or env == "off" or as_a_int == 0:
        return False
    return default


def _register_option(
    name: str, default_value: Any, description: str, validator: Callable
):
    """Register an option.

    Parameters
    ----------
    name : str
        The name of the option.
    default_value : Any
        The default value of the option.
    description : str
        A text description of the option.
    validator : Callable
        Called on the option value to check its validity. Should raise an
        error if the value is invalid.

    Raises
    ------
    BaseException
        Raised by validator if the value is invalid.
    """
    validator(default_value)
    _OPTIONS[name] = Option(
        default_value, default_value, description, validator
    )


def get_option(name: str) -> Any:
    """Get the value of the specified option.

    Parameters
    ----------
    name : str
        The name of the option.

    Returns
    -------
    Any
        The value of the option.

    Raises
    ------
    KeyError
        If option ``name`` does not exist.
    """
    try:
        return _OPTIONS[name].value
    except KeyError:
        raise KeyError(f'"{name}" does not exist.')


def set_option(name: str, val: Any):
    """Set the value of option.

    Parameters
    ----------
    name : str
        The name of the option.
    val : Any
        The value to set.

    Raises
    ------
    KeyError
        If option ``name`` does not exist.
    BaseException
        Raised by validator if the value is invalid.
    """
    try:
        option = _OPTIONS[name]
    except KeyError:
        raise KeyError(f'"{name}" does not exist.')
    option.validator(val)
    option.value = val


def _build_option_description(name, opt):
    return (
        f"{name}:\n"
        f"\t{opt.description}\n"
        f"\t[Default: {opt.default}] [Current: {opt.value}]"
    )


def describe_option(name: str | None = None):
    """Print the description of an option.

    If `name` is unspecified, prints the description of all available options.

    Parameters
    ----------
    name : Optional[str]
        The name of the option.
    """
    names = _OPTIONS.keys() if name is None else [name]
    for name in names:
        print(_build_option_description(name, _OPTIONS[name]))  # noqa: T201


def _make_contains_validator(valid_options: Container) -> Callable:
    """Return a validator that checks if a value is in `valid_options`."""

    def _validator(val):
        if val not in valid_options:
            raise ValueError(
                f"{val} is not a valid option. "
                f"Must be one of {set(valid_options)}."
            )

    return _validator


def _spill_validator(val):
    if val not in {False, True}:
        raise ValueError(
            f"{val} is not a valid option. Must be one of {{False, True}}."
        )


def _integer_validator(val):
    try:
        int(val)
        return True
    except ValueError:
        raise ValueError(f"{val} is not a valid option. Must be an integer.")


def _integer_and_none_validator(val):
    try:
        if val is None or int(val):
            return
    except ValueError:
        raise ValueError(
            f"{val} is not a valid option. Must be an integer or None."
        )


_register_option(
    "default_integer_bitwidth",
    None,
    textwrap.dedent(
        """
        Default bitwidth when the dtype of an integer needs to be
        inferred. If set to `None`, the API will align dtype with pandas.
        APIs that respect this option include:
        \t- cudf object constructors
        \t- cudf.read_csv and cudf.read_json when `dtype` is not specified.
        \t- APIs that require implicit conversion of cudf.RangeIndex to an
        \t  integer index.
        \tValid values are  None, 32 or 64. Default is None.
    """
    ),
    _make_contains_validator([None, 32, 64]),
)


_register_option(
    "default_float_bitwidth",
    None,
    textwrap.dedent(
        """
        Default bitwidth when the dtype of a float needs to be
        inferred. If set to `None`, the API will align dtype with pandas.
        APIs that respect this option include:
        \t- cudf object constructors
        \t- cudf.read_csv and cudf.read_json when `dtype` is not specified.
        \tValid values are None, 32 or 64. Default is None.
    """
    ),
    _make_contains_validator([None, 32, 64]),
)

_register_option(
    "spill",
    _env_get_bool("CUDF_SPILL", False),
    textwrap.dedent(
        """
        Enables spilling.
        \tValid values are True or False. Default is False.
        """
    ),
    _spill_validator,
)


_register_option(
    "spill_on_demand",
    _env_get_bool("CUDF_SPILL_ON_DEMAND", True),
    textwrap.dedent(
        """
        Enables spilling on demand using an RMM out-of-memory error handler.
        This has no effect if spilling is disabled, see the "spill" option.
        \tValid values are True or False. Default is True.
        """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "spill_device_limit",
    _env_get_int("CUDF_SPILL_DEVICE_LIMIT", None),
    textwrap.dedent(
        """
        Enforce a device memory limit in bytes.
        This has no effect if spilling is disabled, see the "spill" option.
        \tValid values are any positive integer or None (disabled).
        \tDefault is None.
        """
    ),
    _integer_and_none_validator,
)

_register_option(
    "spill_stats",
    _env_get_int("CUDF_SPILL_STATS", 0),
    textwrap.dedent(
        """
        If not 0, enables statistics at the specified level:
            0  - disabled (no overhead).
            1+ - duration and number of bytes spilled (very low overhead).
            2+ - a traceback for each time a spillable buffer is exposed
                permanently (potential high overhead).

        Valid values are any positive integer.
        Default is 0 (disabled).
        """
    ),
    _integer_validator,
)

_register_option(
    "mode.pandas_compatible",
    False,
    textwrap.dedent(
        """
        If set to `False`, retains `cudf` specific behavior.
        If set to `True`, enables pandas compatibility mode,
        which will try to match pandas API behaviors in case of
        any inconsistency.
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "memory_profiling",
    _env_get_bool("CUDF_MEMORY_PROFILING", False),
    textwrap.dedent(
        """
        If set to `False`, disables memory profiling.
        If set to `True`, enables memory profiling.
        Read more at: :ref:`memory-profiling-user-doc`
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "io.parquet.low_memory",
    False,
    textwrap.dedent(
        """
        If set to `False`, reads entire parquet in one go.
        If set to `True`, reads parquet file in chunks.
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "io.json.low_memory",
    False,
    textwrap.dedent(
        """
        If set to `False`, reads entire json in one go.
        If set to `True`, reads json file in chunks.
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "kvikio_remote_io",
    _env_get_bool("CUDF_KVIKIO_REMOTE_IO", False),
    textwrap.dedent(
        """
        Whether to use KvikIO's remote IO backend or not.
        \tWARN: this is experimental and may be removed at any time
        \twithout warning or deprecation period.
        \tSet KVIKIO_NTHREADS (default is 8) to change the number of
        \tconcurrent tcp connections, which is important for good performance.
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)


class option_context(ContextDecorator):
    """
    Context manager to temporarily set options in the `with` statement context.

    You need to invoke as ``option_context(pat1, val1, pat2, val2, ...)``.

    Parameters
    ----------
    *args
        Alternating option names and values, e.g.
        ``option_context(name1, val1, name2, val2, ...)``.

    Examples
    --------
    >>> from cudf import option_context
    >>> with option_context('mode.pandas_compatible', True, 'default_float_bitwidth', 32):
    ...     pass
    """

    def __init__(self, *args) -> None:
        if len(args) % 2 != 0:
            raise ValueError(
                "Need to invoke as option_context(pat, val, "
                "[(pat, val), ...])."
            )

        self.ops = tuple(zip(args[::2], args[1::2], strict=True))
        self.undo = tuple((pat, get_option(pat)) for pat, _ in self.ops)

    def __enter__(self) -> None:
        for pat, val in self.ops:
            set_option(pat, val)

    def __exit__(self, *args) -> None:
        for pat, val in self.undo:
            set_option(pat, val)
