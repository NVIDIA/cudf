# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.abc
import importlib.machinery
import os
import pathlib
import sys
import threading
import warnings
from abc import abstractmethod
from collections import defaultdict
from importlib._bootstrap import (  # type: ignore[attr-defined]  # (private importlib lock exists at runtime)
    _ImportLockContext as ImportLock,
)
from types import ModuleType
from typing import Any, ContextManager, NamedTuple, Self  # noqa: UP035

from .fast_slow_proxy import (
    _FunctionProxy,
    _is_function_or_method,
    _Unusable,
    get_final_type_map,
    get_intermediate_type_map,
    get_registered_functions,
)


def rename_root_module(module: str, root: str, new_root: str) -> str:
    """
    Rename a module to a new root.

    Parameters
    ----------
    module
        Module to rename
    root
        Original root
    new_root
        New root

    Returns
    -------
    New module name (if it matches root) otherwise original name.
    """
    if module.startswith(root):
        return new_root + module[len(root) :]
    else:
        return module


class DeducedMode(NamedTuple):
    use_fast_lib: bool
    slow_lib: str
    fast_lib: str


def deduce_cudf_pandas_mode(slow_lib: str, fast_lib: str) -> DeducedMode:
    """
    Determine if cudf.pandas should use the requested fast library.

    Parameters
    ----------
    slow_lib
        Name of the slow library
    fast_lib
        Name of the fast library

    Returns
    -------
    Whether the fast library is being used, and the resulting names of
    the "slow" and "fast" libraries.
    """
    if "CUDF_PANDAS_FALLBACK_MODE" not in os.environ:
        try:
            importlib.import_module(fast_lib)
            return DeducedMode(
                use_fast_lib=True, slow_lib=slow_lib, fast_lib=fast_lib
            )
        except Exception as e:
            warnings.warn(
                f"Exception encountered importing {fast_lib}: {e}."
                f"Falling back to only using {slow_lib}."
            )
    return DeducedMode(
        use_fast_lib=False, slow_lib=slow_lib, fast_lib=slow_lib
    )


class ModuleAcceleratorBase(
    importlib.abc.MetaPathFinder, importlib.abc.Loader
):
    _instance: ModuleAcceleratorBase | None = None
    mod_name: str
    fast_lib: str
    slow_lib: str

    # When walking the module tree and wrapping module attributes,
    # we often will come across the same object more than once. We
    # don't want to create separate wrappers for each
    # instance, so we keep a registry of all module attributes
    # that we can look up to see if we have already wrapped an
    # attribute before
    _wrapped_objs: dict[Any, Any]

    def __new__(
        cls,
        mod_name: str,
        fast_lib: str,
        slow_lib: str,
    ):
        """Build a custom module finder that will provide wrapped modules
        on demand.

        Parameters
        ----------
        mod_name
             Import name to deliver modules under.
        fast_lib
             Name of package that provides "fast" implementation
        slow_lib
             Name of package that provides "slow" fallback implementation
        """
        if ModuleAcceleratorBase._instance is not None:
            raise RuntimeError(
                "Only one instance of ModuleAcceleratorBase allowed"
            )
        self = object.__new__(cls)
        self.mod_name = mod_name
        self.fast_lib = fast_lib
        self.slow_lib = slow_lib

        # When walking the module tree and wrapping module attributes,
        # we often will come across the same object more than once. We
        # don't want to create separate wrappers for each
        # instance, so we keep a registry of all module attributes
        # that we can look up to see if we have already wrapped an
        # attribute before
        self._wrapped_objs = {}
        self._wrapped_objs.update(get_final_type_map())
        self._wrapped_objs.update(get_intermediate_type_map())
        self._wrapped_objs.update(get_registered_functions())

        ModuleAcceleratorBase._instance = self
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(fast={self.fast_lib}, slow={self.slow_lib})"
        )

    def find_spec(
        self, fullname: str, path, target=None
    ) -> importlib.machinery.ModuleSpec | None:
        """Provide ourselves as a module loader.

        Parameters
        ----------
        fullname
            Name of module to be imported, if it starts with the name
            that we are using to wrap, we will deliver ourselves as a
            loader, otherwise defer to the standard Python loaders.

        Returns
        -------
        A ModuleSpec with ourself as loader if we're interposing,
        otherwise None to pass off to the next loader.
        """
        if fullname == self.mod_name or fullname.startswith(
            f"{self.mod_name}."
        ):
            return importlib.machinery.ModuleSpec(
                name=fullname,
                loader=self,
                # Note, this influences the repr of the module, so we may want
                # to change it if we ever want to control that.
                origin=None,
                loader_state=None,
                is_package=True,
            )
        return None

    def create_module(self, spec) -> ModuleType | None:
        return None

    def exec_module(self, mod: ModuleType):
        # importlib calls this function with the global import lock held.
        self._populate_module(mod)

    @abstractmethod
    def disabled(self) -> ContextManager:
        pass

    def _postprocess_module(
        self,
        mod: ModuleType,
        slow_mod: ModuleType,
        fast_mod: ModuleType | None,
    ) -> ModuleType:
        """Ensure that the wrapped module satisfies required invariants.

        Parameters
        ----------
        mod
            Wrapped module to postprocess
        slow_mod
            Slow version that we are mimicking
        fast_mod
            Fast module that provides accelerated implementations (may
            be None

        Returns
        -------
        Checked and validated module

        Notes
        -----
        The implementation of fast-slow proxies imposes certain
        requirements on the wrapped modules that it delivers. This
        function encodes those requirements and raises if the module
        does not satisfy them.

        This post-processing routine should be kept up to date with any
        requirements encoded by fast_slow_proxy.py
        """
        mod.__dict__["_fsproxy_slow"] = slow_mod
        if fast_mod is not None:
            mod.__dict__["_fsproxy_fast"] = fast_mod
        return mod

    @abstractmethod
    def _populate_module(self, mod: ModuleType) -> ModuleType:
        """Populate given module with appropriate attributes.

        This traverses the attributes of the slow module corresponding
        to mod and mirrors those in the provided module in a wrapped
        mode that attempts to execute them using the fast module first.

        Parameters
        ----------
        mod
            Module to populate

        Returns
        -------
        ModuleType
            Populated module

        Notes
        -----
        In addition to the attributes of the slow module,
        the returned module must have the following attributes:

        - '_fsproxy_slow': the corresponding slow module
        - '_fsproxy_fast': the corresponding fast module

        This is necessary for correct rewriting of UDFs when calling
        to the respective fast/slow libraries.

        The necessary invariants are checked and applied in
        :meth:`_postprocess_module`.
        """
        pass

    def _wrap_attribute(
        self,
        slow_attr: Any,
        fast_attr: Any | _Unusable,
        name: str,
    ) -> Any:
        """
        Return the wrapped version of an attribute.

        Parameters
        ----------
        slow_attr : Any
            The attribute from the slow module
        fast_mod : Any (or None)
            The same attribute from the fast module, if it exists
        name
            Name of attribute

        Returns
        -------
        Wrapped attribute
        """
        wrapped_attr: Any
        # TODO: what else should we make sure not to get from the fast
        # library?
        if name in {"__all__", "__dir__", "__file__", "__doc__"}:
            wrapped_attr = slow_attr
        elif self.fast_lib == self.slow_lib:
            # no need to create a fast-slow wrapper
            wrapped_attr = slow_attr
        if any(
            [
                slow_attr in get_registered_functions(),
                slow_attr in get_final_type_map(),
                slow_attr in get_intermediate_type_map(),
            ]
        ):
            # attribute already registered in self._wrapped_objs
            return self._wrapped_objs[slow_attr]
        if isinstance(slow_attr, ModuleType) and slow_attr.__name__.startswith(
            self.slow_lib
        ):
            # attribute is a submodule of the slow library,
            # replace the string "{slow_lib}" in the submodule's
            # name with "{self.mod_name}"
            # now, attempt to import the wrapped module, which will
            # recursively wrap all of its attributes:
            return importlib.import_module(
                rename_root_module(
                    slow_attr.__name__, self.slow_lib, self.mod_name
                )
            )
        if slow_attr in self._wrapped_objs:
            if type(fast_attr) is _Unusable:
                # we don't want to replace a wrapped object that
                # has a usable fast object with a wrapped object
                # with a an unusable fast object.
                return self._wrapped_objs[slow_attr]
        if _is_function_or_method(slow_attr):
            wrapped_attr = _FunctionProxy(fast_attr, slow_attr)
        else:
            wrapped_attr = slow_attr
        return wrapped_attr

    @classmethod
    @abstractmethod
    def install(
        cls, destination_module: str, fast_lib: str, slow_lib: str
    ) -> Self | None:
        """
        Install the loader in sys.meta_path.

        Parameters
        ----------
        destination_module
            Name under which the importer will kick in
        fast_lib
            Name of fast module
        slow_lib
            Name of slow module we are trying to mimic

        Returns
        -------
        Instance of the class (or None if the loader was not installed)

        Notes
        -----
        This function is idempotent. If called with the same arguments
        a second time, it does not create a new loader, but instead
        returns the existing loader from ``sys.meta_path``.

        """
        pass


class ModuleAccelerator(ModuleAcceleratorBase):
    """
    A finder and loader that produces "accelerated" modules.

    When someone attempts to import the specified slow library with
    this finder enabled, we intercept the import and deliver an
    equivalent, accelerated, version of the module. This provides
    attributes and modules that check if they are being used from
    "within" the slow (or fast) library themselves. If this is the
    case, the implementation is forwarded to the actual slow library
    implementation, otherwise a proxy implementation is used (which
    attempts to call the fast version first).
    """

    _denylist: tuple[str]
    _disable_count: defaultdict[int, int]
    _module_cache_prefix: str = "_slow_lib_"

    # TODO: Add possibility for either an explicit allow-list of
    # libraries where the slow_lib should be wrapped, or, more likely
    # a block-list that adds to the set of libraries where no proxying occurs.
    def __new__(
        cls,
        fast_lib,
        slow_lib,
    ):
        self = super().__new__(
            cls,
            slow_lib,
            fast_lib,
            slow_lib,
        )
        # Import the real versions of the modules so that we can
        # rewrite the sys.modules cache.
        slow_module = importlib.import_module(slow_lib)
        fast_module = importlib.import_module(fast_lib)
        # Note, this is not thread safe, but install() below grabs the
        # lock for the whole initialisation and modification of
        # sys.meta_path.
        for mod in sys.modules.copy():
            if mod.startswith(self.slow_lib):
                if mod == "pandas._config.config":
                    # Since it is possible for state to diverge between the proxy and real
                    # module, we skip wrapping this one entirely.
                    # For example, running tests in pandas/tests/config/test_config.py
                    # mutates internal globals like _registered_options on the proxy,
                    # while register_option() reads them from the real module.
                    # Therefore the two get out of sync and raise duplicate registration errors.
                    # Keeping the real module here avoids that split state.
                    continue
                sys.modules[self._module_cache_prefix + mod] = sys.modules[mod]
                del sys.modules[mod]
        self._denylist = (*slow_module.__path__, *fast_module.__path__)

        # This initialization does not need to be protected since a given instance is
        # always being created on a given thread.
        self._disable_count = defaultdict(int)
        return self

    def _populate_module(self, mod: ModuleType):
        mod_name = mod.__name__

        # Here we attempt to import "_fsproxy_slow_lib.x.y.z", but
        # "_fsproxy_slow_lib" does not exist anywhere as a real file, so
        # how does this work?
        # The importer attempts to import ".z" by first importing
        # "_fsproxy_slow_lib.x.y", this recurses until we find
        # "_fsproxy_slow_lib.x" (say), which does exist because we set that up
        # in __init__. Now the importer looks at the __path__
        # attribute of "x" and uses that to find the relative location
        # to look for "y". This __path__ points to the real location
        # of "slow_lib.x". So, as long as we rewire the _already imported_
        # slow_lib modules in sys.modules to _fsproxy_slow_lib, when we
        # get here this will find the right thing.
        # The above exposition is for lazily imported submodules (e.g.
        # avoiding circular imports by putting an import at function
        # level). For everything that is eagerly imported when we do
        # "import slow_lib" this import line is trivial because we
        # immediately pull the correct result out of sys.modules.
        slow_mod = importlib.import_module(
            rename_root_module(
                mod_name,
                self.slow_lib,
                self._module_cache_prefix + self.slow_lib,
            )
        )
        try:
            fast_mod = importlib.import_module(
                rename_root_module(mod_name, self.slow_lib, self.fast_lib)
            )
        except Exception:
            fast_mod = None

        # The version that will be used if called within a denylist
        # package
        real_attributes = {}
        # The version that will be used outside denylist packages
        for key in slow_mod.__dir__():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                slow_attr = getattr(slow_mod, key)
            fast_attr = getattr(fast_mod, key, _Unusable())
            real_attributes[key] = slow_attr
            try:
                wrapped_attr = self._wrap_attribute(slow_attr, fast_attr, key)
                self._wrapped_objs[slow_attr] = wrapped_attr
            except TypeError:
                # slow_attr is not hashable
                pass

        # Our module has (basically) no static attributes and instead
        # always delivers them dynamically where the behaviour is
        # dependent on the calling module.
        setattr(
            mod,
            "__getattr__",
            functools.partial(
                self.getattr_real_or_wrapped,
                real=real_attributes,
                wrapped_objs=self._wrapped_objs,
                loader=self,
            ),
        )

        # ...but, we want to pretend like we expose the same attributes
        # as the equivalent slow module
        setattr(mod, "__dir__", slow_mod.__dir__)

        # We set __path__ to the real path so that importers like
        # jinja2.PackageLoader("slow_mod") work correctly.
        if getattr(slow_mod, "__path__", False):
            assert mod.__spec__
            mod.__path__ = slow_mod.__path__
            mod.__spec__.submodule_search_locations = [*slow_mod.__path__]
        return self._postprocess_module(mod, slow_mod, fast_mod)

    @contextlib.contextmanager
    def disabled(self):
        """Return a context manager for disabling the module accelerator.

        Within the block, any wrapped objects will instead deliver
        attributes from their real counterparts (as if the current
        nested block were in the denylist).

        Returns
        -------
        Context manager for disabling things
        """
        self._disable_count[threading.get_ident()] += 1
        try:
            yield
        finally:
            self._disable_count[threading.get_ident()] -= 1

    @staticmethod
    def getattr_real_or_wrapped(
        name: str,
        *,
        real: dict[str, Any],
        wrapped_objs,
        loader: ModuleAccelerator,
    ) -> Any:
        """
        Obtain an attribute from a module from either the real or
        wrapped namespace.

        Parameters
        ----------
        name
            Attribute to return
        real
            Unwrapped "original" attributes
        wrapped
            Wrapped attributes
        loader
            Loader object that manages denylist and other skipping

        Returns
        -------
        The requested attribute (either real or wrapped)
        """
        use_real = (
            loader._disable_count[threading.get_ident()] > 0
            # If acceleration was disabled on the main thread, we should respect that.
            # This only works because we currently have no way to re-enable other than
            # exiting the disable context, so disabling on the parent thread means that
            # the inner threads will also typically be disabled. This logic breaks if
            # the parent thread queues work on a thread and only then disables
            # acceleration because in that case there is a potential race condition by
            # which the child thread may wind up disabled even though the parent was not
            # disabled when the child was launched. That is a fairly rare pattern though
            # and we can document the limitations.
            # The main thread is always started, so the ident is always an int
            or loader._disable_count[threading.main_thread().ident] > 0  # type: ignore[index]
        )
        if not use_real:
            # Only need to check the denylist if we're not turned off.
            frame = sys._getframe()
            # We cannot possibly be at the top level.
            assert frame.f_back
            calling_module = pathlib.PurePath(frame.f_back.f_code.co_filename)
            use_real = _caller_in_denylist(
                calling_module, tuple(loader._denylist)
            )
        try:
            if use_real:
                return real[name]
            else:
                return wrapped_objs[real[name]]
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")
        except TypeError:
            # real[name] is an unhashable type
            return real[name]

    @classmethod
    def install(
        cls,
        destination_module: str,
        fast_lib: str,
        slow_lib: str,
    ) -> Self | None:
        # This grabs the global _import_ lock to avoid concurrent
        # threads modifying sys.modules.
        # We also make sure that we finish installing ourselves in
        # sys.meta_path before releasing the lock so that there isn't
        # a race between our modification of sys.modules and someone
        # else importing the slow_lib before we have added ourselves
        # to the meta_path
        with ImportLock():
            if destination_module != slow_lib:
                raise RuntimeError(
                    f"Destination module '{destination_module}' must match"
                    f"'{slow_lib}' for this to work."
                )
            mode = deduce_cudf_pandas_mode(slow_lib, fast_lib)
            if mode.use_fast_lib:
                lib_wrappers = importlib.import_module(
                    f".._wrappers.{mode.slow_lib}", __name__
                )
                lib_wrappers.initial_setup()
            try:
                (self,) = (
                    p
                    for p in sys.meta_path
                    if isinstance(p, cls)
                    and p.slow_lib == mode.slow_lib
                    and p.fast_lib == mode.fast_lib
                )
            except ValueError:
                self = cls(mode.fast_lib, mode.slow_lib)
                sys.meta_path.insert(0, self)
            return self


def disable_module_accelerator() -> contextlib.ExitStack:
    """
    Temporarily disable any module acceleration.

    This function only offers limited guarantees of thread safety.
    Cases that will work:
        - multiple threads are launched and each independently turns off acceleration
        - a single thread turns off acceleration and then launches multiple threads
          inside the context manager

    Cases that trigger race conditions:
        - a single thread launches multiple threads and then enters the context manager
          while those threads are still running
        - nested thread launching and acceleration disabling, i.e. if a thread launches
          a thread that disables acceleration and then launches another thread, the
          innermost thread will not have the accelerator disabled.
    """
    with ImportLock(), contextlib.ExitStack() as stack:
        for finder in sys.meta_path:
            if isinstance(finder, ModuleAcceleratorBase):
                stack.enter_context(finder.disabled())
        return stack.pop_all()
    assert False  # pacify type checker


# because this function gets called so often and is quite
# expensive to run, we cache the results:
@functools.lru_cache(maxsize=1024)
def _caller_in_denylist(calling_module, denylist):
    CUDF_PANDAS_PATH = __file__.rsplit("/", 1)[0]
    return not calling_module.is_relative_to(CUDF_PANDAS_PATH) and any(
        calling_module.is_relative_to(path) for path in denylist
    )
