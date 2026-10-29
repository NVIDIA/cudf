# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Find pandas unit tests that reference non-public pandas APIs.

The pandas test files are statically analyzed. A test function is flagged
when its body, its decorators, or a fixture or helper function defined in the
same file references:

* a pandas object, reached through an import or an attribute chain such as
  ``pd.core.common.foo``, that is not in the public API list and is not the
  same object as an entry in that list, or
* an underscore-prefixed attribute such as ``df._mgr`` that is not a
  documented public member name (for example ``_from_sequence``) and is not
  defined by the test code in the same directory.

The installed pandas must be the same version as the analyzed tests and the
public API list.

``--format plugin`` prints entries for ``pandas-testing-plugin.py``. By
default it includes only tests that reference a non-public API directly;
``--include-indirect`` also includes tests that do so only through a fixture
or helper.

Usage:
    python find_private_api_tests.py public_api.txt <tests root> [paths ...]
    python find_private_api_tests.py public_api.txt <tests root> --format plugin
        --collected collected.txt --plugin pandas-testing-plugin.py
"""

import argparse
import ast
import importlib
import inspect
import json
import re
import runpy
import types
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any

SKIP_REASON = "Tests private pandas APIs"

#: Non-public pandas modules and names used as test infrastructure rather
#: than as the subject of a test. References to these are never flagged.
TEST_INFRASTRUCTURE = (
    "pandas._config.using_string_dtype",
    "pandas._testing",
    "pandas.compat",
    "pandas.conftest",
    "pandas.tests",
    "pandas.util._test_decorators",
    "pandas.util.version",
)

#: Public pandas names that the API reference does not list.
UNDOCUMENTED_PUBLIC = (
    "pandas.DataFrame._metadata",
    "pandas.Series._metadata",
    "pandas.api.extensions.ExtensionDtype.construct_from_string",
    "pandas.api.extensions.ExtensionDtype.is_dtype",
    "pandas.api.extensions.no_default",
    "pandas.api.extensions.take",
    "pandas.io.parsers.TextFileReader",
    "pandas.io.stata.StataWriterUTF8",
    "pandas.options",
    "pandas.to_pickle",
    "pandas.tseries.frequencies.is_subperiod",
    "pandas.tseries.frequencies.is_superperiod",
)

#: Namespaces that pandas declares public in full.
PUBLIC_NAMESPACES = ("pandas.api", "pandas.tseries.holiday")

#: Underscore attributes that implement the IPython display protocol.
PROTOCOL_ATTRIBUTES = frozenset(
    {
        "_ipython_display_",
        "_ipython_key_completions_",
        "_repr_html_",
        "_repr_latex_",
        "_repr_mimebundle_",
    }
)

PANDAS_PATH = re.compile(r"^pandas(\.\w+)+$")


class OutputFormat(Enum):
    REPORT = "report"
    PLUGIN = "plugin"


@dataclass(frozen=True, slots=True)
class Finding:
    line: int
    #: A fully qualified pandas name, or an attribute name if ``attribute``.
    name: str
    attribute: bool = False
    #: The fixture or helper that contains the reference, or None if the
    #: test function itself contains it.
    via: str | None = None

    @property
    def description(self) -> str:
        if self.attribute:
            return f"attribute `{self.name}`"
        return f"`{self.name}`"

    @property
    def skip_reason(self) -> str:
        if self.attribute:
            return f"{SKIP_REASON}, for example the {self.name} attribute"
        return f"{SKIP_REASON}, for example {self.name}"

    def __str__(self) -> str:
        if self.via is None:
            return f"line {self.line}: {self.description}"
        return f"line {self.line}: {self.description} (via `{self.via}`)"


@dataclass(frozen=True, slots=True)
class TestFunction:
    test_id: str
    findings: tuple[Finding, ...]

    def is_flagged(self, *, include_indirect: bool) -> bool:
        return any(
            include_indirect or finding.via is None
            for finding in self.findings
        )

    @property
    def skip_reason(self) -> str:
        """The reason naming one finding, preferring a direct one."""
        return self.findings[0].skip_reason


def is_private_name(name: str) -> bool:
    return name.startswith("_") and not (
        name.startswith("__") and name.endswith("__")
    )


def is_under(path: str, namespaces: Iterable[str]) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}.")
        for prefix in namespaces
    )


@cache
def resolve(path: str) -> tuple[object, ...]:
    """Return the objects along ``path``, stopping at the first failure."""
    parts = path.split(".")
    try:
        obj = importlib.import_module(parts[0])
    except Exception:
        return ()
    resolved = [obj]
    for i, part in enumerate(parts[1:], start=2):
        try:
            obj = getattr(obj, part)
        except Exception:
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except Exception:
                break
        resolved.append(obj)
    return tuple(resolved)


class PublicAPI:
    def __init__(self, names: Iterable[str]) -> None:
        self.names = frozenset((*names, *UNDOCUMENTED_PUBLIC))
        objects = tuple(obj for name in self.names for obj in resolve(name))
        # Keep references so the ids stay valid.
        self._objects = objects
        self.object_ids = frozenset(map(id, objects))
        self.modules = frozenset(
            obj.__name__
            for name in self.names
            for obj in resolve(name)
            if isinstance(obj, types.ModuleType)
        )
        self.prefixes = frozenset(
            name.rsplit(".", i)[0]
            for name in self.names
            for i in range(1, name.count("."))
        )
        self.member_names = frozenset(
            name.rsplit(".", 1)[-1] for name in self.names
        )

    def is_private_member(self, name: str) -> bool:
        return (
            is_private_name(name)
            and name not in self.member_names
            and name not in PROTOCOL_ATTRIBUTES
        )

    @cache
    def is_public(self, path: str) -> bool:
        if (
            path in self.names
            or path in self.prefixes
            or is_under(path, TEST_INFRASTRUCTURE)
        ):
            return True
        parts = path.split(".")
        if is_under(path, PUBLIC_NAMESPACES):
            return not any(map(self.is_private_member, parts))
        if (
            len(parts) > 2
            and parts[-1] in self.member_names
            and self.is_public(".".join(parts[:-1]))
        ):
            return True
        resolved = resolve(path)
        if len(resolved) == len(parts):
            obj = resolved[-1]
            if id(obj) in self.object_ids:
                return True
            if isinstance(obj, types.ModuleType):
                return obj.__name__ in self.modules
            if inspect.isclass(obj) or inspect.isroutine(obj):
                module = getattr(obj, "__module__", None) or ""
                return not module.startswith("pandas")
        modules = [
            i
            for i, obj in enumerate(resolved)
            if isinstance(obj, types.ModuleType)
        ]
        if not modules or resolved[modules[-1]].__name__ not in self.modules:
            return False
        return not any(map(self.is_private_member, parts[modules[-1] + 1 :]))


def import_aliases(tree: ast.Module) -> dict[str, str | None]:
    """Map names bound by imports to their pandas path, or None if not pandas."""
    aliases: dict[str, str | None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None:
                    root = alias.name.split(".", 1)[0]
                    aliases[root] = root if root == "pandas" else None
                else:
                    aliases[alias.asname] = (
                        alias.name
                        if alias.name.split(".", 1)[0] == "pandas"
                        else None
                    )
        elif isinstance(node, ast.ImportFrom):
            is_pandas = node.level == 0 and (
                (node.module or "").split(".", 1)[0] == "pandas"
            )
            for alias in node.names:
                aliases[alias.asname or alias.name] = (
                    f"{node.module}.{alias.name}" if is_pandas else None
                )
    return aliases


@cache
def locally_defined_private_names(directory: Path) -> frozenset[str]:
    """Underscore names that the test code in ``directory`` defines itself."""
    names = set()
    for file in directory.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                names.add(node.name)
            elif isinstance(node, ast.Attribute) and isinstance(
                node.ctx, ast.Store
            ):
                names.add(node.attr)
            elif isinstance(node, ast.Name) and isinstance(
                node.ctx, ast.Store
            ):
                names.add(node.id)
    return frozenset(name for name in names if is_private_name(name))


def attribute_chain(node: ast.Attribute) -> tuple[str, ...] | None:
    parts = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


class FindingCollector(ast.NodeVisitor):
    def __init__(
        self,
        public_api: PublicAPI,
        aliases: Mapping[str, str | None],
        local_names: frozenset[str],
    ) -> None:
        self.public_api = public_api
        self.aliases = aliases
        self.local_names = local_names
        self.findings: list[Finding] = []
        self.referenced: set[str] = set()

    def check_path(self, node: ast.AST, path: str) -> None:
        if not self.public_api.is_public(path):
            self.findings.append(Finding(node.lineno, path))

    def check_attribute(self, node: ast.AST, attr: str) -> None:
        if (
            self.public_api.is_private_member(attr)
            and attr not in self.local_names
        ):
            self.findings.append(Finding(node.lineno, attr, attribute=True))

    def visit_Name(self, node: ast.Name) -> None:
        self.referenced.add(node.id)
        if (path := self.aliases.get(node.id)) is not None:
            self.check_path(node, path)

    def visit_arg(self, node: ast.arg) -> None:
        self.referenced.add(node.arg)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = attribute_chain(node)
        if chain is not None:
            root, *attrs = chain
            if root in ("self", "cls"):
                # self._helper is test code, but self.obj._mgr is not.
                self.referenced.add(attrs[0])
                for attr in attrs[1:]:
                    self.check_attribute(node, attr)
                return
            if root in self.aliases:
                self.referenced.add(root)
                if (path := self.aliases[root]) is not None:
                    self.check_path(node, ".".join((path, *attrs)))
                return
        self.check_attribute(node, node.attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "setattr", "hasattr", "delattr")
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self.check_attribute(node, node.args[1].value)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and PANDAS_PATH.match(node.value):
            self.check_path(node, node.value)


@dataclass(frozen=True, slots=True)
class Definition:
    name: str
    findings: tuple[Finding, ...]
    referenced: frozenset[str]


def analyze_definition(
    name: str,
    nodes: Iterable[ast.AST],
    public_api: PublicAPI,
    aliases: Mapping[str, str | None],
    local_names: frozenset[str],
) -> Definition:
    collector = FindingCollector(public_api, aliases, local_names)
    for node in nodes:
        collector.visit(node)
    return Definition(
        name,
        tuple(collector.findings),
        frozenset(collector.referenced - {name}),
    )


def iter_definitions(
    body: Iterable[ast.stmt],
    public_api: PublicAPI,
    aliases: Mapping[str, str | None],
    local_names: frozenset[str],
) -> Iterator[tuple[str, Definition]]:
    """Functions and simple assignments defined directly in ``body``."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes = (*node.decorator_list, node.args, *node.body)
            yield (
                node.name,
                analyze_definition(
                    node.name, nodes, public_api, aliases, local_names
                ),
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else (node.target,)
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    yield (
                        target.id,
                        analyze_definition(
                            target.id,
                            (node.value,),
                            public_api,
                            aliases,
                            local_names,
                        ),
                    )


def gather_findings(
    definition: Definition, scopes: tuple[Mapping[str, Definition], ...]
) -> tuple[Finding, ...]:
    """Findings of ``definition`` and the same-file definitions it uses."""
    findings = list(definition.findings)
    seen = {definition.name}
    pending = sorted(definition.referenced)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        helper = next((scope[name] for scope in scopes if name in scope), None)
        if helper is None:
            continue
        findings.extend(
            Finding(f.line, f.name, f.attribute, name) for f in helper.findings
        )
        pending.extend(sorted(helper.referenced - seen))
    return tuple(
        sorted(set(findings), key=lambda f: (f.via or "", f.line, f.name))
    )


def iter_test_functions(
    body: Iterable[ast.stmt],
    prefix: str,
    scopes: tuple[Mapping[str, Definition], ...],
    public_api: PublicAPI,
    aliases: Mapping[str, str | None],
    local_names: frozenset[str],
) -> Iterator[TestFunction]:
    body = tuple(body)
    definitions = dict(
        iter_definitions(body, public_api, aliases, local_names)
    )
    scopes = (definitions, *scopes)
    for node in body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test"):
            yield TestFunction(
                f"{prefix}::{node.name}",
                gather_findings(definitions[node.name], scopes),
            )
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            yield from iter_test_functions(
                node.body,
                f"{prefix}::{node.name}",
                scopes,
                public_api,
                aliases,
                local_names,
            )


def analyze_file(
    file: Path, root: Path, public_api: PublicAPI
) -> tuple[TestFunction, ...]:
    tree = ast.parse(file.read_text())
    return tuple(
        iter_test_functions(
            tree.body,
            file.relative_to(root).as_posix(),
            (),
            public_api,
            import_aliases(tree),
            locally_defined_private_names(file.parent),
        )
    )


def strip_parameters(node_id: str) -> str:
    return node_id.split("[", 1)[0]


def print_report(results: Mapping[str, tuple[TestFunction, ...]]) -> None:
    tests = tuple(test for tests in results.values() for test in tests)
    direct = sum(test.is_flagged(include_indirect=False) for test in tests)
    indirect = sum(test.is_flagged(include_indirect=True) for test in tests)
    print(  # noqa: T201
        f"# {direct} of {len(tests)} test functions reference non-public "
        f"pandas APIs directly, and {indirect - direct} more only through a "
        "fixture or helper\n"
    )
    for file, file_tests in results.items():
        flagged = tuple(test for test in file_tests if test.findings)
        if not flagged:
            continue
        print(  # noqa: T201
            f"{file}: {len(flagged)} of {len(file_tests)} test functions "
            "flagged"
        )
        for test in flagged:
            print(f"  {test.test_id.removeprefix(file)}")  # noqa: T201
            for finding in test.findings:
                print(f"    {finding}")  # noqa: T201
        print()  # noqa: T201


def print_plugin_entries(
    results: Mapping[str, tuple[TestFunction, ...]],
    collected: Iterable[str],
    plugin: Mapping[str, Any] | None,
    *,
    include_indirect: bool,
) -> None:
    skipped = frozenset(plugin["NODEIDS_TO_SKIP"]) if plugin else frozenset()
    skipped_paths = tuple(plugin["NODEIDS_PATHS_TO_SKIP"]) if plugin else ()
    collected_by_test = defaultdict(list)
    for node_id in collected:
        collected_by_test[strip_parameters(node_id)].append(node_id)
    paths, node_ids = {}, {}
    for file, tests in results.items():
        if any(file.startswith(path) for path in skipped_paths):
            continue
        flagged = tuple(
            test
            for test in tests
            if test.is_flagged(include_indirect=include_indirect)
        )
        if not flagged:
            continue
        if len(flagged) == len(tests):
            paths[file] = flagged[0].skip_reason
            continue
        node_ids.update(
            (node_id, test.skip_reason)
            for test in flagged
            for node_id in collected_by_test.get(test.test_id, ())
            if node_id not in skipped
            and not any(node_id.startswith(path) for path in skipped_paths)
        )
    print(  # noqa: T201
        "# NODEIDS_PATHS_TO_SKIP candidates (every test function in the "
        "file is flagged)"
    )
    for path, reason in sorted(paths.items()):
        print(f"    {json.dumps(path)}: {json.dumps(reason)},")  # noqa: T201
    print("\n# NODEIDS_TO_SKIP candidates")  # noqa: T201
    for node_id, reason in sorted(node_ids.items()):
        print(f"    {json.dumps(node_id)}: {json.dumps(reason)},")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "public_api",
        type=Path,
        help="Output of collect_public_api.py",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory that contains the pandas tests/ directory",
    )
    parser.add_argument(
        "--format",
        type=OutputFormat,
        default=OutputFormat.REPORT,
        choices=tuple(OutputFormat),
        help="report: findings for each flagged test function. "
        "plugin: dictionary entries for pandas-testing-plugin.py",
    )
    parser.add_argument(
        "--collected",
        type=Path,
        help="Output of pytest --collect-only -q, required for --format plugin",
    )
    parser.add_argument(
        "--plugin",
        type=Path,
        help="pandas-testing-plugin.py; omit tests it already skips",
    )
    parser.add_argument(
        "--include-indirect",
        action="store_true",
        help="With --format plugin, also include tests that reference "
        "non-public APIs only through a fixture or helper",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=("tests",),
        help="Test files or directories relative to root (default: tests)",
    )
    args = parser.parse_args()
    if args.format is OutputFormat.PLUGIN and args.collected is None:
        parser.error("--format plugin requires --collected")

    public_api = PublicAPI(args.public_api.read_text().split())
    files = sorted(
        {
            file
            for path in args.paths
            for file in (
                [args.root / path]
                if (args.root / path).is_file()
                else (args.root / path).rglob("test_*.py")
            )
        }
    )
    results = {
        file.relative_to(args.root).as_posix(): analyze_file(
            file, args.root, public_api
        )
        for file in files
    }
    if args.format is OutputFormat.REPORT:
        print_report(results)
    else:
        collected = frozenset(
            line.strip()
            for line in args.collected.read_text().splitlines()
            if "::" in line
        )
        plugin = runpy.run_path(args.plugin) if args.plugin else None
        print_plugin_entries(
            results,
            collected,
            plugin,
            include_indirect=args.include_indirect,
        )


if __name__ == "__main__":
    main()
