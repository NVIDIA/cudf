# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This extension was written entirely by an AI coding agent. Its scope is
# deliberately limited to what is needed to render the cuDF Java API docs; it
# is not a general-purpose Java support layer for Breathe.

"""Render Doxygen XML generated from Java sources with Breathe.

Breathe only ships directives for the C, C++, Python, PHP and C# domains and
falls back to the C++ domain for everything else, which cannot parse Java
declarations (arrays, varargs, ``extends``, ``throws``, ...). This extension
adds a minimal ``java`` Sphinx domain and teaches Breathe to build Java
declarations for nodes whose source file has a ``.java`` extension. It also
adds javadoc-style member summaries, lists of inherited members, deprecation
badges, and links to JDK types.

Usage in conf.py::

    extensions = [..., "breathe_java"]
    breathe_domain_by_extension = {"java": "java"}
    breathe_java_projects = ["cudf-java"]

Each project listed in ``breathe_java_projects`` has its Doxygen XML
preprocessed when the extension is set up (see :func:`clean_java_xml`).

Both Breathe 4 and Breathe 5 are supported. Their renderers and Doxygen
parsers differ, so everything that touches Breathe's node objects goes through
the compatibility helpers below, and the summaries, inherited members and
deprecation information are computed from the XML directly.
"""

from __future__ import annotations

import copy
import dataclasses
import glob
import os
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import breathe
from breathe.renderer import sphinxrenderer
from breathe.renderer.sphinxrenderer import (
    BaseObject,
    DomainDirectiveFactory,
    SphinxRenderer,
)
from docutils import nodes
from packaging.version import Version
from sphinx import addnodes
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.nodes import make_id, make_refnode

if TYPE_CHECKING:
    from sphinx.application import Sphinx

logger = logging.getLogger(__name__)

DOMAIN = "java"

_BREATHE_GE_5 = Version(breathe.__version__) >= Version("5")

# Modifiers shown in declarations. Doxygen also reports ``native`` and
# ``synchronized`` in the type, but like javadoc we treat those as
# implementation details and drop them.
_SHOWN_MODIFIERS = ("abstract", "default", "final", "volatile", "transient")
_HIDDEN_MODIFIERS = ("native", "synchronized", "inline")
_KEYWORDS = frozenset(
    (
        "public",
        "protected",
        "private",
        "static",
        "class",
        "interface",
        "enum",
        "extends",
        "implements",
        "throws",
        "super",
        *_SHOWN_MODIFIERS,
    )
)
_PRIMITIVES = frozenset(
    (
        "void",
        "boolean",
        "byte",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
    )
)
_TOKEN_RE = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<ident>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)"
    r"|(?P<ellipsis>\.\.\.)"
    r"|(?P<punct>.)"
)

# JDK types referenced by the cuDF Java API, linked to the Java 8 API docs
# (the cuDF jar targets Java 8).
_JDK_API_URL = "https://docs.oracle.com/javase/8/docs/api/"
_JDK_TYPES = {
    "java.io": (
        "Closeable",
        "DataInputStream",
        "DataOutputStream",
        "File",
        "IOException",
        "InputStream",
        "OutputStream",
        "Serializable",
        "UncheckedIOException",
    ),
    "java.lang": (
        "ArithmeticException",
        "AutoCloseable",
        "Boolean",
        "Byte",
        "Character",
        "Class",
        "Double",
        "Error",
        "Exception",
        "Float",
        "IllegalArgumentException",
        "IllegalStateException",
        "IndexOutOfBoundsException",
        "Integer",
        "Iterable",
        "Long",
        "NullPointerException",
        "Number",
        "Object",
        "Runnable",
        "RuntimeException",
        "Short",
        "String",
        "Thread",
        "Throwable",
        "UnsupportedOperationException",
        "Void",
    ),
    "java.lang.ref": (
        "PhantomReference",
        "ReferenceQueue",
        "SoftReference",
        "WeakReference",
    ),
    "java.math": ("BigDecimal", "BigInteger", "RoundingMode"),
    "java.nio": ("ByteBuffer",),
    "java.nio.channels": ("FileChannel", "FileChannel.MapMode"),
    "java.util": (
        "ArrayList",
        "Collection",
        "EnumSet",
        "HashMap",
        "Iterator",
        "List",
        "Map",
        "Map.Entry",
        "Optional",
        "Set",
    ),
    "java.util.concurrent": ("TimeUnit",),
    "java.util.concurrent.locks": ("Lock",),
    "java.util.function": ("BiFunction", "Consumer", "Function", "Supplier"),
}
_JDK_ALIASES = {"MapMode": "FileChannel.MapMode", "Entry": "Map.Entry"}


# Doxygen refid prefix marking references to JDK types.
_JDK_REFID_PREFIX = "jdk:"


def _jdk_url(name: str) -> str | None:
    """Return the javadoc URL of a JDK type given by simple or full name."""
    name = re.sub(r"<.*", "", name).strip()
    for package, types in _JDK_TYPES.items():
        if name.startswith(package + "."):
            simple = name[len(package) + 1 :]
            if simple in types:
                return (
                    f"{_JDK_API_URL}{package.replace('.', '/')}/{simple}.html"
                )
    name = _JDK_ALIASES.get(name, name)
    for package, types in _JDK_TYPES.items():
        if name in types:
            return f"{_JDK_API_URL}{package.replace('.', '/')}/{name}.html"
    return None


# ---------------------------------------------------------------------------
# Breathe 4/5 compatibility
# ---------------------------------------------------------------------------


def _value(x) -> str:
    """Return a Doxygen kind or protection as a string.

    Breathe 5 represents these as enums, Breathe 4 as strings.
    """
    return getattr(x, "value", x)


def _compound(self: SphinxRenderer, refid: str):
    """Return ``(file_data, compounddef)`` for the compound with *refid*."""
    if _BREATHE_GE_5:
        file_data = self.parse_compound(refid)
        return file_data, file_data.compounddef[0]
    file_data = self.compound_parser.parse(refid)
    return file_data, file_data.compounddef


def _member_type(node):
    return node.type if _BREATHE_GE_5 else node.type_


def _renderer_tables() -> list[dict]:
    """Return the tables Breathe uses to dispatch nodes to visit methods."""
    names = ("node_handlers", "tagged_node_handlers", "methods")
    return [
        getattr(SphinxRenderer, n) for n in names if hasattr(SphinxRenderer, n)
    ]


def _handle_declaration(
    self: SphinxRenderer, node, objtype, declaration, **kwargs
):
    if _BREATHE_GE_5:
        return self.handle_declaration(node, objtype, declaration, **kwargs)
    return self.handle_declaration(
        node, declaration, obj_type=objtype, **kwargs
    )


# ---------------------------------------------------------------------------
# Doxygen XML preprocessing and the model of the documented API
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Member:
    id: str
    compound: str
    name: str
    kind: str
    prot: str
    static: bool
    args: str
    brief: str
    deprecated: bool
    enum_constant: bool
    constructor: bool

    @property
    def documented(self) -> bool:
        """Whether the member is rendered in the docs, as javadoc would."""
        compound = _compounds.get(self.compound)
        return (
            compound is not None
            and compound.documented
            and self.prot in ("public", "protected")
            and not self.name.startswith("[")
            and not (self.constructor and compound.kind == "enum")
        )


@dataclasses.dataclass
class _Compound:
    id: str
    fullname: str
    kind: str
    prot: str
    deprecated: bool
    members: list[str]
    #: ``(refid, prot)`` of every member, including inherited ones.
    all_members: list[tuple[str, str]]

    @property
    def documented(self) -> bool:
        return self.prot in ("public", "protected")


# The documented Java API, keyed by Doxygen id. Populated by clean_java_xml.
_members: dict[str, _Member] = {}
_compounds: dict[str, _Compound] = {}


def _xml_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _javadoc_link_text(text: str) -> str:
    """Show link text as javadoc does: ``Class.member`` and ``member``.

    Javadoc links are written ``Class#member`` and ``#member``.
    """
    text = re.sub(r"(?<=\w)#(?=\w)", ".", text)
    return re.sub(r"(?<!\w)#(?=\w)", "", text)


def _normalize_type(text: str) -> str:
    # Doxygen pads template brackets and array suffixes with spaces.
    text = re.sub(r"\s*<\s*", "<", text)
    text = re.sub(r"\s*>", ">", text)
    text = re.sub(r"\s+\[\]", "[]", text)
    text = re.sub(r",(?=\S)", ", ", text)
    return " ".join(text.split())


def _source_lines(
    filename: str, _cache: dict[str, list[str]] = {}
) -> list[str]:
    if filename not in _cache:
        try:
            with open(filename) as f:
                _cache[filename] = f.readlines()
        except OSError as e:
            # Visibility and declaration fixes that need the source are
            # skipped for this file.
            # Doxygen uses "[generated]" for members it synthesizes.
            if filename != "[generated]":
                logger.warning("cannot read Java source %r: %s", filename, e)
            _cache[filename] = []
    return _cache[filename]


def _is_annotated_deprecated(location: ET.Element | None) -> bool:
    """Whether the declaration at *location* has a ``@Deprecated`` annotation."""
    if location is None:
        return False
    lines = _source_lines(location.get("file", ""))
    index = int(location.get("line", "0")) - 1
    if not 0 <= index < len(lines):
        return False
    if "@Deprecated" in lines[index]:
        return True
    # Annotations on the lines directly above the declaration.
    index -= 1
    while index >= 0 and lines[index].strip().startswith("@"):
        if lines[index].strip().startswith("@Deprecated"):
            return True
        index -= 1
    return False


def clean_java_xml(path: str) -> None:
    """Fix up Doxygen's Java XML for Breathe and build the API model.

    - Breathe 5 validates compound kinds against the C++ schema, which has no
      top-level ``enum`` compound. Java enums are rewritten as classes; the
      original kind is still recoverable from the ``enum`` prefix of the id.
    - Doxygen fails to resolve some ``{@link Class#member}`` references,
      leaving them with an empty refid, and emits others with a refid that
      lacks the compound prefix. Both are repaired where possible.
      References to JDK types become links to the JDK documentation, and
      other unresolvable references are replaced by their text.
    - Elements that Breathe 4 cannot parse are replaced: ``{@code ...}`` by
      ``<computeroutput>``, ``<pre>`` blocks by ``<verbatim>``, and HTML
      character entities by the characters.
    - Link text is shown as in javadoc, ``Class.member`` instead of
      ``Class#member``.
    - Doxygen marks package-private classes and members of enums as public.
    - Public and protected members of package-private superclasses are
      copied into their public subclasses, as javadoc documents them.
    """
    trees = {
        fn: ET.parse(fn)
        for fn in glob.glob(os.path.join(path, "*.xml"))
        if os.path.basename(fn) not in ("index.xml", "Doxyfile.xml")
    }
    member_ids: dict[str, str] = {}
    qualified_ids: dict[str, list[str]] = {}
    interfaces: set[str] = set()
    for tree in trees.values():
        for compound in tree.getroot().iter("compounddef"):
            name = compound.findtext("compoundname").replace("::", ".")
            qualified_ids.setdefault(name, []).append(compound.get("id"))
            if compound.get("kind") == "interface":
                interfaces.add(name)
        for member in tree.getroot().iter("memberdef"):
            id_ = member.get("id")
            member_ids[id_[id_.rindex("_1") :]] = id_
            qualified_ids.setdefault(
                member.findtext("qualifiedname"), []
            ).append(id_)

    def resolve(text: str) -> str | None:
        target = re.sub(r"\(.*", "", text.strip()).replace("#", ".")
        matches = {
            id_
            for name, ids in qualified_ids.items()
            if name == target or name.endswith("." + target)
            for id_ in ids
        }
        return matches.pop() if len(matches) == 1 else None

    protections = {
        compound.get("id"): _declared_protection(compound, interfaces)
        for tree in trees.values()
        for compound in tree.getroot().iter("compounddef")
    }
    compounds = {
        compound.get("id"): compound
        for tree in trees.values()
        for compound in tree.getroot().iter("compounddef")
    }

    for fn, tree in trees.items():
        root = tree.getroot()
        changed = False
        for compound in root.iter("compounddef"):
            if compound.get("kind") == "enum":
                compound.set("kind", "class")
                changed = True
            if protections[compound.get("id")] in ("public", "protected"):
                changed |= _copy_hidden_base_members(
                    compound, compounds, protections, qualified_ids
                )
            prot = protections[compound.get("id")]
            if prot != compound.get("prot"):
                compound.set("prot", prot)
                changed = True
            # Breathe decides whether to render a nested class by the
            # protection recorded in its outer class.
            for inner in compound.iterfind("innerclass"):
                prot = protections.get(inner.get("refid"))
                if prot is not None and prot != inner.get("prot"):
                    inner.set("prot", prot)
                    changed = True
            if compound.get("id").startswith("enum"):
                changed |= _fix_enum_member_protection(compound)
        # Breathe 4 does not know Doxygen's elements for {@code ...}, <pre>
        # blocks and HTML character entities.
        for code in root.iter("javadoccode"):
            code.tag = "computeroutput"
            changed = True
        for pre in root.iter("preformatted"):
            text, tail = "".join(pre.itertext()), pre.tail
            pre.clear()
            pre.tag, pre.text, pre.tail = "verbatim", text, tail
            changed = True
        for parent in list(root.iter()):
            for child in list(parent):
                if child.tag in _ENTITY_CHARACTERS:
                    _replace_with_text(
                        parent, child, _ENTITY_CHARACTERS[child.tag]
                    )
                    changed = True
        for parent in list(root.iter()):
            for ref in list(parent):
                if ref.tag != "ref":
                    continue
                refid = ref.get("refid", "")
                if refid.startswith(_JDK_REFID_PREFIX):
                    continue
                if refid.startswith("_1"):
                    refid = member_ids.get(refid, "")
                elif not refid:
                    refid = resolve(ref.text or "") or ""
                if ref.text and "#" in ref.text:
                    ref.text = _javadoc_link_text(ref.text)
                    changed = True
                if refid:
                    if refid != ref.get("refid"):
                        ref.set("refid", refid)
                        changed = True
                    continue
                changed = True
                for name in (
                    (ref.text or "").split("(")[0],
                    re.sub(r"[.#(].*", "", ref.text or ""),
                ):
                    if _jdk_url(name):
                        # Resolved by _resolve_external_java_types. Breathe 4
                        # cannot parse links (ulink) inside code markup.
                        ref.set("refid", _JDK_REFID_PREFIX + name)
                        break
                if ref.get("refid"):
                    continue
                _replace_with_text(parent, ref, ref.text or "")
        if changed:
            tree.write(fn)
        _add_to_model(root)

    index = os.path.join(path, "index.xml")
    with open(index) as f:
        text = f.read()
    new_text = re.sub(r'(<compound\s[^>]*\bkind=)"enum"', r'\1"class"', text)
    if new_text != text:
        with open(index, "w") as f:
            f.write(new_text)


_ENTITY_CHARACTERS = {
    "ndash": "\u2013",
    "mdash": "\u2014",
    "nonbreakablespace": "\u00a0",
}


def _replace_with_text(parent: ET.Element, child: ET.Element, text: str):
    """Replace the element *child* of *parent* by *text*."""
    index = list(parent).index(child)
    text += child.tail or ""
    if index == 0:
        parent.text = (parent.text or "") + text
    else:
        previous = parent[index - 1]
        previous.tail = (previous.tail or "") + text
    parent.remove(child)


def _add_to_model(root: ET.Element) -> None:
    for compound in root.iter("compounddef"):
        cid = compound.get("id")
        if not compound.get("language") == "Java":
            continue
        kind = _java_kind(cid, compound.get("kind"))
        member_ids = []
        for member in compound.iter("memberdef"):
            id_ = member.get("id")
            type_ = _normalize_type(_xml_text(member.find("type")))
            for modifier in (*_SHOWN_MODIFIERS, *_HIDDEN_MODIFIERS):
                type_ = re.sub(rf"^{modifier}\b\s*", "", type_)
            detailed = member.find("detaileddescription")
            _members[id_] = _Member(
                id=id_,
                compound=cid,
                name=member.findtext("name"),
                kind=member.get("kind"),
                prot=member.get("prot"),
                static=member.get("static") == "yes",
                args=_normalize_type(member.findtext("argsstring") or ""),
                brief=_xml_text(member.find("briefdescription")),
                deprecated=(
                    detailed is not None
                    and any(
                        x.get("id", "").startswith("deprecated")
                        for x in detailed.iter("xrefsect")
                    )
                )
                or _is_annotated_deprecated(member.find("location")),
                enum_constant=kind == "enum" and not type_,
                constructor=member.get("kind") == "function" and not type_,
            )
            member_ids.append(id_)
        detailed = compound.find("detaileddescription")
        _compounds[cid] = _Compound(
            id=cid,
            fullname=compound.findtext("compoundname").replace("::", "."),
            kind=kind,
            prot=compound.get("prot"),
            deprecated=(
                detailed is not None
                and any(
                    x.get("id", "").startswith("deprecated")
                    for x in detailed.iter("xrefsect")
                )
            )
            or _is_annotated_deprecated(compound.find("location")),
            members=member_ids,
            all_members=[
                (m.get("refid"), m.get("prot"))
                for m in compound.iterfind("listofallmembers/member")
            ],
        )


def _declared_protection(
    compound: ET.Element, interfaces: set[str] | None = None
) -> str:
    """Return the protection of a class as declared in its source file.

    Doxygen reports package-private and some private nested classes as
    public. Classes nested in an interface are implicitly public.
    """
    prot = compound.get("prot")
    if prot != "public":
        return prot
    location = compound.find("location")
    header = _source_header(location.get("file"), int(location.get("line")))
    if header is None:
        return prot
    if modifier := re.search(r"\b(public|protected|private)\b", header):
        return modifier.group(1)
    outer = (
        compound.findtext("compoundname").replace("::", ".").rpartition(".")[0]
    )
    if interfaces is not None and outer in interfaces:
        return prot
    return "package"


def _copy_hidden_base_members(
    compound: ET.Element,
    compounds: dict[str, ET.Element],
    protections: dict[str, str],
    qualified_ids: dict[str, list[str]],
) -> bool:
    """Copy the members *compound* inherits from non-public superclasses.

    Like javadoc, document the public and protected members of a
    package-private superclass as members of the public subclass, since the
    superclass itself is not documented.
    """
    fullname = compound.findtext("compoundname").replace("::", ".")
    own = {
        (m.findtext("name"), m.findtext("argsstring"))
        for m in compound.iter("memberdef")
    }
    changed = False
    base = compound
    while True:
        # Java classes have one superclass; skip interfaces.
        bases = [
            compounds[id_]
            for ref in base.iterfind("basecompoundref")
            for id_ in qualified_ids.get(ref.text.replace("::", "."), [])
            if id_ in compounds and not id_.startswith("interface")
        ]
        if not bases or protections[bases[0].get("id")] in (
            "public",
            "protected",
        ):
            return changed
        base = bases[0]
        base_id = base.get("id")
        base_name = base.findtext("compoundname").replace("::", ".")
        simple_name = base_name.rpartition(".")[2]
        for section in base.iterfind("sectiondef"):
            for member in section.iterfind("memberdef"):
                name = member.findtext("name")
                key = (name, member.findtext("argsstring"))
                if (
                    member.get("prot") not in ("public", "protected")
                    or name in (simple_name, "[static initializer]")
                    or key in own
                ):
                    continue
                own.add(key)
                clone = copy.deepcopy(member)
                old_id = member.get("id")
                new_id = compound.get("id") + old_id[len(base_id) :]
                clone.set("id", new_id)
                for tag in ("definition", "qualifiedname"):
                    element = clone.find(tag)
                    if element is not None and element.text:
                        element.text = element.text.replace(
                            base_name + ".", fullname + "."
                        )
                target = compound.find(
                    f"sectiondef[@kind='{section.get('kind')}']"
                )
                if target is None:
                    target = ET.SubElement(
                        compound, "sectiondef", kind=section.get("kind")
                    )
                    # Keep the DTD's element order: sections come before
                    # the descriptions.
                    compound.remove(target)
                    position = list(compound).index(
                        compound.find("briefdescription")
                    )
                    compound.insert(position, target)
                target.append(clone)
                for ref in compound.iterfind(
                    f"listofallmembers/member[@refid='{old_id}']"
                ):
                    ref.set("refid", new_id)
                changed = True


def _fix_enum_member_protection(compound: ET.Element) -> bool:
    """Mark enum members declared without an access modifier as package-private."""
    changed = False
    for member in compound.iter("memberdef"):
        if member.get("prot") != "public" or not member.findtext("type"):
            # Enum constants have no type and are always public.
            continue
        location = member.find("location")
        lines = _source_lines(location.get("file", ""))
        index = int(location.get("line", "0")) - 1
        if not 0 <= index < len(lines):
            continue
        declaration = lines[index].split(member.findtext("name"), 1)[0]
        if not re.search(r"\b(public|protected)\b", declaration):
            member.set("prot", "package")
            changed = True
    return changed


def documented_java_classes(path: str) -> dict[str, list[dict[str, str]]]:
    """Return the public top-level classes of each package in a Java project.

    The result maps package names to lists of ``{"name", "refid", "kind",
    "brief"}`` dicts sorted by class name. Packages without public classes are
    omitted.
    """
    index = ET.parse(os.path.join(path, "index.xml")).getroot()
    packages: dict[str, list[dict[str, str]]] = {}
    namespaces = {
        c.findtext("name").replace("::", ".")
        for c in index.findall("./compound[@kind='namespace']")
    }
    for compound in index.findall("./compound"):
        # Enums are listed as classes once the XML has been cleaned.
        if compound.get("kind") not in ("class", "interface", "enum"):
            continue
        refid = compound.get("refid")
        qualname = compound.findtext("name").replace("::", ".")
        package, _, name = qualname.rpartition(".")
        if package not in namespaces:
            # Nested classes are documented as part of their outer class.
            continue
        compounddef = ET.parse(os.path.join(path, f"{refid}.xml")).find(
            "./compounddef"
        )
        # This runs before clean_java_xml, so check the source for
        # package-private classes too.
        if _declared_protection(compounddef) != "public":
            continue
        # This runs before clean_java_xml, so fix up link text here too.
        brief = _javadoc_link_text(
            _xml_text(compounddef.find("./briefdescription"))
        )
        kind = _java_kind(refid, compounddef.get("kind"))
        location = compounddef.find("./location")
        header = _source_header(
            location.get("file"), int(location.get("line"))
        )
        if kind == "interface" and header and "@interface" in header:
            kind = "annotation"
        packages.setdefault(package, []).append(
            {"name": name, "refid": refid, "kind": kind, "brief": brief}
        )
    for classes in packages.values():
        classes.sort(key=lambda c: c["name"])
    return dict(sorted(packages.items()))


def _java_kind(refid: str, kind: str) -> str:
    if refid.startswith("enum"):
        return "enum"
    return kind


_RST_SPECIAL = re.compile(r"([\\`*_|])")

# Package summary sections, in the order javadoc uses.
_SUMMARY_SECTIONS = (
    ("Interfaces", lambda c: c["kind"] == "interface"),
    ("Classes", lambda c: c["kind"] == "class" and not _is_throwable(c)),
    ("Enums", lambda c: c["kind"] == "enum"),
    ("Exceptions", lambda c: c["kind"] == "class" and _is_throwable(c)),
    ("Annotation Types", lambda c: c["kind"] == "annotation"),
)


def _is_throwable(cls: dict[str, str]) -> bool:
    return cls["name"].endswith(("Exception", "Error"))


def _heading(text: str, underline: str) -> list[str]:
    return [text, underline * len(text), ""]


def generate_api_pages(
    xml_dir: str, outdir: str, project: str, title: str
) -> None:
    """Write reST pages documenting every public class of a Java project.

    *outdir* receives an ``index.rst`` that lists the packages, one page per
    package summarizing its classes, and one page per top-level class. Files
    are only rewritten when their content changes so that Sphinx does not
    reread unchanged pages, and stale pages are removed.
    """
    packages = documented_java_classes(xml_dir)
    pages: dict[str, list[str]] = {}

    index = [*_heading(title, "="), ".. toctree::", "   :maxdepth: 1", ""]
    index += [f"   {package}" for package in packages]
    pages["index"] = index

    for package, classes in packages.items():
        page = [*_heading(package, "=")]
        for section, predicate in _SUMMARY_SECTIONS:
            members = [c for c in classes if predicate(c)]
            if not members:
                continue
            page += _heading(section, "-")
            page += [".. list-table::", "   :widths: 30 70", ""]
            for cls in members:
                brief = _RST_SPECIAL.sub(r"\\\1", cls["brief"])
                page.append(
                    f"   * - :java:type:`{cls['name']} <{package}.{cls['name']}>`"
                )
                page.append(f"     - {brief}")
            page.append("")
        page += [".. toctree::", "   :hidden:", ""]
        page += [f"   {package}.{cls['name']}" for cls in classes]
        pages[package] = page

        for cls in classes:
            qualname = f"{package}.{cls['name']}".replace(".", "::")
            directive = "doxygenclass"
            if cls["kind"] in ("interface", "annotation"):
                directive = "doxygeninterface"
            pages[f"{package}.{cls['name']}"] = [
                *_heading(cls["name"], "="),
                f".. {directive}:: {qualname}",
                f"   :project: {project}",
                "   :members:",
                "   :protected-members:",
                "   :undoc-members:",
            ]

    os.makedirs(outdir, exist_ok=True)
    for name, lines in pages.items():
        path = os.path.join(outdir, f"{name}.rst")
        content = "\n".join(lines).rstrip() + "\n"
        try:
            with open(path) as f:
                if f.read() == content:
                    continue
        except FileNotFoundError:
            pass
        with open(path, "w") as f:
            f.write(content)
    for path in glob.glob(os.path.join(outdir, "*.rst")):
        if os.path.basename(path)[: -len(".rst")] not in pages:
            os.remove(path)


# ---------------------------------------------------------------------------
# The java domain
# ---------------------------------------------------------------------------


def _split_signature(sig: str) -> list[tuple[str, str]]:
    return [
        (m.lastgroup, m.group())
        for m in _TOKEN_RE.finditer(sig)
        if m.lastgroup
    ]


class JavaObject(ObjectDescription[str]):
    """Describe a Java declaration given as a single line of Java source.

    The declaration is tokenized rather than parsed: keywords and primitive
    types are highlighted, the declared name is marked up as such, and any
    other identifier becomes a cross-reference to a Java type, which links to
    the JDK documentation for JDK types and is rendered as plain text if no
    documented type of that name exists.
    """

    doc_field_types = [
        GroupedField(
            "parameter",
            label="Parameters",
            names=("param", "parameter", "arg", "argument"),
            can_collapse=True,
        ),
        Field(
            "returnvalue",
            label="Returns",
            has_arg=False,
            names=("return", "returns"),
        ),
        GroupedField(
            "exceptions",
            label="Throws",
            rolename="type",
            names=("throws", "throw", "exception", "raises"),
            can_collapse=True,
        ),
        GroupedField(
            "typeparameter",
            label="Type Parameters",
            names=("tparam",),
            can_collapse=True,
        ),
    ]

    #: Fully qualified name of the declared object, e.g.
    #: ``ai.rapids.cudf.Table.concatenate``. Defaults to the signature.
    fullname: str | None = None
    #: Whether the declared object is deprecated.
    deprecated: bool = False

    def _name_index(self, tokens: list[tuple[str, str]], name: str) -> int:
        if self.objtype in ("method", "constructor"):
            paren = next(
                i for i, (_, text) in enumerate(tokens) if text == "("
            )
            candidates = range(paren - 1, -1, -1)
        elif self.objtype == "field":
            stop = next(
                (i for i, (_, text) in enumerate(tokens) if text == "="),
                len(tokens),
            )
            candidates = range(stop - 1, -1, -1)
        else:
            keyword = next(
                i
                for i, (_, text) in enumerate(tokens)
                if text in ("class", "interface", "enum")
            )
            candidates = range(keyword + 1, len(tokens))
        return next(i for i in candidates if tokens[i][1] == name)

    def handle_signature(self, sig: str, signode: addnodes.desc_signature):
        fullname = self.fullname or sig
        prefix, _, name = fullname.rpartition(".")
        tokens = _split_signature(sig)
        try:
            name_index = self._name_index(tokens, name)
        except StopIteration:
            logger.warning(
                "Cannot find %r in Java %s declaration %r",
                name,
                self.objtype,
                sig,
                location=self.get_location(),
            )
            name_index = -1

        self._param_types: list[str] = []
        i = 0
        while i < len(tokens):
            kind, text = tokens[i]
            if i == name_index:
                if prefix and self.objtype in ("class", "interface", "enum"):
                    signode += addnodes.desc_addname(
                        prefix + ".", prefix + "."
                    )
                signode += addnodes.desc_name(text, text)
            elif (
                i == name_index + 1
                and text == "("
                and self.objtype in ("method", "constructor")
            ):
                i = self._parameter_list(tokens, i, sig, signode)
            else:
                signode += self._token_node(kind, text)
            i += 1
        if self.deprecated:
            signode += addnodes.desc_sig_space()
            signode += nodes.inline(
                "Deprecated", "Deprecated", classes=["java-deprecated-badge"]
            )
        return fullname

    def _token_node(self, kind: str, text: str) -> nodes.Node:
        if kind == "space":
            return addnodes.desc_sig_space()
        if text in _KEYWORDS:
            return addnodes.desc_sig_keyword(text, text)
        if text in _PRIMITIVES:
            return addnodes.desc_sig_keyword_type(text, text)
        if kind == "ident":
            return self._type_xref(text)
        return addnodes.desc_sig_punctuation(text, text)

    def _parameter_list(
        self,
        tokens: list[tuple[str, str]],
        start: int,
        sig: str,
        signode: addnodes.desc_signature,
    ) -> int:
        """Render the parameters starting at ``tokens[start] == "("``.

        Returns the index of the closing parenthesis.
        """
        params: list[list[tuple[str, str]]] = [[]]
        depth = 0
        i = start + 1
        while i < len(tokens):
            kind, text = tokens[i]
            if text in "(<":
                depth += 1
            elif text == ">" or (text == ")" and depth > 0):
                depth -= 1
            elif text == ")":
                break
            elif text == "," and depth == 0:
                params.append([])
                i += 1
                continue
            params[-1].append((kind, text))
            i += 1

        paramlist = addnodes.desc_parameterlist()
        for param in params:
            while param and param[0][0] == "space":
                param.pop(0)
            if not param:
                continue
            node = addnodes.desc_parameter()
            # The last identifier is the parameter name.
            name_index = max(
                j for j, (kind, _) in enumerate(param) if kind == "ident"
            )
            for j, (kind, text) in enumerate(param):
                if j == name_index:
                    node += addnodes.desc_sig_name(text, text)
                else:
                    node += self._token_node(kind, text)
            paramlist += node
            self._param_types.append(
                "".join(t for _, t in param[:name_index]).strip()
            )

        max_len = self.config.maximum_signature_line_length or 0
        paramlist["multi_line_parameter_list"] = (
            "single-line-parameter-list" not in self.options
            and max_len > 0
            and len(" ".join(sig.split())) > max_len
        )
        signode += paramlist
        return i

    def _type_xref(self, target: str) -> addnodes.pending_xref:
        ref = addnodes.pending_xref(
            "",
            refdomain=DOMAIN,
            reftype="type",
            reftarget=target,
            refwarn=False,
        )
        ref["java:scope"] = self.env.ref_context.get("java:class")
        ref += addnodes.desc_sig_name(target, target)
        return ref

    def _node_id(self, name: str) -> str:
        """Return a stable anchor for *name*.

        Overloads after the first one are distinguished by their parameter
        types, e.g. ``java-ai.rapids.cudf.Table.Table-long-Arr``.
        """
        document = self.state.document
        node_id = f"{DOMAIN}-{name}"
        if node_id in document.ids and self.objtype in (
            "method",
            "constructor",
        ):
            types = (
                "-".join(
                    t.replace("[]", "-Arr").replace("...", "-Arr")
                    for t in self._param_types
                )
                or "noargs"
            )
            node_id = re.sub(r"[^\w.-]+", "-", f"{node_id}-{types}").rstrip(
                "-"
            )
        if node_id in document.ids:
            node_id = make_id(self.env, document, DOMAIN, name)
        return node_id

    def add_target_and_index(
        self, name: str, sig: str, signode: addnodes.desc_signature
    ) -> None:
        node_id = self._node_id(name)
        signode["ids"].append(node_id)
        self.state.document.note_explicit_target(signode)
        domain: JavaDomain = self.env.get_domain(DOMAIN)
        domain.note_object(name, self.objtype, node_id, location=signode)
        if "no-index-entry" not in self.options:
            short = name.rpartition(".")[2]
            self.indexnode["entries"].append(
                ("single", f"{short} ({name})", node_id, "", None)
            )

    def run(self) -> list[nodes.Node]:
        result = super().run()
        if self.deprecated:
            for node in result:
                if isinstance(node, addnodes.desc):
                    node["classes"].append("java-deprecated")
        return result

    def before_content(self) -> None:
        if self.objtype in ("class", "interface", "enum") and self.names:
            self.env.ref_context.setdefault("java:classes", []).append(
                self.env.ref_context.get("java:class")
            )
            self.env.ref_context["java:class"] = self.names[-1]

    def after_content(self) -> None:
        if self.objtype in ("class", "interface", "enum") and self.names:
            stack = self.env.ref_context.get("java:classes", [])
            self.env.ref_context["java:class"] = stack.pop() if stack else None

    def _object_hierarchy_parts(self, sig_node: addnodes.desc_signature):
        return tuple(sig_node.get("_toc_parts", ()))

    def _toc_entry_name(self, sig_node: addnodes.desc_signature) -> str:
        return sig_node.get("_toc_name", "")


class JavaDomain(Domain):
    name = DOMAIN
    label = "Java"
    object_types = {
        "class": ObjType("class", "type", "class"),
        "interface": ObjType("interface", "type", "class"),
        "enum": ObjType("enum", "type", "class"),
        "method": ObjType("method", "meth"),
        "constructor": ObjType("constructor", "meth"),
        "field": ObjType("field", "field"),
    }
    directives = {kind: JavaObject for kind in object_types}
    roles = {
        "type": XRefRole(),
        "class": XRefRole(),
        "meth": XRefRole(),
        "field": XRefRole(),
    }
    initial_data: dict[str, Any] = {"objects": {}}

    @property
    def objects(self) -> dict[str, tuple[str, str, str]]:
        """Map of full name to ``(docname, node_id, objtype)``."""
        return self.data["objects"]

    def note_object(self, name, objtype, node_id, location=None) -> None:
        # Overloaded methods share a name; link to the first one.
        self.objects.setdefault(name, (self.env.docname, node_id, objtype))

    def clear_doc(self, docname: str) -> None:
        for name, (doc, _, _) in list(self.objects.items()):
            if doc == docname:
                del self.objects[name]

    def merge_domaindata(self, docnames, otherdata) -> None:
        for name, entry in otherdata["objects"].items():
            if entry[0] in docnames:
                self.objects.setdefault(name, entry)

    def _find(self, target: str, scope: str | None) -> str | None:
        target = target.split("<", 1)[0]
        # Search the enclosing classes and their packages from the inside out.
        while scope:
            if f"{scope}.{target}" in self.objects:
                return f"{scope}.{target}"
            scope = scope.rpartition(".")[0]
        if target in self.objects:
            return target
        matches = [n for n in self.objects if n.endswith("." + target)]
        return matches[0] if len(matches) == 1 else None

    def resolve_xref(
        self, env, fromdocname, builder, typ, target, node, contnode
    ):
        if typ == "doxygen":
            # Breathe registers Doxygen ids as anonymous labels.
            label = env.get_domain("std").anonlabels.get(target)
            if label is None:
                return None
            docname, label_id = label
            return make_refnode(
                builder, fromdocname, docname, label_id, contnode
            )
        name = self._find(target, node.get("java:scope"))
        if name is None:
            return None
        docname, node_id, _ = self.objects[name]
        return make_refnode(
            builder, fromdocname, docname, node_id, contnode, name
        )

    def resolve_any_xref(
        self, env, fromdocname, builder, target, node, contnode
    ):
        refnode = self.resolve_xref(
            env, fromdocname, builder, "type", target, node, contnode
        )
        return [] if refnode is None else [(f"{DOMAIN}:type", refnode)]

    def get_objects(self):
        for name, (docname, node_id, objtype) in self.objects.items():
            yield name, name, objtype, docname, node_id, 1


def _resolve_external_java_types(app, env, node, contnode):
    target = node.get("reftarget") or ""
    if target.startswith(_JDK_REFID_PREFIX):
        name = target[len(_JDK_REFID_PREFIX) :]
        return nodes.reference(
            "", "", contnode, refuri=_jdk_url(name), reftitle=name
        )
    if node.get("refdomain") == DOMAIN and node.get("reftype") == "type":
        # Link JDK types to their documentation.
        if (url := _jdk_url(node["reftarget"])) is not None:
            return nodes.reference(
                "", "", contnode, refuri=url, reftitle=node["reftarget"]
            )
        # Signatures also reference type parameters and package-private
        # types, which are not documented. Render those as plain text.
        if not node.get("refexplicit"):
            return contnode
    # Javadoc may link to package-private objects, which are not rendered.
    if node.get("refdomain") in ("std", DOMAIN) and (
        node.get("reftarget") in _members
        or node.get("reftarget") in _compounds
    ):
        return contnode
    return None


# ---------------------------------------------------------------------------
# Breathe integration
# ---------------------------------------------------------------------------


class JavaBreatheObject(BaseObject, JavaObject):
    pass


_original_create = DomainDirectiveFactory.create


# Attributes of the next directive created by _create_directive. Breathe
# reuses one options dict for nested directives, so these cannot be passed as
# directive options.
_next_directive_attrs: dict[str, Any] = {}


def _create_directive(domain: str, args):
    if domain != DOMAIN:
        return _original_create(domain, args)
    args = [f"{DOMAIN}:{args[0]}", *args[1:]]
    directive = JavaBreatheObject(*args)
    for name, value in _next_directive_attrs.items():
        setattr(directive, name, value)
    _next_directive_attrs.clear()
    return directive


def _declare(
    self: SphinxRenderer,
    node,
    objtype: str,
    declaration: str,
    fullname: str,
    toc_name: str,
    deprecated: bool = False,
    **kwargs,
) -> list[nodes.Node]:
    _next_directive_attrs.update(fullname=fullname, deprecated=deprecated)
    nodes_ = _handle_declaration(self, node, objtype, declaration, **kwargs)
    # handle_declaration sets these from Doxygen's names.
    for n in nodes_:
        if isinstance(n, addnodes.desc):
            n[0]["_toc_parts"] = tuple(fullname.split("."))
            n[0]["_toc_name"] = toc_name
            break
    return nodes_


def _text(renderer: SphinxRenderer, node) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return _normalize_type(node)
    return _normalize_type("".join(n.astext() for n in renderer.render(node)))


def _split_modifiers(type_: str) -> tuple[list[str], str]:
    """Split leading modifiers that Doxygen leaves in a member's type."""
    modifiers = []
    words = type_.split(" ")
    while words and words[0] in (*_SHOWN_MODIFIERS, *_HIDDEN_MODIFIERS):
        word = words.pop(0)
        if word in _SHOWN_MODIFIERS:
            modifiers.append(word)
    return modifiers, " ".join(words)


def _protection(node) -> list[str]:
    prot = _value(node.prot) if node.prot is not None else "public"
    return [prot] if prot in ("public", "protected") else []


def _is_static(node) -> bool:
    # Breathe 5 parses the attribute as a bool, Breathe 4 leaves it a string.
    return node.static is True or node.static == "yes"


def _source_header(filename: str, line: int) -> str | None:
    """Return the declaration of a class as written in its source file.

    Doxygen does not record whether a base type outside the documented
    sources (e.g. ``AutoCloseable``) is a class or an interface, so the
    ``extends`` and ``implements`` clauses cannot be rebuilt from the XML.
    """
    lines = _source_lines(filename)[int(line) - 1 :]
    header = []
    for line_ in lines:
        line_ = line_.split("//", 1)[0]
        header.append(line_.split("{", 1)[0])
        if "{" in line_:
            break
    text = " ".join(" ".join(header).split())
    # Drop annotations, which may precede the modifiers on the same line.
    text = re.sub(r"@(?!interface\b)[\w.]+(\([^)]*\))?\s*", "", text)
    if not re.search(r"\b(class|interface|enum)\b", text):
        return None
    return text


def _ref(
    refid: str, text: str, classes: list[str] | None = None
) -> nodes.Node:
    """Return a link to the Doxygen target *refid*.

    Breathe links Doxygen targets with ``std:ref``, but its resolution
    replaces the link text by plain text, so the java domain resolves these.
    """
    ref = addnodes.pending_xref(
        "",
        refdomain=DOMAIN,
        reftype="doxygen",
        reftarget=refid,
        refexplicit=True,
    )
    ref += nodes.literal(text, text, classes=classes or [])
    return ref


def _summary_table(title: str, members: list[_Member]) -> list[nodes.Node]:
    """Return a javadoc-style table of *members* with their summaries."""
    table = nodes.table(classes=["java-summary"])
    tgroup = nodes.tgroup(cols=2)
    table += tgroup
    tgroup += nodes.colspec(colwidth=40)
    tgroup += nodes.colspec(colwidth=60)
    tbody = nodes.tbody()
    tgroup += tbody
    for member in members:
        text = member.name
        if member.kind == "function":
            text += member.args or "()"
        classes = ["java-deprecated"] if member.deprecated else []
        description = nodes.paragraph()
        if member.deprecated:
            description += nodes.emphasis("Deprecated.", "Deprecated.")
            if member.brief:
                description += nodes.Text(" ")
        description += nodes.Text(member.brief)
        row = nodes.row()
        row += nodes.entry(
            "", nodes.paragraph("", "", _ref(member.id, text, classes))
        )
        row += nodes.entry("", description)
        tbody += row
    rubric = nodes.rubric(text=title, classes=["java-summary-title"])
    return [rubric, table]


def _class_summary(compound: _Compound) -> list[nodes.Node]:
    """Return summaries of the members of *compound* and those it inherits."""
    own = [
        _members[id_]
        for id_ in compound.members
        if id_ in _members and _members[id_].documented
    ]

    def key(member: _Member):
        return (member.name.lower(), member.args)

    groups = (
        ("Enum Constant Summary", [m for m in own if m.enum_constant]),
        (
            "Field Summary",
            [m for m in own if m.kind == "variable" and not m.enum_constant],
        ),
        ("Constructor Summary", [m for m in own if m.constructor]),
        (
            "Method Summary",
            [m for m in own if m.kind == "function" and not m.constructor],
        ),
    )
    result: list[nodes.Node] = []
    for title, members in groups:
        if members:
            result += _summary_table(title, sorted(members, key=key))

    # Inherited members, grouped by the class that declares them, skipping
    # overridden ones.
    declared = {(m.name, m.args) for m in own}
    inherited: dict[str, dict[str, list[_Member]]] = {}
    for refid, _ in compound.all_members:
        member = _members.get(refid)
        if (
            member is None
            or member.compound == compound.id
            or not member.documented
            or member.constructor
            or (member.name, member.args) in declared
            # Static methods of interfaces are not inherited.
            or (
                member.static
                and member.kind == "function"
                and _compounds[member.compound].kind == "interface"
            )
        ):
            continue
        kind = "Fields" if member.kind == "variable" else "Methods"
        by_kind = inherited.setdefault(member.compound, {})
        by_kind.setdefault(kind, []).append(member)
    for compound_id, by_kind in inherited.items():
        base = _compounds[compound_id]
        for kind in ("Fields", "Methods"):
            members = by_kind.get(kind)
            if not members:
                continue
            para = nodes.paragraph(classes=["java-inherited"])
            label = f"{kind} inherited from {base.kind} "
            para += nodes.strong(label, label)
            para += _ref(base.id, base.fullname)
            para += nodes.strong(":", ":")
            para += nodes.Text(" ")
            # Like javadoc, list overloaded methods once.
            seen: dict[str, _Member] = {}
            for member in sorted(members, key=key):
                seen.setdefault(member.name, member)
            for i, member in enumerate(seen.values()):
                if i:
                    para += nodes.Text(", ")
                classes = ["java-deprecated"] if member.deprecated else []
                para += _ref(member.id, member.name, classes)
            result.append(para)
    return result


def _insert_summary(rendered: list[nodes.Node], summary: list[nodes.Node]):
    """Insert *summary* between a class's description and its members."""
    for i, node in enumerate(rendered):
        if isinstance(node, nodes.rubric) or (
            isinstance(node, nodes.container)
            and "breathe-sectiondef" in node["classes"]
        ):
            rendered[i:i] = summary
            return
    rendered.extend(summary)


def _java_class(self: SphinxRenderer, node) -> list[nodes.Node]:
    file_data, nodeDef = _compound(self, node.refid)
    parent_context = self.context.create_child_context(file_data)
    new_context = parent_context.create_child_context(nodeDef)

    with sphinxrenderer.WithContext(self, new_context):
        kind = _java_kind(nodeDef.id, _value(nodeDef.kind))
        fullname = nodeDef.compoundname.replace("::", ".")
        name = fullname.rpartition(".")[2]
        declaration = _source_header(
            nodeDef.location.file, nodeDef.location.line
        )
        if declaration is None:
            declaration = " ".join([*_protection(nodeDef), kind, name])
        compound = _compounds.get(nodeDef.id)

        def content(contentnode) -> None:
            rendered = self.render(file_data, parent_context)
            # Code blocks in javadoc are Java.
            for node_ in rendered:
                if isinstance(node_, nodes.Element):
                    for block in node_.findall(nodes.literal_block):
                        block["language"] = "java"
            if compound is not None:
                _insert_summary(rendered, _class_summary(compound))
            contentnode.extend(rendered)

        nodes_ = _declare(
            self,
            nodeDef,
            kind,
            declaration,
            fullname,
            name,
            deprecated=compound is not None and compound.deprecated,
            content_callback=content,
        )
        if "members-only" in self.context.directive_args[2]:
            return list(nodes_[1][1].children)
        return nodes_


def _member_fullname(node) -> str:
    member = _members.get(node.id)
    if member is not None and member.compound in _compounds:
        return f"{_compounds[member.compound].fullname}.{node.name}"
    return getattr(node, "qualifiedname", None) or node.name


def _java_function(self: SphinxRenderer, node) -> list[nodes.Node]:
    modifiers, type_ = _split_modifiers(_text(self, _member_type(node)))
    if node.name.startswith("["):
        # Static initializer blocks are not API.
        return []
    is_constructor = not type_
    if is_constructor and node.id.startswith("enum"):
        # Enum constructors are always private.
        return []
    decl = _protection(node)
    if _is_static(node):
        decl.append("static")
    decl += modifiers
    if type_:
        decl.append(type_)
    args = _text(self, node.argsstring) if node.argsstring else "()"
    decl.append(node.name + args)
    if node.exceptions:
        decl.append(_text(self, node.exceptions))

    member = _members.get(node.id)
    return _declare(
        self,
        node,
        "constructor" if is_constructor else "method",
        " ".join(decl),
        _member_fullname(node),
        node.name + "()",
        deprecated=member is not None and member.deprecated,
    )


def _java_variable(self: SphinxRenderer, node) -> list[nodes.Node]:
    modifiers, type_ = _split_modifiers(_text(self, _member_type(node)))
    is_enum_constant = not type_ and node.id.startswith("enum")
    decl = [] if is_enum_constant else _protection(node)
    if _is_static(node):
        decl.append("static")
    decl += modifiers
    if type_:
        decl.append(type_)
    decl.append(node.name)
    # Like javadoc, show the values of primitive and string constants. Enum
    # constants have no type and their "initializer" is the constructor
    # argument list, which is omitted.
    initializer = _text(self, node.initializer)
    is_constant = _is_static(node) and "final" in modifiers
    if is_constant and initializer and type_ in (*_PRIMITIVES, "String"):
        if not initializer.startswith("="):
            initializer = "= " + initializer
        decl.append(initializer)

    member = _members.get(node.id)
    nodes_ = _declare(
        self,
        node,
        "field",
        " ".join(decl),
        _member_fullname(node),
        node.name,
        deprecated=member is not None and member.deprecated,
    )
    if is_enum_constant:
        for n in nodes_:
            if isinstance(n, addnodes.desc):
                n["java:enum-constant"] = True
    return nodes_


# Breathe's member sections in javadoc order: fields, then constructors and
# methods.
_SECTION_ORDER = [
    (kind, title)
    for kind, title in SphinxRenderer.sections
    if "attrib" in _value(kind)
] + [
    (kind, title)
    for kind, title in SphinxRenderer.sections
    if "attrib" not in _value(kind)
]

# Javadoc-style titles for Breathe's member sections.
_SECTION_TITLES = {
    "Public Functions": "Methods",
    "Public Static Functions": "Static Methods",
    "Public Members": "Fields",
    "Public Static Attributes": "Static Fields",
    "Protected Functions": "Protected Methods",
    "Protected Static Functions": "Protected Static Methods",
    "Protected Attributes": "Protected Fields",
    "Protected Static Attributes": "Protected Static Fields",
}


def _java_compound(self: SphinxRenderer, node, *args, **kwargs):
    _, compounddef = _compound(self, node.refid)
    if _value(compounddef.kind) in ("class", "interface"):
        return _java_class(self, node)
    return None


def _split_section(
    rubric: nodes.rubric,
    members: list[nodes.Node],
    predicate,
    first_title: str,
) -> list[nodes.Node]:
    """Move the members matching *predicate* into a section of their own.

    Each member is rendered as an index node followed by a desc node.
    """
    first: list[nodes.Node] = []
    rest: list[nodes.Node] = []
    pending: list[nodes.Node] = []
    for n in members:
        pending.append(n)
        if isinstance(n, addnodes.desc):
            (first if predicate(n) else rest).extend(pending)
            pending = []
    rest += pending
    if not first:
        return [rubric, *rest]
    result = [
        nodes.rubric(
            text=first_title,
            classes=["breathe-sectiondef-title"],
            ids=[
                "breathe-section-title-"
                + first_title.replace(" ", "-").lower()
            ],
        ),
        *first,
    ]
    if any(isinstance(n, addnodes.desc) for n in rest):
        result += [rubric, *rest]
    return result


def _java_sectiondef(self: SphinxRenderer, node, nodes_: list[nodes.Node]):
    if not nodes_ or not isinstance(nodes_[0], nodes.rubric):
        return nodes_
    rubric, members = nodes_[0], nodes_[1:]
    title = rubric.astext()
    rubric.clear()
    rubric += nodes.Text(_SECTION_TITLES.get(title, title))

    # Doxygen lists enum constants with fields and constructors with
    # methods; javadoc separates them.
    kind = _value(node.kind)
    if kind == "public-attrib":
        return _split_section(
            rubric,
            members,
            lambda n: n.get("java:enum-constant"),
            "Enum Constants",
        )
    if kind in ("public-func", "protected-func"):
        title = "Constructors"
        if kind == "protected-func":
            title = "Protected Constructors"
        return _split_section(
            rubric, members, lambda n: n.get("objtype") == "constructor", title
        )
    return nodes_


def _is_java(self: SphinxRenderer, node) -> bool:
    if self.context is not None and self.context.domain == DOMAIN:
        return True
    # Breathe cannot detect the domain of a compound referenced from the
    # index, so look up the compound's location directly.
    location = getattr(node, "location", None)
    if location is None and getattr(node, "refid", None):
        try:
            location = _compound(self, node.refid)[1].location
        except Exception:
            return False
    return (
        location is not None
        and self.project_info.domain_for_file(location.file) == DOMAIN
    )


def _replace_method(name: str, original, method) -> None:
    setattr(SphinxRenderer, name, method)
    # Breathe also dispatches through tables of handlers.
    for table in _renderer_tables():
        for key, handler in table.items():
            if handler is original:
                table[key] = method


def _patch(name: str, java_impl, *, post: bool = False) -> None:
    """Route ``SphinxRenderer.<name>`` to *java_impl* for the java domain.

    With ``post=True``, *java_impl* instead receives the nodes rendered by the
    original method and returns replacement nodes. If *java_impl* returns
    ``None``, the original method is used.
    """
    original = getattr(SphinxRenderer, name)

    def method(self, node, *args, **kwargs):
        if not _is_java(self, node):
            return original(self, node, *args, **kwargs)
        if post:
            return java_impl(self, node, original(self, node, *args, **kwargs))
        result = java_impl(self, node, *args, **kwargs)
        return (
            original(self, node, *args, **kwargs) if result is None else result
        )

    _replace_method(name, original, method)


def _patch_compounddef() -> None:
    original = SphinxRenderer.visit_compounddef

    def visit_compounddef(self, node, *args, **kwargs):
        if not _is_java(self, node):
            return original(self, node, *args, **kwargs)
        # visit_compounddef emits sections in the order of self.sections.
        previous = self.sections
        self.sections = _SECTION_ORDER
        try:
            return original(self, node, *args, **kwargs)
        finally:
            self.sections = previous

    _replace_method("visit_compounddef", original, visit_compounddef)


def _patch_renderer() -> None:
    _patch_compounddef()
    _patch("visit_compound", _java_compound)
    _patch("visit_class", _java_class)
    _patch("visit_function", _java_function)
    _patch("visit_variable", _java_variable)
    _patch("visit_sectiondef", _java_sectiondef, post=True)
    DomainDirectiveFactory.create = staticmethod(_create_directive)


def _clean_projects(app: Sphinx, config) -> None:
    for project in config.breathe_java_projects:
        path = config.breathe_projects[project]
        if not os.path.isabs(path):
            path = os.path.join(app.confdir, path)
        clean_java_xml(path)


def _add_static_files(app: Sphinx) -> None:
    if app.builder.format == "html":
        static = os.path.join(os.path.dirname(__file__), "breathe_java_static")
        app.config.html_static_path.append(static)
        app.add_css_file("breathe_java.css")


def setup(app: Sphinx) -> dict[str, Any]:
    app.setup_extension("breathe")
    app.add_config_value("breathe_java_projects", [], "env")
    app.add_domain(JavaDomain)
    app.connect("config-inited", _clean_projects)
    app.connect("builder-inited", _add_static_files)
    app.connect(
        "missing-reference", _resolve_external_java_types, priority=400
    )
    _patch_renderer()
    return {"parallel_read_safe": True, "parallel_write_safe": True}
