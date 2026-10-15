# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render Doxygen XML generated from Java sources with Breathe.

Breathe only ships directives for the C, C++, Python, PHP and C# domains and
falls back to the C++ domain for everything else, which cannot parse Java
declarations (arrays, varargs, ``extends``, ``throws``, ...). This extension
adds a minimal ``java`` Sphinx domain and teaches Breathe to build Java
declarations for nodes whose source file has a ``.java`` extension.

Usage in conf.py::

    extensions = [..., "breathe_java"]
    breathe_domain_by_extension = {"java": "java"}
    breathe_java_projects = ["libcudf-java"]

Each project listed in ``breathe_java_projects`` has its Doxygen XML
preprocessed when the extension is set up (see :func:`clean_java_xml`).
"""

from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

from breathe import parser
from breathe.renderer import sphinxrenderer
from breathe.renderer.sphinxrenderer import (
    BaseObject,
    DomainDirectiveFactory,
    SphinxRenderer,
)
from docutils import nodes
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


# ---------------------------------------------------------------------------
# Doxygen XML preprocessing
# ---------------------------------------------------------------------------


# Doxygen ids of every compound and member in the Java projects, used to
# tell references to undocumented Java objects apart from broken references.
_java_ids: set[str] = set()


def clean_java_xml(path: str) -> None:
    """Fix up Doxygen's Java XML for Breathe.

    - Breathe 5 validates compound kinds against the C++ schema, which has no
      top-level ``enum`` compound. Java enums are rewritten as classes; the
      original kind is still recoverable from the ``enum`` prefix of the id.
    - Doxygen fails to resolve some ``{@link Class#member}`` references,
      leaving them with an empty refid, and emits others with a refid that
      lacks the compound prefix. Both are repaired where possible, and
      unresolvable references are replaced by their text.
    - Link text is shown as in javadoc, ``Class.member`` instead of
      ``Class#member``.
    - Doxygen marks package-private members of enums as public.
    """
    trees = {
        fn: ET.parse(fn)
        for fn in glob.glob(os.path.join(path, "*.xml"))
        if os.path.basename(fn) not in ("index.xml", "Doxyfile.xml")
    }
    member_ids: dict[str, str] = {}
    qualified_ids: dict[str, list[str]] = {}
    for tree in trees.values():
        for compound in tree.getroot().iter("compounddef"):
            _java_ids.add(compound.get("id"))
            name = compound.findtext("compoundname").replace("::", ".")
            qualified_ids.setdefault(name, []).append(compound.get("id"))
        for member in tree.getroot().iter("memberdef"):
            id_ = member.get("id")
            _java_ids.add(id_)
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

    for fn, tree in trees.items():
        root = tree.getroot()
        changed = False
        for compound in root.iter("compounddef"):
            if compound.get("kind") == "enum":
                compound.set("kind", "class")
                changed = True
            if compound.get("id").startswith("enum"):
                changed |= _fix_enum_member_protection(compound)
        for parent in list(root.iter()):
            for ref in list(parent):
                if ref.tag != "ref":
                    continue
                changed = True
                refid = ref.get("refid", "")
                if refid.startswith("_1"):
                    refid = member_ids.get(refid, "")
                elif not refid:
                    refid = resolve(ref.text or "") or ""
                if ref.text and "#" in ref.text:
                    ref.text = ref.text.lstrip("#").replace("#", ".")
                if refid:
                    ref.set("refid", refid)
                    continue
                # Replace the reference by its text.
                index = list(parent).index(ref)
                text = (ref.text or "") + (ref.tail or "")
                if index == 0:
                    parent.text = (parent.text or "") + text
                else:
                    previous = parent[index - 1]
                    previous.tail = (previous.tail or "") + text
                parent.remove(ref)
        if changed:
            tree.write(fn)

    index = os.path.join(path, "index.xml")
    with open(index) as f:
        text = f.read()
    new_text = re.sub(r'(<compound\s[^>]*\bkind=)"enum"', r'\1"class"', text)
    if new_text != text:
        with open(index, "w") as f:
            f.write(new_text)


def _fix_enum_member_protection(compound: ET.Element) -> bool:
    """Mark enum members declared without an access modifier as package-private."""
    changed = False
    for member in compound.iter("memberdef"):
        if member.get("prot") != "public" or not member.findtext("type"):
            # Enum constants have no type and are always public.
            continue
        location = member.find("location")
        try:
            with open(location.get("file")) as f:
                line = f.readlines()[int(location.get("line")) - 1]
        except (OSError, IndexError, TypeError, ValueError):
            continue
        declaration = line.split(member.findtext("name"), 1)[0]
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
        if compounddef.get("prot") != "public":
            continue
        brief = " ".join(
            "".join(compounddef.find("./briefdescription").itertext()).split()
        )
        # Show link text as in javadoc, as clean_java_xml does.
        brief = re.sub(r"(\w)#(\w)", r"\1.\2", brief)
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
    other identifier becomes a cross-reference to a Java type, which is
    rendered as plain text if no documented type of that name exists.
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


def _ignore_unknown_java_types(app, env, node, contnode):
    # Signatures reference JDK types (String, Optional, ...) and type
    # parameters that are not documented here. Render those as plain text.
    if node.get("refdomain") == DOMAIN and node.get("reftype") == "type":
        if not node.get("refexplicit"):
            return contnode
    # Javadoc may link to package-private objects, which are not rendered.
    if node.get("refdomain") == "std" and node.get("reftarget") in _java_ids:
        return contnode
    return None


# ---------------------------------------------------------------------------
# Breathe integration
# ---------------------------------------------------------------------------


class JavaBreatheObject(BaseObject, JavaObject):
    pass


_original_create = DomainDirectiveFactory.create


# Full name of the next directive created by _create_directive. Breathe
# reuses one options dict for nested directives, so the name cannot be passed
# as a directive option.
_next_fullname: str | None = None


def _create_directive(domain: str, args):
    global _next_fullname
    if domain != DOMAIN:
        return _original_create(domain, args)
    args = [f"{DOMAIN}:{args[0]}", *args[1:]]
    directive = JavaBreatheObject(*args)
    directive.fullname, _next_fullname = _next_fullname, None
    return directive


def _handle_declaration(
    self: SphinxRenderer, node, objtype, declaration, fullname, **kwargs
):
    global _next_fullname
    _next_fullname = fullname
    return self.handle_declaration(node, objtype, declaration, **kwargs)


def _text(renderer: SphinxRenderer, node) -> str:
    if node is None:
        return ""
    text = "".join(n.astext() for n in renderer.render(node))
    # Doxygen pads template brackets and array suffixes with spaces.
    text = re.sub(r"\s*<\s*", "<", text)
    text = re.sub(r"\s*>", ">", text)
    text = re.sub(r"\s+\[\]", "[]", text)
    text = re.sub(r",(?=\S)", ", ", text)
    return " ".join(text.split())


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
    prot = node.prot.value if node.prot is not None else "public"
    return [prot] if prot in ("public", "protected") else []


def _set_toc_name(nodes_: list[nodes.Node], fullname: str, toc_name: str):
    for node in nodes_:
        if isinstance(node, addnodes.desc):
            sig = node[0]
            sig["_toc_parts"] = tuple(fullname.split("."))
            sig["_toc_name"] = toc_name
            return


def _source_header(filename: str, line: int) -> str | None:
    """Return the declaration of a class as written in its source file.

    Doxygen does not record whether a base type outside the documented
    sources (e.g. ``AutoCloseable``) is a class or an interface, so the
    ``extends`` and ``implements`` clauses cannot be rebuilt from the XML.
    """
    try:
        with open(filename) as f:
            lines = f.readlines()[line - 1 :]
    except (OSError, TypeError):
        return None
    header = []
    for line in lines:
        line = line.split("//", 1)[0]
        header.append(line.split("{", 1)[0])
        if "{" in line:
            break
    text = " ".join(" ".join(header).split())
    # Drop annotations, which may precede the modifiers on the same line.
    text = re.sub(r"@(?!interface\b)[\w.]+(\([^)]*\))?\s*", "", text)
    if not re.search(r"\b(class|interface|enum)\b", text):
        return None
    return text


def _java_class(self: SphinxRenderer, node) -> list[nodes.Node]:
    file_data = self.parse_compound(node.refid)
    nodeDef = file_data.compounddef[0]
    parent_context = self.context.create_child_context(file_data)
    new_context = parent_context.create_child_context(nodeDef)

    with sphinxrenderer.WithContext(self, new_context):
        kind = _java_kind(nodeDef.id, nodeDef.kind.value)
        fullname = nodeDef.compoundname.replace("::", ".")
        name = fullname.rpartition(".")[2]
        declaration = _source_header(
            nodeDef.location.file, nodeDef.location.line
        )
        if declaration is None:
            declaration = " ".join([*_protection(nodeDef), kind, name])

        def content(contentnode) -> None:
            contentnode.extend(self.render(file_data, parent_context))

        nodes_ = _handle_declaration(
            self,
            nodeDef,
            kind,
            declaration,
            fullname,
            content_callback=content,
        )
        # handle_declaration stores Doxygen's "::"-separated name.
        _set_toc_name(nodes_, fullname, name)
        if "members-only" in self.context.directive_args[2]:
            return list(nodes_[1][1].children)
        return nodes_


def _java_function(self: SphinxRenderer, node) -> list[nodes.Node]:
    modifiers, type_ = _split_modifiers(_text(self, node.type))
    if node.name.startswith("["):
        # Static initializer blocks are not API.
        return []
    is_constructor = not type_
    if is_constructor and node.id.startswith("enum"):
        # Enum constructors are always private.
        return []
    decl = _protection(node)
    if node.static:
        decl.append("static")
    decl += modifiers
    if type_:
        decl.append(type_)
    args = _text(self, node.argsstring) if node.argsstring else "()"
    decl.append(node.name + args)
    if node.exceptions:
        decl.append(_text(self, node.exceptions))

    fullname = node.qualifiedname or node.name
    objtype = "constructor" if is_constructor else "method"
    nodes_ = _handle_declaration(self, node, objtype, " ".join(decl), fullname)
    _set_toc_name(nodes_, fullname, node.name + "()")
    return nodes_


def _java_variable(self: SphinxRenderer, node) -> list[nodes.Node]:
    modifiers, type_ = _split_modifiers(_text(self, node.type))
    is_enum_constant = not type_ and node.id.startswith("enum")
    decl = [] if is_enum_constant else _protection(node)
    if node.static:
        decl.append("static")
    decl += modifiers
    if type_:
        decl.append(type_)
    decl.append(node.name)
    # Like javadoc, show the values of primitive and string constants. Enum
    # constants have no type and their "initializer" is the constructor
    # argument list, which is omitted.
    initializer = _text(self, node.initializer)
    is_constant = node.static and "final" in modifiers
    if is_constant and initializer and type_ in (*_PRIMITIVES, "String"):
        if not initializer.startswith("="):
            initializer = "= " + initializer
        decl.append(initializer)

    fullname = node.qualifiedname or node.name
    nodes_ = _handle_declaration(self, node, "field", " ".join(decl), fullname)
    _set_toc_name(nodes_, fullname, node.name)
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
    if "attrib" in kind.value
] + [
    (kind, title)
    for kind, title in SphinxRenderer.sections
    if "attrib" not in kind.value
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
    kind = self.parse_compound(node.refid).compounddef[0].kind
    if kind in (
        parser.DoxCompoundKind.class_,
        parser.DoxCompoundKind.interface,
    ):
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
    kind = parser.DoxSectionKind
    if node.kind == kind.public_attrib:
        return _split_section(
            rubric,
            members,
            lambda n: n.get("java:enum-constant"),
            "Enum Constants",
        )
    if node.kind in (kind.public_func, kind.protected_func):
        title = "Constructors"
        if node.kind == kind.protected_func:
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
            location = self.parse_compound(node.refid).compounddef[0].location
        except Exception:
            return False
    return (
        location is not None
        and self.project_info.domain_for_file(location.file) == DOMAIN
    )


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


def _replace_method(name: str, original, method) -> None:
    setattr(SphinxRenderer, name, method)
    # Breathe also dispatches through handler tables built at class creation.
    for table in (
        SphinxRenderer.node_handlers,
        SphinxRenderer.tagged_node_handlers,
    ):
        for key, handler in table.items():
            if handler is original:
                table[key] = method


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


def setup(app: Sphinx) -> dict[str, Any]:
    app.setup_extension("breathe")
    app.add_config_value("breathe_java_projects", [], "env")
    app.add_domain(JavaDomain)
    app.connect("config-inited", _clean_projects)
    app.connect("missing-reference", _ignore_unknown_java_types, priority=400)
    _patch_renderer()
    return {"parallel_read_safe": True, "parallel_write_safe": True}
