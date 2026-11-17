# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is adapted from official sphinx tutorial for `todo` extension:
# https://www.sphinx-doc.org/en/master/development/tutorials/todo.html
from __future__ import annotations

from typing import cast

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from sphinx import addnodes
from sphinx.domains import Domain
from sphinx.errors import NoUri
from sphinx.locale import _ as get_translation_sphinx
from sphinx.util.docutils import SphinxDirective, new_document


class PandasCompat(nodes.Admonition, nodes.Element):
    pass


class PandasCompatList(nodes.General, nodes.Element):
    pass


def visit_PandasCompat_node(self, node):
    self.visit_admonition(node)


def depart_PandasCompat_node(self, node):
    self.depart_admonition(node)


class PandasCompatListDirective(Directive):
    def run(self):
        return [PandasCompatList("")]


class PandasCompatDirective(BaseAdmonition, SphinxDirective):
    # this enables content in the directive
    has_content = True

    def run(self):
        targetid = "PandasCompat-%d" % self.env.new_serialno("PandasCompat")
        targetnode = nodes.target("", "", ids=[targetid])

        PandasCompat_node = PandasCompat("\n".join(self.content))
        PandasCompat_node += nodes.title(
            get_translation_sphinx("Pandas Compatibility Note"),
            get_translation_sphinx("Pandas Compatibility Note"),
        )
        PandasCompat_node["docname"] = self.env.docname
        PandasCompat_node["target"] = targetnode
        self.state.nested_parse(
            self.content, self.content_offset, PandasCompat_node
        )

        if not hasattr(self.env, "PandasCompat_all_pandas_compat"):
            self.env.PandasCompat_all_pandas_compat = []

        self.env.PandasCompat_all_pandas_compat.append(
            {
                "docname": self.env.docname,
                "PandasCompat": PandasCompat_node.deepcopy(),
                "target": targetnode,
            }
        )

        return [targetnode, PandasCompat_node]


def purge_PandasCompats(app, env, docname):
    if not hasattr(env, "PandasCompat_all_pandas_compat"):
        return

    env.PandasCompat_all_pandas_compat = [
        PandasCompat
        for PandasCompat in env.PandasCompat_all_pandas_compat
        if PandasCompat["docname"] != docname
    ]


def merge_PandasCompats(app, env, docnames, other):
    if not hasattr(env, "PandasCompat_all_pandas_compat"):
        env.PandasCompat_all_pandas_compat = []
    if hasattr(other, "PandasCompat_all_pandas_compat"):
        env.PandasCompat_all_pandas_compat.extend(
            other.PandasCompat_all_pandas_compat
        )


class PandasCompatDomain(Domain):
    name = "pandascompat"
    label = "pandascompat"

    @property
    def pandascompats(self):
        return self.data.setdefault("pandascompats", {})

    def clear_doc(self, docname):
        self.pandascompats.pop(docname, None)

    def merge_domaindata(self, docnames, otherdata):
        for docname in docnames:
            self.pandascompats[docname] = otherdata["pandascompats"][docname]

    def process_doc(self, env, docname, document):
        pandascompats = self.pandascompats.setdefault(docname, [])
        for pandascompat in document.findall(PandasCompat):
            env.app.emit("pandascompat-defined", pandascompat)
            pandascompats.append(pandascompat)


class PandasCompatListProcessor:
    def __init__(self, app, doctree, docname):
        self.builder = app.builder
        self.config = app.config
        self.env = app.env
        self.domain = cast(
            "PandasCompatDomain", app.env.get_domain("pandascompat")
        )
        self.document = new_document("")
        self.process(doctree, docname)

    def process(self, doctree: nodes.document, docname: str) -> None:
        pandascompats = [
            v for vals in self.domain.pandascompats.values() for v in vals
        ]
        for node in doctree.findall(PandasCompatList):
            if not self.config.include_pandas_compat:
                node.parent.remove(node)
                continue

            content: list[nodes.Element | None] = (
                [nodes.target()] if node.get("ids") else []
            )

            for pandascompat in pandascompats:
                # Create a copy of the pandascompat node
                new_pandascompat = pandascompat.deepcopy()
                new_pandascompat["ids"].clear()

                self.resolve_reference(new_pandascompat, docname)
                content.append(new_pandascompat)

                ref = self.create_reference(pandascompat, docname)
                content.append(ref)

            node.replace_self(content)

    def create_reference(self, pandascompat, docname):
        para = nodes.paragraph()
        newnode = nodes.reference("", "")
        innernode = nodes.emphasis(
            get_translation_sphinx("[source]"),
            get_translation_sphinx("[source]"),
        )
        newnode["refdocname"] = pandascompat["docname"]
        try:
            newnode["refuri"] = (
                self.builder.get_relative_uri(docname, pandascompat["docname"])
                + "#"
                + pandascompat["target"]["refid"]
            )
        except NoUri:
            # ignore if no URI can be determined, e.g. for LaTeX output
            pass
        newnode.append(innernode)
        para += newnode
        return para

    def resolve_reference(self, todo, docname: str) -> None:
        """Resolve references in the todo content."""
        for node in todo.findall(addnodes.pending_xref):
            if "refdoc" in node:
                node["refdoc"] = docname

        # Note: To resolve references, it is needed to wrap it with document node
        self.document += todo
        self.env.resolve_references(self.document, docname, self.builder)
        self.document.remove(todo)


def setup(app):
    app.add_config_value("include_pandas_compat", False, "html")
    app.add_node(PandasCompatList)
    app.add_node(
        PandasCompat,
        html=(visit_PandasCompat_node, depart_PandasCompat_node),
        latex=(visit_PandasCompat_node, depart_PandasCompat_node),
        text=(visit_PandasCompat_node, depart_PandasCompat_node),
        man=(visit_PandasCompat_node, depart_PandasCompat_node),
        texinfo=(visit_PandasCompat_node, depart_PandasCompat_node),
    )
    app.add_directive("pandas-compat", PandasCompatDirective)
    app.add_directive("pandas-compat-list", PandasCompatListDirective)
    app.add_domain(PandasCompatDomain)
    app.connect("doctree-resolved", PandasCompatListProcessor)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
