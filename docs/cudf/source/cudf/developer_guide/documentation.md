# Writing documentation

cuDF documentation is comprised of:

- Docstrings written using [reStructuredText (reST)](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) and inline with the code in `python/cudf/cudf`
- Documents such as the API reference or user guides written using [MyST](https://myst-parser.readthedocs.io/en/latest/) in `docs/cudf/source/cudf`

The documentation containing both components is built using [Sphinx](https://www.sphinx-doc.org/en/master/).

## Docstrings

cuDF public and private docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style
and should include all applicable [`numpydoc sections`](https://numpydoc.readthedocs.io/en/latest/format.html#sections).

For example:

```python
class A:
    """Brief description of A.

    Longer description of A.

    Parameters
    ----------
    x : int
        Description of x, the first constructor parameter.
    """
    def __init__(self, x: int):
        pass

    def foo(self, bar: str):
        """Short description of foo.

        Longer description of foo.

        Parameters
        ----------
        bar : str
            Description of bar.

        Returns
        -------
        float
            Description of the return value of foo.

        Raises
        ------
        ValueError
            Explanation of when a ValueError is raised.
            In this case, a ValueError is raised if bar is "fail".

        Examples
        --------
        The examples section is _strongly_ encouraged.
        Where appropriate, it may mimic the examples for the corresponding pandas API.
        >>> a = A()
        >>> a.foo('baz')
        0.0
        >>> a.foo('fail')
        ...
        ValueError: Failed!
        """
        if bar == "fail":
            raise ValueError("Failed!")
        return 0.0
```

Additionally, class properties, which are not explicitly covered, should be documented in the getter function.

Docstrings are validated using [`ruff pydocstyle rules`](https://docs.astral.sh/ruff/rules/#pydocstyle-d) to ensure consistent docstring formatting.

```{note}
Docstrings of private functions and classes are not linted or validated to follow the numpydoc style.

These docstrings should still follow the numpydoc style, but a single line `#` comment is acceptable
if a private function or class only has minimal use in 1 file.
```

## Published documentation

To mirror the format of the [pandas API documentation](https://pandas.pydata.org/docs/reference/index.html),
the API docs are organized into specific pages and sections in `docs/cudf/source/cudf/api_docs`.
For example, all `DataFrame` documentation is contained in `docs/cudf/source/cudf/api_docs/dataframe.rst` which mirrors
[pandas DataFrame API documentation page](https://pandas.pydata.org/docs/reference/frame.html)

Within each section, documentation is created using the Sphinx [`autosummary extension`](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html) to generate pages for each documented API. Each section of the docs looks like the following:

```
Section name
~~~~~~~~~~~~
.. autosummary::
   API1
   API2
   ...
```

Each listed will automatically have its docstring rendered into a separate page.
This layout comes from [NVIDIA Sphinx Theme](https://pypi.org/project/nvidia-sphinx-theme/).

````{note}
Autosummary generates stub pages that look like this (using `cudf.concat` as an example):

```
cudf.concat
===========

.. currentmodule:: cudf

.. autofunction:: concat
```

Commands like `autofunction` come from the Sphinx [`autodoc extension`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html).
This directive will import cudf and pull the docstring from `cudf.concat`.
This approach allows us to do the minimal amount of manual work in organizing our docs,
while still matching the pandas layout as closely as possible.
````

When adding a new API, include the API name to the appropriate page.

### Documenting classes

Python classes and the Sphinx plugins used in RAPIDS interact in nontrivial ways.
`autosummary`'s default page generated for a class uses [`autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) to automatically detect and document all methods of a class.
That means that in addition to the manually created `autosummary` pages where class methods are grouped into sections of related features, there is another page for each class where all the methods of that class are automatically summarized in a table for quick access.
However, we also use the [`numpydoc`](https://numpydoc.readthedocs.io/en/latest/) extension, which offers the same feature.
We use both in order to match the contents and style of the pandas documentation as closely as possible.

pandas is also particular about what information is included in a class's documentation.
While the documentation pages for the major user-facing classes like `DataFrame`, `Series`, and `Index` contain all APIs, less visible classes or subclasses (such as subclasses of `Index`) only include the methods that are specific to those subclasses.
For example, {py:class}`cudf.CategoricalIndex` only includes `codes` and `categories` on its page, not the entire set of `Index` functionality.

To accommodate these requirements, we take the following approach:
1. The default `autosummary` template for classes is overridden with a [simpler template that does not generate method or attribute documentation](https://github.com/rapidsai/cudf/blob/main/docs/cudf/source/_templates/autosummary/class.rst). In other words, we disable `autosummary`'s generation of Methods and Attributes lists.
2. We rely on `numpydoc` entirely for the classes that need their entire APIs listed (`DataFrame`/`Series`/etc). `numpydoc` will automatically populate Methods and Attributes section if (and only if) they are not already defined in the class's docstring.
3. For classes that should only include a subset of APIs, we include those explicitly in the class's documentation. When those lists exist, `numpydoc` will not override them. If either the Methods or Attributes section should be empty, that section must still be included but should simply contain "None". For example, the class documentation for `CategoricalIndex` could include something like the following:

```
    Attributes
    ----------
    codes
    categories

    Methods
    -------
    None

```

## Comparing to pandas

When an API intentionally deviates in signature or behavior from pandas, use the custom `pandas-compat` Sphinx directive
inside the API docstring to describe the difference. For example:

```python
def foo(self):
    """
    Returns result from foo.

    .. pandas-compat::
        :meth:`pandas.DataFrame.foo`

        Explanation of differences
    """
```

All API compatibility differences from pandas will be rendered in the [pandas comparison](../PandasCompat.md) page.

## Writing documentation pages

In addition to docstrings, our docs also contain a number of more dedicated user guides in `docs/cudf/source/cudf` written in [MyST markdown](https://myst-parser.readthedocs.io/en/latest/). Since the [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html) is largely applicable to cuDF as the APIs are similar, a dedicated user guide should be written to describe:

- Concepts specific to cuDF that do not exist in pandas
- Implementations that differ from pandas that impact the user.
- Additional functionality or ecosystem support that is specific to cuDF and not pandas.

```{note}
Add links between documentation pages and specific section where applicable with
[Myst auto-generated anchors](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#auto-generated-header-anchors),
when possible.
```

## Building documentation

### Requirements

The following are required to build the documentation:
- A RAPIDS-compatible GPU. This is necessary because the documentation execute code.
- A working copy of cudf in the same build environment.
  If you are only making changes to documentation we recommend following the
  [Documentation contributions guide](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#documentation-contributions) otherwise follow the
  [build instructions](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#setting-up-your-build-environment).
- Sphinx, numpydoc, and MyST-NB.
  Assuming you follow the instructions in the previous step, these should automatically be installed into your environment.

### Building and viewing docs

Once you have a working copy of cudf, building the docs is straightforward:
1. Navigate to `/path/to/cudf/docs/cudf/`.
2. Execute `make html`

This will run Sphinx in your shell and generate outputs at `build/html/index.html`.
To view the results.
1. Navigate to `build/html`
2. Execute `python -m http.server`

Then, open a web browser and go to `https://localhost:8000`.
If something else is currently running on port 8000,
`python -m http.server` will automatically find the next available port.
Alternatively, you may specify a port with `python -m http.server $PORT`.

You may build docs on a remote machine but want to view them locally.
Assuming the other machine's IP address is visible on your local network,
you can view the docs by replacing `localhost` with the IP address of the host machine.
Alternatively, you may also forward the port using e.g.
`ssh -N -f -L localhost:$LOCAL_PORT:localhost:$REMOTE_PORT $REMOTE_IP`.
That will make `$REMOTE_IP:$REMOTE_PORT` visible at `localhost:$LOCAL_PORT`.
