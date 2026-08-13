# libcudf C++ Documentation Guide

These guidelines apply to documenting all libcudf C++ source files using doxygen style formatting although only public APIs and classes are actually [published](https://docs.rapids.ai/api/libcudf/stable/index.html).

## Copyright License

The copyright comment is included here but may also be mentioned in a coding guideline document as well.
The following is the license header comment that should appear at the beginning of every C++ source file.

    /*
     * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
     * SPDX-License-Identifier: Apache-2.0
     */

The comment should start with `/*` and not `/**` so it is not processed by doxygen.

Also, here are the rules for the copyright year.

- A new file should have the year in which it was created
- A modified file should span the year it was created and the year it was modified (e.g. `2019-2021`)

Changing the copyright year may not be necessary if no content has changed (e.g. reformatting only).

## Doxygen

The [doxygen tool](https://www.doxygen.nl/manual/index.html) is used to generate HTML pages from the C++ comments in the source code.
Doxygen recognizes and parses block comments and performs specialized output formatting when it encounters [doxygen commands](https://www.doxygen.nl/manual/commands.html).

There are almost 200 commands (also called tags in this document) that doxygen recognizes in comment blocks.
This document provides guidance on which commands/tags to use and how to use them in the libcudf C++ source code.

The doxygen process can be customized using options in the [Doxyfile](../Doxyfile).

Here are some of the custom options in the Doxyfile for libcudf.
| Option | Setting | Description |
| ------ | ------- | ----------- |
| PROJECT_NAME | libcudf | Title used on the main page |
| PROJECT_NUMBER | 22.02.00 | Version number |
| EXTENSION_MAPPING | cu=C++ cuh=C++ | Process `cu` and `cuh` as C++ |
| INPUT | main_page.md regex.md unicode.md ../include | Embedded markdown files and source code directories to process |
| FILE_PATTERNS | *.cpp *.hpp *.h *.c *.cu *.cuh | File extensions to process |

## Block Comments

Use the following style for block comments describing functions, classes and other types, groups, and files.

    /**
     * description text and
     * doxygen tags go here
     */

Doxygen comment blocks start with `/**` and end with `*/` only, and with nothing else on those lines.
Do not add dashes `-----` or extra asterisks `*****` to the first and last lines of a doxygen block.
The block must be placed immediately before the source code line to which it refers.
The block may be indented to line up vertically with the item it documents as appropriate.
See the [Example](#the-example) section below.

Each line in the comment block between the `/**` and `*/` lines should start with a space followed by an asterisk.
Any text on these lines, including tag declarations, should start after a single space after the asterisk.

## Tag/Command names

Use @ to prefix doxygen commands (e.g. \@brief, \@code, etc.)

## Markdown

The doxygen tool supports a limited set of markdown format in the comment block including links, tables, lists, etc.
In some cases a trade-off may be required for readability in the source text file versus the readability in the doxygen formatted web pages.
For example, there are some limitations on readability with '%' character and pipe character '|' within a markdown table.

Avoid using direct HTML tags.
Although doxygen supports markdown and markdown supports HTML tags, the HTML support for doxygen's markdown is also limited.

## The Example

The following example covers most of the doxygen block comment and tag styles
for documenting C++ code in libcudf.

    /**
     * @file source_file.cpp
     * @brief Description of source file contents
     *
     * Longer description of the source file contents.
     */

    /**
     * @brief One line description of the class
     *
     * @ingroup optional_predefined_group_id
     *
     * Longer, more detailed description of the class.
     *
     * @tparam T Short description of each template parameter
     * @tparam U Short description of each template parameter
     */
    template <typename T, typename U>
    class example_class {

      void get_my_int();            ///< Simple members can be documented like this
      void set_my_int( int value ); ///< Try to use descriptive member names

      /**
       * @brief Short, one line description of the member function
       *
       * A more detailed description of what this function does and what
       * its logic does.
       *
       * @code
       * example_class<int> inst;
       * inst.set_my_int(5);
       * int output = inst.complicated_function(1,dptr,fptr);
       * @endcode
       *
       * @param[in]     first  This parameter is an input parameter to the function
       * @param[in,out] second This parameter is used both as an input and output
       * @param[out]    third  This parameter is an output of the function
       *
       * @return The result of the complex function
       */
      T complicated_function(int first, double* second, float* third)
      {
          // Do not use doxygen-style block comments
          // for code logic documentation.
      }

     private:
      int my_int;                ///< An example private member variable
    };

    /**
     * @brief Short, one line description of this free function
     *
     * @ingroup optional_predefined_group_id
     *
     * A detailed description must start after a blank line.
     *
     * @code
     * template<typename T>
     * struct myfunctor {
     *   bool operator()(T input) { return input % 2 > 0; }
     * };
     * free_function<myfunctor,int>(myfunctor{},12);
     * @endcode
     *
     * @throw cudf::logic_error if `input_argument` is negative or zero
     *
     * @tparam functor_type The type of the functor
     * @tparam input_type The datatype of the input argument
     *
     * @param[in] functor        The functor to be called on the input argument
     * @param[in] input_argument The input argument passed into the functor
     * @return The result of calling the functor on the input argument
     */
    template <class functor_type, typename input_type>
    bool free_function(functor_type functor, input_type input_argument)
    {
      CUDF_EXPECTS( input_argument > 0, "input_argument must be positive");
      return functor(input_argument);
    }

    /**
     * @brief Short, one line description
     *
     * @ingroup optional_predefined_group_id
     *
     * Optional, longer description.
     */
    enum class example_enum {
      first_enum,   ///< Description of the first enum
      second_enum,  ///< Description of the second enum
      third_enum    ///< Description of the third enum
    };

## Descriptions

The comment description should clearly detail how the output(s) are created from any inputs.
Include any performance and any boundary considerations.
Also include any limits on parameter values and if any default values are declared.
Don't forget to specify how nulls are handled or produced.
Also, try to include a short [example](#inline-examples) if possible.

### @brief

The [\@brief](https://www.doxygen.nl/manual/commands.html#cmdbrief) text should be a short, one line description.
Doxygen does not provide much space to show this text in the output pages.
Always follow the \@brief line with a blank comment line. Normally this is like a title and not sentence
and therefore does not need a period. Only use a period if it is a sentence.

The longer description is the rest of the comment text that is not tagged with any doxygen command.

    /**
     * @brief Short description or title
     *
     * Long description.
     *

### \@copydoc

Documentation for declarations in headers should be clear and complete.
You can use the [\@copydoc](https://www.doxygen.nl/manual/commands.html#cmdcopydoc) tag to avoid duplicating the comment block for a function definition.

      /**
       * @copydoc complicated_function(int,double*,float*)
       *
       * Any extra documentation.
       */

Also, \@copydoc is useful when documenting a `detail` function that differs only by the `stream` parameter.

    /**
     * @copydoc cudf::segmented_count_set_bits(bitmask_type const*,std::vector<size_type> const&)
     *
     * @param[in] stream Optional CUDA stream on which to execute kernels
     */
    std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                    std::vector<size_type> const& indices,
                                                    rmm::cuda_stream_view stream = cudf::get_default_stream());

Note, you must specify the whole signature of the function, including optional parameters, so that doxygen will be able to locate it.

### Function parameters

The following tags should appear near the end of function comment block in the order specified here:

| Command | Description |
| ------- | ----------- |
| [\@throw](#throw) | Specify the conditions in which the function may throw an exception |
| [\@tparam](#tparam) | Description for each template parameter |
| [\@param](#param) | Description for each function parameter |
| [\@return](#return) | Short description of object or value returned |

#### \@throw

Add an [\@throw](https://www.doxygen.nl/manual/commands.html#cmdthrow) comment line in the doxygen block for each exception that the function may throw.
You only need to include exceptions thrown by the function itself.
If the function calls another function that may throw an exception, you do not need to document those exceptions here.

Include the name of the exception without backtick marks so doxygen can add reference links correctly.

     *
     * @throw cudf::logic_error if `input_argument` is negative or zero
     *

Using \@throws is also acceptable but VS Code and other tools only do syntax highlighting on \@throw.

#### @tparam

Add a [\@tparam](https://www.doxygen.nl/manual/commands.html#cmdtparam) comment line for each template parameter declared by this function.
The name of the parameter specified after the doxygen tag must match exactly to the template parameter name.

     *
     * @tparam functor_type The type of the functor
     * @tparam input_type The datatype of the input argument
     *

The definition should detail the requirements of the parameter.
For example, if the template is for a functor or predicate, then describe the expected input types and output.

#### @param

Add a [\@param](https://www.doxygen.nl/manual/commands.html#cmdparam) comment line for each function parameter passed to this function.
The name of the parameter specified after the doxygen tag must match the function's parameter name.
Also include append `[in]`, `[out]` or `[in,out]` to the `@param` if it is not clear from the declaration and the parameter name itself.

     *
     * @param[in]     first  This parameter is an input parameter to the function
     * @param[in,out] second This parameter is used both as an input and output
     * @param[out]    third  This parameter is an output of the function
     *

It is also recommended to vertically aligning the 3 columns of text if possible to make it easier to read in a source code editor.
Finally, the description is normally like a title and only needs a period if it is a sentence.

#### @return

Add a single [\@return](https://www.doxygen.nl/manual/commands.html#cmdreturn) comment line at the end of the comment block if the function returns an object or value.
Include a brief description of what is returned.

    /**
     * ...
     *
     * @return A new column of type INT32 and no nulls
     */

Do not include the type of the object returned with the `@return` comment.

### Inline Examples

It is usually helpful to include a source code example inside your comment block when documenting a function or other declaration.
Use the [\@code](https://www.doxygen.nl/manual/commands.html#cmdcode) and [\@endcode](https://www.doxygen.nl/manual/commands.html#cmdendcode) pair to include inline examples.

Doxygen supports syntax highlighting for C++ and several other programming languages (e.g. Python, Java).
By default, the \@code tag uses syntax highlighting based on the source code in which it is found.

     *
     * @code
     * auto result = cudf::make_column( );
     * @endcode
     *

You can specify a different language by indicating the file extension in the tag:

     *
     * @code{.py}
     * import cudf
     * s = cudf.Series([1,2,3])
     * @endcode
     *

If you wish to use pseudocode in your example, use the following:

     *
     * Sometimes pseudocode is clearer.
     * @code{.pseudo}
     * s = int column of [ 1, 2, null, 4 ]
     * r = fill( s, [1, 2], 0 )
     * r is now [ 1, 0, 0, 4 ]
     * @endcode
     *

When writing example snippets, using fully qualified class names allows doxygen to add reference links to the example.

     *
     * @code
     * auto result1 = make_column( ); // reference link will not be created
     * auto result2 = cudf::make_column( ); // reference link will be created
     * @endcode
     *

Although using 3 backtick marks \`\`\` for example blocks will work too, they do not stand out as well in VS Code and other source editors.

Do not use the `@example` tag in the comments for a declaration, or doxygen will interpret the entire source file as example source code.
The source file is then published under a separate _Examples_ page in the output.

### Deprecations

Add a single [\@deprecated](https://www.doxygen.nl/manual/commands.html#cmddeprecated) comment line
to comment blocks for APIs that will be removed in future releases. Mention alternative /
replacement APIs in the deprecation comment.

    /**
     * ...
     *
     * @deprecated This function is deprecated. Use another new function instead.
     */

## Namespaces

Doxygen output includes a _Namespaces_ page that shows all the namespaces declared with comment blocks in the processed files.
Here is an example of a doxygen description comment for a namespace declaration.

    /**
     * @brief cuDF interfaces
     *
     * This is the top-level namespace which contains all cuDF functions and types.
     */
    namespace CUDF_EXPORT cudf {

A description comment should be included only once for each unique namespace declaration.
Otherwise, if more than one description is found, doxygen aggregates the descriptions in an arbitrary order in the output pages.

If you introduce a new namespace, provide a description block for only one declaration and not for every occurrence.

## Groups/Modules

Grouping declarations into modules helps users to find APIs in the doxygen pages.
Generally, common functions are already grouped logically into header files but doxygen does not automatically group them this way in its output.
The doxygen output includes a _Modules_ page that organizes items into groups specified using the [Grouping doxygen commands](https://www.doxygen.nl/manual/grouping.html).
These commands can group common functions across header files, source files, and even namespaces.
Groups can also be nested by defining new groups within existing groups.

For libcudf, all the group hierarchy is defined in the [doxygen_groups.h](../../include/doxygen_groups.h) header file.
The [doxygen_groups.h](../../include/doxygen_groups.h) file does not need to be included in any other source file, because the definitions in this file are used only by the doxygen tool to generate groups in the _Modules_ page.
Modify this file only to add or update groups.
The existing groups have been carefully structured and named, so new groups should be added thoughtfully.

When creating a new API, specify its group using the [\@ingroup](https://www.doxygen.nl/manual/commands.html#cmdingroup) tag and the group reference id from the [doxygen_groups.h](../../include/doxygen_groups.h) file.

    namespace CUDF_EXPORT cudf {

    /**
     * @brief ...
     *
     * @ingroup transformation_fill
     *
     * @param ...
     * @return ...
     */
    std::unique_ptr<column> fill(table_view const& input,...);

    }  // namespace cudf

You can also use the \@addtogroup with a `@{ ... @}` pair to automatically include doxygen comment blocks as part of a group.

    namespace CUDF_EXPORT cudf {
    /**
     * @addtogroup transformation_fill
     * @{
     */

    /**
     * @brief ...
     *
     * @param ...
     * @return ...
     */
    std::unique_ptr<column> fill(table_view const& input,...);

    /** @} */
    }  // namespace cudf

This just saves adding \@ingroup to individual doxygen comment blocks within a file.
Make sure a blank line is included after the \@addtogroup command block so doxygen knows it does not apply to whatever follows in the source code.
Note that doxygen will not assign groups to items if the \@addtogroup with `@{ ... @}` pair includes a namespace declaration.
So include the `@addtogroup` and `@{ ... @}` between the namespace declaration braces as shown in the example above.

Summary of groups tags
| Tag/Command | Where to use |
| ----------- | ------------ |
| `@defgroup` | For use only in [doxygen_groups.h](../../include/doxygen_groups.h) and should include the group's title. |
| `@ingroup` | Use inside individual doxygen block comments for declaration statements in a header file. |
| `@addtogroup` | Use instead of `@ingroup` for multiple declarations in the same file within a namespace declaration. Do not specify a group title. |
| `@{ ... @}` |  Use only with `@addtogroup`. |

## Build Doxygen Output

We recommend installing Doxygen using conda (`conda install doxygen`) or a Linux package manager (`sudo apt install doxygen`).
Alternatively you can [build and install doxygen from source](https://www.doxygen.nl/manual/install.html).

To build the libcudf HTML documentation simply run the `doxygen` command from the `cpp/doxygen` directory containing the `Doxyfile`.
The libcudf documentation can also be built using `cmake --build . --target docs_cudf` from the cmake build directory (e.g. `cpp/build`).
Doxygen reads and processes all appropriate source files under the `cpp/include/` directory.
The output is generated in the `cpp/doxygen/html/` directory.
You can load the local `index.html` file generated there into any web browser to view the result.

To view docs built on a remote server, you can run a simple HTTP server using Python: `cd html && python -m http.server`.
Then open `<IP address>:8000` in your local web browser, inserting the IP address of the machine on which you ran the HTTP server.

The doxygen output is intended for building documentation only for the public APIs and classes.
For example, the output should not include documentation for `detail` or `/src` files, and these directories are excluded in the `Doxyfile` configuration.
When published by the build/CI system, the doxygen output will appear on our external [RAPIDS web site](https://docs.rapids.ai/api/libcudf/stable/index.html).
