# Unicode Limitations

The strings column currently supports only UTF-8 characters internally.
For functions that require character testing (e.g. cudf::strings::all_characters_of_type()) or
case conversion (e.g. cudf::strings::capitalize(), etc) only the 16-bit [Unicode 13.0](https://www.unicode.org/versions/Unicode13.0.0/)
character code-points (0-65535) values are supported.
Case conversion and character testing on characters above code-point 65535 are not supported.

Case conversions that are context-sensitive are not supported. Also, case conversions that result
in multiple characters are not reversible. That is, adjacent individual characters will not be case converted
to a single character. For example, converting character ß to upper case will result in the characters "SS". But converting "SS" to lower case will produce "ss".

Strings case and type APIs:

- cudf::strings::all_characters_of_type()
- cudf::strings::to_upper()
- cudf::strings::to_lower()
- cudf::strings::capitalize()
- cudf::strings::title()
- cudf::strings::swapcase()

Also, using regex patterns that use the shorthand character classes `\d \D \w \W \s \S` will include only appropriate characters with
code-points between (0-65535).
