# Regex Features {#md_regex}

This page specifies which regular expression (regex) features are currently supported by libcudf strings column APIs that accept regex patterns:

- cudf::strings::contains_re()
- cudf::strings::matches_re()
- cudf::strings::count_re()
- cudf::strings::extract()
- cudf::strings::extract_all_record()
- cudf::strings::findall()
- cudf::strings::find_re()
- cudf::strings::replace_re()
- cudf::strings::replace_with_backrefs()
- cudf::strings::split_re()
- cudf::strings::split_record_re()

The details are based on features documented at https://www.regular-expressions.info/reference.html

**Note:** The alternation character is the pipe character `|` and not the character included in the tables on this page. There is an issue including the pipe character inside the table markdown that is rendered by doxygen.

By default, only the `\n` character is recognized as a line break. The [cudf::strings::regex_flags::EXT_NEWLINE](@ref cudf::strings::regex_flags) increases the set of line break characters to include:
- Paragraph separator (Unicode: `2029`, UTF-8: `E280A9`)
- Line separator (Unicode: `2028`, UTF-8: `E280A8`)
- Next line (Unicode: `0085`, UTF-8: `C285`)
- Carriage return (Unicode: `000D`, UTF-8: `0D`)

With `EXT_NEWLINE`, a carriage-return/line-feed pair (`\r\n`) is treated as a single line terminator: the `^` and `$` anchors match before the `\r` and after the `\n`, and never between them.

**Invalid regex patterns will result in undefined behavior**. This includes but is not limited to the following:
- Unescaped special characters (listed in the third row of the Characters table below) when they are intended to match as literals.
- Unmatched paired special characters like `()`, `[]`, and `{}`.
- Empty groups, classes, or quantifiers. That is, `()` and `[]` without an enclosing expression and `{}` without a valid integer.
- Unqualified quantifiers. That is, a quantifier with no preceding item to match like `*a`, `a⎮?`, `(+)`, `{2}a`, etc.

## Features Supported

### Characters

| Feature  | Syntax | Description | Example |
| ---------- | ------------- | ------------- | ------------- |
| Literal character | Any character except `[\^$.⎮?*+()` | All characters except the listed special characters match a single instance of themselves | `a` matches `a` |
| Literal curly braces | `{` and `}` | `{` and `}` are literal characters, unless they are part of a valid regular expression token such as a quantifier `{3}` | `{` matches `{` |
| Backslash escapes a metacharacter | `\` followed by any of `[\^$.⎮?*+(){}` | A backslash escapes special characters to suppress their special meaning | `\*` matches `*` |
| Hexadecimal escape | `\xFF` where `FF` are 2 hexadecimal digits | Matches the character at the specified position in the ASCII table | `\x40` matches `@` |
| Character escape | `\n`, `\r` and `\t` | Match an line-feed (LF) character, carriage return (CR) character and a tab character respectively | `\r\n` matches a Windows CRLF line break |
| Character escape | `\a` | Match the "alert" or "bell" control character (ASCII 0x07) | |
| Character escape | `\f` | Match the form-feed control character (ASCII 0x0C) | |
| NULL escape      | `\0` | Match the NULL character ||
| Octal escape     | `\100` through `\177` <br/> `\200` through `\377` <br/> `\01` through `\07` <br/> `\010` through `\077` | Matches the character at the specified position in the ASCII table | `\100` matches `@` |

### Basic Features

| Feature  | Syntax | Description | Example |
| ---------- | ------------- | ------------- | ------------- |
| Dot | . (dot) | Matches any single character except line break characters. Optionally match line break characters. The behavior of the dot when encountering a  `\n` character can be controlled by cudf::strings::regex_flags for some regex APIs. | . matches x or (almost) any other character |
| Alternation | `⎮` (pipe) | Causes the regex engine to match either the part on the left side, or the part on the right side. Can be strung together into a series of alternations. | `abc⎮def⎮xyz` matches `abc`, `def` or `xyz` |


### Character Classes

| Feature  | Syntax | Description | Example |
| ---------- | ------------- | ------------- | ------------- |
| Character class | `[` | `[` begins a character class. | |
| Literal character | Any character except `\^-]` | All characters except the listed special characters are literal characters that add themselves to the character class. | `[abc]` matches `a`, `b` or `c` |
| Backslash escapes a metacharacter	| `\` (backslash) followed by any of `\^-]` | A backslash escapes special characters to suppress their special meaning. | `[\^\]]` matches `^` or `]` |
| Range | `-` (hyphen) between two tokens that each specify a single character. | Adds a range of characters to the character class. If '`-`' is the first or last character (e.g. `[a-]` or `[-z]`), it will match a literal '`-`' and not infer a range. | `[a-zA-Z0-9]` matches any ASCII letter or digit |
| Negated character class | `^` (caret) immediately after the opening `[` | Negates the character class, causing it to match a single character not listed in the character class. | `[^a-d]` matches `x` (any character except `a`, `b`, `c` or `d`) |
| Literal opening bracket | `[` | An opening square bracket is a literal character that adds an opening square bracket to the character class. | `[ab[cd]ef]` matches `aef]`, `bef]`, `[ef]`, `cef]`, and `def]` |
| Character escape | `\n`, `\r` and `\t` | Add an LF character, a CR character, or a tab character to the character class, respectively. | `[\n\r\t]` matches a line feed, a carriage return, or a tab character |
| Character escape | `\a` | Add the "alert" or "bell" control character (ASCII 0x07) to the character class. | `[\a\t]` matches a bell or a tab character |
| Character escape | `\b` | Add the backspace control character (ASCII 0x08) to the character class. | `[\b\t]` matches a backspace or a tab character |
| Character escape | `\f` | Add the form-feed control character (ASCII 0x0C) to the character class. | `[\f\t]` matches a form-feed or a tab character |

### Shorthand Character Classes

| Feature  | Syntax | Description | Example |
| ---------- | ------------- | ------------- | ------------- |
| Shorthand | `\d` | Adds all digits to the character class. Matches a single digit if used outside character classes. The behavior can be controlled by [cudf::strings::regex_flags::ASCII](@ref cudf::strings::regex_flags) to include only `[0-9]` | `\d` matches a character that is a digit |
| Shorthand | `\w` | Adds all word characters to the character class. Matches a single word character if used outside character classes. The behavior can be controlled by [cudf::strings::regex_flags::ASCII](@ref cudf::strings::regex_flags) to include only `[0-9A-Za-z_]` | `\w` matches any single word character |
| Shorthand | `\s` | Adds all whitespace to the character class. Matches a single whitespace character if used outside character classes. The behavior can be controlled by [cudf::strings::regex_flags::ASCII](@ref cudf::strings::regex_flags) to include only `[\t- ]` | `\s` matches any single whitespace character |
| Shorthand | `\D` | Adds all non-digits to the character class. Matches a single character that is not a digit character if used outside character classes. The behavior can be controlled by [cudf::strings::regex_flags::ASCII](@ref cudf::strings::regex_flags) | `[\D]` matches a single character that is not a digit character |
| Shorthand | `\W` | Adds all non-word characters to the character class. Matches a single character that is not a word character if used outside character classes. The behavior can be controlled by [cudf::strings::regex_flags::ASCII](@ref cudf::strings::regex_flags) | [`\W`] matches a single character that is not a word character |
| Shorthand | `\S` | Adds all non-whitespace to the character class. Matches a single character that is not a whitespace character if used outside character classes. The behavior can be controlled by [cudf::strings::regex_flags::ASCII](@ref cudf::strings::regex_flags) | `[\S]` matches a single character that is not a whitespace character |

### Anchors

| Feature  | Syntax | Description | Example |
| ---------- | ------------- | ------------- | ------------- |
| String anchor | `^` (caret) | Matches at the start of the string | `^.` matches `a` in `abcdef` |
| Line anchor | `^` (caret) | When [cudf::strings::regex_flags::MULTILINE](@ref cudf::strings::regex_flags) is specified: Matches after each line break in addition to matching at the start of the string, thus matching at the start of each line in the string. | `^.` matches `a` and `d` in `abc\ndef` |
| String anchor | `$` (dollar) | Matches at the end of the string as well as before the final line break in the string | `.$` matches `f` in `abcdef` and in `abcdef\n` |
| Line anchor | `$` (dollar) | When [cudf::strings::regex_flags::MULTILINE](@ref cudf::strings::regex_flags) is specified: Matches before each line break in addition to matching at the end of the string, thus matching at the end of each line in the string. | `.$` matches `c` and `f` in `abc\ndef` and in `abc\ndef\n`　|
| String anchor | `\A` | Matches at the start of the string | `\A\w` matches only `a` in `abc` |
| String anchor | `\Z` | Matches at the end of the string | `\w\Z` matches `f` in `abc\ndef` but fails to match `abc\ndef\n` |

### Word Boundaries

| Feature  | Syntax | Description | Example |
| ---------- | ------------- | ------------- | ------------- |
| Word boundary | `\b` | Matches at a position that is followed by a word character but not preceded by a word character, or that is preceded by a word character but not followed by a word character. | `\b.` matches `a`, the space, and `d` in `abc def` |
| Word boundary | `\B`　| Matches at a position that is preceded and followed by a word character, or that is not preceded and not followed by a word character. | `\B.` matches `b`, `c`, `e`, and `f` in `abc def` |

### Quantifiers

| Feature  | Syntax | Description | Example |
| ---------- | ------------- | ------------- | ------------- |
| Greedy quantifier | `?` (question mark) | Makes the preceding item optional. Greedy, so the optional item is included in the match if possible. | `abc?` matches `abc` or `ab` |
| Greedy quantifier | `*` (star) | Repeats the previous item zero or more times. Greedy, so as many items as possible will be matched before trying permutations with fewer matches of the preceding item, up to the point where the preceding item is not matched at all. | `".*"` matches `"def"` and `"ghi"` in `abc "def" "ghi" jkl` |
| Greedy quantifier | `+` (plus)　| Repeats the previous item once or more. Greedy, so as many items as possible will be matched before trying permutations with fewer matches of the preceding item, up to the point where the preceding item is matched only once. | `".+"` matches `"def"` and `"ghi"` in `abc "def" "ghi" jkl` |
| Lazy quantifier | `??` | Makes the preceding item optional. Lazy, so the optional item is excluded in the match if possible. | `abc??` matches `ab` or `abc` |
| Lazy quantifier | `*?` | Repeats the previous item zero or more times. Lazy, so the engine first attempts to skip the previous item, before trying permutations with ever increasing matches of the preceding item. | `".*?"` matches `"def"` and `"ghi"` in `abc "def" "ghi" jkl` |
| Lazy quantifier | `+?` | Repeats the previous item once or more. Lazy, so the engine first matches the previous item only once, before trying permutations with ever increasing matches of the preceding item. | `".+?"` matches `"def"` and `"ghi"` in `abc "def" "ghi" jkl` |
| Fixed quantifier | `{n}` where `n` is an integer: `0 ≤ n ≤ 999` | Repeats the previous item exactly `n` times. | `a{5}` matches `aaaaa` |
| Greedy quantifier | `{n,m}` where `n` and `m` are integers: `0 ≤ n ≤ m ≤ 999` | Repeats the previous item between `n` and `m` times. Greedy, so repeating `m` times is tried before reducing the repetition to `n` times. | `a{2,4}` matches `aaaa`, `aaa` or `aa` |
| Greedy quantifier | `{n,}` where `n` is an integer: `0 ≤ n ≤ 999` | Repeats the previous item at least `n` times. Greedy, so as many items as possible will be matched before trying permutations with fewer matches of the preceding item, up to the point where the preceding item is matched only `n` times. | `a{2,}` matches `aaaaa` in `aaaaa` |
| Lazy quantifier | `{n,m}?` where `n` and `m` are integers `0 ≤ n ≤ m ≤ 999` | Repeats the previous item between `n` and `m` times. Lazy, so repeating `n` times is tried before increasing the repetition to `m` times. | `a{2,4}?` matches `aa`, `aaa`, or `aaaa` |
| Lazy quantifier | `{n,}?` where `n` is an integer: `0 ≤ n ≤ 999` | Repeats the previous item `n` or more times. Lazy, so the engine first matches the previous item `n` times, before trying permutations with ever increasing matches of the preceding item. | `a{2,}?` matches `aa` in `aaaaa` |

### Groups

| Feature  | Syntax | Description | Example |
| ---------- | ------------- | ------------- | ------------- |
| Capturing group | `(regex)` | Parentheses group the regex between them. They capture the text matched by the regex inside them into a numbered group. They allow you to apply regex operators to the entire grouped regex. | `(abc⎮def)ghi` matches `abcghi` or `defghi` |
| Non-capturing group | `(?:regex)` | Non-capturing parentheses group the regex so you can apply regex operators, but do not capture anything. | `(?:abc⎮def)ghi` matches `abcghi` or `defghi` |

### Replacement Backreferences

| Feature  | Syntax | Description | Example |
| ---------- | ------------- | ------------- | ------------- |
| Backreference | `\1` through `\99` | Insert the text matched by capturing groups 1 through 99 | Replacing `(a)(b)(c)` with `\3\3\1` in `abc` yields `cca` |
| Backreference | `${1}` through `${99}` | Insert the text matched by capturing groups 1 through 99 | Replacing `(a)(b)(c)` with `${2}.${2}:{$3}` in `abc` yields `b.b:c` |
| Whole match | `${0}` | Insert the whole regex match | Replacing `(\d)(a)` with `[${0}]:-${2}_${1};` in `123abc` yields `12[3a]:-a_3;bc`

### Ignore case

The [cudf::strings::regex_flags::IGNORECASE](@ref cudf::strings::regex_flags) can be used to match literal characters and characters within a character class
to a limited extent. Only those characters supported by the C++ STL [`std::tolower`](https://en.cppreference.com/cpp/locale/tolower)
and [`std::toupper`](https://en.cppreference.com/cpp/locale/toupper)
when used with [`std::locale`](https://en.cppreference.com/cpp/locale/locale) with locale name`C.UTF-8` are supported.
For example, this feature does not include matching characters which have multiple character counterparts.

Character classes (defined by `[]`) have the further restriction that any range of characters are expected to map appropriately between
lower case and upper case ranges. For example the range `[a-z]` would also try to match characters in `[A-Z]` when the `IGNORECASE` is specified.

The inline `(?i...)` ignore case format pattern is not supported.
