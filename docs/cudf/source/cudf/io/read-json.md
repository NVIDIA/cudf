# Working with JSON data

This page contains a tutorial about reading and manipulating JSON data in cuDF.

## Reading JSON data

By default, the cuDF JSON reader expects input data using the
"records" orientation. Records-oriented JSON data comprises
an array of objects at the root level, and each object in the
array corresponds to a row. Records-oriented JSON data begins
with `[`, ends with `]` and ignores unquoted whitespace.
Another common variant for JSON data is "JSON Lines", where
JSON objects are separated by new line characters (`\n`), and
each object corresponds to a row.

```python
>>> j = '''[
    {"a": "v1", "b": 12},
    {"a": "v2", "b": 7},
    {"a": "v3", "b": 5}
]'''
>>> df_records = cudf.read_json(j, engine='cudf')

>>> j = '\n'.join([
...     '{"a": "v1", "b": 12}',
...     '{"a": "v2", "b": 7}',
...     '{"a": "v3", "b": 5}'
... ])
>>> df_lines = cudf.read_json(j, lines=True)

>>> df_lines
    a   b
0  v1  12
1  v2   7
2  v3   5
>>> df_records.equals(df_lines)
True
```

The cuDF JSON reader also supports arbitrarily-nested combinations
of JSON objects and arrays, which map to struct and list data types.
The following examples demonstrate the inputs and outputs for
reading nested JSON data.

```python
# Example with columns types:
# list<int> and struct<k:string>
>>> j = '''[
    {"list": [0,1,2], "struct": {"k":"v1"}},
    {"list": [3,4,5], "struct": {"k":"v2"}}
]'''
>>> df = cudf.read_json(j, engine='cudf')
>>> df
        list       struct
0  [0, 1, 2]  {'k': 'v1'}
1  [3, 4, 5]  {'k': 'v2'}

# Example with columns types:
# list<struct<k:int>> and struct<k:list<int>, m:int>
>>> j = '\n'.join([
...     '{"a": [{"k": 0}], "b": {"k": [0, 1], "m": 5}}',
...     '{"a": [{"k": 1}, {"k": 2}], "b": {"k": [2, 3], "m": 6}}',
... ])
>>> df = cudf.read_json(j, lines=True)
>>> df
                      a                      b
0            [{'k': 0}]  {'k': [0, 1], 'm': 5}
1  [{'k': 1}, {'k': 2}]  {'k': [2, 3], 'm': 6}
```

## Handling large and small JSON Lines files

For workloads based on JSON Lines data, cuDF includes reader options
to assist with data processing: byte range support for large files,
and multi-source support for small files.

Some workflows require processing large JSON Lines files that may
exceed GPU memory capacity. The JSON reader in cuDF supports a byte
range argument that specifies a starting byte offset and size in bytes.
The reader parses each record that begins within the byte range,
and for this reason byte ranges do not need to align with record
boundaries. To avoid skipping rows or reading duplicate rows, byte ranges
should be adjacent, as shown in the following example.

```python
>>> num_rows = 10
>>> j = '\n'.join([
...     '{"id":%s, "distance": %s, "unit": "m/s"}' % x \
...     for x in zip(range(num_rows), cupy.random.rand(num_rows))
... ])

>>> chunk_count = 4
>>> chunk_size = len(j) // chunk_count + 1
>>> data = []
>>> for x in range(chunk_count):
...    d = cudf.read_json(
...         j,
...         lines=True,
...         byte_range=(chunk_size * x, chunk_size),
...     )
...     data.append(d)
>>> df = cudf.concat(data)
```

By contrast, some workflows require processing many small JSON
Lines files. Rather than looping through the sources and
concatenating the resulting dataframes, the JSON reader in
cuDF accepts an iterable of data sources. Then the raw inputs
are concatenated and processed as a single source. Please
note that the JSON reader in cuDF accepts sources as file paths,
raw strings, or file-like objects, as well as iterables of these sources.

```python
>>> j1 = '{"id":0}\n{"id":1}\n'
>>> j2 = '{"id":2}\n{"id":3}\n'

>>> df = cudf.read_json([j1, j2], lines=True)
```

## Unpacking list and struct data

After reading JSON data into a cuDF dataframe with list/struct
column types, the next step in many workflows extracts or
flattens the data into simple types. For struct columns, one
solution is extracting the data with the `struct.explode`
accessor and joining the result to the parent dataframe. The
following example demonstrates how to extract data from a struct column.

```python
>>> j = '\n'.join([
...    '{"x": "Tokyo", "y": {"country": "Japan", "iso2": "JP"}}',
...    '{"x": "Jakarta", "y": {"country": "Indonesia", "iso2": "ID"}}',
...    '{"x": "Shanghai", "y": {"country": "China", "iso2": "CN"}}'
... ])
>>> df = cudf.read_json(j, lines=True)
>>> df = df.drop(columns='y').join(df['y'].struct.explode())
>>> df
          x    country iso2
0     Tokyo      Japan   JP
1   Jakarta  Indonesia   ID
2  Shanghai      China   CN
```

For list columns where the order of the elements is meaningful,
the `list.get` accessor extracts the elements from specific
positions. The resulting `cudf.Series` object can then be assigned
to a new column in the dataframe. The following example
demonstrates how to extract the first and second elements from a
list column.

```python
>>> j = '\n'.join([
...    '{"name": "Peabody, MA", "coord": [42.53, -70.98]}',
...    '{"name": "Northampton, MA", "coord": [42.32, -72.66]}',
...    '{"name": "New Bedford, MA", "coord": [41.63, -70.93]}'
... ])

>>> df = cudf.read_json(j, lines=True)
>>> df['latitude'] = df['coord'].list.get(0)
>>> df['longitude'] = df['coord'].list.get(1)
>>> df = df.drop(columns='coord')
>>> df
              name  latitude  longitude
0      Peabody, MA     42.53     -70.98
1  Northampton, MA     42.32     -72.66
2  New Bedford, MA     41.63     -70.93
```

Finally, for list columns with variable length, the `explode`
method creates a new dataframe with each element as a row.
Joining the exploded dataframe on the parent dataframe yields
an output with all simple types. The following example flattens
a list column and joins it to the index and additional data from
the parent dataframe.

```python
>>> j = '\n'.join([
...    '{"product": "socks", "ratings": [2, 3, 4]}',
...    '{"product": "shoes", "ratings": [5, 4, 5, 3]}',
...    '{"product": "shirts", "ratings": [3, 4]}'
... ])

>>> df = cudf.read_json(j, lines=True)
>>> df = df.drop(columns='ratings').join(df['ratings'].explode())
>>> df
  product  ratings
0   socks        2
0   socks        4
0   socks        3
1   shoes        5
1   shoes        5
1   shoes        4
1   shoes        3
2  shirts        3
2  shirts        4
```

## Building JSON data solutions

Sometimes a workflow needs to process JSON data with an object
root and cuDF provides tools to build solutions for this kind
of data. If you need to process JSON data with an object root,
we recommend reading the data as a single JSON Line and then
unpacking the resulting dataframe. The following example
reads a JSON object as a single line and then extracts the
"results" field into a new dataframe.

```python
>>> j = '''{
    "metadata" : {"vehicle":"car"},
    "results": [
        {"id": 0, "distance": 1.2},
        {"id": 1, "distance": 2.4},
        {"id": 2, "distance": 1.7}
    ]
}'''

# first read the JSON object with line=True
>>> df = cudf.read_json(j, lines=True)
>>> df
             metadata                                            results
0  {'vehicle': 'car'}  [{'id': 0, 'distance': 1.2}, {'id': 1, 'distan...

# then explode the 'results' column
>>> df = df['results'].explode().struct.explode()
>>> df
   id  distance
0   0       1.2
1   1       2.4
2   2       1.7
```
