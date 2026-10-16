# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import random

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


def reference_bpe(text, ranks, separator=" "):
    """Reference BPE: repeatedly merge all (left-to-right, non-overlapping)
    occurrences of the lowest ranked adjacent pair of tokens.
    """
    if text is None:
        return None
    tokens = list(text)
    while len(tokens) > 1:
        candidates = [
            ranks[pair] for pair in itertools.pairwise(tokens) if pair in ranks
        ]
        if not candidates:
            break
        best = min(candidates)
        merged = []
        i = 0
        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and ranks.get((tokens[i], tokens[i + 1])) == best
            ):
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        tokens = merged
    return separator.join(tokens)


def to_ranks(merge_pairs):
    return {
        tuple(pair.split(" ", 1)): rank
        for rank, pair in enumerate(merge_pairs)
    }


def encode(strings, merge_pairs, separator=None):
    return plc.nvtext.byte_pair_encode.byte_pair_encoding(
        plc.Column.from_arrow(pa.array(strings, type=pa.string())),
        plc.nvtext.byte_pair_encode.BPEMergePairs(
            plc.Column.from_arrow(pa.array(merge_pairs))
        ),
        None
        if separator is None
        else plc.Scalar.from_arrow(pa.scalar(separator)),
    )


@pytest.fixture(scope="module")
def merge_pairs():
    return [
        "e n",
        "i t",
        "i s",
        "e s",
        "en t",
        "c e",
        "es t",
        "en ce",
        "t h",
        "h i",
        "th is",
        "t est",
        "s i",
        "s ent",
    ]


@pytest.mark.parametrize(
    "separator, expected",
    [
        (None, ["test   sent ence", "this is   test", "this is it"]),
        ("_", ["test_ _sent_ence", "this_is_ _test", "this_is_it"]),
    ],
)
def test_byte_pair_encoding(merge_pairs, separator, expected):
    result = encode(
        ["test sentence", "thisis test", "thisisit"], merge_pairs, separator
    )
    assert_column_eq(result, pa.array(expected))


def test_byte_pair_encoding_nulls_and_slices(merge_pairs):
    strings = ["thisisit", None, "", "test sentence", None, "thisis"]
    expected = ["this is it", None, "", "test   sent ence", None, "this is"]
    plc_col = plc.Column.from_arrow(pa.array(strings))
    mps = plc.nvtext.byte_pair_encode.BPEMergePairs(
        plc.Column.from_arrow(pa.array(merge_pairs))
    )
    result = plc.nvtext.byte_pair_encode.byte_pair_encoding(plc_col, mps)
    assert_column_eq(result, pa.array(expected))

    sliced = plc.copying.slice(plc_col, [1, 5])[0]
    result = plc.nvtext.byte_pair_encode.byte_pair_encoding(sliced, mps)
    assert_column_eq(result, pa.array(expected[1:5]))


def test_byte_pair_encoding_empty(merge_pairs):
    result = encode([], merge_pairs)
    assert_column_eq(result, pa.array([], type=pa.string()))
    result = encode(["", None, ""], merge_pairs)
    assert_column_eq(result, pa.array(["", None, ""]))


def test_byte_pair_encoding_rank_order():
    # pairs are merged by rank even if a lower ranked pair needs
    # a token that is created by a higher ranked pair
    merge_pairs = ["t he", "h e", "e n", "e s", "en t", "c e", "en ce"]
    strings = ["thetest", "sentence", "thesentence"]
    expected = [reference_bpe(s, to_ranks(merge_pairs), "|") for s in strings]
    assert expected == ["the|t|es|t", "s|ent|ence", "the|s|ent|ence"]
    assert_column_eq(encode(strings, merge_pairs, "|"), pa.array(expected))


def random_merge_pairs(rng, alphabet, count, max_token=8):
    """Builds a table where every half exists before it is used,
    like tables created by BPE training.
    """
    vocab = list(alphabet)
    pairs = []
    seen = set()
    while len(pairs) < count:
        left = vocab[max(rng.randrange(len(vocab)), rng.randrange(len(vocab)))]
        right = vocab[
            max(rng.randrange(len(vocab)), rng.randrange(len(vocab)))
        ]
        if len(left) + len(right) > max_token or (left, right) in seen:
            continue
        seen.add((left, right))
        pairs.append(f"{left} {right}")
        vocab.append(left + right)
    return pairs


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_byte_pair_encoding_random(seed):
    rng = random.Random(seed)
    # includes multi-byte characters; space and '-' never appear in the table
    alphabet = ["a", "b", "c", "d", "e", "é", "ü", "中"]
    merge_pairs = random_merge_pairs(rng, alphabet, 60)
    characters = [*alphabet, " ", "-"]
    strings = [
        "".join(rng.choice(characters) for _ in range(rng.randrange(300)))
        for _ in range(200)
    ]
    strings += ["a" * n for n in (1, 2, 3, 7, 64, 65, 300)]
    strings += [None, ""]
    ranks = to_ranks(merge_pairs)
    expected = [reference_bpe(s, ranks) for s in strings]
    assert_column_eq(encode(strings, merge_pairs), pa.array(expected))


def test_byte_pair_encoding_errors(merge_pairs):
    with pytest.raises(ValueError, match="unique"):
        plc.nvtext.byte_pair_encode.BPEMergePairs(
            plc.Column.from_arrow(pa.array(["a b", "c d", "a b"]))
        )
    for separator in ["", "ab", "é"]:
        with pytest.raises(RuntimeError, match="separator"):
            encode(["thisisit"], merge_pairs, separator)
