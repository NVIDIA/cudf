# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run with Jython and zstd-jni on the classpath."""

import os
import shutil
import stat
import tempfile
import unittest
import zipfile

import native_chunks
from com.github.luben.zstd import ZstdIOException


def write(path, contents):
    with open(path, "wb") as output:
        output.write(contents)


class NativeChunkPackagingTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_removes_stale_chunks_when_source_is_small_or_missing(self):
        for source_exists in (True, False):
            library = os.path.join(self.root, "libcudf.so")
            write(library, b"a" * 2048)
            self.assertTrue(native_chunks.produce(library, 0, 1024, 9, 2))
            chunks = library + ".chunks"
            manifest = library + ".chunks.properties"
            self.assertTrue(os.path.isdir(chunks))
            self.assertTrue(os.path.isfile(manifest))

            if source_exists:
                write(library, b"b" * 1024)
            self.assertFalse(native_chunks.produce(library, 2048, 1024, 9, 2))
            self.assertEqual(source_exists, os.path.exists(library))
            self.assertFalse(os.path.exists(chunks))
            self.assertFalse(os.path.exists(manifest))

    def test_replaces_stale_chunks_after_source_recopy(self):
        library = os.path.join(self.root, "libcudf.so")
        write(library, b"a" * 2048)
        self.assertTrue(native_chunks.produce(library, 0, 1024, 9, 2))
        with open(library + ".chunks/00000", "rb") as input_file:
            old_chunk = input_file.read()

        write(library, b"b" * 2048)
        self.assertTrue(native_chunks.produce(library, 0, 1024, 9, 2))
        with open(library + ".chunks/00000", "rb") as input_file:
            self.assertNotEqual(old_chunk, input_file.read())
        self.assertFalse(os.path.exists(library))

    def test_round_trip_and_corruption(self):
        directory = os.path.join(self.root, "amd64", "Linux")
        os.makedirs(directory)
        library = os.path.join(directory, "libcudf.so")
        write(library, b"0123456789abcdef" * 200000)
        self.assertTrue(native_chunks.produce(library, 0, 1024 * 1024, 9, 2))
        self.assertFalse(os.path.exists(library))
        archive = os.path.join(self.root, "test.jar")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as jar:
            jar.comment = "build metadata"
            jar.writestr("other.txt", "untouched")
            for directory, unused_dirs, names in os.walk(self.root):
                for name in names:
                    path = os.path.join(directory, name)
                    if path != archive:
                        jar.write(path, os.path.relpath(path, self.root))
        os.chmod(archive, 0o644)
        self.assertEqual(4, native_chunks.store(self.root, archive))
        self.assertEqual(4, native_chunks.verify(archive))
        self.assertEqual(4, native_chunks.store(self.root, archive))
        with zipfile.ZipFile(archive) as jar:
            names = jar.namelist()
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual("untouched", jar.read("other.txt"))
            self.assertEqual("build metadata", jar.comment)
            self.assertEqual(0o644, stat.S_IMODE(os.stat(archive).st_mode))
            self.assertEqual(
                zipfile.ZIP_DEFLATED, jar.getinfo("other.txt").compress_type
            )

        write(library + ".chunks/00000", b"broken")
        native_chunks.store(self.root, archive)
        with self.assertRaises(ZstdIOException):
            native_chunks.verify(archive)


if __name__ == "__main__":
    unittest.main()
