#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce zstd native-library chunks and store their JAR entries without ZIP compression."""

import argparse
import os
import shutil
import stat
import sys
import tempfile
import threading
import zipfile

try:
    import Queue as queue
except ImportError:
    import queue

CHUNK_SIZE = 32 * 1024 * 1024
MINIMUM_SIZE = 256 * 1024 * 1024


def _file_state(path):
    stat = os.stat(path)
    modified = getattr(stat, "st_mtime_ns", None)
    if modified is None:
        from java.nio.file import Files, Paths
        from java.util.concurrent import TimeUnit

        modified = Files.getLastModifiedTime(Paths.get(path)).to(
            TimeUnit.NANOSECONDS
        )
    return stat.st_size, modified


def _compress(source, output, offset, length, level):
    from com.github.luben.zstd import ZstdOutputStreamNoFinalizer
    from jarray import zeros
    from java.io import FileInputStream, FileOutputStream
    from java.util.zip import CRC32

    checksum = CRC32()
    input_stream = FileInputStream(source)
    try:
        input_stream.getChannel().position(offset)
        output_stream = FileOutputStream(output)
        try:
            compressed = ZstdOutputStreamNoFinalizer(output_stream, level)
            try:
                buffer = zeros(1024 * 1024, "b")
                while length:
                    read = input_stream.read(
                        buffer, 0, min(len(buffer), length)
                    )
                    if read < 0:
                        raise IOError(
                            "Native library changed during chunking: %s"
                            % source
                        )
                    compressed.write(buffer, 0, read)
                    checksum.update(buffer, 0, read)
                    length -= read
            finally:
                compressed.close()
        finally:
            output_stream.close()
    finally:
        input_stream.close()
    return checksum.getValue()


def _compress_all(source, temporary, size, chunk_size, level, workers):
    count = (size + chunk_size - 1) // chunk_size
    tasks = queue.Queue()
    for index in range(count):
        tasks.put(index)
    checksums = [None] * count
    errors = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                index = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                checksums[index] = _compress(
                    source,
                    os.path.join(temporary, "%05d" % index),
                    index * chunk_size,
                    min(chunk_size, size - index * chunk_size),
                    level,
                )
            except BaseException as error:
                lock.acquire()
                try:
                    errors.append(error)
                finally:
                    lock.release()
            finally:
                tasks.task_done()

    threads = [
        threading.Thread(target=worker)
        for unused in range(min(workers, count))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return checksums


def produce(
    source,
    minimum_size=MINIMUM_SIZE,
    chunk_size=CHUNK_SIZE,
    level=9,
    workers=12,
):
    source = str(source)
    chunks = source + ".chunks"
    manifest = source + ".chunks.properties"
    # Maven may copy a rebuilt library over the previous output while leaving
    # its old chunks in target/classes. Replace those chunks for this source.
    shutil.rmtree(chunks, ignore_errors=True)
    if os.path.exists(manifest):
        os.unlink(manifest)
    if not os.path.isfile(source) or os.path.getsize(source) < minimum_size:
        return False
    size, modified = _file_state(source)
    temporary = tempfile.mkdtemp(
        prefix=os.path.basename(source) + ".chunks.tmp-",
        dir=os.path.dirname(source) or ".",
    )
    temporary_manifest = temporary + ".properties"
    try:
        checksums = _compress_all(
            source, temporary, size, chunk_size, level, workers
        )
        if _file_state(source) != (size, modified):
            raise IOError(
                "Native library changed during chunking: %s" % source
            )
        lines = [
            "format.version=1",
            "library.size=%d" % size,
            "chunk.size=%d" % chunk_size,
            "chunk.count=%d" % len(checksums),
            "chunk.compression=zstd",
        ]
        lines.extend(
            "chunk.%05d.crc32=%08x" % (index, crc)
            for index, crc in enumerate(checksums)
        )
        with open(temporary_manifest, "wb") as output:
            output.write(("\n".join(lines) + "\n").encode("ascii"))
        os.rename(temporary, chunks)
        os.rename(temporary_manifest, manifest)
        os.unlink(source)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if os.path.exists(temporary_manifest):
            os.unlink(temporary_manifest)
        shutil.rmtree(chunks, ignore_errors=True)
        if os.path.exists(manifest):
            os.unlink(manifest)
        raise
    return True


def _properties(contents):
    return dict(
        line.split("=", 1)
        for line in contents.decode("ascii").splitlines()
        if "=" in line
    )


def store(root, archive):
    from jarray import zeros
    from java.io import FileInputStream, FileOutputStream
    from java.util.jar import JarFile, JarOutputStream
    from java.util.zip import CRC32, ZipEntry

    root, archive = str(root), str(archive)
    entries = []
    for directory, unused_dirs, names in os.walk(root):
        for name in sorted(names):
            if not name.endswith(".chunks.properties"):
                continue
            manifest = os.path.join(directory, name)
            with open(manifest, "rb") as input_file:
                properties = _properties(input_file.read())
            if properties.get("chunk.compression") != "zstd":
                continue
            library = manifest[: -len(".chunks.properties")]
            for index in range(int(properties["chunk.count"])):
                chunk = os.path.join(library + ".chunks", "%05d" % index)
                if not os.path.isfile(chunk):
                    raise IOError("Missing native chunk %s" % chunk)
                entries.append(
                    os.path.relpath(chunk, root).replace(os.sep, "/")
                )
    if not entries:
        return 0
    replacements = set(entries)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(archive) + ".tmp-",
        dir=os.path.dirname(os.path.abspath(archive)),
    )
    os.close(descriptor)
    buffer = zeros(1024 * 1024, "b")
    try:
        original = JarFile(archive)
        try:
            output = JarOutputStream(FileOutputStream(temporary))
            try:
                if original.getComment() is not None:
                    output.setComment(original.getComment())
                for old_entry in original.entries():
                    name = old_entry.getName()
                    if name in replacements:
                        continue
                    entry = ZipEntry(name)
                    entry.setMethod(old_entry.getMethod())
                    if old_entry.getTime() >= 0:
                        entry.setTime(old_entry.getTime())
                    if old_entry.getComment() is not None:
                        entry.setComment(old_entry.getComment())
                    if old_entry.getExtra() is not None:
                        entry.setExtra(old_entry.getExtra())
                    if old_entry.getMethod() == ZipEntry.STORED:
                        entry.setSize(old_entry.getSize())
                        entry.setCrc(old_entry.getCrc())
                    output.putNextEntry(entry)
                    if not old_entry.isDirectory():
                        input_stream = original.getInputStream(old_entry)
                        try:
                            _copy_stream(input_stream, output, buffer)
                        finally:
                            input_stream.close()
                    output.closeEntry()
                for name in entries:
                    path = os.path.join(root, name.replace("/", os.sep))
                    size = os.path.getsize(path)
                    checksum = CRC32()
                    input_stream = FileInputStream(path)
                    try:
                        while True:
                            read = input_stream.read(buffer, 0, len(buffer))
                            if read < 0:
                                break
                            checksum.update(buffer, 0, read)
                    finally:
                        input_stream.close()
                    entry = ZipEntry(name)
                    entry.setMethod(ZipEntry.STORED)
                    entry.setSize(size)
                    entry.setCrc(checksum.getValue())
                    output.putNextEntry(entry)
                    input_stream = FileInputStream(path)
                    try:
                        _copy_stream(input_stream, output, buffer)
                    finally:
                        input_stream.close()
                    output.closeEntry()
            finally:
                output.close()
        finally:
            original.close()
        os.chmod(temporary, stat.S_IMODE(os.stat(archive).st_mode))
        os.rename(temporary, archive)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    with zipfile.ZipFile(archive) as jar:
        for name in entries:
            if jar.getinfo(name).compress_type != zipfile.ZIP_STORED:
                raise RuntimeError("Native chunk is not STORED: %s" % name)
    return len(entries)


def _copy_stream(source, output, buffer):
    while True:
        read = source.read(buffer, 0, len(buffer))
        if read < 0:
            return
        output.write(buffer, 0, read)


def verify(archive):
    from com.github.luben.zstd import ZstdInputStreamNoFinalizer
    from jarray import zeros
    from java.util.jar import JarFile
    from java.util.zip import CRC32

    checked = 0
    with zipfile.ZipFile(str(archive)) as jar:
        names = set(jar.namelist())
        native_jar = JarFile(str(archive))
        try:
            for manifest_name in sorted(
                name for name in names if name.endswith(".chunks.properties")
            ):
                properties = _properties(jar.read(manifest_name))
                if properties.get("chunk.compression") != "zstd":
                    continue
                if properties.get("format.version") != "1":
                    raise RuntimeError(
                        "Unsupported chunk format: %s" % manifest_name
                    )
                library_name = manifest_name[: -len(".chunks.properties")]
                if library_name in names:
                    raise RuntimeError(
                        "Conventional library beside %s" % manifest_name
                    )
                size = int(properties["library.size"])
                chunk_size = int(properties["chunk.size"])
                count = int(properties["chunk.count"])
                if (
                    size <= 0
                    or chunk_size <= 0
                    or count != (size + chunk_size - 1) // chunk_size
                ):
                    raise RuntimeError(
                        "Invalid chunk layout: %s" % manifest_name
                    )
                for index in range(count):
                    name = library_name + ".chunks/%05d" % index
                    entry = jar.getinfo(name)
                    if entry.compress_type != zipfile.ZIP_STORED:
                        raise RuntimeError(
                            "Native chunk is not STORED: %s" % name
                        )
                    compressed = ZstdInputStreamNoFinalizer(
                        native_jar.getInputStream(native_jar.getJarEntry(name))
                    )
                    try:
                        buffer = zeros(1024 * 1024, "b")
                        checksum = CRC32()
                        total = 0
                        while True:
                            read = compressed.read(buffer, 0, len(buffer))
                            if read < 0:
                                break
                            total += read
                            if total > min(
                                chunk_size, size - index * chunk_size
                            ):
                                raise RuntimeError(
                                    "Oversized native chunk: %s" % name
                                )
                            checksum.update(buffer, 0, read)
                    finally:
                        compressed.close()
                    expected_size = min(chunk_size, size - index * chunk_size)
                    expected_crc = int(
                        properties["chunk.%05d.crc32" % index], 16
                    )
                    if (
                        total != expected_size
                        or checksum.getValue() != expected_crc
                    ):
                        raise RuntimeError(
                            "Native chunk size or CRC mismatch: %s" % name
                        )
                    checked += 1
        finally:
            native_jar.close()
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    pack = sub.add_parser("produce")
    pack.add_argument("library")
    pack.add_argument("--minimum-size", type=int, default=MINIMUM_SIZE)
    pack.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    pack.add_argument("--level", type=int, default=9)
    pack.add_argument("--workers", type=int, default=12)
    pack.add_argument("--require-loader")
    jar = sub.add_parser("store")
    jar.add_argument("root")
    jar.add_argument("archive")
    check = sub.add_parser("verify")
    check.add_argument("archive")
    args = parser.parse_args()
    if args.command is None:
        parser.error("A command is required")
    if args.command == "produce":
        if args.minimum_size < 0 or args.chunk_size <= 0 or args.workers <= 0:
            parser.error("Invalid chunk size or worker count")
        if args.require_loader is not None:
            if not os.path.isfile(args.require_loader):
                parser.error(
                    "cuDF submodule must contain the zstd NativeDepsLoader"
                )
            with open(args.require_loader) as loader:
                if "CHUNK_COMPRESSION_KEY" not in loader.read():
                    parser.error(
                        "cuDF submodule must contain the zstd NativeDepsLoader"
                    )
        sys.stdout.write(
            "Chunked native library: %s\n"
            % produce(
                args.library,
                args.minimum_size,
                args.chunk_size,
                args.level,
                args.workers,
            )
        )
    elif args.command == "store":
        sys.stdout.write(
            "Stored %d zstd native chunks\n" % store(args.root, args.archive)
        )
    else:
        sys.stdout.write(
            "Verified %d zstd native chunks\n" % verify(args.archive)
        )


if __name__ == "__main__":
    main()
