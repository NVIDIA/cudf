/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLStreamHandler;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.CRC32;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class NativeDepsLoaderExtractionTest {
  private static final String TEST_ARCH = "native-deps-loader-tests";
  private static final String TEST_OS = "TestOS";

  private Path resourceDir;

  @BeforeEach
  void setupResources() throws IOException, URISyntaxException {
    Path classPathRoot = Paths.get(
        NativeDepsLoaderExtractionTest.class.getProtectionDomain()
            .getCodeSource().getLocation().toURI());
    resourceDir = classPathRoot.resolve(TEST_ARCH).resolve(TEST_OS);
    Files.createDirectories(resourceDir);
  }

  @AfterEach
  void removeResources() throws IOException {
    Path testRoot = resourceDir.getParent();
    if (Files.exists(testRoot)) {
      try (Stream<Path> paths = Files.walk(testRoot)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).collect(Collectors.toList())) {
          Files.delete(path);
        }
      }
    }
  }

  @Test
  void extractsConventionalResource() throws IOException {
    String baseName = "conventionaltest";
    byte[] expected = "conventional native library".getBytes(StandardCharsets.UTF_8);
    Files.write(libraryPath(baseName), expected);

    File extracted = NativeDepsLoader.createFile(TEST_OS, TEST_ARCH, baseName);
    try {
      assertArrayEquals(expected, Files.readAllBytes(extracted.toPath()));
    } finally {
      Files.deleteIfExists(extracted.toPath());
    }
  }

  @Test
  void extractsChunkedResourceAndPrefersManifest() throws IOException {
    String baseName = "chunkedtest";
    byte[] expected = "chunked native library contents".getBytes(StandardCharsets.UTF_8);
    Files.write(libraryPath(baseName),
        "wrong conventional contents".getBytes(StandardCharsets.UTF_8));
    writeChunkedResource(baseName, expected, 8, "1", null, null);

    File extracted = NativeDepsLoader.createFile(TEST_OS, TEST_ARCH, baseName);
    try {
      assertArrayEquals(expected, Files.readAllBytes(extracted.toPath()));
    } finally {
      Files.deleteIfExists(extracted.toPath());
    }
  }

  @Test
  void extractsChunksConcurrently() throws Exception {
    String mappedName = "libconcurrent.so";
    byte[] expected = "0123456789abcdef".getBytes(StandardCharsets.UTF_8);
    int chunkSize = 4;
    Map<String, byte[]> resources = new HashMap<>();
    String resourceRoot = "/native/" + mappedName;
    StringBuilder chunkCrcProperties = new StringBuilder();
    for (int i = 0; i < expected.length / chunkSize; i++) {
      byte[] chunk = new byte[chunkSize];
      System.arraycopy(expected, i * chunkSize, chunk, 0, chunkSize);
      resources.put(String.format(Locale.ROOT, "%s.chunks/%05d", resourceRoot, i), chunk);
      CRC32 chunkCrc = new CRC32();
      chunkCrc.update(chunk);
      chunkCrcProperties.append(String.format(
          Locale.ROOT, "chunk.%05d.crc32=%08x%n", i, chunkCrc.getValue()));
    }
    String manifest = String.format(Locale.ROOT,
        "format.version=1%n"
            + "library.size=%d%n"
            + "chunk.size=%d%n"
            + "chunk.count=%d%n",
        expected.length, chunkSize, expected.length / chunkSize)
        + chunkCrcProperties;
    resources.put(resourceRoot + ".chunks.properties",
        manifest.getBytes(StandardCharsets.ISO_8859_1));

    CountDownLatch concurrentReaders = new CountDownLatch(2);
    AtomicInteger activeReaders = new AtomicInteger();
    AtomicInteger maxReaders = new AtomicInteger();
    URL manifestUrl = new URL(null, "memory:" + resourceRoot + ".chunks.properties",
        new URLStreamHandler() {
          @Override
          protected URLConnection openConnection(URL url) {
            return new URLConnection(url) {
              @Override
              public void connect() {
              }

              @Override
              public InputStream getInputStream() throws IOException {
                byte[] contents = resources.get(url.getPath());
                if (contents == null) {
                  throw new FileNotFoundException(url.toString());
                }
                if (url.getPath().contains(".chunks/")) {
                  return coordinatedStream(
                      contents, concurrentReaders, activeReaders, maxReaders);
                }
                return new ByteArrayInputStream(contents);
              }
            };
          }
        });

    File extracted = File.createTempFile("concurrent-chunks", ".so");
    try {
      NativeDepsLoader.extractChunkedResource(manifestUrl, mappedName, extracted, 2);
      assertArrayEquals(expected, Files.readAllBytes(extracted.toPath()));
      assertEquals(0, concurrentReaders.getCount());
      assertEquals(2, maxReaders.get());
    } finally {
      Files.deleteIfExists(extracted.toPath());
    }
  }

  @Test
  void rejectsUnsupportedManifestVersionAndDeletesPartialFile() throws IOException {
    String baseName = "badversiontest";
    byte[] expected = "versioned contents".getBytes(StandardCharsets.UTF_8);
    writeChunkedResource(baseName, expected, 8, "2", null, null);
    Set<Path> before = temporaryFiles(baseName);

    assertThrows(IOException.class,
        () -> NativeDepsLoader.createFile(TEST_OS, TEST_ARCH, baseName));

    assertEquals(before, temporaryFiles(baseName));
  }

  @Test
  void rejectsInvalidChunkCount() throws IOException {
    String baseName = "badcounttest";
    byte[] expected = "invalid chunk count".getBytes(StandardCharsets.UTF_8);
    writeChunkedResource(baseName, expected, 8, "1", null, 99L);

    assertThrows(IOException.class,
        () -> NativeDepsLoader.createFile(TEST_OS, TEST_ARCH, baseName));
  }

  @Test
  void rejectsMissingChunkCrc() throws IOException {
    String baseName = "missingcrctest";
    byte[] expected = "missing chunk crc".getBytes(StandardCharsets.UTF_8);
    writeChunkedResource(baseName, expected, 8, "1", null, null);
    Path manifestPath = resourceDir.resolve(
        System.mapLibraryName(baseName) + ".chunks.properties");
    Properties manifest = new Properties();
    try (InputStream in = Files.newInputStream(manifestPath)) {
      manifest.load(in);
    }
    manifest.remove("chunk.00000.crc32");
    try (OutputStream out = Files.newOutputStream(manifestPath)) {
      manifest.store(out, null);
    }

    assertThrows(IOException.class,
        () -> NativeDepsLoader.createFile(TEST_OS, TEST_ARCH, baseName));
  }

  @Test
  void rejectsMissingChunkAndDeletesPartialFile() throws IOException {
    String baseName = "missingchunktest";
    byte[] expected = "a library with three chunks".getBytes(StandardCharsets.UTF_8);
    writeChunkedResource(baseName, expected, 10, "1", null, null);
    Files.delete(chunkDirectory(baseName).resolve("00002"));
    Set<Path> before = temporaryFiles(baseName);

    assertThrows(IOException.class,
        () -> NativeDepsLoader.createFile(TEST_OS, TEST_ARCH, baseName));

    assertEquals(before, temporaryFiles(baseName));
  }

  @Test
  void rejectsOversizedChunk() throws IOException {
    String baseName = "longchunktest";
    byte[] expected = "a library with long data".getBytes(StandardCharsets.UTF_8);
    writeChunkedResource(baseName, expected, 8, "1", null, null);
    Files.write(chunkDirectory(baseName).resolve("00000"),
        new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9});

    assertThrows(IOException.class,
        () -> NativeDepsLoader.createFile(TEST_OS, TEST_ARCH, baseName));
  }

  @Test
  void rejectsShortChunk() throws IOException {
    String baseName = "shortchunktest";
    byte[] expected = "a library with short data".getBytes(StandardCharsets.UTF_8);
    writeChunkedResource(baseName, expected, 8, "1", null, null);
    Files.write(chunkDirectory(baseName).resolve("00000"), new byte[]{1, 2, 3});

    assertThrows(IOException.class,
        () -> NativeDepsLoader.createFile(TEST_OS, TEST_ARCH, baseName));
  }

  @Test
  void rejectsCrcMismatch() throws IOException {
    String baseName = "badcrctest";
    byte[] expected = "crc checked contents".getBytes(StandardCharsets.UTF_8);
    writeChunkedResource(baseName, expected, 8, "1", 0L, null);

    assertThrows(IOException.class,
        () -> NativeDepsLoader.createFile(TEST_OS, TEST_ARCH, baseName));
  }

  private static InputStream coordinatedStream(
      byte[] contents, CountDownLatch concurrentReaders,
      AtomicInteger activeReaders, AtomicInteger maxReaders) {
    return new InputStream() {
      private final InputStream delegate = new ByteArrayInputStream(contents);
      private boolean entered;

      private void awaitConcurrentReader() throws IOException {
        if (!entered) {
          entered = true;
          int active = activeReaders.incrementAndGet();
          maxReaders.updateAndGet(previous -> Math.max(previous, active));
          concurrentReaders.countDown();
          try {
            if (!concurrentReaders.await(5, TimeUnit.SECONDS)) {
              throw new IOException("Timed out waiting for concurrent chunk extraction");
            }
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted waiting for concurrent chunk extraction", e);
          }
        }
      }

      @Override
      public int read() throws IOException {
        awaitConcurrentReader();
        return delegate.read();
      }

      @Override
      public int read(byte[] buffer, int offset, int length) throws IOException {
        awaitConcurrentReader();
        return delegate.read(buffer, offset, length);
      }

      @Override
      public void close() throws IOException {
        delegate.close();
        if (entered) {
          entered = false;
          activeReaders.decrementAndGet();
        }
      }
    };
  }

  private Path libraryPath(String baseName) {
    return resourceDir.resolve(System.mapLibraryName(baseName));
  }

  private Path chunkDirectory(String baseName) {
    return resourceDir.resolve(System.mapLibraryName(baseName) + ".chunks");
  }

  private void writeChunkedResource(String baseName, byte[] contents, int chunkSize,
                                    String version, Long firstChunkCrcOverride,
                                    Long countOverride) throws IOException {
    String mappedName = System.mapLibraryName(baseName);
    Path chunks = chunkDirectory(baseName);
    Files.createDirectories(chunks);
    int chunkCount = (contents.length + chunkSize - 1) / chunkSize;
    StringBuilder chunkCrcProperties = new StringBuilder();
    for (int i = 0; i < chunkCount; i++) {
      int offset = i * chunkSize;
      int length = Math.min(chunkSize, contents.length - offset);
      byte[] chunk = new byte[length];
      System.arraycopy(contents, offset, chunk, 0, length);
      Files.write(chunks.resolve(String.format(Locale.ROOT, "%05d", i)), chunk);
      CRC32 chunkCrc = new CRC32();
      chunkCrc.update(chunk);
      long crcValue = i == 0 && firstChunkCrcOverride != null
          ? firstChunkCrcOverride : chunkCrc.getValue();
      chunkCrcProperties.append(String.format(
          Locale.ROOT, "chunk.%05d.crc32=%08x%n", i, crcValue));
    }

    long manifestCount = countOverride == null ? chunkCount : countOverride;
    String manifest = String.format(Locale.ROOT,
        "format.version=%s%n"
            + "library.size=%d%n"
            + "chunk.size=%d%n"
            + "chunk.count=%d%n",
        version, contents.length, chunkSize, manifestCount)
        + chunkCrcProperties;
    Files.write(resourceDir.resolve(mappedName + ".chunks.properties"),
        manifest.getBytes(StandardCharsets.ISO_8859_1));
  }

  private static Set<Path> temporaryFiles(String prefix) throws IOException {
    Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"));
    try (Stream<Path> paths = Files.list(tempDir)) {
      return paths
          .filter(path -> path.getFileName().toString().startsWith(prefix))
          .collect(Collectors.toSet());
    }
  }
}
