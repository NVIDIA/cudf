/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import com.github.luben.zstd.Zstd;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import java.util.zip.CRC32;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Guard for {@code -Ppackaged-jar-tests}: fail fast if cuDF classes were loaded
 * from {@code target/classes} (or any other path) instead of the packaged
 * classifier JAR supplied via {@code -Dcudf.jar.path} / the
 * {@code cudf.packaged.jar} system property.
 */
class PackagedJarOriginCheck {
  @TempDir
  Path tempDir;

  @Test
  void cudfClassesAreLoadedFromPackagedJar() throws Exception {
    Path expected = packagedJar();
    Path actual = Paths.get(
        Cuda.class.getProtectionDomain().getCodeSource().getLocation().toURI())
        .toRealPath();
    assertEquals(expected, actual);
  }

  @Test
  void packagedJarDoesNotEmbedZstdJni() throws Exception {
    try (JarFile jar = new JarFile(packagedJar().toFile())) {
      assertFalse(jar.stream().anyMatch(entry ->
          entry.getName().startsWith("com/github/luben/zstd/")
              || (entry.getName().startsWith("linux/")
              && entry.getName().contains("libzstd-jni"))));
    }
  }

  @Test
  void nativeDepUtilExtractsConventionalWithoutZstdJni() throws Exception {
    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");
    String baseName = "standalonetest";
    byte[] expected = "standalone native library".getBytes(StandardCharsets.UTF_8);
    Path resource = tempDir.resolve(arch).resolve(os).resolve(System.mapLibraryName(baseName));
    Files.createDirectories(resource.getParent());
    Files.write(resource, expected);
    assertPackagedExtract(baseName, expected, false);
  }

  @Test
  void nativeDepUtilExtractsZstdChunksWithRuntimeDependency() throws Exception {
    String baseName = "standalonezstdtest";
    byte[] expected = "standalone compressed native library"
        .getBytes(StandardCharsets.UTF_8);
    Path resource = tempDir.resolve(System.getProperty("os.arch"))
        .resolve(System.getProperty("os.name"))
        .resolve(System.mapLibraryName(baseName));
    Path chunks = resource.resolveSibling(resource.getFileName() + ".chunks");
    Files.createDirectories(chunks);
    Files.write(chunks.resolve("00000"), Zstd.compress(expected, 9));
    CRC32 crc = new CRC32();
    crc.update(expected);
    String manifest = String.format(Locale.ROOT,
        "format.version=1%nlibrary.size=%d%nchunk.size=%d%nchunk.count=1%n"
            + "chunk.compression=zstd%nchunk.00000.crc32=%08x%n",
        expected.length, expected.length, crc.getValue());
    Files.write(resource.resolveSibling(resource.getFileName() + ".chunks.properties"),
        manifest.getBytes(StandardCharsets.ISO_8859_1));
    assertPackagedExtract(baseName, expected, true);
  }

  private void assertPackagedExtract(String baseName, byte[] expected,
      boolean includeZstdJni) throws Exception {
    Path destination = tempDir.resolve("extracted").resolve(System.mapLibraryName(baseName));
    Files.createDirectories(destination.getParent());

    Path java = Paths.get(System.getProperty("java.home"), "bin", "java");
    if (!Files.isExecutable(java)) {
      java = java.resolveSibling("java.exe");
    }
    String classPath = packagedJar() + File.pathSeparator + tempDir;
    if (includeZstdJni) {
      classPath += File.pathSeparator + zstdJniJar();
    }
    Process process = new ProcessBuilder(
        java.toString(),
        "-cp", classPath,
        NativeDepUtil.class.getName(),
        "extract", baseName, destination.toString())
        .redirectErrorStream(true)
        .start();
    if (!process.waitFor(30, TimeUnit.SECONDS)) {
      process.destroyForcibly();
      fail("NativeDepUtil did not exit within 30 seconds");
    }
    String output;
    try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
      output = reader.lines().collect(Collectors.joining(System.lineSeparator()));
    }

    assertEquals(0, process.exitValue(), output);
    assertEquals(destination.toAbsolutePath().toString(), output);
    assertArrayEquals(expected, Files.readAllBytes(destination));
  }

  private static Path packagedJar() throws Exception {
    return Paths.get(System.getProperty("cudf.packaged.jar")).toRealPath();
  }

  private static Path zstdJniJar() throws Exception {
    Path jar = Paths.get(Zstd.class.getProtectionDomain().getCodeSource()
        .getLocation().toURI()).toRealPath();
    if (!Files.isRegularFile(jar)) {
      throw new IllegalStateException("zstd-jni must be loaded from its own JAR: " + jar);
    }
    return jar;
  }
}
