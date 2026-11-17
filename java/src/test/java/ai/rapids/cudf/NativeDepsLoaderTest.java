/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Validates the lib-native-dir validation paths in {@link NativeDepsLoader}.
 *
 * <p>This test class must run in its own forked JVM (configured via the
 * {@code native-deps-loader-test} surefire execution in {@code java/pom.xml})
 * because it sets the {@code ai.rapids.cudf.lib-native-dir} system property
 * before any cudf class is touched, and {@link NativeDepsLoader} captures
 * that value into a {@code static final} field on first load.
 */
class NativeDepsLoaderTest {
  private static final String PROP = "ai.rapids.cudf.lib-native-dir";

  @TempDir
  static Path libDir;

  @BeforeAll
  static void setup() {
    String expected = libDir.toString();
    System.setProperty(PROP, expected);

    // Sanity #1: setProperty actually took effect.
    assertEquals(expected, System.getProperty(PROP),
        "lib-native-dir system property was not set as expected");

    // Sanity #2: nothing has triggered a successful load yet. If this fails,
    // some prior code in this JVM already touched a cudf class that calls
    // loadNativeDeps() in its static initializer, which means
    // NativeDepsLoader's static-final libNativeDir field captured null
    // before our setProperty above. The forked-JVM surefire execution must
    // have been broken (e.g. reuseForks=true added). Abort with a clear
    // message rather than running tests that would silently misbehave.
    assertFalse(NativeDepsLoader.getLoaded(),
        "NativeDepsLoader.loaded == true at @BeforeAll time. " +
        "This test class must run in a fresh forked JVM where no other " +
        "cudf class has been touched. Check the surefire execution.");
  }

  @AfterAll
  static void teardown() {
    System.clearProperty(PROP);
  }

  @AfterEach
  void cleanContents() throws IOException {
    try {
      try (Stream<Path> s = Files.list(libDir)) {
        s.forEach(p -> {
          try (Stream<Path> walk = Files.walk(p)) {
            walk.sorted(Comparator.reverseOrder()).forEach(q -> {
              try {
                Files.delete(q);
              } catch (IOException e) {
                throw new UncheckedIOException(e);
              }
            });
          } catch (IOException e) {
            throw new UncheckedIOException(e);
          }
        });
      }
    } finally {
      NativeDepsLoader.resetLoaded();
    }
  }

  @Test
  void flatLoad_throws_whenDirIsEmpty() {
    IOException ex = assertThrows(IOException.class,
        () -> NativeDepsLoader.loadNativeDeps(new String[]{"cudf"}, false));
    assertTrue(ex.getMessage().contains("libcudf.so"),
        "expected message to mention libcudf.so, got: " + ex.getMessage());
  }

  @Test
  void singleLoad_throws_whenDirIsEmpty() {
    IOException ex = assertThrows(IOException.class,
        () -> NativeDepsLoader.loadNativeDep("cudf", false));
    assertTrue(ex.getMessage().contains("libcudf.so"),
        "expected message to mention libcudf.so, got: " + ex.getMessage());
  }

  @Test
  void flatLoad_throws_whenLibIsMissing() throws IOException {
    Files.createFile(libDir.resolve("libnvcomp.so"));
    IOException ex = assertThrows(IOException.class,
        () -> NativeDepsLoader.loadNativeDeps(new String[]{"nvcomp", "cudf"}, false));
    assertTrue(ex.getMessage().contains("libcudf.so"),
        "expected message to mention libcudf.so, got: " + ex.getMessage());
  }

  @Test
  void flatLoad_throws_whenLibIsADirectory() throws IOException {
    Files.createDirectory(libDir.resolve("libcudf.so"));
    IOException ex = assertThrows(IOException.class,
        () -> NativeDepsLoader.loadNativeDeps(new String[]{"cudf"}, false));
    assertTrue(ex.getMessage().contains("libcudf.so"),
        "expected message to mention libcudf.so, got: " + ex.getMessage());
  }

  @Test
  void noArgLoad_failsSilently_andLeavesLibraryNotLoaded() {
    // The no-arg loadNativeDeps swallows Throwable and only sets loaded=true
    // on success. With an empty libDir validation will fail; we expect no
    // exception to escape and the loaded flag to stay false.
    NativeDepsLoader.loadNativeDeps();
    assertFalse(NativeDepsLoader.getLoaded(),
        "loaded flag should remain false after a failed load");
  }
}
