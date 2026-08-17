/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Guard for {@code -Ppackaged-jar-tests}: fail fast if cuDF classes were loaded
 * from {@code target/classes} (or any other path) instead of the packaged
 * classifier JAR supplied via {@code -Dcudf.jar.path} / the
 * {@code cudf.packaged.jar} system property.
 */
class PackagedJarOriginCheck {
  @Test
  void cudfClassesAreLoadedFromPackagedJar() throws Exception {
    Path expected = Paths.get(System.getProperty("cudf.packaged.jar")).toRealPath();
    Path actual = Paths.get(
        Cuda.class.getProtectionDomain().getCodeSource().getLocation().toURI())
        .toRealPath();
    assertEquals(expected, actual);
  }
}
