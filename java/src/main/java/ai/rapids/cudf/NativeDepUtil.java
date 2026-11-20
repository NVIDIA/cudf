/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import java.io.File;
import java.io.IOException;

/** Command-line utilities for inspecting and extracting packaged native dependencies. */
public final class NativeDepUtil {
  private NativeDepUtil() {
  }

  /**
   * Extract a packaged native dependency without loading it.
   *
   * @param args {@code extract <library-base-name> <destination>}
   * @throws IOException if the native dependency cannot be extracted
   */
  public static void main(String[] args) throws IOException {
    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");
    File destination = execute(args, os, arch);
    System.out.println(destination);
  }

  static File execute(String[] args, String os, String arch) throws IOException {
    if (args.length != 3 || !"extract".equals(args[0])) {
      throw new IllegalArgumentException(
          "Usage: NativeDepUtil extract <library-base-name> <destination>");
    }
    return NativeDepsLoader.extractNativeDep(os, arch, args[1], new File(args[2]));
  }
}
