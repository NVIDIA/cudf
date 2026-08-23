/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

/**
 * This class will load the native dependencies.
 */
public class NativeDepsLoader {
  private static final Logger log = LoggerFactory.getLogger(NativeDepsLoader.class);

  /**
   * Set this system property to true to prevent unpacked dependency files from
   * being deleted immediately after they are loaded. The files will still be
   * scheduled for deletion upon exit.
   */
  private static final Boolean preserveDepsAfterLoad = Boolean.getBoolean(
      "ai.rapids.cudf.preserve-dependencies");

  /**
   * Optional path to a directory of pre-extracted native libraries. When set,
   * those files are loaded directly and the JAR-extraction step is skipped.
   * The directory must contain every requested library (validated up front);
   * pre-unpacked files are never deleted by the application.
   * Override with {@code -Dai.rapids.cudf.lib-native-dir=<path>}.
   */
  private static final String libNativeDir = System.getProperty(
      "ai.rapids.cudf.lib-native-dir");

  /**
   * When true, log per-library extraction/load timings and an aggregate
   * summary at INFO level. Disabled by default to keep startup quiet.
   * Override with {@code -Dai.rapids.cudf.lib-log-load-timing=true}.
   */
  private static final boolean libLogLoadTiming = Boolean.getBoolean(
      "ai.rapids.cudf.lib-log-load-timing");

  // Indices into the long[2] timing slots stored in the per-load timings map.
  private static final int EXTRACT_MS_IDX = 0;
  private static final int LOAD_MS_IDX = 1;

  /**
   * Defines the loading order for the dependencies. Dependencies are loaded in
   * stages where all the dependencies in a stage are not interdependent and
   * therefore can be loaded in parallel. All dependencies within an earlier
   * stage are guaranteed to have finished loading before any dependencies in
   * subsequent stages are loaded. nvcomp is skipped if its resource is absent
   * because static libcudf builds include it.
   */
  private static final String[][] loadOrder = new String[][]{
      new String[]{
          "nvcomp"
      },
      new String[]{
          "cudf"
      },
      new String[]{
          "cudfjni"
      }
  };
  private static final ClassLoader loader = NativeDepsLoader.class.getClassLoader();

  private static boolean loaded = false;

  /**
   * Load the native libraries needed for libcudf, if not loaded already.
   */
  public static synchronized void loadNativeDeps() {
    if (!loaded) {
      try {
        String[][] deps = loadOrder;
        if (!hasNativeResource("nvcomp")) {
          log.info("Skipping optional native dependency {}", System.mapLibraryName("nvcomp"));
          deps = Arrays.stream(loadOrder)
              .filter(stage -> Arrays.stream(stage).noneMatch("nvcomp"::equals))
              .toArray(String[][]::new);
        }
        loadNativeDeps(deps, preserveDepsAfterLoad);
        loaded = true;
      } catch (Throwable t) {
        log.error("Could not load cudf jni library...", t);
      }
    }
  }

  private static boolean hasNativeResource(String baseName) {
    String mapped = System.mapLibraryName(baseName);
    if (libNativeDir != null) {
      return new File(libNativeDir, mapped).isFile();
    }
    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");
    String path = arch + "/" + os + "/" + mapped;
    return loader.getResource(path) != null;
  }

  /**
   * Allows other libraries to reuse the same native deps loading logic. Libraries will be searched
   * for under ${os.arch}/${os.name}/ in the class path using the class loader for this class.
   * <br/>
   * Because this just loads the libraries and loading the libraries themselves needs to be a
   * singleton operation it is recommended that any library using this provide their own wrapper
   * function similar to
   * <pre>
   *     private static boolean loaded = false;
   *     static synchronized void loadNativeDeps() {
   *         if (!loaded) {
   *             try {
   *                 // If you also depend on the cudf liobrary being loaded, be sure it is loaded
   *                 // first
   *                 ai.rapids.cudf.NativeDepsLoader.loadNativeDeps();
   *                 ai.rapids.cudf.NativeDepsLoader.loadNativeDeps(new String[]{...});
   *                 loaded = true;
   *             } catch (Throwable t) {
   *                 log.error("Could not load ...", t);
   *             }
   *         }
   *     }
   * </pre>
   * This function should be called from the static initialization block of any class that uses
   * JNI. For example
   * <pre>
   *     public class UsesJNI {
   *         static {
   *             MyNativeDepsLoader.loadNativeDeps();
   *         }
   *     }
   * </pre>
   * @param loadOrder the base name of the libraries. For example libfoo.so would be passed in as
   *                  "foo".  The libraries are loaded in the order provided.
   * @throws IOException on any error trying to load the libraries.
   */
  public static void loadNativeDeps(String[] loadOrder) throws IOException {
    loadNativeDeps(loadOrder, preserveDepsAfterLoad);
  }

  /**
   * Allows other libraries to reuse the same native deps loading logic. Libraries will be searched
   * for under ${os.arch}/${os.name}/ in the class path using the class loader for this class.
   * <br/>
   * Because this just loads the libraries and loading the libraries themselves needs to be a
   * singleton operation it is recommended that any library using this provide their own wrapper
   * function similar to
   * <pre>
   *     private static boolean loaded = false;
   *     static synchronized void loadNativeDeps() {
   *         if (!loaded) {
   *             try {
   *                 // If you also depend on the cudf liobrary being loaded, be sure it is loaded
   *                 // first
   *                 ai.rapids.cudf.NativeDepsLoader.loadNativeDeps();
   *                 ai.rapids.cudf.NativeDepsLoader.loadNativeDeps(new String[]{...});
   *                 loaded = true;
   *             } catch (Throwable t) {
   *                 log.error("Could not load ...", t);
   *             }
   *         }
   *     }
   * </pre>
   * This function should be called from the static initialization block of any class that uses
   * JNI. For example
   * <pre>
   *     public class UsesJNI {
   *         static {
   *             MyNativeDepsLoader.loadNativeDeps();
   *         }
   *     }
   * </pre>
   * @param loadOrder the base name of the libraries. For example libfoo.so would be passed in as
   *                  "foo".  The libraries are loaded in the order provided.
   * @param preserveDeps if false the dependencies will be deleted immediately after loading
   *                     rather than on exit.
   * @throws IOException on any error trying to load the libraries.
   */
  public static void loadNativeDeps(String[] loadOrder, boolean preserveDeps) throws IOException {
    if (libNativeDir != null) {
      validateLibNativeDir(loadOrder);
    }

    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");

    for (String toLoad : loadOrder) {
      loadDep(os, arch, toLoad, preserveDeps);
    }
  }

  /**
   * Optionally load native dependencies. This method attempts to load the specified libraries
   * but does not throw exceptions on failure. Instead, it returns true if all libraries were
   * loaded successfully, false otherwise.
   * @param loadOrder the base name of the libraries. For example libfoo.so would be passed in as
   *                  "foo". The libraries are loaded in the order provided.
   * @return true if all libraries were loaded successfully, false otherwise
   */
  public static boolean loadOptionalNativeDeps(String[] loadOrder) {
    try {
      loadNativeDeps(loadOrder, preserveDepsAfterLoad);
      return true;
    } catch (Throwable t) {
      log.warn("Could not load optional native dependencies: " + t.getMessage());
      return false;
    }
  }

  /**
   * Load native dependencies in stages, where the dependency libraries in each stage
   * are loaded only after all libraries in earlier stages have completed loading.
   * @param loadOrder array of stages with an array of dependency library names in each stage
   * @param preserveDeps if false the dependencies will be deleted immediately after loading
   *                     rather than on exit.
   * @throws IOException on any error trying to load the libraries
   */
  private static void loadNativeDeps(String[][] loadOrder, boolean preserveDeps) throws IOException {
    if (libNativeDir != null) {
      validateLibNativeDir(loadOrder);
    }

    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");

    long t0 = System.currentTimeMillis();
    // When timing is enabled, collect per-library extract/load durations into
    // a shared map so the summary line can include a breakdown. The map is
    // keyed by the platform-specific library name (e.g. "libcudf.so").
    Map<String, long[]> timings = libLogLoadTiming ? new ConcurrentHashMap<>() : null;

    ExecutorService executor = Executors.newCachedThreadPool();
    List<List<Future<File>>> allFileFutures = new ArrayList<>();

    // Start unpacking and creating the temporary files for each dependency.
    // Unpacking a dependency does not depend on stage order.
    for (String[] stageDependencies : loadOrder) {
      List<Future<File>> stageFileFutures = new ArrayList<>();
      allFileFutures.add(stageFileFutures);
      for (String name : stageDependencies) {
        stageFileFutures.add(executor.submit(() -> createFileTimed(os, arch, name, timings)));
      }
    }

    List<Future<?>> loadCompletionFutures = new ArrayList<>();

    // Proceed stage-by-stage waiting for the dependency file to have been
    // produced then submit them to the thread pool to be loaded.
    for (int i = 0; i < allFileFutures.size(); i++) {
      List<Future<File>> stageFileFutures = allFileFutures.get(i);
      String[] stageNames = loadOrder[i];
      // Submit all dependencies in the stage to be loaded in parallel
      loadCompletionFutures.clear();
      for (int j = 0; j < stageFileFutures.size(); j++) {
        Future<File> fileFuture = stageFileFutures.get(j);
        String name = stageNames[j];
        loadCompletionFutures.add(
            executor.submit(() -> loadDepTimed(fileFuture, preserveDeps, name, timings)));
      }

      // Wait for all dependencies in this stage to have been loaded
      for (Future<?> loadCompletionFuture : loadCompletionFutures) {
        try {
          loadCompletionFuture.get();
        } catch (ExecutionException | InterruptedException e) {
          throw new IOException("Error loading dependencies", e);
        }
      }
    }

    executor.shutdownNow();

    if (libLogLoadTiming) {
      logLoadSummary(loadOrder, timings, System.currentTimeMillis() - t0);
    }
  }

  /**
   * Allows other libraries to reuse the same native deps loading logic. Library will be searched
   * for under ${os.arch}/${os.name}/ in the class path using the class loader for this class.
   * @param depName the base name of the library. For example libfoo.so would be passed in as
   *                "foo".  The libraries are loaded in the order provided.
   * @param preserveDep if false the dependencies will be deleted immediately after loading
   *                    rather than on exit.
   * @return path where the dependency was loaded
   * @throws IOException on any error trying to load the libraries.
   */
  public static File loadNativeDep(String depName, boolean preserveDep) throws IOException {
    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");
    return loadDep(os, arch, depName, preserveDep);
  }

  private static File loadDep(String os, String arch, String baseName, boolean preserveDep)
      throws IOException {
    File path = createFile(os, arch, baseName);
    loadDep(path, preserveDep);
    return path;
  }

  /** Load a library at the specified path */
  private static void loadDep(File path, boolean preserveDep) {
    System.load(path.getAbsolutePath());
    // Pre-unpacked libraries live in a user-supplied directory and must never
    // be deleted by the application, regardless of the preserveDep flag.
    if (!preserveDep && libNativeDir == null) {
      path.delete();
    }
  }

  /** Records elapsed time for {@code baseName} at the given index. No-op when timings is null. */
  private static void recordTiming(Map<String, long[]> timings, String baseName,
                                   int idx, long elapsed) {
    if (timings != null) {
      timings.computeIfAbsent(System.mapLibraryName(baseName), k -> new long[2])[idx] = elapsed;
    }
  }

  /** Awaits the file future then loads it, recording the load wall time into {@code timings}. */
  private static void loadDepTimed(Future<File> fileFuture, boolean preserveDep,
                                   String baseName, Map<String, long[]> timings) {
    File path;
    try {
      path = fileFuture.get();
    } catch (ExecutionException | InterruptedException e) {
      throw new RuntimeException("Error loading dependencies", e);
    }
    long t0 = System.currentTimeMillis();
    loadDep(path, preserveDep);
    recordTiming(timings, baseName, LOAD_MS_IDX, System.currentTimeMillis() - t0);
  }

  /** Calls {@link #createFile} and records the extraction wall time into {@code timings}. */
  private static File createFileTimed(String os, String arch, String baseName,
                                      Map<String, long[]> timings) throws IOException {
    long t0 = System.currentTimeMillis();
    File loc = createFile(os, arch, baseName);
    recordTiming(timings, baseName, EXTRACT_MS_IDX, System.currentTimeMillis() - t0);
    return loc;
  }

  /** Extract the contents of a library resource into a temporary file */
  private static File createFile(String os, String arch, String baseName) throws IOException {
    String mappedName = System.mapLibraryName(baseName);
    // Fast path: when ai.rapids.cudf.lib-native-dir is set, the loader skips
    // JAR extraction entirely and uses the pre-unpacked file from the
    // user-supplied directory. Existence is already validated up front by
    // validateLibNativeDir(); we just hand back the path here.
    if (libNativeDir != null) {
      File loc = new File(libNativeDir, mappedName);
      if (libLogLoadTiming) {
        log.info("Skipped JAR extraction for {} (using lib-native-dir={})",
            mappedName, libNativeDir);
      }
      return loc;
    }
    String path = arch + "/" + os + "/" + mappedName;
    File loc;
    URL resource = loader.getResource(path);
    if (resource == null) {
      throw new FileNotFoundException("Could not locate native dependency " + path);
    }
    long t0 = System.currentTimeMillis();
    try (InputStream in = resource.openStream()) {
      loc = File.createTempFile(baseName, ".so");
      loc.deleteOnExit();
      try (OutputStream out = new FileOutputStream(loc)) {
        byte[] buffer = new byte[1024 * 16];
        int read = 0;
        while ((read = in.read(buffer)) >= 0) {
          out.write(buffer, 0, read);
        }
      }
    }
    if (libLogLoadTiming) {
      long elapsed = System.currentTimeMillis() - t0;
      long sizeMB = loc.length() / (1024L * 1024L);
      log.info("Extracted {} in {} ms (size={} MB)", mappedName, elapsed, sizeMB);
    }
    return loc;
  }

  /**
   * Verify that every library named in {@code order} exists as a regular file
   * inside {@link #libNativeDir}. Throws {@link IOException} listing the first
   * missing library if validation fails.
   */
  private static void validateLibNativeDir(String[][] order) throws IOException {
    for (String[] stage : order) {
      validateLibNativeDir(stage);
    }
  }

  /**
   * Flat-array variant of {@link #validateLibNativeDir(String[][])}.
   */
  private static void validateLibNativeDir(String[] names) throws IOException {
    File dir = new File(libNativeDir);
    if (!dir.isDirectory()) {
      throw new IOException(
          "ai.rapids.cudf.lib-native-dir validation failed: not a directory: "
              + dir.getAbsolutePath());
    }
    for (String name : names) {
      File f = new File(dir, System.mapLibraryName(name));
      if (!f.isFile()) {
        throw new IOException(
            "ai.rapids.cudf.lib-native-dir validation failed: expected library not found: "
                + f.getAbsolutePath());
      }
    }
  }

  /**
   * Emit a multi-line summary describing total load time plus per-library
   * extract and load durations. Library rows are right-padded so the
   * {@code extract=}/{@code load=} columns line up in the output.
   */
  private static void logLoadSummary(String[][] loadOrder, Map<String, long[]> timings,
                                     long totalMs) {
    List<String> names = Arrays.stream(loadOrder)
        .flatMap(Arrays::stream)
        .map(System::mapLibraryName)
        .collect(Collectors.toList());
    int width = names.stream().mapToInt(String::length).max().orElse(0);
    String body = names.stream()
        .map(n -> {
          long[] t = timings.getOrDefault(n, new long[2]);
          return String.format("  %-" + width + "s  extract=%d ms  load=%d ms",
              n, t[EXTRACT_MS_IDX], t[LOAD_MS_IDX]);
        })
        .collect(Collectors.joining("\n"));
    log.info("Native dependency load complete  total={} ms\n{}", totalMs, body);
  }

  public static boolean libraryLoaded() {
    if (!loaded) {
      loadNativeDeps();
    }
    return loaded;
  }

  /** Test hook: read the loaded flag without triggering a load attempt. */
  static boolean getLoaded() {
    return loaded;
  }

  /** Test hook: force the next no-arg loadNativeDeps() call to re-run. */
  static void resetLoaded() {
    loaded = false;
  }
}
