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
import java.io.RandomAccessFile;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.zip.CRC32;

/**
 * This class will load the native dependencies.
 */
public class NativeDepsLoader {
  private static final Logger log = LoggerFactory.getLogger(NativeDepsLoader.class);
  private static final int COPY_BUFFER_SIZE = 1024 * 1024;
  // Positional extraction uses one copy buffer per worker.
  private static final int MAX_CONCURRENT_CHUNK_READS =
      Math.max(1, Math.min(12, Runtime.getRuntime().availableProcessors()));
  private static final String CHUNK_MANIFEST_SUFFIX = ".chunks.properties";
  private static final String CHUNK_DIRECTORY_SUFFIX = ".chunks/";
  private static final String CHUNK_FORMAT_VERSION = "1";

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
   * subsequent stages are loaded.
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
        loadNativeDeps(loadOrder, preserveDepsAfterLoad);
        loaded = true;
      } catch (Throwable t) {
        log.error("Could not load cudf jni library...", t);
      }
    }
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
    if (libNativeDir != null) {
      validateLibNativeDir(new String[]{depName});
    }
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

  /** Extract the contents of a library resource into a temporary file. */
  static File createFile(String os, String arch, String baseName) throws IOException {
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
    URL chunkManifestResource = loader.getResource(path + CHUNK_MANIFEST_SUFFIX);
    URL resource = chunkManifestResource == null ? loader.getResource(path) : null;
    if (chunkManifestResource == null && resource == null) {
      throw new FileNotFoundException("Could not locate native dependency " + path);
    }
    long t0 = System.currentTimeMillis();
    File loc = File.createTempFile(baseName, ".so");
    loc.deleteOnExit();
    boolean success = false;
    try {
      if (chunkManifestResource == null) {
        extractConventionalResource(resource, loc);
      } else {
        extractChunkedResource(chunkManifestResource, mappedName, loc);
      }
      success = true;
    } finally {
      if (!success && loc.exists() && !loc.delete()) {
        log.warn("Could not delete partial native dependency {}", loc);
      }
    }
    if (libLogLoadTiming) {
      long elapsed = System.currentTimeMillis() - t0;
      long sizeMB = loc.length() / (1024L * 1024L);
      log.info("Extracted {} in {} ms (size={} MB)", mappedName, elapsed, sizeMB);
    }
    return loc;
  }

  private static void extractConventionalResource(URL resource, File loc) throws IOException {
    try (InputStream in = resource.openStream();
         OutputStream out = new FileOutputStream(loc)) {
      copy(in, out, new byte[COPY_BUFFER_SIZE]);
    }
  }

  private static void extractChunkedResource(URL manifestResource, String mappedName, File loc)
      throws IOException {
    extractChunkedResource(
        manifestResource, mappedName, loc, MAX_CONCURRENT_CHUNK_READS);
  }

  static void extractChunkedResource(URL manifestResource, String mappedName, File loc,
                                     int maxConcurrentReads) throws IOException {
    if (maxConcurrentReads <= 0) {
      throw new IllegalArgumentException("maxConcurrentReads must be positive");
    }
    ChunkManifest manifest = ChunkManifest.load(manifestResource);
    int concurrentReads = Math.min(maxConcurrentReads, manifest.chunkCount);
    ExecutorService executor = Executors.newFixedThreadPool(concurrentReads);
    List<Future<?>> chunkFutures = new ArrayList<>(manifest.chunkCount);
    try (RandomAccessFile out = new RandomAccessFile(loc, "rw")) {
      out.setLength(manifest.librarySize);
      FileChannel outputChannel = out.getChannel();
      try {
        for (int i = 0; i < manifest.chunkCount; i++) {
          chunkFutures.add(submitChunkExtraction(
              executor, manifestResource, mappedName, manifest, outputChannel, i));
        }
        executor.shutdown();
        awaitChunks(chunkFutures, mappedName);
      } finally {
        shutdownAndAwait(executor);
      }
    }
  }

  private static Future<?> submitChunkExtraction(
      ExecutorService executor, URL manifestResource, String mappedName,
      ChunkManifest manifest, FileChannel outputChannel, int chunkIndex) throws IOException {
    String chunkName = String.format(Locale.ROOT, "%05d", chunkIndex);
    URL chunkResource = new URL(
        manifestResource, mappedName + CHUNK_DIRECTORY_SUFFIX + chunkName);
    long outputOffset = manifest.chunkSize * chunkIndex;
    long expectedSize = manifest.expectedChunkSize(chunkIndex);
    long expectedCrc32 = manifest.expectedChunkCrc32(chunkIndex);
    return executor.submit(() -> {
      extractChunk(chunkResource, outputChannel, outputOffset, expectedSize, expectedCrc32);
      return null;
    });
  }

  private static void awaitChunks(List<Future<?>> futures, String mappedName)
      throws IOException {
    IOException failure = null;
    for (int i = 0; i < futures.size(); i++) {
      try {
        futures.get(i).get();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException(String.format(Locale.ROOT,
            "Interrupted while extracting native dependency chunk %s/%05d",
            mappedName, i), e);
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        IOException chunkFailure = cause instanceof IOException
            ? (IOException) cause
            : new IOException(String.format(Locale.ROOT,
                "Could not extract native dependency chunk %s/%05d", mappedName, i), cause);
        if (failure == null) {
          failure = chunkFailure;
        } else {
          failure.addSuppressed(chunkFailure);
        }
      }
    }
    if (failure != null) {
      throw failure;
    }
  }

  private static void shutdownAndAwait(ExecutorService executor) {
    executor.shutdownNow();
    boolean interrupted = Thread.interrupted();
    while (!executor.isTerminated()) {
      try {
        executor.awaitTermination(1, TimeUnit.SECONDS);
      } catch (InterruptedException e) {
        interrupted = true;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private static void extractChunk(
      URL resource, FileChannel outputChannel, long outputOffset,
      long expectedSize, long expectedCrc32) throws IOException {
    byte[] buffer = new byte[COPY_BUFFER_SIZE];
    ByteBuffer bytes = ByteBuffer.wrap(buffer);
    CRC32 crc = new CRC32();
    long totalBytes = 0;
    try (InputStream in = resource.openStream()) {
      int read;
      while ((read = in.read(buffer)) != -1) {
        if (read == 0) {
          continue;
        }
        if (read > expectedSize - totalBytes) {
          throw new IOException(String.format(Locale.ROOT,
              "Native dependency chunk %s exceeds expected size of %d bytes",
              resource, expectedSize));
        }

        crc.update(buffer, 0, read);
        bytes.clear();
        bytes.limit(read);
        long writeOffset = outputOffset + totalBytes;
        while (bytes.hasRemaining()) {
          int written = outputChannel.write(bytes, writeOffset);
          if (written <= 0) {
            throw new IOException("Could not make progress writing native dependency " + resource);
          }
          writeOffset += written;
        }
        totalBytes += read;
      }
    } catch (FileNotFoundException e) {
      throw new IOException("Could not locate native dependency chunk " + resource, e);
    }

    if (totalBytes != expectedSize) {
      throw new IOException(String.format(Locale.ROOT,
          "Native dependency chunk %s has size %d bytes, expected %d",
          resource, totalBytes, expectedSize));
    }
    if (crc.getValue() != expectedCrc32) {
      throw new IOException(String.format(Locale.ROOT,
          "Native dependency chunk CRC32 mismatch for %s: expected %08x but extracted %08x",
          resource, expectedCrc32, crc.getValue()));
    }
  }

  private static void copy(InputStream in, OutputStream out, byte[] buffer)
      throws IOException {
    int read;
    while ((read = in.read(buffer)) != -1) {
      if (read > 0) {
        out.write(buffer, 0, read);
      }
    }
  }

  private static final class ChunkManifest {
    private static final String FORMAT_VERSION_KEY = "format.version";
    private static final String LIBRARY_SIZE_KEY = "library.size";
    private static final String CHUNK_SIZE_KEY = "chunk.size";
    private static final String CHUNK_COUNT_KEY = "chunk.count";

    private final long librarySize;
    private final long chunkSize;
    private final int chunkCount;
    private final long[] chunkCrc32;

    private ChunkManifest(long librarySize, long chunkSize, int chunkCount, long[] chunkCrc32) {
      this.librarySize = librarySize;
      this.chunkSize = chunkSize;
      this.chunkCount = chunkCount;
      this.chunkCrc32 = chunkCrc32;
    }

    private static ChunkManifest load(URL resource) throws IOException {
      Properties properties = new Properties();
      try (InputStream in = resource.openStream()) {
        properties.load(in);
      } catch (IllegalArgumentException e) {
        throw new IOException("Malformed native dependency chunk manifest " + resource, e);
      }
      String version = require(properties, FORMAT_VERSION_KEY, resource);
      if (!CHUNK_FORMAT_VERSION.equals(version)) {
        throw new IOException("Unsupported native dependency chunk manifest version " + version
            + " in " + resource);
      }
      long librarySize = parsePositiveLong(properties, LIBRARY_SIZE_KEY, resource);
      long chunkSize = parsePositiveLong(properties, CHUNK_SIZE_KEY, resource);
      long chunkCountLong = parsePositiveLong(properties, CHUNK_COUNT_KEY, resource);
      if (chunkCountLong > Integer.MAX_VALUE) {
        throw new IOException("Native dependency chunk count is too large in " + resource);
      }
      long expectedChunkCount = 1 + ((librarySize - 1) / chunkSize);
      if (chunkCountLong != expectedChunkCount) {
        throw new IOException(String.format(Locale.ROOT,
            "Invalid native dependency chunk count in %s: expected %d but found %d",
            resource, expectedChunkCount, chunkCountLong));
      }
      int chunkCount = (int) chunkCountLong;
      long[] chunkCrc32 = new long[chunkCount];
      for (int i = 0; i < chunkCount; i++) {
        String key = String.format(Locale.ROOT, "chunk.%05d.crc32", i);
        chunkCrc32[i] = parseCrc32(properties, key, resource);
      }
      return new ChunkManifest(librarySize, chunkSize, chunkCount, chunkCrc32);
    }

    private static long parseCrc32(Properties properties, String key, URL resource)
        throws IOException {
      String value = require(properties, key, resource);
      if (!value.matches("[0-9a-fA-F]{8}")) {
        throw new IOException("Invalid " + key + " in " + resource + ": " + value);
      }
      try {
        return Long.parseLong(value, 16);
      } catch (NumberFormatException e) {
        throw new IOException("Invalid " + key + " in " + resource + ": " + value, e);
      }
    }

    private static long parsePositiveLong(Properties properties, String key, URL resource)
        throws IOException {
      String value = require(properties, key, resource);
      try {
        long parsed = Long.parseLong(value);
        if (parsed <= 0) {
          throw new NumberFormatException("value must be positive");
        }
        return parsed;
      } catch (NumberFormatException e) {
        throw new IOException("Invalid " + key + " in " + resource + ": " + value, e);
      }
    }

    private static String require(Properties properties, String key, URL resource)
        throws IOException {
      String value = properties.getProperty(key);
      if (value == null || value.trim().isEmpty()) {
        throw new IOException("Missing " + key + " in native dependency chunk manifest "
            + resource);
      }
      return value.trim();
    }

    private long expectedChunkSize(int chunkIndex) {
      if (chunkIndex == chunkCount - 1) {
        return librarySize - (chunkSize * chunkIndex);
      }
      return chunkSize;
    }

    private long expectedChunkCrc32(int chunkIndex) {
      return chunkCrc32[chunkIndex];
    }
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
