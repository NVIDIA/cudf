/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.ast.CompiledExpression;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * ColumnVectors may store data off heap, and because of complicated processing the life time of
 * an individual vector can vary a lot.  Typically a java finalizer could be used for this but
 * they can cause a number of performance issues related to gc, and in some cases may effectively
 * leak resources if the heap is large and GC's end up being delayed.
 * <p>
 * To address these issues the primary way to releasing the resources of a ColumnVector that is
 * stored off of the java heap should be through reference counting. Because memory leaks are
 * really bad for long lived daemons this is intended to be a backup.
 * <p>
 * When a ColumnVector first allocates off heap resources it should register itself with this
 * along with a Cleaner instance. The Cleaner instance should have no direct links to the
 * ColumnVector that would prevent the ColumnVector from being garbage collected. This will
 * use WeakReferences internally to know when the resources have been leaked.
 * A ColumnVector may keep a reference to the Cleaner instance and either update it as new
 * resources are allocated or use it to release the resources it is holding.  Once the
 * ColumnVector's reference count reaches 0 and the resources are released. At some point
 * later the Cleaner itself will be released.
 */
public final class MemoryCleaner {
  private static final boolean REF_COUNT_DEBUG = Boolean.getBoolean("ai.rapids.refcount.debug");
  private static final Logger log = LoggerFactory.getLogger(MemoryCleaner.class);
  private static final AtomicLong idGen = new AtomicLong(0);

  /**
   * Check if configured the shutdown hook which checks leaks at shutdown time.
   *
   * @return true if configured, false otherwise.
   */
  public static boolean configuredDefaultShutdownHook() {
    return REF_COUNT_DEBUG;
  }

  /**
   * API that can be used to clean up the resources for a vector, even if there was a leak
   */
  public static abstract class Cleaner {
    private final List<RefCountDebugItem> refCountDebug;
    public final long id = idGen.incrementAndGet();
    private boolean leakExpected = false;

    public Cleaner() {
      if (REF_COUNT_DEBUG) {
        refCountDebug = new LinkedList<>();
      } else {
        refCountDebug = null;
      }
    }

    public final void addRef() {
      if (REF_COUNT_DEBUG && refCountDebug != null) {
        synchronized(this)  {
          refCountDebug.add(new MemoryCleaner.RefCountDebugItem("INC"));
        }
      }
    }

    public final void delRef() {
      if (REF_COUNT_DEBUG && refCountDebug != null) {
        synchronized(this) {
          refCountDebug.add(new MemoryCleaner.RefCountDebugItem("DEC"));
        }
      }
    }

    public final void logRefCountDebug(String message) {
      if (REF_COUNT_DEBUG && refCountDebug != null) {
        synchronized(this) {
          log.error("{} (ID: {}): {}", message, id, MemoryCleaner.stringJoin("\n", refCountDebug));
        }
      }
    }

    /**
     * Clean up any resources not previously released.
     * @param logErrorIfNotClean if true we should log a leak unless it is expected.
     * @return true if resources were cleaned up else false.
     */
    public final boolean clean(boolean logErrorIfNotClean) {
      boolean cleaned = cleanImpl(logErrorIfNotClean && !leakExpected);
      if (cleaned) {
        all.remove(id);
      }
      return cleaned;
    }

    /**
     * Return true if a leak is expected for this object else false.
     */
    public final boolean isLeakExpected() {
      return leakExpected;
    }

    /**
     * Clean up any resources not previously released.
     * @param logErrorIfNotClean if true and there are resources to clean up a leak has happened
     *                           so log it.
     * @return true if resources were cleaned up else false.
     */
    protected abstract boolean cleanImpl(boolean logErrorIfNotClean);

    public void noWarnLeakExpected() {
      leakExpected = true;
    }

    /**
     * Check if the underlying memory has been cleaned up or not.
     * @return true this is clean else false.
     */
    public abstract boolean isClean();
  }

  static final AtomicLong leakCount = new AtomicLong();
  private static final Map<Long, CleanerWeakReference> all =
      new ConcurrentHashMap(); // We want to be thread safe
  private static final ReferenceQueue<?> collected = new ReferenceQueue<>();

  private static class CleanerWeakReference<T> extends WeakReference<T> {

    // For avoiding double closes
    private volatile boolean isCleaned = false;

    private final Cleaner cleaner;
    final boolean isRmmBlocker;

    public CleanerWeakReference(T orig, Cleaner cleaner, ReferenceQueue collected, boolean isRmmBlocker) {
      super(orig, collected);
      this.cleaner = cleaner;
      this.isRmmBlocker = isRmmBlocker;
    }

    public synchronized void clean() {
      if (isCleaned) {
        return;
      }
      isCleaned = true;

      if (cleaner.clean(true)) {
        leakCount.incrementAndGet();
      }
    }
  }

  /**
   * The default GPU as set by user threads.
   */
  private static volatile int defaultGpu = -1;

  /**
   * This should be called from RMM when it is initialized.
   */
  static void setDefaultGpu(int defaultGpuId) {
    defaultGpu = defaultGpuId;
  }

  private static class CleanerThread extends Thread {
    private volatile boolean stopFlag = false;
    private volatile boolean stopped = false;

    CleanerThread() {
      super("Cleaner Thread");
    }

    void cleanCollected() {
      int currentGpuId = -1;
      while (true) {
        CleanerWeakReference next = null;
        try {
          next = (CleanerWeakReference)collected.remove(100);
        } catch (InterruptedException e)  {
          // ignore
        }

        if (next == null) {
          // queue is empty, stop this loop
          return;
        }

        try {
          if (currentGpuId != defaultGpu) {
            Cuda.setDevice(defaultGpu);
            currentGpuId = defaultGpu;
          }
        } catch (Throwable t) {
          log.error("ERROR TRYING TO SET GPU ID TO " + defaultGpu, t);
        }
        try {
          next.clean();
        } catch (Throwable t) {
          log.error("CAUGHT EXCEPTION WHILE TRYING TO CLEAN " + next, t);
        }
        all.remove(next.cleaner.id);
      }
    }

    @Override
    public void run() {
      try {
        while (!stopFlag) {
          cleanCollected();
        }
      } finally {
        stopped = true;
      }
    }

    void stopGracefully() {
      stopFlag = true;
      while (!stopped) {
        try {
          Thread.sleep(10);
        } catch (InterruptedException e) {
          // Ignored
        }
      }
    }
  }

  /**
   * This is used to close all Rmm resources to check leaks.
   * Should invoke this before closing Rmm.
   */
  public static void cleanAllRmmBlockers() {
    // set the device for the current thread
    if (defaultGpu >= 0) {
      Cuda.setDevice(defaultGpu);
    }

    for (CleanerWeakReference cwr : MemoryCleaner.all.values()) {
      if (cwr.isRmmBlocker) {
        cwr.clean();
      }
    }
  }

  private static final CleanerThread t = new CleanerThread();

  /**
   * Default shutdown runnable used to be added to Spark shutdown hook.
   * Spark-Rapids get this hook and register it into Spark shutdown hooks.
   * Note: `cleanAllRmmBlocker` should be called to clean up RMM resources.
   * It checks the leaks at shutdown time for non-RMM resources.
   */
  static class ShutdownHookThread implements Runnable {

    CleanerThread ct;

    ShutdownHookThread() {
      this.ct = t;
    }

    public void cleanAtShutdown() {
      // stop cleaner thread first
      ct.stopGracefully();

      // set the device for the current thread
      if (defaultGpu >= 0) {
        Cuda.setDevice(defaultGpu);
      }

      // force gc
      System.gc();

      // clean up anything that got collected
      ct.cleanCollected();

      // Check the remaining references. Here only closes non-RMM resources
      // MUST call `cleanAllRmmBlockers` first to close RMM resources, or
      // core dump occurs.
      for (CleanerWeakReference cwr : MemoryCleaner.all.values()) {
        cwr.clean();
      }
    }

    @Override
    public void run() {
      cleanAtShutdown();
    }
  }

  static {
    t.setDaemon(true);
    t.start();
  }

  /**
   * De-register the default shutdown hook from Java default Runtime, then return the corresponding
   * shutdown runnable.
   * If you want to register the default shutdown runnable in a custom shutdown hook manager
   * instead of Java default Runtime, should first remove it using this method and then add it
   *
   * @return the default shutdown runnable
   */
  public static Runnable removeDefaultShutdownHook() {
    return new ShutdownHookThread();
  }

  static void register(ColumnVector vec, Cleaner cleaner) {
    // It is now registered...
    all.put(cleaner.id, new CleanerWeakReference(vec, cleaner, collected, true));
  }

  static void register(Scalar s, Cleaner cleaner) {
    // It is now registered...
    all.put(cleaner.id, new CleanerWeakReference(s, cleaner, collected, true));
  }

  static void register(HostColumnVectorCore vec, Cleaner cleaner) {
    // It is now registered...
    all.put(cleaner.id, new CleanerWeakReference(vec, cleaner, collected, false));
  }

  static void register(MemoryBuffer buf, Cleaner cleaner) {
    // It is now registered...
    all.put(cleaner.id, new CleanerWeakReference(buf, cleaner, collected, buf instanceof BaseDeviceMemoryBuffer));
  }

  static void register(Cuda.Stream stream, Cleaner cleaner) {
    // It is now registered...
    all.put(cleaner.id, new CleanerWeakReference(stream, cleaner, collected, false));
  }

  static void register(Cuda.Event event, Cleaner cleaner) {
    // It is now registered...
    all.put(cleaner.id, new CleanerWeakReference(event, cleaner, collected, false));
  }

  static void register(CuFileDriver driver, Cleaner cleaner) {
    // It is now registered...
    all.put(cleaner.id, new CleanerWeakReference(driver, cleaner, collected, false));
  }

  static void register(CuFileBuffer buffer, Cleaner cleaner) {
    // It is now registered...
    all.put(cleaner.id, new CleanerWeakReference(buffer, cleaner, collected, false));
  }

  static void register(CuFileHandle handle, Cleaner cleaner) {
    // It is now registered...
    all.put(cleaner.id, new CleanerWeakReference(handle, cleaner, collected, false));
  }

  public static void register(CompiledExpression expr, Cleaner cleaner) {
    all.put(cleaner.id, new CleanerWeakReference(expr, cleaner, collected, false));
  }

  static void register(HybridScanReader reader, Cleaner cleaner) {
    // RMM blocker: wrapper can hold device memory (chunked_filter_row_mask).
    all.put(cleaner.id, new CleanerWeakReference(reader, cleaner, collected, true));
  }

  static void register(HashJoin hashJoin, Cleaner cleaner) {
    all.put(cleaner.id, new CleanerWeakReference(hashJoin, cleaner, collected, true));
  }

  static void register(KeyRemapping keyRemapping, Cleaner cleaner) {
    all.put(cleaner.id, new CleanerWeakReference(keyRemapping, cleaner, collected, true));
  }

  /**
   * This is not 100% perfect and we can still run into situations where RMM buffers were not
   * collected and this returns false because of thread race conditions. This is just a best effort.
   * @return true if there are rmm blockers else false.
   */
  static boolean bestEffortHasRmmBlockers() {
    return all.values().stream().anyMatch(cwr -> cwr.isRmmBlocker && !cwr.cleaner.isClean());
  }

  /**
   * Convert elements in it to a String and join them together. Only use for debug messages
   * where the code execution itself can be disabled as this is not fast.
   */
  private static <T> String stringJoin(String delim, Iterable<T> it) {
    return String.join(delim,
        StreamSupport.stream(it.spliterator(), false)
            .map((i) -> i.toString())
            .collect(Collectors.toList()));
  }

  /**
   * When debug is enabled holds information about inc and dec of ref count.
   */
  private static final class RefCountDebugItem {
    final StackTraceElement[] stackTrace;
    final long timeMs;
    final String op;

    public RefCountDebugItem(String op) {
      this.stackTrace = Thread.currentThread().getStackTrace();
      this.timeMs = System.currentTimeMillis();
      this.op = op;
    }

    public String toString() {
      Date date = new Date(timeMs);
      // Simple Date Format is horribly expensive only do this when debug is turned on!
      SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSSS z");
      return dateFormat.format(date) + ": " + op + "\n"
          + stringJoin("\n", Arrays.asList(stackTrace))
          + "\n";
    }
  }
}
