/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <ucontext.h>
#include <unistd.h>

static struct sigaction previous_action;

static void write_hex(uintptr_t value)
{
  char text[2 + 2 * sizeof(value) + 1];
  char const digits[] = "0123456789abcdef";
  text[0]             = '0';
  text[1]             = 'x';
  for (unsigned i = 0; i < 2 * sizeof(value); ++i) {
    text[2 + i] = digits[(value >> (4 * (2 * sizeof(value) - i - 1))) & 15];
  }
  text[sizeof(text) - 1] = '\n';
  (void)write(STDERR_FILENO, text, sizeof(text));
}

static void record_crash(int signal_number, siginfo_t* info, void* context)
{
  char const marker[] = "CUDF_DIAGNOSTIC_SIGSEGV fault=";
  (void)write(STDERR_FILENO, marker, sizeof(marker) - 1);
  write_hex((uintptr_t)info->si_addr);
  (void)write(STDERR_FILENO, "instruction=", 12);
#if defined(__x86_64__)
  write_hex((uintptr_t)((ucontext_t*)context)->uc_mcontext.gregs[REG_RIP]);
#elif defined(__aarch64__)
  write_hex((uintptr_t)((ucontext_t*)context)->uc_mcontext.pc);
#else
  write_hex(0);
#endif
  /* Record the instruction and mappings before unwinding: even an unwind
     failure inside the loader must leave enough information to find the DSO. */
  int fd = open("/proc/self/maps", O_RDONLY);
  if (fd >= 0) {
    char buffer[4096];
    ssize_t size;
    while ((size = read(fd, buffer, sizeof(buffer))) > 0) {
      (void)write(STDERR_FILENO, buffer, (size_t)size);
    }
    (void)close(fd);
  }
  void* frames[64];
  int count = backtrace(frames, 64);
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  /* A library may already own SIGSEGV for recoverable faults. Do not turn
     one of those into a new failure introduced by the diagnostic itself. */
  if (previous_action.sa_handler != SIG_DFL && previous_action.sa_handler != SIG_IGN) {
    (void)sigaction(signal_number, &previous_action, 0);
    if (previous_action.sa_flags & SA_SIGINFO) {
      previous_action.sa_sigaction(signal_number, info, context);
    } else {
      previous_action.sa_handler(signal_number);
    }
    return;
  }
  /* Preserve signal termination rather than turning a crash into a test
     assertion; SA_RESETHAND has already restored the default disposition. */
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, signal_number);
  sigprocmask(SIG_UNBLOCK, &signals, 0);
  raise(signal_number);
  _exit(128 + signal_number);
}

__attribute__((constructor)) static void install_crash_recorder(void)
{
  /* Load the unwinder before entering a signal handler, where lazy loading
     and allocation would otherwise make backtrace collection less reliable. */
  void* frames[1];
  (void)backtrace(frames, 1);
  struct sigaction action = {0};
  action.sa_sigaction     = record_crash;
  action.sa_flags         = SA_SIGINFO | SA_RESETHAND;
  sigemptyset(&action.sa_mask);
  (void)sigaction(SIGSEGV, &action, &previous_action);
}
