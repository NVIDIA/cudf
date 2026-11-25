/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

static int report_fd;
static struct sigaction previous;

static ssize_t write_field(char const* name, uintptr_t value)
{
  static char const digits[] = "0123456789abcdef";
  char text[64];
  unsigned int offset = 0;
  while (*name != '\0') {
    text[offset++] = *name++;
  }
  text[offset++] = '=';
  text[offset++] = '0';
  text[offset++] = 'x';
  for (unsigned int i = 2 * sizeof(value); i > 0; --i) {
    text[offset + i - 1] = digits[value & 15];
    value >>= 4;
  }
  offset += 2 * sizeof(value);
  text[offset++] = '\n';
  return write(report_fd, text, offset);
}

static void report_segv(int sig, siginfo_t* info, void* context)
{
  int saved_errno         = errno;
  ucontext_t const* frame = context;
  struct timespec now;
  if (clock_gettime(CLOCK_REALTIME, &now) == 0) {
    (void)write_field("epoch_seconds", now.tv_sec);
    (void)write_field("nanoseconds", now.tv_nsec);
  }
  (void)write_field("si_code", (uintptr_t)info->si_code);
  (void)write_field("address", info->si_code > 0 ? (uintptr_t)info->si_addr : 0);
#if defined(__aarch64__)
  (void)write_field("pc", frame->uc_mcontext.pc);
  (void)write_field("lr", frame->uc_mcontext.regs[30]);
#elif defined(__x86_64__)
  (void)write_field("pc", frame->uc_mcontext.gregs[REG_RIP]);
#else
#error Unsupported fault context
#endif

  // Retain the load offsets for symbolization without uploading process memory.
  int maps = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
  if (maps >= 0) {
    char buffer[2048];
    ssize_t bytes;
    while ((bytes = read(maps, buffer, sizeof(buffer))) > 0) {
      if (write(report_fd, buffer, bytes) != bytes) { break; }
    }
    (void)close(maps);
  }

  // A synchronous fault repeats on return and reaches Python's handler with
  // the original signal information. An explicitly raised signal needs replay.
  if (sigaction(sig, &previous, 0) != 0) { _exit(128 + sig); }
  if (info->si_code <= 0) { (void)raise(sig); }
  errno = saved_errno;
}

int install_fault_report(int fd)
{
  struct sigaction action = {0};
  action.sa_flags         = SA_SIGINFO | SA_ONSTACK;
  action.sa_sigaction     = report_segv;
  sigemptyset(&action.sa_mask);
  report_fd = fd;
  return sigaction(SIGSEGV, &action, &previous);
}
