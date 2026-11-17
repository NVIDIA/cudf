#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module for running commands with timeout and capturing stack traces.

This module provides functionality to run commands with a timeout, capture stack traces
of processes that exceed the timeout, and properly terminate process trees.

See Also
--------
subprocess.Popen : For running subprocesses without timeout.
psutil.Process : For process management and information.

Examples
--------
>>> from timeout_with_stack import run_with_timeout
>>> exit_code = run_with_timeout(["sleep", "10"], timeout=5)
>>> print(f"Process exited with code: {exit_code}")
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from enum import IntEnum
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import FrameType


# Time to give a command that has timed out to exit after SIGINT.
DEFAULT_GRACE_PERIOD = 30.0


class StackType(IntEnum):
    """Enum representing the type of stack trace to capture."""

    C = 0
    Python = 1


def get_child_pids(pid: int) -> list[int]:
    """
    Get all child PIDs of a given process.

    This function retrieves all child process IDs (PIDs) of a given process,
    including recursively nested child processes.

    Parameters
    ----------
    pid
        The process ID of the parent process.

    Returns
    -------
    A list of child process IDs. Returns an empty list if the parent process
    does not exist.

    See Also
    --------
    psutil.Process.children : For getting child processes.

    Examples
    --------
    >>> from timeout_with_stack import get_child_pids
    >>> child_pids = get_child_pids(1234)
    >>> print(f"Child PIDs: {child_pids}")
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        return [p.pid for p in children]
    except psutil.NoSuchProcess:
        return []


def capture_stack_trace(pid: int, stack_type=StackType.C) -> None:
    """
    Capture stack trace for a given process.

    This function captures the stack trace of a process using GDB. It prints the
    stack trace to stdout, which can be useful for debugging hanging or long-running
    processes.

    Parameters
    ----------
    pid
        The process ID of the process to capture stack trace for.
    stack_type
        The stack type to extract, either C or Python.

    See Also
    --------
    capture_all_stacks : For capturing stack traces of a process and its children.

    Examples
    --------
    >>> from timeout_with_stack import capture_stack_trace
    >>> capture_stack_trace(1234)
    """
    if stack_type is StackType.C:
        bt_command = "thread apply all bt"
        print(f"\nCapturing C stack trace for process {pid}:")
    else:
        bt_command = "thread apply all py-bt"
        print(f"\nCapturing Python stack trace for process {pid}:")
    gdb = shutil.which("gdb")
    if gdb is None:
        print(f"Skipping stack trace for process {pid}: gdb not found")
        return

    gdb_args = [gdb, "--quiet"]
    try:
        executable = psutil.Process(pid).exe()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    else:
        auto_load_script = f"{executable}-gdb.py"
        if os.path.isfile(auto_load_script):
            # GDB attaches while processing --pid, before regular -ex commands
            # run. Trust only this executable's helper early enough for it to
            # register Python commands such as py-bt during the attach.
            gdb_args.extend(
                [
                    "-iex",
                    f"add-auto-load-safe-path {auto_load_script}",
                ]
            )

    gdb_args.extend(
        [
            "--pid",
            str(pid),
            "-ex",
            "set pagination off",
            "-ex",
            "set confirm off",
            "-ex",
            bt_command,
            "-ex",
            "quit",
        ]
    )

    try:
        proc = subprocess.run(
            gdb_args,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"Timed out capturing stack trace for process {pid}")
        return

    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)


def capture_all_stacks(pid: int, *, enable_python: bool = False) -> None:
    """
    Capture stack traces for parent and all child processes.

    This function captures stack traces for both the parent process and all its
    child processes. It first captures the parent's stack trace, then recursively
    captures stack traces for all child processes.

    Parameters
    ----------
    pid
        The process ID of the parent process.
    enable_python
        Whether to capture Python stack traces.

    See Also
    --------
    capture_stack_trace : For capturing stack trace of a single process.
    get_child_pids : For getting child process IDs.

    Examples
    --------
    >>> from timeout_with_stack import capture_all_stacks
    >>> capture_all_stacks(1234, enable_python=True)
    """
    # Capture parent process stack
    if enable_python:
        capture_stack_trace(pid, stack_type=StackType.Python)
    capture_stack_trace(pid, stack_type=StackType.C)

    # Get and capture all child processes
    child_pids = get_child_pids(pid)
    for child_pid in child_pids:
        if enable_python:
            capture_stack_trace(child_pid, stack_type=StackType.Python)
        capture_stack_trace(child_pid, stack_type=StackType.C)


def terminate_process_tree(pid: int) -> None:
    """
    Terminate a process and all its children.

    This function terminates a process and all its child processes. It first
    attempts to gracefully terminate all processes, then forcefully kills any
    remaining processes after a timeout.

    Parameters
    ----------
    pid
        The process ID of the parent process to terminate.

    See Also
    --------
    psutil.Process.terminate : For gracefully terminating a process.
    psutil.Process.kill : For forcefully killing a process.

    Examples
    --------
    >>> from timeout_with_stack import terminate_process_tree
    >>> terminate_process_tree(1234)
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Terminate children first
        for child in children:
            with suppress(psutil.NoSuchProcess):
                child.terminate()

        # Create a copy of children list
        terminated_children = list(children)

        # Wait for all children to terminate
        for child in terminated_children:
            with suppress(psutil.TimeoutExpired):
                child.wait(timeout=3)

        # Kill any remaining children
        for child in terminated_children:
            with suppress(psutil.NoSuchProcess):
                child.kill()

        # Terminate parent
        parent.terminate()
        try:
            parent.wait(timeout=3)
        except psutil.TimeoutExpired:
            parent.kill()

    except psutil.NoSuchProcess:
        pass


def is_alive(proc: psutil.Process) -> bool:
    """Whether a process is still running (a zombie counts as exited)."""
    try:
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def interrupt_process(process: subprocess.Popen, grace_period: float) -> bool:
    """
    Ask a process to stop by sending it SIGINT, and wait for it to do so.

    SIGINT is what pressing Ctrl-C sends. A Python program such as pytest turns
    it into a ``KeyboardInterrupt``, which lets it report the tests that ran,
    write its reports, and shut down its own children (for example pytest-xdist
    workers) before exiting. Only ``process`` itself is signalled: it is
    responsible for forwarding the request to anything it started.

    This returns once ``process`` and every process it had started when this
    function was called have exited, or after ``grace_period`` seconds,
    whichever is first.

    Parameters
    ----------
    process
        The process to interrupt.
    grace_period
        Maximum time in seconds to wait for the processes to exit.

    Returns
    -------
    True if the process and all of its descendants exited within the grace
    period, False if any of them is still running.

    See Also
    --------
    terminate_process_tree : For forcefully stopping a process tree.
    """
    deadline = time.monotonic() + grace_period
    # Take the list now. Once the parent exits, its children are re-parented
    # and can no longer be found from it.
    descendants = get_child_pids(process.pid)
    with suppress(ProcessLookupError):
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=grace_period)
    except subprocess.TimeoutExpired:
        return False

    remaining = []
    for pid in descendants:
        with suppress(psutil.NoSuchProcess):
            remaining.append(psutil.Process(pid))
    while any(is_alive(proc) for proc in remaining):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def install_signal_handler(pid: int, *signals: signal.Signals) -> None:
    """
    Install signal handler that terminates the given pid on receiving any of signals
    """

    def handler(signum: int, frame: FrameType | None) -> None:
        print(f"Received {signum=}, terminating process tree")
        terminate_process_tree(pid)
        sys.exit(signum)

    for sig in signals:
        signal.signal(sig, handler)


def run_with_timeout(
    cmd: Sequence[str],
    timeout: float,
    *,
    enable_python: bool = False,
    grace_period: float = DEFAULT_GRACE_PERIOD,
) -> int:
    """
    Run a command with a timeout and capture stack traces if it exceeds the timeout.

    This function runs a command with a specified timeout. If the command exceeds
    the timeout, it captures stack traces of the process and its children, then
    stops it in stages:

    1. Send SIGINT to the command, as Ctrl-C would, and give it ``grace_period``
       seconds to exit. A Python program such as pytest handles this by
       reporting the tests that ran and shutting down its own child processes
       (e.g. pytest-xdist workers).
    2. If the command, or anything it started, is still running after that,
       terminate the whole process tree with SIGTERM and, after a short wait,
       SIGKILL.

    It handles keyboard interrupts gracefully.

    Parameters
    ----------
    cmd
        The command and its arguments to run.
    timeout
        Maximum time in seconds to allow the command to run.
    enable_python
        Whether to capture Python stack traces.
    grace_period
        Seconds to wait for the command to exit after SIGINT before
        terminating its process tree.

    Returns
    -------
    Return code of the command, or 124 if timeout occurred, or signal.SIGINT
    if interrupted by keyboard.

    See Also
    --------
    subprocess.Popen : For running subprocesses without timeout.
    capture_all_stacks : For capturing stack traces of processes.

    Examples
    --------
    >>> from timeout_with_stack import run_with_timeout
    >>> exit_code = run_with_timeout(["sleep", "10"], timeout=5, enable_python=True)
    >>> print(f"Process exited with code: {exit_code}")
    """
    # Start the process with a new process group
    # Note: preexec_fn is used here as we need to create a new process group
    # for proper termination of child processes
    process = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
    )
    install_signal_handler(
        process.pid,
        signal.SIGTERM,
        signal.SIGABRT,
        signal.SIGHUP,
        signal.SIGQUIT,
    )
    start_time = time.time()

    try:
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                return process.returncode
            time.sleep(0.1)

        print(f"\nProcess timed out after {timeout} seconds")
        print("Capturing stack traces for all processes...")

        # Capture stacks for parent and all children
        capture_all_stacks(process.pid, enable_python=enable_python)

        # Ask the command to stop first, so that it can clean up after itself.
        print(
            f"\nSending SIGINT to process {process.pid}, "
            f"waiting up to {grace_period} seconds for it to exit..."
        )
        if interrupt_process(process, grace_period):
            print("Process exited after SIGINT")
        else:
            print(
                f"Process tree still running {grace_period} seconds after SIGINT"
            )
            print("\nTerminating process tree...")
            terminate_process_tree(process.pid)
        # Sweep up anything left in the process group created for the command
        # above, whether or not SIGINT was enough. This includes processes that
        # were re-parented away from the tree, or started after the descendant
        # snapshot in interrupt_process was taken.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt")
        terminate_process_tree(process.pid)
        return signal.SIGINT
    else:
        return 124


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a command with timeout and capture stack traces"
    )
    parser.add_argument("timeout", type=float, help="Timeout in seconds")
    parser.add_argument(
        "--enable-python",
        action="store_true",
        help="Enable Python stack trace capture",
    )
    parser.add_argument(
        "--grace-period",
        type=float,
        default=DEFAULT_GRACE_PERIOD,
        help=(
            "Seconds to wait for the command to exit after it is sent SIGINT "
            "on timeout, before its process tree is terminated"
        ),
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to run"
    )

    args = parser.parse_args()

    if not args.command:
        parser.error("No command specified")

    exit_code = run_with_timeout(
        args.command,
        args.timeout,
        enable_python=args.enable_python,
        grace_period=args.grace_period,
    )
    sys.exit(exit_code)
