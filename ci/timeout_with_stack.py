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

    try:
        proc = subprocess.run(
            [
                gdb,
                "--quiet",
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
            ],
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
    cmd: Sequence[str], timeout: float, *, enable_python: bool = False
) -> int:
    """
    Run a command with a timeout and capture stack traces if it exceeds the timeout.

    This function runs a command with a specified timeout. If the command exceeds
    the timeout, it captures stack traces of the process and its children before
    terminating them. It handles keyboard interrupts gracefully.

    Parameters
    ----------
    cmd
        The command and its arguments to run.
    timeout
        Maximum time in seconds to allow the command to run.
    enable_python
        Whether to capture Python stack traces.

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

        # Terminate the entire process tree
        print("\nTerminating process tree...")
        terminate_process_tree(process.pid)
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
        "command", nargs=argparse.REMAINDER, help="Command to run"
    )

    args = parser.parse_args()

    if not args.command:
        parser.error("No command specified")

    exit_code = run_with_timeout(
        args.command, args.timeout, enable_python=args.enable_python
    )
    sys.exit(exit_code)
