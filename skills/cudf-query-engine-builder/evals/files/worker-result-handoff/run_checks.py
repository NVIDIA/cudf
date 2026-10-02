#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run CPU ownership checks inside an isolated Linux evaluation runner.

This compiles and executes submitted C++ code. Environment filtering, temporary
storage and limits below reduce exposure; they do not provide a sandbox or prove
CUDA execution. The caller must isolate filesystem, network and credentials.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

OUTPUT_LIMIT = 128 * 1024
FILE_LIMIT = 8 * 1024 * 1024
MEMORY_LIMIT = 2 * 1024 * 1024 * 1024


def failure(stage, message, exit_code=1):
    print(json.dumps({'passed': False, 'stage': stage, 'error': message,
                      'native_execution': False}))
    raise SystemExit(exit_code)


def limits(cpu_seconds):
    import resource
    for kind, wanted in ((resource.RLIMIT_CPU, cpu_seconds),
                         (resource.RLIMIT_FSIZE, FILE_LIMIT),
                         (resource.RLIMIT_AS, MEMORY_LIMIT),
                         (resource.RLIMIT_CORE, 0)):
        _, hard = resource.getrlimit(kind)
        bounded = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
        resource.setrlimit(kind, (bounded, bounded))


def invoke(command, work, seconds):
    # Do not inherit provider tokens, compiler overrides or preload variables.
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(work),
                   'TMPDIR': str(work), 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}
    with tempfile.TemporaryFile(dir=work) as output, tempfile.TemporaryFile(dir=work) as errors:
        process = subprocess.Popen(command, cwd=work, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                                   shell=False, start_new_session=True,
                                   preexec_fn=lambda: limits(seconds))
        timed_out = False
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also stop ordinary descendants that kept the process group alive.
            # This is cleanup, not confinement against hostile native code.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        output.seek(0)
        errors.seek(0)
        stdout = output.read(OUTPUT_LIMIT + 1)
        stderr = errors.read(OUTPUT_LIMIT + 1)
        if timed_out:
            raise TimeoutError(f'{seconds} second timeout')
        if len(stdout) > OUTPUT_LIMIT or len(stderr) > OUTPUT_LIMIT:
            raise RuntimeError('command output exceeded the 128 KiB per-stream limit')
        return process.returncode, stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')


def main():
    if sys.platform != 'linux':
        failure('setup', 'Use the isolated Linux evaluation environment.', 2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation', type=Path,
                        help='Adapter to test with this copy of the fixed checks')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    # Use the system toolchain supplied by the image, never a caller's PATH.
    compiler = next((path for path in ('/usr/bin/g++', '/usr/bin/clang++')
                     if Path(path).is_file() and os.access(path, os.X_OK)), None)
    if compiler is None:
        failure('setup', '/usr/bin/g++ or /usr/bin/clang++ is required', 2)
    stage = 'setup'
    try:
        with tempfile.TemporaryDirectory(prefix='cudf-host-contract-', dir='/tmp') as directory:
            work = Path(directory)
            for name in ('sim_backend.hpp', 'test_adapter.cpp'):
                shutil.copyfile(base / name, work / name)
            shutil.copyfile(args.implementation or base / 'adapter.hpp', work / 'adapter.hpp')
            stage = 'compile'
            command = [compiler, '-std=c++20', '-O0', '-g', '-Wall', '-Wextra',
                       '-Werror', '-pedantic', str(work / 'test_adapter.cpp'),
                       '-o', str(work / 'checks')]
            code, output, errors = invoke(command, work, 60)
            if code:
                failure(stage, errors or output or f'compiler exited {code}')
            stage = 'execute'
            code, output, errors = invoke([str(work / 'checks')], work, 20)
            if code < 0:
                failure(stage, f'test process terminated by signal {-code}')
            sys.stdout.write(output)
            sys.stderr.write(errors)
            return code
    except (OSError, TimeoutError, RuntimeError, subprocess.SubprocessError) as error:
        failure(stage, str(error))


if __name__ == '__main__':
    sys.exit(main())
