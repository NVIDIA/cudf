#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add independently rerun CPU-fixture results to standard SkillEvaluator grading.

The verifier snapshots the submitted adapter, rejects changed protected fixtures,
then reruns the fixed checker. It does not trust the agent's claimed result.
This checks generated repairs; it is not a security sandbox or a proof of CUDA
execution. The evaluation operator must provide an isolated Linux environment.
Standard SkillEvaluator scores and verdicts remain owned by the default grader.
"""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile

METRIC = "compiled_fixture_correctness"
MAX_FILE_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 128 * 1024
FIXTURE_HASHES = {
    "async-copy-cleanup": {
        "TASK.md": "43f3709cbe7e3d64cd3556e8464914d388efebd59f9817b0c311b0881d113a19",
        "sim_backend.hpp": "c0a4dffb6f5b240fc9b339c6c6e88f5b56f35ca048a08f63ddc2b9032a5830d4",
        "test_adapter.cpp": "b7bcdd02e5563a19e3d96484606927e4c8b2107fb114cea300a9e8c5ad2146e9",
        "run_checks.py": "0e198858a4fa02c995bd2ea92841e924cdc928ae19445f76f2e8826df01979f3"
    },
    "worker-result-handoff": {
        "TASK.md": "0f88d729f9acfea3e55a7e702b826bdc940059b05adcaa366490aa9c9a87ddbe",
        "sim_backend.hpp": "c0a4dffb6f5b240fc9b339c6c6e88f5b56f35ca048a08f63ddc2b9032a5830d4",
        "test_adapter.cpp": "a36b780cc47e110d48f1b03497331b49fd49b3922d3a6b9413bdd6a35d86073d",
        "run_checks.py": "0e198858a4fa02c995bd2ea92841e924cdc928ae19445f76f2e8826df01979f3"
    }
}
EXPECTED_CHECKS = {
    "async-copy-cleanup": (
        "nullable_normal_result", "empty_result", "all_null_result",
        "allocation_before_submission_and_recovery",
        "output_allocation_failure_and_recovery",
        "result_wrapper_failure_and_recovery",
    ),
    "worker-result-handoff": (
        "worker_handoff_after_caller_owners_expire", "empty_worker_result",
        "all_null_worker_result", "discard_before_get",
        "two_outstanding_results", "explicit_cpu_fallback_is_labeled",
    ),
}


def result(score, status, **details):
    metrics = {} if score is None else {METRIC: float(score)}
    return {"custom_metrics": metrics, "details": {
        METRIC: {"status": status, "native_execution": False, **details}
    }}


def read_input_file(input_root, case_id, name):
    """Read one bounded regular file without following directory or file links."""
    with ExitStack() as stack:
        directory = os.open(input_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, directory)
        case = os.open(case_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                       dir_fd=directory)
        stack.callback(os.close, case)
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=case)
        stack.callback(os.close, descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError("Input must be a bounded regular file")
        content = bytearray()
        while len(content) <= MAX_FILE_BYTES:
            block = os.read(descriptor, min(65536, MAX_FILE_BYTES + 1 - len(content)))
            if not block:
                break
            content.extend(block)
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("Input file exceeds the size limit")
        return bytes(content)


def fixture_snapshot(input_root, case_id):
    snapshot = {}
    for name, digest in FIXTURE_HASHES[case_id].items():
        content = read_input_file(input_root, case_id, name)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Protected fixture changed")
        snapshot[name] = content
    snapshot["adapter.hpp"] = read_input_file(input_root, case_id, "adapter.hpp")
    return snapshot


def process_limits():
    import resource
    for kind, requested in (
        (resource.RLIMIT_CPU, 90), (resource.RLIMIT_FSIZE, 8 * 1024 * 1024),
        (resource.RLIMIT_AS, 2 * 1024 * 1024 * 1024), (resource.RLIMIT_CORE, 0),
    ):
        _, hard = resource.getrlimit(kind)
        limit = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        resource.setrlimit(kind, (limit, limit))


def invoke_checker(directory):
    """Run the trusted checker with bounded output and no inherited environment."""
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(directory),
                   "TMPDIR": str(directory), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    with tempfile.TemporaryFile(dir=directory) as output, tempfile.TemporaryFile(dir=directory) as errors:
        process = subprocess.Popen(
            [sys.executable, "-I", str(directory / "run_checks.py")],
            cwd=directory, env=environment, stdin=subprocess.DEVNULL,
            stdout=output, stderr=errors, shell=False, start_new_session=True,
            preexec_fn=process_limits,
        )
        try:
            process.wait(timeout=100)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        output.seek(0)
        errors.seek(0)
        stdout = output.read(MAX_OUTPUT_BYTES + 1)
        stderr = errors.read(MAX_OUTPUT_BYTES + 1)
        if len(stdout) > MAX_OUTPUT_BYTES or len(stderr) > MAX_OUTPUT_BYTES:
            raise ValueError("Checker output exceeds the size limit")
        return process.returncode, stdout


def parse_checker_result(case_id, returncode, stdout):
    try:
        payload = json.loads(stdout)
    except (ValueError, UnicodeError):
        return result(0, "invalid_checker_output", returncode=returncode)
    if not isinstance(payload, dict):
        return result(0, "invalid_checker_output", returncode=returncode)
    checks = payload.get("checks")
    if not isinstance(checks, list) or len(checks) != len(EXPECTED_CHECKS[case_id]):
        return result(0, "incomplete_checks", returncode=returncode)
    observed = {}
    for check in checks:
        if (not isinstance(check, dict)
                or check.get("name") not in EXPECTED_CHECKS[case_id]
                or check["name"] in observed
                or not isinstance(check.get("passed"), bool)):
            return result(0, "invalid_checker_output", returncode=returncode)
        observed[check["name"]] = check["passed"]
    valid = (payload.get("backend") == "deterministic-cpu-simulator"
             and payload.get("native_execution") is False
             and isinstance(payload.get("passed"), bool))
    passed = valid and returncode == 0 and payload["passed"] and all(observed.values())
    return result(int(passed), "passed" if passed else "failed",
                  returncode=returncode, checks=observed)


def grade(entry, input_root=Path("/workspace/input")):
    case_id = entry.get("id")
    if case_id not in EXPECTED_CHECKS:
        return result(None, "not_applicable",
                      reason="This task is assessed by the standard graders.")
    if sys.platform != "linux":
        return result(0, "unsupported_environment")
    try:
        snapshot = fixture_snapshot(input_root, case_id)
    except (OSError, ValueError):
        return result(0, "invalid_or_changed_input")
    try:
        with tempfile.TemporaryDirectory(prefix="cudf-fixture-grade-", dir="/tmp") as temp:
            directory = Path(temp)
            for name, content in snapshot.items():
                (directory / name).write_bytes(content)
            code, stdout = invoke_checker(directory)
            reward = parse_checker_result(case_id, code, stdout)
        # The submitted implementation must not alter the public protected files.
        for name, digest in FIXTURE_HASHES[case_id].items():
            if hashlib.sha256(read_input_file(input_root, case_id, name)).hexdigest() != digest:
                return result(0, "protected_input_changed_during_check")
        reward["details"][METRIC]["adapter_sha256"] = hashlib.sha256(snapshot["adapter.hpp"]).hexdigest()
        return reward
    except subprocess.TimeoutExpired:
        return result(0, "checker_timeout")
    except (OSError, ValueError, subprocess.SubprocessError):
        return result(0, "checker_error")


def reward_filename(raw_path):
    """Accept only a direct JSON output inside the container verifier directory."""
    path = Path(raw_path)
    if (not path.is_absolute() or ".." in path.parts
            or path.parent != Path("/logs/verifier")
            or path.suffix != ".json" or len(path.name) > 128):
        raise ValueError("Reward output must be a direct JSON file under /logs/verifier")
    return path.name


def write_reward(reward, raw_path="/logs/verifier/reward.json"):
    """Write atomically through trusted directory descriptors without following links.

    This grader targets Harbor's Linux containers. Arbitrary local-mode log
    directories are unsupported; an override cannot select another output root.
    The standard verifier runs first and must already have created /logs/verifier.
    """
    name = reward_filename(raw_path)
    body = (json.dumps(reward, indent=2) + "\n").encode("utf-8")
    if len(body) > 64 * 1024:
        raise ValueError("Custom reward exceeds the size limit")
    with ExitStack() as stack:
        logs = os.open("/logs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, logs)
        directory = os.open("verifier", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=logs)
        stack.callback(os.close, directory)

        def check_destination():
            try:
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                return
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Existing reward output must be a regular file")

        check_destination()
        import uuid
        temporary = ".cudf-reward-" + uuid.uuid4().hex + ".tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            check_destination()
            # Replacing the directory entry never writes through a final symlink
            # or an existing hard link, even if it changed after the last check.
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass


def main():
    entry_path = Path(os.environ.get("HARBOR_ENTRY_JSON", "/tests/entry.json"))
    try:
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        if not isinstance(entry, dict):
            raise ValueError("Eval entry must be an object")
        reward = grade(entry)
    except (OSError, ValueError):
        reward = result(0, "invalid_eval_entry")
    write_reward(reward, os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json"))
    # default_plus_custom restores the standard reward and appends custom_metrics.
    # Do not write a replacement overall score or reward.txt.


if __name__ == "__main__":
    main()
