---
name: mylws-test
description: >-
  Verify mylws learning examples and code changes with lightweight local checks or
  remote Gemini CPU, GPU, and multi-node tests. Use when asked to run tests or
  pytest, or to verify a mylws example. Maintain reproducible commands and results.
---

# mylws-test

Use [mylws-develop](../mylws-develop/SKILL.md) to identify the example entry point, dependencies, and test objectives. Run existing tests for routine regression checks. Verify real communication and GPU behavior in the corresponding environment.

Keep test code simple. Call the example directly and use the necessary `assert` statements to check results and reject invalid cases. Avoid production-grade exception hierarchies, compatibility branches, and generic test wrappers. Assert behavior in the existing example when possible so the test does not duplicate the implementation.

Keep inputs, calls, expected results, and assertions easy to identify. Use clear names and intermediate variables where helpful. Give each check room to express its intent clearly.

Assume GPU examples run in environments with GPUs. Test the CUDA path directly and keep the necessary GPU count and device assertions. Omit CPU fallback, automatic device selection, and failure-skipping branches for local environments without GPUs. Report a blocker if the environment falls short of the requirements.

## Checks before remote testing

1. Use `wx-remote-gemini` to connect to the task specified by the user. Use current session evidence to confirm the container and avoid reusing historical tokens or relying on old session IDs.
2. Use `wx-detect-my-dfs` to confirm `MY_DFS` in the current session. Use `wx-sync-to-gpu-servers` with the project's existing sync configuration, confirm the code is present, and verify checksums for key files.
3. Check Python, dependency versions, and the required APIs. Prefer existing versions that meet the requirements and install missing dependencies within the task's authorization. Real communication requires evidence from a real run.
4. Use `wx-mpirun` to confirm the node count from `/etc/mpi/hostfile` and run once per node. Verify which nodes participated using their hostnames, including nodes beyond the launcher.
5. Check GPU occupancy with `nvidia-smi` or `dguard status`. Follow `wx-remote-gemini` to pause a running dguard on the launcher, typically for 10 minutes for a short test. Leave other workloads alone.

Before testing models or data, check local paths, versions, and the loading call chain to avoid implicit network access. Use `wx-download` for model and dataset downloads. Install Python packages from a verified package source.

## Environment and bash

Use variables verified in the current session for long-lived runners, with simple existence checks and no complex path discovery. Reactivate the environment in each independent shell so it does not depend on the image's PATH or a previous command's `source`.

```bash
set -Eeuo pipefail
MYLWS_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/mylws"
source "${MYLWS_VENV:?set the verified venv path}/bin/activate"
cd "$MYLWS_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
test "$(command -v python)" = "$MYLWS_VENV/bin/python"
python -c 'import sys; print(sys.executable, sys.prefix)'
```

The Ray/NIXL example was verified in `/opt/venvs/ray-nixl-test`, which used a `.pth` file to reuse Ray and PyTorch from `/opt/venvs/carrot`.
On a different machine, verify the environment and its base paths again. If the expected interpreter is unavailable, stop and diagnose the environment.

Keep one-off bash commands for the user short and easy to copy. They may include remote absolute paths verified for the current task. Use clear placeholders for unknown paths. Keep DFS discovery out of these commands and keep the commands out of long-lived repository runners and public documentation.

## Test files and records

- Use `tests/<topic>/` for standalone tests and their required resources. Running an existing example under `examples/` requires no new test code, directory, or Markdown record.
- Record an example's verification commands, results, and relevant limitations directly in `examples/<topic>/README.md`. For standalone tests that need documentation, prefer an existing relevant README.
- Keep only the latest date and status, reproduction commands, key results, relevant limitations, and one log location. Reuse commands already in the README and record environment details that affect reproduction in one line.
- Update the existing record after a rerun. Keep a sentence explaining any unresolved failure or blocker.
- Keep raw samples, diagnostics, hashes, PIDs, and full environment details in logs. Omit empty fields such as "not recorded."
- For documentation-only changes, check the documentation locally.

## Ray and multi-node testing

- Confirm ownership and occupancy of the target cluster first. Before starting a test cluster, follow `wx-remote-gemini` to stop Ray and then start it. Leave clusters with unrelated running tasks alone.
- Temporary single-node examples may start a local instance via `ray.init()`. For multi-node tests, start the cluster through `wx-mpirun` and explicitly pass `--address` or `RAY_ADDRESS` so all nodes join the same cluster.
- Set the source `PYTHONPATH` before starting Ray. Pass `PATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`, and any other required variables through mpirun. Use the same verified venv for all workers.
- Start and stop Ray with `python -m ray.scripts.scripts`. When reusing an environment, the `ray` script on PATH can have a shebang pointing to the base venv, leaving newly installed NIXL unavailable to workers.
- Normalize every hostfile `slots` value to 1 and use the `bond1` and `routed direct` options required by `wx-mpirun`. Check for slot values other than `slots=8`.
- Check `ray.nodes()` and available resources to confirm registration after MPI succeeds. For cross-node cases, verify that the two endpoints have distinct node IDs.

## Ray/NIXL example verification scope

After the environment and resource checks above, run the existing regression entry point.

```bash
python examples/ray_nixl/main.py --size-mib 64
```

- The case needs 2 CPUs and 1 GPU on the same node. The two actors share the GPU. Check that both endpoints have the same node ID and Ray GPU ID, since `cuda:0` is process-local. The example currently covers same-node transfers only.
- Check the exit code, shape, dtype, device, tensor contents, and the actual NIXL backend in the logs. The fixed output field `transport` is insufficient evidence of the underlying communication path.
- Confirm RDMA with transport logs and corresponding device evidence. Successful NIXL/UCX initialization or cross-node execution alone is insufficient. If evidence is missing, report RDMA as unverified. Identify TCP, shared memory, and CUDA IPC results by the actual transport.
- Report correctness and performance separately. Exclude initial connection, memory registration, and cold start when measuring steady-state bandwidth. Treat timings that include those costs as startup-inclusive measurements. Add warmup, repeated samples, and statistics only for performance tasks.

## Execution, artifacts, and cleanup

- Run necessary local checks first and then remote cases relevant to the change. Verify that the tests and required plugins exist before running pytest. Add `--timeout` only when applicable.
- For long tests, use an MCP background task and check output at each stage. Report new progress or blockers every minute while the test runs. On failure, diagnose the cause, fix the local source, and sync it again. Keep source edits off the receiver.
- Save results and logs that need retention under `$MY_DFS/work/mylws/outputs/<topic>/<run-id>/`. Keep tokens and keys out of artifacts and keep artifacts out of source sync and commits.
- Use MCP background tasks without nohup or shell redirection. After completion, save needed output with `remote_task_dump` and release the task. Purge tasks whose output need not be retained after collecting it.
- Clean up only the actors and processes created for this task. Prefer `python -m ray.scripts.scripts stop --force` on a dedicated test cluster. Avoid `pkill -9 python` and automatic deletion of files on the sync receiver based on run duration.
- If this task paused dguard, resume it or verify that automatic resumption is active. Report actual passes, coverage gaps, environment changes, and the location of evidence or archived results.
