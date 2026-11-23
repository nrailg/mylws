---
name: mylws-develop
description: >-
  Plan, implement, and test changes to lightweight implementations, learning examples,
  and documentation in mylws.
---

# mylws-develop

Follow Plan → Code → Test.

## Plan

- Identify the goal, scope, relevant README files and dependencies, existing workspace changes, and acceptance checks. Keep the plan brief and match its detail to the task. Use `mymemo` when a plan needs to be saved.
- Proceed directly for routine work. For exploratory work such as new algorithms, optimizations, architectures, CUDA kernels, or Torch operators, search the web and GitHub for prior work, including issues and PRs. Present the findings and wait for the user to confirm the direction before planning or coding. Reuse confirmation already given for the same direction.
- For exploratory work, identify the reference or baseline and correctness criteria. Include metrics and measurement conditions for optimizations. If the evidence calls for a different algorithm or architecture, research it and confirm the new direction before proceeding.

## Code

- Apply the [Primary principle, Style, and Coding (Design) rules](../mylws-code-review/SKILL.md#primary-principle) during implementation. Keep reusable implementations under `src/` and learning examples under `examples/`.
- Preserve unrelated workspace changes. Update relevant documentation and add or adjust tests where they protect behavior. For exploratory work, verify the smallest critical hypothesis first.

## Test

- Use [mylws-test](../mylws-test/SKILL.md) to run the smallest checks that cover the change. Use applicable checks for documentation and low-impact changes.
- Fix failures caused by the change and rerun affected checks. Return to Plan if a fundamental assumption, requirement, or interface goal changes.
- Report the result, the checks that ran, and anything the available environment could not verify. Commit and push as authorized by the user.

## Dependencies and development environment

- Use the local Mac for development and available lightweight checks. Verify Linux, CUDA, and NIXL behavior in the relevant remote environment.
- Check existing versions and required APIs before deciding whether to install or upgrade. Keep installed versions that meet the requirements.
- Install, upgrade, and download within the authorization for the current task. If additional authorization is needed, identify the package, version, and target environment. Reuse authorization already granted.
- Prefer an isolated venv for incompatible dependencies. Keep trial installations out of shared training environments. Install and run with the same interpreter, for example using `python -m pip`.
- Run from source. Use editable installs, `uv sync`, or an implicitly installing `uv run` only when packaging is requested.
- Record verified versions and environments in the relevant implementation or example README. A temporary venv in one container applies to that environment.

Use `wx-remote-gemini` to connect, `wx-detect-my-dfs` to confirm `MY_DFS` for the current session, and `wx-sync-to-gpu-servers` to sync.
Edit source in the Mac workspace and treat the sync receiver as a copy. Explicitly activate the verified environment in each independent remote shell.

```bash
MYLWS_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/mylws"
source "${MYLWS_VENV:?set the verified venv path}/bin/activate"
cd "$MYLWS_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
command -v python
python -c 'import sys; print(sys.executable, sys.prefix)'
```

Extend `PYTHONPATH` according to the actual import layout. Check whether `src/` or `tests/` exists before relying on either path. Ray workers must inherit the same interpreter, source paths, and required environment variables.
