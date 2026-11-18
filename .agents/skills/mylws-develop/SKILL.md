---
name: mylws-develop
description: >-
  Develop lightweight implementations, learning examples, and documentation in mylws.
  Use for code or documentation additions, edits, and refactors. Choose the appropriate
  planning, implementation, testing, and independent review steps. Follow the shared rules
  in mylws-code-review and use mylws-test to run verification.
---

# mylws-develop

## Scope and rules

This workflow applies only to `mylws`. Read the root README, existing README files for the topic, dependency declarations, and relevant past decisions. Confirm the requirements and identify existing workspace changes. Develop reusable implementations such as `src/myfa` and learning examples under `examples/` according to their respective goals.

Before coding, read the primary principle, Style, and Coding (Design) in [mylws-code-review](../mylws-code-review/SKILL.md). Use these shared rules for development and review. Use [mylws-test](../mylws-test/SKILL.md) to run verification. For review-only requests or requests to run existing tests, go directly to the corresponding skill.

## Parameters and comments in learning examples

- Use distinct values when illustrating different dimensions or concepts, such as a tile grid of `(2, 3)`, so equal values cannot obscure the distinction. Fixed loop bounds are fine when intuitive and derived in a nearby comment.
- Use the Markdown markers for the relevant language in VS Code In Comment Preview for multiline derivations or structured explanations. The body can contain headings, lists, and `$...$` formulas. In C++, start with `/* md`, prefix each body line with `*`, and end with `*/`. In Python, start with `''' md` and end with `'''`. Omit extra Markdown fences in the source.
- Keep LaTeX readable in the source. Put the `$$`, `\begin{...}`, and `\end{...}` delimiters on separate lines and indent the contents of environments. Write each matrix row on its own line and align the `&` column separators with spaces. Break long formulas at logical points. Distinguish source line breaks from LaTeX `\\` and preserve the formula's meaning and rendered layout when adjusting the formatting.

```cpp
/* md
 * # Tile counts
 * Rows are $4 \times 2 = 8$ and columns are $y = 3$.
 */
```

```python
''' md
# Tile counts
Rows are $4 \times 2 = 8$ and columns are $y = 3$.
'''
```

## Choosing a workflow

Classify the task as exploratory or routine. New algorithms, optimizations, architectures, CUDA kernels, and Torch operators are exploratory and require the research and confirmation process below. Proceed directly with routine development.

| Task | Workflow |
| --- | --- |
| Exploratory development | Web and GitHub research → user confirmation → short Plan (Plan Review as needed) → Code + Test → independent Code Review → fixes and verification |
| Documentation, formatting, or small mechanical changes that preserve behavior | Edit → applicable checks → primary agent self-review |
| Routine features, fixes, refactors, or coordinated changes across files | Short Plan → Code + Test → independent Code Review → fixes and verification |
| An undecided design or higher-risk changes to interfaces, concurrency, or resource lifetimes | Add one independent Plan Review as needed before implementation, then proceed with Code + Test and Code Review |

Use behavior impact and uncertainty to choose the workflow. File count and diff size alone are insufficient. The user's explicit review scope takes precedence.

## Exploratory development

1. **Find prior work.** Search the web and GitHub for the problem, algorithm, or operator, including relevant implementations, issues, and PRs. Check official documentation, papers, and source code for conditions of applicability. Reuse existing research after checking that its versions and findings still fit the current requirements.
2. **Present the research findings.** Link to sources. Explain what existing approaches solve, how they differ from the current requirements, and why direct reuse, adaptation, or a new implementation makes sense. Identify important unknowns. If the search yielded no implementation, limit the claim to what the search found.
3. **Wait for the user to confirm the direction.** The project's Research Before Development rule requires confirmation of the research findings before implementation planning or coding. Continue read-only research and evidence checks if useful while waiting. Prototypes, experimental code, and implementation plans also require confirmation. Reuse explicit confirmation already given for the same direction.
4. **Plan and verify after confirmation.** Identify the hypotheses to verify, the reference implementation or baseline, and the correctness criteria. For optimization work, include performance metrics and measurement conditions. Add a Plan Review for higher-risk designs, then follow the shared Code + Test, Code Review, and fix workflow.
5. **Let evidence guide the next step.** Verify the smallest critical hypothesis first, then expand the implementation. Record disproven hypotheses. Make local adjustments within the confirmed direction. If the work calls for a different algorithm or architecture or expanded scope, do the additional research and confirm the direction again. Limit trial and error and report failed verification as such.

## Plan

- Record the goal, scope, key assumptions, and acceptance checks. Reuse existing conclusions and keep the plan short. If the plan needs to be saved, use `mymemo` and keep temporary plans out of the source tree.
- Proceed directly with authorized routine work when the scope is clear. Ask only for missing information that affects the design, significant scope changes, or tradeoffs that require the user's decision. Follow the confirmation process above for exploratory work.
- When a Plan Review is needed, the reviewer checks requirements coverage, interface and resource constraints, feasibility, and verification, and reports concrete problems with evidence. The primary agent verifies the findings and adjusts the plan. Default to one review and proceed without requiring repeated APPROVED verdicts. Investigate unresolved fundamental issues or bring them to the user for a decision.

## Code + Test

- Apply the shared rules to the implementation, tests, and necessary documentation. Preserve unrelated user changes. Tests should protect actual behavior. Documentation or low-impact changes need only applicable checks.
- Run the smallest effective verification for the change and fix known failures before handing off to an independent reviewer. If hardware or dependencies are unavailable, code review can proceed while explicitly identifying what remains unverified. Review alone provides no runtime evidence.
- Keep the reviewed content stable during review. Identify the change with an explicit baseline and diff, and distinguish pre-existing staged, unstaged, and untracked content. A stable review snapshot does not require an unauthorized commit or branch switch.

## Independent review

The primary agent initiates independent review at the stage specified by the workflow. The review skill defines the rules and output. It does not start agents or review loops. Follow session and project instructions for model selection and delegation. If tooling is unavailable or delegation conditions are unmet, proceed with the implementation, verification, and primary agent self-review that can be completed. Report that independent review remains outstanding.

Include the following in each assignment.

- **Objective and scope.** Provide the review type (Plan or Code Review), user requirements, acceptance checks, repository path, applicable rules, comparison baseline, and diff or new files. For Code Review, also provide actual test results and gaps in coverage.
- **Context.** Use a separate context, with `fork_turns="none"` when supported. Omit the implementer's reasoning and self-assessment. Allow the reviewer to read relevant definitions, callers, configurations, and tests so they can reason about behavior beyond the diff.
- **Permitted operations.** Read source, diffs, and existing verification logs. The primary agent runs tests through `mylws-test`. The reviewer may identify additional checks to run.
- **Prohibited side effects.** No file changes, remote operations, commits, published comments, or further delegation.
- **Deliverables and stopping conditions.** Return once the agreed scope is covered. Report what was read, findings with code locations and evidence, coverage gaps, and actions needed from the primary agent. Leave fixes and any further review to the primary agent.

Follow the [review process](../mylws-code-review/SKILL.md#review-process) to cover Style and Coding (Design) in one pass and merge findings with the same root cause. The primary agent independently checks each finding and its evidence, fixes confirmed issues, and cites code or test evidence when rejecting a finding. The final decision belongs to the primary agent.

## Fixes, follow-up, and completion

- Fix local issues locally and rerun affected checks. Return to Plan only if a fundamental assumption, requirement, or interface goal changes. Handle local user feedback in the same way.
- Default to one full Code Review. When a fix affects behavior or design, add a focused follow-up review if needed and cover only the fix and affected call paths. The primary agent can self-review style-only corrections.
- Limit independent Code Review to two rounds per task. Restarting an agent or rewriting the Plan does not reset the count. After two rounds, the primary agent continues diagnosis and verification and consults the user only if a decision is needed. Unresolved issues still count toward the completion criteria. Arrange additional review only if the user explicitly expands the task or requests it.
- The task is complete when the requirements are implemented, applicable checks pass, and confirmed blockers are resolved. Report anything unfinished because the required environment or verification was unavailable. Treat pure style suggestions according to their severity.
- Report the result, essential verification, and unresolved issues. State whether independent review or self-review was performed. Commit and push as authorized by the user.

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
