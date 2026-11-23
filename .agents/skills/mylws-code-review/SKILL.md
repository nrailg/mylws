---
name: mylws-code-review
description: >-
  Review mylws commits, branch diffs, or local changes for Style and Coding (Design),
  including readability, design, behavioral correctness, and project rules. Also
  serves as the development standard. Reviews are read-only by default, with
  lighter checks for learning examples.
---

# mylws code review

These rules apply to development and review. Before development, read Style and Coding (Design) and apply the relevant rules during implementation and self-review. Use [mylws-develop](../mylws-develop/SKILL.md) to decide when to start an independent review. For user-requested reviews, pre-commit reviews, or the Code Review stage of development, use the [review process](#review-process) to confirm scope, check both categories, and merge duplicate findings. If the user specifies a single category, check only that category. The skill itself starts no agents or review loops.

Review `src/`, `examples/`, `tests/`, and related documentation within the requested scope. Understand the implementation requirements or the example's learning goal, interfaces, and call paths before assessing correctness, design, and style.

## Lightweight review for examples/

Review `examples/` for simplicity, readability, and a clear demonstration of the intended concept. These conventions take precedence over the stricter Style and Coding (Design) rules below. Apply the full rules to implementations under `src/`, including `src/myfa`. For mixed changes, apply the rules for each path.

- Focus on material issues affecting comprehension, demonstration correctness, execution, or information protection. Skip the full code quality checklist.
- Allow straight-line code, fixed parameters, modest duplication, and explanatory comments that aid understanding. Split functions, extract shared modules, add abstractions, or generalize interfaces only when the example needs it.
- Treat type hints, `@override`, docstring templates, line width, and uniform formatting as optional. Report import placement, naming, or comment format only when it materially affects comprehension or behavior.
- Verify the example's stated purpose. When existing run checks, output, or assertions are sufficient, skip separate unit tests, exhaustive edge cases, and support for environments outside the stated scope.
- Explain what is hard to understand, which concept is demonstrated incorrectly, or how execution would fail. Suggest the smallest fix. Omit style advice with no practical impact.

## Primary principle

Keep code simple and easy to read. mylws contains reusable lightweight implementations under `src/` and standalone learning examples under `examples/`. A reader should be able to follow the core logic, data flow, and necessary constraints. Prefer clarity over a lower line count. These rules take precedence over the skill's other style guidance.

- Keep only the logic required by the implementation or demonstration and the `assert` statements needed to reject invalid cases. Every branch, parameter, helper, and abstraction must serve a concrete need in the current task.
- Express checks directly with `assert` when one assertion suffices. Avoid `if ...: raise ...` blocks, custom exceptions, and layers of exception wrapping for these checks.
- Let original exceptions from third-party calls propagate. Add `try/except`, retries, fallback, version compatibility branches, or automatic environment detection only when an explicit recovery requirement calls for them.
- Prefer straight-line code and direct calls. Keep simple examples focused on their demonstration and avoid scaffolding for future extensions. Use constants for clear, fixed example parameters.
- Keep necessary assertions for inputs, resources, shape, dtype, devices, and results. Release resources where required.
- For GPU examples, assume GPUs are available in the runtime and test environment. Use CUDA and an explicit GPU count. Check necessary resource constraints with `assert` and let failures surface. Omit CPU fallback, device selection based on `torch.cuda.is_available()`, skip branches, and compatibility logic for a GPU-free Mac. Add a CPU path only when the user explicitly requests one.
- Use clear names and order code by execution and data flow. Keep useful intermediate variables and expand complex expressions or split assertions for readability. Avoid obscure abbreviations, nested comprehensions, and complicated one-liners.
- In review, look for unused code and check whether a reader can follow the implementation directly. Keep clear lines that improve readability. Avoid using extra abstractions or comments to compensate for confusing code.

## Style

### Naming, types, and imports

- Import dependencies at the top of the module. Imports under `TYPE_CHECKING` are allowed. Avoid function-local lazy imports and `try/except ImportError` that defers dependency errors.
- Add type hints to function signatures. Use the `@override` supported by the project's Python version for overridden methods.
- Follow the project's existing formatting configuration. With no configuration, aim for a 100-character line width. Verify whether a ruff configuration exists before relying on one. Preserve the minimum Python version declared for the implementation or example.
- Write comments and documentation that explain purpose and behavior in Chinese. Keep code terms such as `staging buffer` and `metadata`, and identifiers, in their original form.

### Comments and docstrings

- Use comments for non-obvious intent, workarounds, reference lifetimes, shape, units, layout, or other constraints the code cannot express directly.
- Put background, full workflows, and run instructions in the relevant implementation or example README. Omit line-by-line paraphrases, commented-out old code, and debugging history.
- Helpers, private functions, and one-line wrappers usually need no docstring. Briefly document non-obvious interface constraints when needed. Avoid applying the full NumPy docstring template to simple examples.
- Aim for one line per comment within a function. Omit comments that repeat names or obvious steps.
- Flag meaningless comments or docstrings, unjustified interface probing, lazy imports, silent error recovery, and unrelated changes. Fix these issues in your own changes during self-review.

### Documentation and progress records

Keep Markdown brief. Include only what readers need to run the work, understand it, or assess the result. Progress records keep the latest status, commands, results, and relevant limitations. Put details in logs and let Git retain history. Omit work diaries, repeated summaries, and empty template fields. Maintain each piece of information in one place and link to it elsewhere. See [mylws-test](../mylws-test/SKILL.md) for test records.

### Review judgment

- Distinguish readability issues from behavioral defects. Mark pure style suggestions as Minor.
- Treat `getattr`, `.get()`, and `try/except` as cues for inspection. Evaluate their use against the interface contract.
- Keep intermediate variables and expanded steps that improve readability. Avoid mechanically shortening code or imposing uniformity on historical style.

## Coding (Design)

### Design and behavior

- **Requirements and scope.** Check whether the implementation meets the feature, learning, or verification goal. Limit changes to the task. Avoid incidental renames, signature changes, whole-file formatting, and logic reordering. When adapting upstream work, preserve the structure relevant to the implementation or learning goal.
- **Responsibilities and abstractions.** Check whether the entry point leads the reader through the core computation, data flow, and resource use. Split code for actual reuse or interface boundaries. Omit frameworks, general-purpose helpers, and inheritance layers for hypothetical extensions.
- **Interfaces and state.** Check inputs, outputs, side effects, and resource ownership. Trace interface changes through callers, examples, and tests. Check cleanup, asynchronous completion, and error paths against the contract. Verify that resources are released after their final use.
- **Correctness and edge cases.** Check constraints relevant to the change, including empty input, length and index boundaries, shape, dtype, device, layout, stride, masks, numerical tolerance, and affected gradient paths. For GPU kernels, also check thread-to-data mapping, out-of-bounds access, synchronization, and data races.
- **Communication and concurrency.** For Ray, NIXL, MPI, or asynchronous GPU operations, trace process and node placement, buffer lifetimes, execution order, and completion conditions. Verify when data is actually safe to use after an API returns successfully.
- **Verification and performance.** Test observable behavior using an independent reference or the contract's expected results. Avoid computing expected results by duplicating the implementation, layering mocks, or adding generic wrappers. Base performance conclusions on measurements and the environment, and distinguish cold starts from steady-state runs.
- **Applicability of rules.** Apply new rules to additions and changes in scope. Keep historical issues separate from current findings and avoid repository-wide refactoring.

### Explicit interfaces and early failures

- Access required fields directly. Use explicit data contracts such as `batch["tokens"]`. Avoid probing established interfaces with `getattr` or `hasattr` by default.
- Call the API for the known third-party version directly. Add adapters, `Protocol`, or interface probing only for actual version requirements.
- Let failures surface. Do not swallow exceptions, guess substitutes, skip failed items, or automatically switch device, transport, data source, or path. Define any allowed defaults or recovery behavior in the interface or configuration contract in advance.
- Use `assert condition` for internal invariants and user input validation. Add a brief message when needed. Avoid `raise ValueError`, `raise TypeError`, or `raise RuntimeError` for these checks. Run assertion-dependent checks with assertions enabled, without `python -O`.
- Catch exceptions only when there is a purpose beyond rewording the error. Make sure necessary cleanup preserves the original task failure.
- Distinguish missing fields from `None`, `False`, and `0`. Keep valid values when applying defaults and avoid `or default` when it would overwrite them.

### Project purpose and layout

mylws contains lightweight implementations, standalone learning examples, and notes. Treat `src/myfa` as an implementation module and develop and review it according to its interface contract, correctness, and reuse requirements.

```text
src/<topic>/        Reusable lightweight implementations, such as src/myfa
examples/<topic>/
  main.py           Example entry point (existing names are also fine)
  requirements.txt  Dependencies for this example
  README.md         Concepts, environment, run commands, and verification scope
tests/<topic>/       Standalone tests
.agents/skills/     Development, review, and testing skills for this project
```

- Put importable or callable implementations in `src/<topic>/`, standalone programs that explain concepts or demonstrate usage in `examples/<topic>/`, and standalone tests in `tests/<topic>/`. Keep existing tests where they are.
- Document `src` interfaces, dependencies, and usage in an existing relevant README, such as `src/README.md`. A separate README for each topic is optional.
- Apply the following organization rules to `examples/`. Standalone demonstration entry points and example-specific dependency layouts are optional for `src/` modules.
- Keep each example focused on one question and make it understandable and runnable on its own. Add new examples to the root README.
- Isolate dependencies by example. Avoid forcing all examples into a single package or environment.
- Use third-party APIs directly when demonstrating mechanisms such as Ray or NIXL. Avoid copying Carrot's private Ray layer or its trainer and distributed module boundaries.
- Separate argument parsing from demonstration logic as needed. Extract shared modules only for actual reuse or interface boundaries. Keep two-line operations inline when a wrapper serves no purpose.
- Document devices, node counts, resources, data paths, run commands, and success criteria in the README. Distinguish illustrative, measured, and unverified capabilities.
- Keep example assertions, run checks, and verification results in `examples/<topic>/`. Record results briefly in the example README. Running an example alone requires no separate test files or records in `tests/`.

### Testing and readability

Provide lightweight verification that protects actual behavior for features and fixes. Follow [mylws-test](../mylws-test/SKILL.md) to run it. Verify logic independently where possible and reserve long end-to-end runs for behavior that requires them. Add tests for documentation or low-impact changes only when they serve a purpose.

When adding or changing tests, follow these rules.

- Start each test function with one sentence in Chinese explaining what it verifies. Then construct the inputs, make the call, and assert the result directly.
- Explain only non-obvious inputs, mock boundaries, and numerical tolerances. Keep obvious blocks free of unnecessary comments and simple assertions free of extra fixtures, helpers, or framework wrappers.
- Limit conclusions to the environment actually tested. Report syntax, communication, single-node, and cross-node results according to what ran.

### Information protection

Keep Gemini tokens, secrets, proxy credentials, personal absolute DFS paths, and raw environment dumps out of code, README files, test records, and commits.
Use verified variables such as `MY_DFS` in reusable commands. See `mylws-test` for path exceptions in one-off user commands.
Distinguish official sources from internal deployment details in public references. Check logs for sensitive content before adding them to the repository. Store raw remote artifacts in the project directory on the user's DFS.

## Review process

Use this process for both Style and Coding (Design), including independent Code Review during development. Apply only the relevant rules when self-reviewing small changes.

### Scope and evidence

- Confirm the repository root, purpose of the change, and comparison baseline. Report a blocker if the requested project does not match the Git root or the ref is missing. Stay within the requested repository.
- Include staged, unstaged, and relevant untracked files in a local review. Read the full diff and inspect new files separately. Verify whether dependencies, generated files, and nested repositories are part of the project scope before reviewing them.
- For branch reviews, confirm source and target and pin their SHAs. Use `git diff <target_sha>...<source_sha>` unless the user specifies a commit or comparison method. Confirm the default branch or source explicitly when needed. State whether workspace changes are included.
- For PR or MR reviews, verify the repository, source, target, and head SHA from the page. If a fetch is needed, verify the remote first. Review a pinned version without switching the user's workspace. Reconfirm scope if the version changes.
- Apply lightweight review to `examples/` and the full rules to other implementations in scope. Read the changes and the definitions, callers, configuration, README files, and tests needed to understand their behavior. Distinguish an empty diff from a missing comparison baseline.
- Default to read-only review. Modify files only when the user also asks for fixes. Commits and published comments require authorization. A review request alone grants no permission to publish.
- When verification is needed, use [mylws-test](../mylws-test/SKILL.md) to choose the smallest effective scope. Report checks as not run, failed, skipped, or passed as appropriate. A local syntax check verifies syntax only.

### Findings

- Report only concrete issues introduced by the change that are worth fixing. Include the file and smallest useful line range, triggering condition, impact, and smallest fix.
- Merge Style and Coding (Design) findings with the same root cause. Cite multiple locations for design issues when needed to show actual coupling, repeated changes, or a broken contract. Require abstractions only for demonstrated needs.
- For project-rule findings, cite the applicable rule. Ordinary defects need no forced rule reference. Keep other repositories' conventions and personal preferences out of this project's requirements.
- **Blocker** covers correctness, security, or data corruption issues that must be fixed. **Major** covers design, maintenance, or verification issues with concrete impact. **Minor** covers style suggestions without behavioral impact. Use the user's priority format when specified.
- Order the merged findings by severity, then briefly describe scope, verification, and limitations. Report historical issues and open questions separately from findings. If no issue can be confirmed, say so directly.
- Limit single-category review conclusions to that category. A Style review with no findings covers Style alone.
