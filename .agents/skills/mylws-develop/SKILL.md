---
name: mylws-develop
description: >-
  开发 mylws 的轻量实现、学习示例和配套说明，按任务选择 Plan、实现、测试和独立 review，应用 mylws-code-review 的共享规则。
  在 mylws 中新增、修改或重构代码与文档时使用；代码审查使用 mylws-code-review，运行验证配合 mylws-test。
---

# mylws-develop

## 范围与规则

本流程仅用于 `mylws`。先读根 README、目标主题的已有 README、依赖声明和相关历史决策，确认需求与工作区已有改动。`src/myfa` 等属于可复用实现，`examples/` 属于学习示例，按各自目标开发。

编码前阅读 [mylws-code-review](../mylws-code-review/SKILL.md) 的首要原则、Style 和 Coding（Design），开发和审查共用该处规则。验证执行使用 [mylws-test](../mylws-test/SKILL.md)。纯 review 或运行已有测试直接使用对应 skill，不启动开发流程。

## 任务分流

先区分探索性工作与常规开发。新算法、新优化、新架构、CUDA kernel 和 Torch operator 开发属于探索性工作，必须先走下面的调研与确认流程；其余常规开发直接推进。

| 任务 | 流程 |
| --- | --- |
| 探索性开发 | 调研 Web + GitHub → 用户确认方向 → 短 Plan（按需 Plan Review）→ Code + Test → 独立 Code Review → 修复与验证 |
| 文档、格式及不改变行为的小范围机械修改 | 修改 → 必要检查 → 主 agent 自查 |
| 常规功能、修复、重构或需要跨文件协调的改动 | 短 Plan → Code + Test → 独立 Code Review → 修复与验证 |
| 关键方案未确定，或接口、并发、资源生命周期改动风险较高 | 在实现前按需增加一次独立 Plan Review，之后走 Code + Test 与 Code Review |

按行为影响与不确定性分流，不以文件数或 diff 行数机械判断；用户明确指定的审查范围优先。

## 探索性开发：先调研，再确认

1. **查找已有工作**：围绕问题、算法或算子搜索 Web 和 GitHub，覆盖相关实现、issues 和 PRs；核对官方文档、论文及源码中的适用条件。已有调研可复用，但需检查其版本与当前需求是否仍匹配。
2. **给出调研结论**：附来源链接，说明已有方案解决了什么、与当前需求的差异，以及直接复用、适配或自行实现的理由；明确仍未知的关键点。没有找到实现只能表述为本次检索未发现，不断言不存在。
3. **等待用户确认方向**：按项目 Research Before Development 规则，调研结论交用户确认后才能制定实现 Plan 或编码。确认前可以继续只读检索和核对证据，不以原型、试验代码或实现计划绕过该步骤；同一方向已有明确确认时不重复询问。
4. **确认后规划与验证**：Plan 写清要验证的假设、参考实现或 baseline、正确性标准；优化任务另列性能指标和测量条件。方案风险较高时增加 Plan Review，随后进入统一的 Code + Test、Code Review 和修复流程。
5. **按证据决定下一步**：先验证最小关键假设，再扩展实现。假设被推翻时记录结果；在已确认方向内局部调整，若需换算法、架构或扩大范围，补充调研并重新确认方向，不无界试错。验证失败不能包装成优化成功。

## Plan

- 只写目标、改动范围、关键假设和验收方式，复用已有结论，不另造长篇模板。需要持久化时按 `mymemo` 保存，不在源码目录留临时计划。
- 已授权且边界清楚的常规任务直接执行；只有影响方案的缺失信息、重大范围变化或需要用户选择的取舍才询问。探索性工作的确认要求仍按上述规则执行。
- 需要 Plan Review 时，reviewer 检查需求覆盖、接口与资源约束、方案可行性和验证方式，给出具体问题及依据。主 agent 核实并调整计划；默认只审一次，不以反复获得 APPROVED 作为开工条件。未解决的根本问题先定位或交用户决策。

## Code + Test

- 按共享规则编写实现、测试与必要说明，保留无关的用户改动。测试保护实际行为，不为文档或低影响改动凑数。
- 先运行与改动相符的最小有效验证，修复已知失败，再交独立 reviewer。缺设备或依赖时可以先审代码，但必须注明未验证项，不把 review 当作运行证据。
- Review 期间保持被审查内容稳定；以明确的基准和 diff 标识本次改动，区分已有 staged、unstaged 和 untracked 内容，不为固定版本擅自提交或切换分支。

## 独立 Review

独立 review 由主 agent 在分流后的相应阶段发起；review skill 只定义规则与输出，不自行启动 agent 或循环。模型选择与委派条件遵守当前会话和项目指令；工具不可用或委派条件不满足时，继续能完成的实现、验证与主 agent 自查，并明确报告未完成独立审查，不能称为独立 review 通过。

每次委派明确提供：

- **目标与范围**：Plan 或 Code Review、用户需求、验收方式、仓库路径、适用规则、对比基准及本次 diff/新增文件；Code Review 另附实际测试结果与未覆盖项。
- **上下文**：使用独立上下文（支持时用 `fork_turns="none"`），不传实现者的推理过程或自我评价；允许读取相关定义、调用方、配置与测试，不能只看 diff 猜行为。
- **允许操作**：只读源码、diff 和已有验证日志。测试由主 agent 按 `mylws-test` 统一执行，reviewer 可提出需要补充的验证。
- **禁止副作用**：不改文件、不操作远端环境、不提交或发布评论、不继续委派其他 agent。
- **交付与停止条件**：完成约定范围即返回；说明读了什么、发现什么、代码位置与依据、未覆盖项及主 agent 需要采取的动作。不自行修复或循环审查。

Code Review 按 [审查流程](../mylws-code-review/SKILL.md#审查流程) 一次检查 Style 和 Coding（Design），合并同根因 findings。主 agent 独立核对每条问题及证据；成立则修复，不采纳时给出代码或测试依据，reviewer 的 verdict 不替代最终判断。

## 修复、复核与完成

- 局部问题局部修复，并重跑受影响的验证；只有根本假设、需求或接口目标变化才回 Plan。用户的局部反馈同样不重走全部阶段。
- Code Review 默认一轮完整审查；涉及行为或设计的修复，必要时增加一轮针对性复核，只检查修复和受影响调用链。纯风格修正由主 agent 自查即可。
- 同一任务的独立 Code Review 最多两轮，不通过重启 agent 或重写 Plan 重置轮次。两轮后仍有问题，由主 agent 继续定位与验证；需要用户决策时才询问，不因轮次耗尽宣称完成。只有用户明确扩大任务或要求追加审查时才另行安排。
- 完成条件是需求已实现、适用检查通过、确认的阻塞问题已解决。缺少必要环境或验证时明确报告未完成项；纯风格建议不升级为正确性阻塞。
- 最终只报告结果、必要验证和未解决问题；说明实际采用独立 review 还是自查，不重复转贴全部审查记录。提交、推送等动作按用户授权执行。

## 依赖与开发环境

- 本地 Mac 用于开发及可运行的轻量检查；Linux、CUDA、NIXL 的实际行为在对应远端环境验证。
- 先核对已有依赖版本和所需 API，再决定安装或升级。满足要求的现有版本不因上游更新而升级。
- 安装、升级、下载按当前任务授权执行；需要补充授权时给出具体包、版本和目标环境，已有授权不重复询问。
- 优先为不兼容的依赖创建独立 venv，不在共享训练环境中试装依赖。安装与运行使用同一解释器，例如 `python -m pip`。
- 保持源码态运行，未要求打包时不执行 editable install、`uv sync` 或隐式安装项目的 `uv run`。
- 已验证版本和环境记录在对应实现或示例的 README；不把一次容器里的临时 venv 当作所有机器的既定配置。

上机按 `wx-remote-gemini` 连接，按 `wx-detect-my-dfs` 确认当前会话的 `MY_DFS`，按 `wx-sync-to-gpu-servers` 同步。
Mac 工作区是源码源头，不在同步接收端修改源码。每个独立远端 shell 显式激活已确认的环境：

```bash
MYLWS_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/mylws"
source "${MYLWS_VENV:?set the verified venv path}/bin/activate"
cd "$MYLWS_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
command -v python
python -c 'import sys; print(sys.executable, sys.prefix)'
```

按实际 import 布局补充 `PYTHONPATH`，不假定存在 `src/` 或 `tests/`。Ray worker 必须继承相同的解释器、源码路径和必要环境变量。
