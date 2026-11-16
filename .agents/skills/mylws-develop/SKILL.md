---
name: mylws-develop
description: >-
  开发 mylws 的轻量实现、学习示例和配套说明，按 mylws-code-review 的 Style 与 Coding（Design）规则实现。
  在 mylws 中新增、修改或重构代码与文档时使用；代码审查使用 mylws-code-review，运行验证配合 mylws-test。
---

# mylws-develop

## 开发流程

1. 先读根 README、目标主题的已有 README 和依赖声明，确认实现需求或学习目标、入口和目录边界；区分 `src/` 实现与 `examples/` 示例。
2. 编码前阅读 [mylws-code-review](../mylws-code-review/SKILL.md) 的首要原则、Style 和 Coding（Design）。开发和审查共用该处规则：按规则设计接口、编写实现、测试与说明，不另维护一套编码规范。
3. 保持实现与示例简洁、便于人类阅读。完成改动后按相关规则自查，只修正本次改动的问题；普通开发无须启动完整审查流程或输出 findings。
4. 按 [mylws-test](../mylws-test/SKILL.md) 选择与改动相符的验证，报告实际结果与未覆盖项。文档或低影响改动不为凑数添加测试。

用户请求 review 或提交前审查时，使用 [mylws-code-review 的审查流程](../mylws-code-review/SKILL.md#审查流程)，默认同时检查 Style 和 Coding（Design）；指定单类时仅检查该部分。

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
