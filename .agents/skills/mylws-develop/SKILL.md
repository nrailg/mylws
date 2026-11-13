---
name: mylws-develop
description: >-
  mylws 示例项目的目录边界、依赖、编码、测试和信息保护规范。
  在 mylws 中新增、修改或审查学习示例与配套说明时使用；运行验证配合 mylws-test。
---

# mylws-develop

## 首要原则：代码必须非常简洁、便于人类阅读

mylws 是学习和验证机制的示例仓库，不是生产 code base。人类可读性是写代码的重要目标：读者应能顺着代码理解核心逻辑、数据流和必要约束。简洁以容易理解为前提，不追求最少行数。以下原则优先于本 skill 的其他风格建议：

- 只保留演示目标必需的逻辑和排除非法情况的 `assert`。每个分支、参数、helper 和抽象都必须有当前示例的实际需要。
- 能用一条 `assert` 表达的检查，就直接写 `assert`；不要展开为 `if ...: raise ...`，不要增加自定义异常或多层异常包装。
- 第三方调用失败时让原始异常直接暴露。没有明确恢复需求，不加 `try/except`、重试、fallback、版本兼容分支或环境自动探测。
- 优先直线代码和直接调用，不为未来扩展搭框架，不把简单示例写成可配置的通用工具。固定且清楚的示例参数可以直接写常量。
- 简洁不能省掉关键约束：保留必要的输入、资源、shape/dtype、设备和结果断言，以及有实际必要的资源释放。
- GPU 示例默认测试环境有 GPU，直接使用 CUDA 和明确的 GPU 数量；不写 CPU fallback、设备自动选择或为了兼容本地无 GPU 而增加分支。资源不足用必要的 `assert` 直接失败；只有用户明确要求比较 CPU/GPU 时才添加 CPU 路径。
- 使用含义明确的命名，按执行和数据流顺序组织代码；必要时保留中间变量、展开表达式或拆开断言，让读者无需在脑中解码。避免晦涩缩写、嵌套推导式和复杂的一行表达式。
- Review 同时检查“能否删除无用代码”和“人能否直接读懂”。为可读性保留的几行清晰代码不算冗余；不靠增加抽象或堆砌注释弥补难懂的实现。

## 项目定位与目录

mylws 是独立学习示例和笔记的集合。先读根 README、目标示例 README 和依赖声明，再修改代码。

```text
examples/<主题>/
  main.py           示例入口，沿用已有入口名也可以
  requirements.txt  该示例的依赖
  README.md         原理、环境、运行命令和验证边界
tests/<主题>/       需要独立维护的测试与档案
.agents/skills/     当前项目的开发和测试 skill
```

- 一个示例围绕一个明确的问题，保持能单独理解和运行；新增示例在根 README 添加入口。
- 依赖按示例隔离，不把所有示例强行合并为一个安装包或统一环境。
- 演示 Ray/NIXL 等机制时可以直接使用第三方 API；不要照搬 Carrot 的私有 Ray 层、trainer、distributed 模块边界。
- 参数解析与演示逻辑按需要分开。只在真实复用或接口边界需要时提取公共模块，不为两行代码增加 wrapper。
- README 写清设备、节点数、资源、数据路径、运行命令和成功条件，区分示意、实测和未验证的能力。

## 依赖与开发环境

- 本地 Mac 用于开发及可运行的轻量检查；Linux、CUDA、NIXL 的实际行为在对应远端环境验证。
- 先核对已有依赖版本和所需 API，再决定安装或升级。满足要求的现有版本不因上游更新而升级。
- 安装、升级、下载按当前任务授权执行；需要补充授权时给出具体包、版本和目标环境，已有授权不重复询问。
- 优先为不兼容的依赖创建独立 venv，不在共享训练环境中试装依赖。安装与运行使用同一解释器，例如 `python -m pip`。
- 保持源码态运行，未要求打包时不执行 editable install、`uv sync` 或隐式安装项目的 `uv run`。
- 已验证版本和环境记录在各示例 README；不把一次容器里的临时 venv 当作所有机器的既定配置。

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

## 编码规范

### 明确接口，及时失败

- 必需字段直接访问，数据字段用 `batch["tokens"]` 这样的显式契约；默认不用 `getattr` / `hasattr` 探测已有接口。
- 针对明确的第三方版本直接调用 API，不为假想的版本差异增加 adapter、`Protocol` 或接口探测分支。
- 禁止吞异常、猜替代值、跳过失败项，或在失败后自动切换设备、transport、数据来源和路径。允许的默认值和恢复行为必须事先写入接口或配置契约。
- 内部不变式和用户输入合法性使用 `assert condition`，需要解释时附简短消息；禁止为这类检查改用 `raise ValueError` / `raise TypeError` / `raise RuntimeError`。不要用 `python -O` 运行依赖断言的检查。
- 不捕获异常只为换一种报错措辞；必要的清理逻辑不能掩盖主任务失败。

### import 与风格

- 依赖在模块顶部 import；禁止函数内 lazy import，也不用 `try/except ImportError` 延迟依赖错误。`TYPE_CHECKING` 下的 import 可以。
- 函数签名写 type hints；覆盖方法使用当前 Python 版本支持的 `@override`。
- 对齐项目已有格式配置；未配置时以 100 字符行宽为目标，不虚构已有 ruff 配置，不擅自提高示例声明的 Python 最低版本。
- 只做任务必需的修改，不顺便改名、调整签名、格式化全文件或重排逻辑。借鉴上游时保留与学习目标有关的结构。
- 新规范用于本次新增或修改的部分；发现历史代码不一致时记录具体位置，不借此扩大任务为全仓重构。

## 注释与 docstring

- 注释只解释非直观意图、workaround、引用生命周期、shape、单位、布局或其他代码无法直接表达的约束。
- 学习背景、完整流程和运行讲解放在示例 README；不要逐行翻译实现、保留注释掉的旧代码或加入调试历史。
- helper、私有函数、一行 wrapper 通常不写 docstring。确有非直观接口约束时简短说明，不为简单示例套完整的 NumPy docstring 模板。
- 函数体中的单条注释尽量一行，不复述函数名、变量名或显然的步骤。
- 无意义注释/docstring、无理由的接口探测、lazy import、静默容错和无关改动均须在 review 中指出并修正本次改动。

## 测试与可读性

功能与修复提供能保护实际行为的轻量验证，具体执行见 [mylws-test](../mylws-test/SKILL.md)。不要用长时间 e2e 替代可以独立验证的逻辑；文档或低影响改动不为凑数添加测试。

新增或修改测试时：

- 每个测试函数开头用一句中文说明验证意图，随后直接构造输入、调用和断言。
- 只为非直观的输入、mock 边界或数值容差补充说明；不要求每个显然的逻辑块写注释，不为简单断言增加 fixture、helper 或测试框架封装。
- 测试结论只覆盖实际执行的环境；语法通过不能写成通信通过，单机通过不能写成跨节点通过。

## 信息保护

不把 Gemini token、密钥、代理认证、个人绝对 DFS 路径或原始环境变量转储写入代码、README、测试档案和提交。
通用命令使用 `MY_DFS` 等已确认变量；一次性用户命令的路径例外见 `mylws-test`。
公开引用区分官方来源和内部部署信息，日志入库前检查敏感内容；原始远端产物保存在个人 DFS 的项目目录。
