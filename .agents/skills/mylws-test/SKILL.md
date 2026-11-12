---
name: mylws-test
description: >-
  验证 mylws 学习示例与代码改动，运行本地轻量检查及远程 Gemini CPU/GPU/多节点测试，
  维护可复现命令和测试档案。用户要求测试、跑测、pytest 或验证 mylws 示例时使用。
---

# mylws-test

先按 [mylws-develop](../mylws-develop/SKILL.md) 确认示例入口、依赖和测试目标。常规回归直接运行已有测试；真实通信和 GPU 行为必须在对应环境验证。

测试代码同样必须非常简洁：直接调用示例并用必要的 `assert` 校验结果和非法情况，不增加生产级异常体系、兼容分支或通用测试封装。能在现有示例里直接断言的行为，不再复制一套实现来测试。

GPU 示例的测试环境默认有 GPU。直接运行 CUDA 路径，保留必要的 GPU 数量和设备断言；不为本地无 GPU 增加 CPU fallback、自动设备选择或跳过失败的分支。实际环境不满足要求时报告阻塞，不降级运行。

## 上机前检查

1. 用 `wx-remote-gemini` 连接用户指定的任务；不要复用历史 token 或凭旧 session ID 推断仍连接正确容器。
2. 用 `wx-detect-my-dfs` 在当前会话确认 `MY_DFS`；用 `wx-sync-to-gpu-servers` 复用当前项目同步配置，确认代码已到位并核对关键文件校验值。
3. 检查 Python、依赖版本和所需 API；优先复用满足要求的已有版本，不为跑测试直接升级整个环境。缺少依赖时按任务授权补齐，不能用 mock 的结果冒充真实通信。
4. 按 `wx-mpirun` 从 `/etc/mpi/hostfile` 确认节点数；每节点只执行一次。用 hostname 验证实际参与节点，不把 launcher 当作全部节点信息。
5. 用 `nvidia-smi` 或 `dguard status` 检查 GPU 占用。按 `wx-remote-gemini` 在 launcher 暂停正在运行的 dguard，短测通常 10 分钟；其他工作负载不能当作看护进程清理。

模型或数据测试先核对本地路径、版本和加载调用链，避免隐式联网。下载模型/数据按 `wx-download`；Python 包使用确认的包源，不把 Hugging Face 下载流程套到 pip 安装上。

## 环境与 bash

长期 runner 使用当前会话确认的变量，只做简单存在性检查，不嵌入复杂路径搜索。每个独立 shell 重新激活，不依赖镜像 PATH 或上一条命令的 `source`：

```bash
set -Eeuo pipefail
MYLWS_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/mylws"
source "${MYLWS_VENV:?set the verified venv path}/bin/activate"
cd "$MYLWS_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
test "$(command -v python)" = "$MYLWS_VENV/bin/python"
python -c 'import sys; print(sys.executable, sys.prefix)'
```

Ray/NIXL 示例曾在 `/opt/venvs/ray-nixl-test` 验证，该环境通过 `.pth` 复用 `/opt/venvs/carrot` 的 Ray/PyTorch。
这只是已知环境：换机后必须重新核对环境及其基础路径，不能假定存在或悄悄回退到其他 Python。

给用户自行执行的一次性 bash 应短、可复制，可写入当前任务已确认的远端绝对路径；未知路径用清楚的待替换项。
这类命令不包含 DFS 发现过程，也不直接复制进仓库长期 runner 或公开文档。

## 测试文件与记录

新增或调整需要留档的测试事项，在 `tests/<主题>/` 维护同名三件套，例如：

```text
tests/ray_nixl/test_transfer.py
tests/ray_nixl/test_transfer.sh
tests/ray_nixl/test_transfer.md
```

- `.py` 承载断言与结果校验，不把核心判断藏在 shell heredoc 中。
- `.sh` 保持简单，检查已准备的环境/资源、设置参数、调用测试；不隐式安装依赖或下载数据。测试专用配置放在同一目录。
- `.md` 先写目的、命令和预期，结束后回填 `PASS`、`FAIL`、`NOT RUN` 或 `BLOCKED` 及实际证据。
- 记录实际 Python/依赖版本、硬件/节点数、命令/退出码、关键断言、镜像 tag、源码 commit 和证据位置。未知信息写“未记录”；未提交源码补充关键文件 hash，不能用 HEAD 冒充被测源码版本。
- 同一事项重跑在原 `.md` 追加日期小节，不新建按日期命名的档案，不覆盖历史失败或未执行项。
- 只跑已有示例作常规回归时不强制新增三件套，可沿用示例 README 的实测记录。语法检查和纯文档验证也不建立远端实验档案。

运行前按 `mylws-develop` 的测试可读性要求检查中文意图说明；档案中的关键断言必须与代码实际验证的内容一致。

## Ray 与多节点

- 先确认目标集群归属和占用；起测试集群前按 `wx-remote-gemini` 先 stop 再 start。已有无关任务运行时不能直接清空集群。
- 临时单机示例可以由 `ray.init()` 启动本地实例；多节点测试先按 `wx-mpirun` 起集群，再显式传 `--address` 或 `RAY_ADDRESS`，不能各节点各起一个互不连接的实例。
- 设置源码 `PYTHONPATH` 后再启动 Ray，mpirun 传递 `PATH`、`LIBRARY_PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH` 及该测试需要的其他变量。worker 使用同一个已确认的 venv。
- 使用 `python -m ray.scripts.scripts` 起停 Ray，避免复用环境时 PATH 中的 `ray` 脚本 shebang 指向基础 venv，导致新装的 NIXL 对 worker 不可见。
- hostfile 的全部 `slots` 归一为 1，并使用 `wx-mpirun` 要求的 `bond1` 和 `routed direct` 参数；不要仅把 `slots=8` 改成 1 后假定其他 slot 值不存在。
- MPI 命令成功不代表 Ray 所有节点已注册；检查 `ray.nodes()` 和资源。跨节点用例必须验证两端实际 node ID 不同。

## Ray/NIXL 示例的验证范围

下面是已有入口的回归命令，执行前完成上述环境和资源检查：

```bash
python examples/ray_nixl/main.py --size-mib 64

# 仅在两个合适节点已加入同一个 Ray 集群后执行
python examples/ray_nixl/main.py --address auto --cross-node --size-mib 64
```

- 用例需两个 Actor 的 CPU 资源和两张可用 GPU。只有一个节点时，将跨节点传输标为 `NOT RUN`，可独立验证 `--cross-node` 的拒绝行为。
- 同时检查退出码、shape、dtype、设备和数据内容，以及日志中的实际 NIXL 后端；输出中的固定 `transport` 字段不能独立证明底层通信路径。
- NIXL/UCX 初始化成功、跨节点成功均不等于已证明 RDMA；需要实际 transport 日志和相应设备证据。没有证据就写“未验证”，不把 TCP、共享内存或 CUDA IPC 结果表述成 RDMA。
- 正确性和性能分开报告。包含首次连接、内存注册或冷启动的计时不能作为稳定带宽；只有性能任务才增加预热、重复采样与对应统计。

## 执行、产物与清理

- 先做能在本地完成的必要检查，再跑与改动相符的远端用例。使用 pytest 前确认测试存在及所用插件可用，不盲加 `--timeout`。
- 长测试用 MCP 后台任务，按阶段检查输出；运行中每分钟报告新进展或阻塞。失败先定位原因，修复本地源码并重新同步，避免接收端热改。
- 需要保留的结果和日志放在 `$MY_DFS/work/mylws/outputs/<主题>/<运行标识>/`，不存 token/密钥；确保运行产物不被同步回源码或误提交。
- MCP 后台任务不使用 nohup 或 shell 重定向，完成后用 `remote_task_dump` 保存所需输出，再释放任务；没有留存价值的任务输出收集后 purge。
- 只清理本次创建的 Actor/进程。独占测试集群优先 `python -m ray.scripts.scripts stop --force`，禁止 `pkill -9 python`；不要按运行时长自动删除接收端文件。
- 若本次暂停了 dguard，结束后恢复看护或确认其定时恢复有效。最终报告实际通过项、未覆盖项、环境变更及证据/档案位置。
