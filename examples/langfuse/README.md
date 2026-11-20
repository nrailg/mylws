# Langfuse quickstart

Langfuse 记录 LLM 应用的一次请求经过了哪些步骤。打开一条 trace，可以沿着步骤看输入、输出和耗时，定位问题出在检索、模型调用还是后处理。需要比较 prompt 或模型版本时，可以用 Dataset 保存评估样本，再用 Experiment 批量运行并记录评分。

## 运行示例

在目标 Langfuse 项目的 **Settings → API Keys** 创建一对 Project API Keys。`LANGFUSE_BASE_URL` 填部署提供的 SDK 地址，包含协议、域名和端口（如有）。示例使用 SDK v4，适配上游自托管 Langfuse v3.63.0+ 和 v4。

在仓库根目录执行。示例在单机 CPU 上运行，使用固定的问题标签和模拟回答。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/langfuse/requirements.txt

export LANGFUSE_BASE_URL='https://your-langfuse.example.com'
export LANGFUSE_PUBLIC_KEY='pk-lf-...'
read -rs LANGFUSE_SECRET_KEY  # 粘贴 Secret Key 后按回车，输入不会回显
export LANGFUSE_SECRET_KEY
python examples/langfuse/main.py
```

在项目的 **Traces** 页面按终端打印的 `Trace ID` 查找根 observation `Q1`。下面是第一轮展开，每个后续 Q 会继续生成自己的 `A1` 和 `A2`。

```text
Q1
├── A1
│   ├── Q2
│   └── Q3
└── A2
    ├── Q3
    └── Q4
```

从 `Q1` 往下展开三轮。最后一层的 Q 仍生成两个回答，到此结束。整棵树共 85 个 Q span 和 170 个 A generation，属于同一条 trace。两个 `Q3` 是不同的 observation，各自挂在对应的 A 下。

每个 Q 和 A 的 Input 都记录从根到当前问题的 `messages`。例如，`A1` 分支下的 `Q3` 收到 `Q1 → A1 for Q1 → Q3`，`A2` 分支下的 `Q3` 收到 `Q1 → A2 for Q1 → Q3`。每条分支只继承自己的历史。

接入真实应用时，把模型调用放进 generation 的 `with` 块。模型返回 token 用量时，可通过 `generation.update(usage_details={"input": ..., "output": ...})` 记录。

`Trace ID` 在客户端生成。以 Traces 页面出现该 ID 为验证成功的标准。

SDK 通过 `POST /api/public/otel/v1/traces` 上报，并使用 Project API Keys 做 HTTP Basic Auth。如果该请求返回 HTTP 302 并跳转到公司 SSO，向部署维护者确认 SDK 运行环境应使用的入口或机器认证方式。若返回 HTTP 401 或 403，检查值是否来自目标项目，以及接入网关的权限。

## 验证

2026-11-20 在 macOS、Python 3.12.14、Langfuse SDK 4.18.0 下运行上述递归示例。本地 HTTP 接收端解码实际 OTLP 请求，验证 255 个 observation 的父子关系、输入和输出，以及各分支的 `messages`。

同日在单台 Gemini launcher、Linux x86_64、Python 3.12.13、Langfuse SDK 4.18.0 下验证了递归树的内部实例上报。在 Tracing 页面分别打开 `A1` 和 `A2` 下的 `Q3`，确认两条分支的 Input 展示了各自的历史。

官方文档 [Tracing quickstart](https://langfuse.com/docs/observability/get-started) · [自托管版本兼容表](https://langfuse.com/self-hosting/upgrade/versioning) · [Dataset 评估](https://langfuse.com/docs/evaluation/get-started/offline)
