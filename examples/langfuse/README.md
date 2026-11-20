# Langfuse quickstart

Langfuse 记录 LLM 应用的一次请求经过了哪些步骤。打开一条 trace，可以沿着步骤看输入、输出和耗时，定位问题出在检索、模型调用还是后处理。需要比较 prompt 或模型版本时，可以用 Dataset 保存评估样本，再用 Experiment 批量运行并记录评分。

## 运行示例

在目标 Langfuse 项目的 **Settings → API Keys** 创建一对 Project API Keys。`LANGFUSE_BASE_URL` 填部署提供的 SDK 地址，包含协议、域名和端口（如有）。示例使用 SDK v4，适配上游自托管 Langfuse v3.63.0+ 和 v4。

在仓库根目录执行。示例在单机 CPU 上运行，使用固定的检索结果和模拟回答。

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

在项目的 **Traces** 页面按终端打印的 `Trace ID` 查找 `demo-answer-question`。点开后应看到两个子 observation。

| Observation | 在页面上看什么 |
| --- | --- |
| `retrieve-policy` | 用户问题，以及检索返回的退货规则 |
| `generate-answer` | 带检索上下文的 messages、模拟模型名 `demo-stub` 和回答 |

接入真实应用时，把实际检索和模型调用放到对应的 `with` 块里。模型返回 token 用量时，可通过 `generation.update(usage_details={"input": ..., "output": ...})` 记录。

`Trace ID` 在客户端生成。以 Traces 页面出现该 ID 为验证成功的标准。

SDK 通过 `POST /api/public/otel/v1/traces` 上报，并使用 Project API Keys 做 HTTP Basic Auth。如果该请求返回 HTTP 302 并跳转到公司 SSO，向部署维护者确认 SDK 运行环境应使用的入口或机器认证方式。若返回 HTTP 401 或 403，检查值是否来自目标项目，以及接入网关的权限。

## 验证

2026-11-20 在 Linux x86_64、Python 3.12.13、Langfuse SDK 4.18.0 下，从单台 Gemini launcher 向内部 Langfuse v4.82.0 实例运行上述示例。Tracing 页面显示了终端打印的 Trace ID，以及 `retrieve-policy` 和 `generate-answer` 两个子 observation。

同日在 macOS、Python 3.12.14、Langfuse SDK 4.18.0 下，通过本地 HTTP 接收端解码实际 OTLP 请求，确认父子关系、输入输出和模型名。

官方文档 [Tracing quickstart](https://langfuse.com/docs/observability/get-started) · [自托管版本兼容表](https://langfuse.com/self-hosting/upgrade/versioning) · [Dataset 评估](https://langfuse.com/docs/evaluation/get-started/offline)
