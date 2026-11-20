import os

from langfuse import Langfuse


def main() -> None:
    langfuse = Langfuse(
        base_url=os.environ["LANGFUSE_BASE_URL"],
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    )
    question = "How long is the return window?"

    with langfuse.start_as_current_observation(
        name="demo-answer-question", input=question
    ) as request:
        with langfuse.start_as_current_observation(
            name="retrieve-policy", as_type="retriever", input=question
        ) as retrieval:
            policy = "Orders can be returned within 30 days of delivery."
            retrieval.update(output=policy)

        messages = [
            {"role": "system", "content": f"Answer using this policy: {policy}"},
            {"role": "user", "content": question},
        ]
        with langfuse.start_as_current_observation(
            name="generate-answer", as_type="generation", model="demo-stub", input=messages
        ) as generation:
            answer = "You can return your order within 30 days of delivery."
            generation.update(output=answer)

        request.update(output=answer)
        trace_id = request.trace_id

    # 短生命周期脚本退出前主动导出 SDK 缓冲的 observations。
    langfuse.flush()
    print(answer)
    print(f"Trace ID: {trace_id}")


if __name__ == "__main__":
    main()
