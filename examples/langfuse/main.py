import os

from langfuse import Langfuse


def trace_question(
    langfuse: Langfuse,
    question_number: int,
    remaining_depth: int,
    history: list[dict[str, str]],
) -> str:
    question = f"Q{question_number}"
    messages = [*history, {"role": "user", "content": question}]
    answers = []

    with langfuse.start_as_current_observation(name=question, input=messages) as question_span:
        for answer_number in (1, 2):
            answer_name = f"A{answer_number}"
            answer = f"{answer_name} for {question}"
            with langfuse.start_as_current_observation(
                name=answer_name, as_type="generation", model="demo-stub", input=messages
            ) as generation:
                generation.update(output=answer)
                if remaining_depth > 0:
                    branch_history = [*messages, {"role": "assistant", "content": answer}]
                    first_follow_up = question_number + answer_number
                    for follow_up in (first_follow_up, first_follow_up + 1):
                        trace_question(langfuse, follow_up, remaining_depth - 1, branch_history)
            answers.append(answer)

        question_span.update(output=answers)
        return question_span.trace_id


def main() -> None:
    langfuse = Langfuse(
        base_url=os.environ["LANGFUSE_BASE_URL"],
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    )
    trace_id = trace_question(langfuse, question_number=1, remaining_depth=3, history=[])

    # 短生命周期脚本退出前主动导出 SDK 缓冲的 observations。
    langfuse.flush()
    print(f"Trace ID: {trace_id}")


if __name__ == "__main__":
    main()
