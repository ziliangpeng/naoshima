import openai
import os
import sys

openai.api_key = os.getenv("OPENAI_API_KEY", "your_api_key_here")

GPT_4 = "gpt-4"
GPT_3_TURBO = "gpt-3.5-turbo"


def chat(prompt, initial_history, history, model="gpt-3.5-turbo"):
    print()
    history.append({"role": "user", "content": prompt})

    full_response_text = ""
    for i, resp in enumerate(
        openai.ChatCompletion.create(
            model=model, messages=initial_history + history, stream=True
        )
    ):
        if i == 0:
            assert resp.choices[0].delta.role == "assistant"
            continue
        if resp.choices[0].finish_reason == "stop":
            continue
        assert hasattr(resp.choices[0].delta, "content")
        text = resp.choices[0].delta.content
        sys.stdout.write(text)
        sys.stdout.flush()
        full_response_text += text

    print()
    history.append({"role": "assistant", "content": full_response_text})
    return full_response_text


def trim_history(history):
    history = history[2:]


if __name__ == "__main__":
    initial_history = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    history = []

    while True:
        try:
            prompt = input("\n========================================\nSay: ")
            generated_text = chat(prompt, initial_history, history, model=GPT_3_TURBO)
        except openai.error.InvalidRequestError as e:
            print(type(e))
            if (
                "This model's maximum context length is 4097 tokens. However, your messages resulted in 4108 tokens. Please reduce the length of the messages."
                in e.user_message
            ):
                print("InvalidRequestError. Trimming history..")
                trim_history(history)
            else:
                print("InvalidRequestError but I don't know what to do")
                print(e)
        except Exception as e:
            print(type(e).__name__)
            print(e)
            print("there is an error")
            continue
