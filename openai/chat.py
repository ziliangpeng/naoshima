import openai
import os
import sys

openai.api_key = os.getenv("OPENAI_API_KEY", "your_api_key_here")

GPT_4 = "gpt-4"
GPT_3 = "gpt-3.5-turbo"

token_count = 0


def chat(prompt, initial_history, history, model="gpt-3.5-turbo"):
    global token_count
    print()
    history.append({"role": "user", "content": prompt})
    full_response_text = ""
    session_token_count = 0
    too_long = False
    """
    NOTE: when the response is hitting length, GPT will stop. but if you include that in the
    prompt history message and hit length in request, you get invalid request.
    TODO: something clever to auto trim and retry if length is hit.
    """
    for i, resp in enumerate(
        openai.ChatCompletion.create(
            model=model, messages=initial_history + history, stream=True
        )
    ):
        if i == 0:
            model = resp.model
            assert resp.choices[0].delta.role == "assistant"
            continue
        if resp.choices[0].finish_reason != None:
            if resp.choices[0].finish_reason == "stop":
                continue
            elif resp.choices[0].finish_reason == "length":
                # TODO: some handling of over length
                print(resp)
                print("finish due to length.")
                too_long = True
                continue
            else:
                print("finish due to others")
                print(resp.choices[0].finish_reason)

        if not hasattr(resp.choices[0].delta, "content"):
            print(resp)
        assert hasattr(resp.choices[0].delta, "content")
        session_token_count += 1
        text = resp.choices[0].delta.content
        sys.stdout.write(text)
        sys.stdout.flush()
        full_response_text += text

    if too_long:
        print("Encountered some toolong error. trimming")
        trim_history(history)
        # TODO: if too long, remove the last user question.
        return ""

    token_count += session_token_count
    print()
    print()
    history.append({"role": "assistant", "content": full_response_text})
    print("Model: " + model)
    print(
        f"Total length: {sum(map(lambda x: len(x['content']), history))}. Response length: {len(full_response_text)}."
    )
    print(f"Total token: {token_count}")
    return full_response_text


def trim_history(history):
    print("Yes. actually trimming now")
    del history[:2]


if __name__ == "__main__":
    initial_history = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    history = []

    while True:
        try:
            prompt = input("\n========================================\nSay: ")

            # For automated testing
            # prompt = "Choose a topic in natural sceience and write a 1000 word essasy to phrase it"
            # print(f"Automating using prompt: {prompt}:")

            generated_text = chat(prompt, initial_history, history, model=GPT_4)
        except openai.error.InvalidRequestError as e:
            print(type(e))
            if "This model's maximum context length is" in e.user_message:
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
