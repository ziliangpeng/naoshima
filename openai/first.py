import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY", "your_api_key_here")

GPT_4 = "gpt-4"
GPT_3_TURBO = "gpt-3.5-turbo"


def chat(prompt, history, model="gpt-3.5-turbo"):
    history.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model=model,
        messages=history,
    )

    print(response["model"])
    print(response["usage"])
    history.append(response.choices[0]["message"])

    return response.choices[0]["message"]["content"]


if __name__ == "__main__":
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    while True:
        try:
            prompt = input("\nSay: ")
            generated_text = chat(prompt, history, model=GPT_3_TURBO)
            print(generated_text)
        except Exception as e:
            print(e)
            print("there is an error")
            continue
