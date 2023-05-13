import openai
import os
import sys

from chat import chat

openai.api_key = os.getenv("OPENAI_API_KEY", "your_api_key_here")

GPT_4 = "gpt-4"
GPT_3 = "gpt-3.5-turbo"

if __name__ == "__main__":
    initial_history = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    history = []

    # ending_token = input("What's your ending token: ")
    ending_token = "x"
    # text = "Summarize this into 3 paragraphs with moderate amount of details:\n\n"
    text = "In this session, you will be an assistant that help me summarize YouTube transcripts. Please focus on the meaningful content, and skip low information density text. I will paste the youtube transcript, and please summarize for me into a few paragraphs with moderate level of details. After that, I will ask ask you questions about the context and please answer based on your knowledge reading the transcript. Here's the transcript:\n\n"

    while True:
        prompt = input("Say: ")
        print(prompt)
        if prompt.strip() == ending_token.strip():
            break
        text += prompt + "\n"

    print(f"Total Text:\n{text}\n================================\n")
    try:
        # For automated testing
        # prompt = "Choose a topic in natural sceience and write a 1000 word essasy to phrase it"
        # print(f"Automating using prompt: {prompt}:")

        generated_text = chat(text, initial_history, history, model=GPT_4)
    except openai.error.InvalidRequestError as e:
        print(type(e))
        # if "This model's maximum context length is" in e.user_message:
        #     print("InvalidRequestError. Trimming history..")
        #     trim_history(history)
        # else:
        #     print("InvalidRequestError but I don't know what to do")
        #     print(e)
    except Exception as e:
        print(type(e).__name__)
        print(e)
        print("there is an error")

    while True:
        prompt = input("Say: ")
        chat(prompt, initial_history, history, model=GPT_4)
