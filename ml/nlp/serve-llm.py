from torch import bfloat16, float16
import transformers

# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=float16, device_map="auto"
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)


generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,  # if using langchain set True
    task="text-generation",
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1,  # if output begins repeating increase
)


res = generate_text("hello there")
print(res[0]["generated_text"])
