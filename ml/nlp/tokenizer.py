import tiktoken
import sys

# Convert a string to a list of token IDs
def string_to_token_ids(text, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(text)
    return token_ids

# Convert a list of token IDs back to a string
def token_ids_to_string(token_ids, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    text = encoding.decode(token_ids)
    return text

# Example usage
encoding_name = "cl100k_base"  # Replace with the appropriate encoding for your model
encoding_name = "r50k_base"
with open(sys.argv[1]) as f:
    text = f.read()
# text = "Hello, world!"

token_ids = string_to_token_ids(text, encoding_name)
print(f"Token IDs: {token_ids}")
print(f"number of uniqie tokens: {len(set(token_ids))}")
print(f"max token id: {max(token_ids)}")

# reconstructed_text = token_ids_to_string(token_ids, encoding_name)
# print(f"Reconstructed Text: {reconstructed_text}")