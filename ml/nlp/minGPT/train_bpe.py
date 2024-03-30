"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer
import sys

directory = sys.argv[1]
text = ""
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        if filename.endswith(".txt"):
            with open(os.path.join(dirpath, filename), "r", encoding="utf-8") as f:
                print(f"Reading {filename}")
                text += f.read() + '\n\n'
# open some text and train a vocab of 512 tokens
# text = open(text, "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer], ["basic"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 1024 * 8, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")