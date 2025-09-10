from radix import RadixTree
import random
import string
import math

"""
    How to generate many strings that have meaningful prefix overlap?
    We can borrow ideas from LLM actually.
    The problem can be formualted as, given a sequence, produce a distribution for next character.
    Then if you keep sampling from the distribution following the probability, you will get a sequence that has meaningful prefix overlap.
    The sequence doesn't have to be meaningful, we just need to fingerprint an input sequence, and convert it into a distribution.
"""

SCALE = 2
REPEAT = 5

def gen_str(length=10):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def softmax(x):
    exp_x = [math.exp(x) for x in x]
    sum_exp = sum(exp_x)
    return [x / sum_exp for x in exp_x]

def gen_str_smart(length=10):
    s = ''
    for i in range(length):
        rand_state = random.getstate()

        h = hash(s)
        probs = softmax([x * SCALE for x in range(26)])
        random.seed(h)
        random.shuffle(probs)

        random.setstate(rand_state)
        c = random.choices(string.ascii_lowercase, probs, k=1)[0]
        s += c
    return s

print(softmax([x * SCALE for x in range(26)]))

tree = RadixTree()
words = []
for i in range(1000):
    s = gen_str_smart(100)
    words.append(s)
    tree.insert(s)

words.sort()
for w in words:
    print(w)

print(tree.root.stats())