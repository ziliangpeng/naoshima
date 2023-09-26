import requests


def load_names():
    karpathy_names_url = (
        "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    )
    names = requests.get(karpathy_names_url).text.split("\n")
    return names


def load_tokenized_names(names, pair_len=2):
    tokenized_names = []
    for name in names:
        tokens = []
        for i in range(0, len(name) - pair_len + 1):
            tokens.append(name[i : i + pair_len])
        tokenized_names.append(tokens)
    return tokenized_names


if __name__ == "__main__":
    names = load_names()
    print(names[:10])
    print(len(names))
    print(load_tokenized_names(names[:2]))
