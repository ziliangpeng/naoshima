import requests


def load_names():
    karpathy_names_url = (
        "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    )
    names = requests.get(karpathy_names_url).text.split("\n")
    return names


if __name__ == "__main__":
    names = load_names()
    print(names[:10])
    print(len(names))
