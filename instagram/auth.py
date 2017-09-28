import json
import sys
import os
sys.path.append(os.path.join(sys.path[0], 'instabot.py/src'))


def load_secrets():
    with open('secret.local', 'r') as f:
        secret_data = json.loads(f.read())
        return secret_data["login"], secret_data["password"]


def auth():
    login, password = load_secrets()
    # use instabot
    from instabot import InstaBot
    return InstaBot(login=login, password=password, log_mod=1)
