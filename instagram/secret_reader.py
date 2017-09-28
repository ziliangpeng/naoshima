import json


def load_secrets():
    with open('secret.local', 'r') as f:
        secret_data = json.loads(f.read())
        return secret_data["login"], secret_data["password"]


def load_user_id():
    with open('secret.local', 'r') as f:
        secret_data = json.loads(f.read())
        return secret_data["id"]


def load_whitelist():
    with open('secret.local', 'r') as f:
        secret_data = json.loads(f.read())
        return secret_data["whitelist"]
