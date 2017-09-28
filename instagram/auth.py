import secret_reader
import sys
import os
sys.path.append(os.path.join(sys.path[0], 'instabot.py/src'))


def auth():
    login, password = secret_reader.load_secrets()
    # use instabot
    from instabot import InstaBot
    return InstaBot(login=login, password=password, log_mod=1)
