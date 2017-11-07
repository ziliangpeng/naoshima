import secret_reader
import sys
import os
sys.path.append(os.path.join(sys.path[0], 'instabot_libs/src'))
from instabot import InstaBot


def auth(log_mod=0):
    login, password = secret_reader.load_secrets()
    # use instabot
    return InstaBot(login=login, password=password, log_mod=log_mod)
