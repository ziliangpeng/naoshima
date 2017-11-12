import secret_reader
from lib.instabot import InstaBot


def auth(log_mod=0):
    login, password = secret_reader.load_secrets()
    # use instabot
    return InstaBot(login=login, password=password, log_mod=log_mod)
