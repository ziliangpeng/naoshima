from mylib import user_config_reader, data
import requests
import sys
from requests.cookies import RequestsCookieJar
from lib.instabot import InstaBot
from http import cookiejar


def auth(log_mod=0, session=None):
    login, password = user_config_reader.load_secrets()

    if session:
        print('auth: using session in param to create bot')
        bot = InstaBot(login=login, password=password, session=session, log_mod=log_mod)
    else:
        saved_session = data.get_session(login)
        if saved_session:
            print('auth: using session in redis to create bot')
            bot = InstaBot(login=login, password=password, session=saved_session, log_mod=log_mod)
        else:
            print('auth: using user/password to create bot')
            bot = InstaBot(login=login, password=password, log_mod=log_mod)

    if bot.login_status:
        print('auth: done. saving session into redis')
        data.save_session(login, bot.s)
    else:
        print('auth: loging failed')

    return bot


def auth_from_cookies_file(filename, log_mod=0):
    cookies = cookiejar.MozillaCookieJar(filename)
    cookies.load()

    requests_cookies = RequestsCookieJar()
    for c in cookies:
        requests_cookies.set_cookie(c)
    session = requests.Session()
    session.cookies = requests_cookies
    return auth(log_mod=log_mod, session=session)


if __name__ == '__main__':
    # upload cookies from file to redis
    filename = sys.argv[1]
    auth_from_cookies_file(filename)
