import string
import time
from util import load_cookies, follow_user

cookies = load_cookies()
for user in map(string.strip, open('users.txt').readlines()):
    print user
    follow_user(user, cookies)
    time.sleep(10)

