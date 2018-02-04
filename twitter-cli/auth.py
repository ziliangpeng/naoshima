import tweepy
import json


def load_secrets():
    with open('secret.local', 'r') as f:
        d = json.loads(f.read())
        return d["consumer_key"], d["consumer_secret"], d["access_token"], d["access_token_secret"]


def save_secrets(j):
    with open('secret.local', 'w') as f:
        context = json.dumps(j, indent=4)
        print(context)
        f.write(context)


def auth():
    consumer_key, consumer_secret, access_token, access_token_secret = load_secrets()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    return tweepy.API(auth)


def reauth():
    import webbrowser
    ck, cs, at, ats = load_secrets()
    auth = tweepy.OAuthHandler(ck, cs)
    url = auth.get_authorization_url(access_type='Write')
    print(url)
    webbrowser.open(url)
    pin = input('PIN: ').strip()
    access_token = auth.get_access_token(pin)
    print(access_token)
    j = {
        "consumer_key": ck,
        "consumer_secret": cs,
        "access_token": access_token[0],
        "access_token_secret": access_token[1]
    }
    save_secrets(j)

api = auth()


if __name__ == '__main__':
    reauth()
