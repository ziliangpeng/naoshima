import tweepy
import json


def load_secrets():
    with open('secret.local', 'r') as f:
        d = json.loads(f.read())
        return d["consumer_key"], d["consumer_secret"], d["access_token"], d["access_token_secret"]


def auth():
    consumer_key, consumer_secret, access_token, access_token_secret = load_secrets()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    return tweepy.API(auth)


api = auth()
