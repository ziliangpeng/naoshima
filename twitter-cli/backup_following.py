import tweepy
import time
from auth import api

u = api.get_user(screen_name='recklessdesuka')
my_id = u.id

read_list = []
for l in api.lists_all():
    print(l.id, l.name)
    n = l.name
    if 'backup' in n and '2018' in n:
        read_list.append(l.id)
        print('read list id is', read_list)

write_list = None
for l in api.lists_all():
    print(l.id, l.name)
    n = l.name
    if 'backup' in n and '2018' in n and '04-14' in n:
        write_list = l.id
        print('write list id is', write_list)

# list_id = '977990964890411013'


def fo():
    in_list_users = []
    for list_id in read_list:
        in_list_users += list(tweepy.Cursor(api.list_members, list_id=list_id, count=5000).items())
    in_list_ids = list(map(lambda x:x.id, in_list_users))
    print(in_list_ids)
    print(len(in_list_users))

    ids = []
    for i, id in enumerate(list(tweepy.Cursor(api.friends_ids, count=5000).items())):
        if id in in_list_ids:
            print(id, 'already in list')
            continue
        # if not (i >= 0 and i < 1000):
        #     continue
        # if i < 2000:
            # continue
        #u = api.get_user(i)
        #print(i, u.name)
        print('adding', id)
        # ids.append(i)
        api.add_list_member(user_id=id, list_id=write_list)
        time.sleep(1)
        # if len(ids) > 64:
        #    print('adding ids', ids)
        #    print(api.add_list_members(user_id=ids, list_id=list_id))
        #    ids = []
        #    time.sleep(30)
    # print(api.add_list_members(user_id=ids, list_id=list_id))


def main():
    while True:
        try:
            fo()
        except BaseException as e:
            print(e)
            print("sleeping for 1 day")
            time.sleep(3600 * 24) # sleep for a day


if __name__ == '__main__':
    main()
