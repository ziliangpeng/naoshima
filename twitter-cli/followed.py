FOLLOWED_FILENAME = 'followed;local'


followed = set()

with open(FOLLOWED_FILENAME, 'r+') as f:
    for id in f.readlines():
        followed.add(id.strip())


def follow(id):
    id = str(id)
    with open(FOLLOWED_FILENAME, 'a+') as f:
        f.write(id + '\n')
        followed.add(id)


def is_followed(id):
    id = str(id)
    return id in followed
