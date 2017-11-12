data_filename = 'followed.txt'
followed = set()

with open(data_filename, 'r+') as fr:
    for id in fr.readlines():
        followed.add(id.strip())

f = open(data_filename, 'a+')


def follow(id):
    followed.add(id)
    f.write(str(id) + '\n')
    f.flush()


def is_followed(id):
    return id in followed
