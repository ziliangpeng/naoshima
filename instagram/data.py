
data_filename = 'followed.txt'
f = open(data_filename, 'a+')


def followed(id):
    f.write(str(id) + '\n')
    f.flush()
