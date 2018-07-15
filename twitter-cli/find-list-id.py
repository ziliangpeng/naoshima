from auth import api
import sys

def find_list_id(list_name):
    for l in api.lists_all():
        if list_name in l.name:
            return l.id

list_name = sys.argv[1]
print(find_list_id(list_name))

