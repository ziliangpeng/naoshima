import auth
import time
import requests


jpg_content = open('blue.jpg', 'rb')

"""
url = 'http://httpbin.org/post'
files = {
    'photo': ('airbnb.jpg', 'jpg-content', 'image/jpeg'),
}
data = {
    'upload_id': 1511240426024,
    'media_type': 1
}
req = requests.Request('POST', url, files=files, data=data)
prepared = req.prepare()
print(prepared.body)
"""

bot = auth.auth()
upload_photo_url = 'https://www.instagram.com/create/upload/photo/'
configure_photo_url = 'https://www.instagram.com/create/configure/'
epoch = int(time.time() * 1000)
files = {
    'upload_id': (None, str(epoch), None),
    'photo': ('photo.jpg', jpg_content, 'application/octet-stream'),  # 'image/jpeg'),
    'media_type': (None, '1', None)
}
# data = {
#     'upload_id': epoch,
#     'media_type': 1
# }

print('starting to upload photo')
print('epoch', epoch)
# print('sleeping for epoch')
# time.sleep(10)

req = requests.Request('POST', upload_photo_url, files=files)  # , data=data)
prepared = req.prepare()
print('request headers', prepared.headers)
# print('request body', prepared.body)
bot.s.headers['referer'] = 'https://www.instagram.com/create/style/'
print('session headers', bot.s.headers)
r = bot.s.post(upload_photo_url, files=files)  # , data=data)
print('status code', r.status_code)
print('headers', r.headers)
print('body', r.text)

# print('sleeping after upload...')
# time.sleep(10)

print('starting to configure photo')
# tag = 'airbnb'
# payload = 'upload_id=%d&caption=123' % (epoch)
payload = {'upload_id': epoch}  # , 'caption': 'test'}
print('payload', payload)
# r = bot.s.post(configure_photo_url, data=payload)
# req = requests.Request('POST', configure_photo_url, data=payload)
# prepared = req.prepare()
# print('request headers', prepared.headers)
# print('request body', prepared.body)
# bot.s.headers['referer'] = 'https://www.instagram.com/create/details/'
print('session headers', bot.s.headers)

# payload = 'upload_id=%d&caption=test' % (epoch)
r = bot.s.post(configure_photo_url, data=payload)
print('status code', r.status_code)
print('headers', r.headers)
print('body', r.text)
