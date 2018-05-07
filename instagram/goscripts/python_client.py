import ig_pb2_grpc
import ig_pb2
import grpc

import sys


host = 'localhost'
port = 8888
channel = grpc.insecure_channel('%s:%d' % (host, port))
stub = ig_pb2_grpc.InstagramStub(channel)

get_user_request = ig_pb2.GetUserRequest()
get_user_request.username = sys.argv[1]
user = stub.GetUser(get_user_request)
print(user)
