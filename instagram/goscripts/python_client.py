import ig_pb2_grpc
import ig_pb2
import grpc


host = 'localhost'
port = 8888
channel = grpc.insecure_channel('%s:%d' % (host, port))
stub = ig_pb2_grpc.InstagramStub(channel)

user = stub.GetUser(ig_pb2.GetUserRequest(username="aaa"))
print(user)
