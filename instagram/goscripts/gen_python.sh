#!/bin/bash

# need to `pip install grpcio-tools`


python -m grpc_tools.protoc -I. --python_out=ig --grpc_python_out=ig ig.proto
