#!/bin/bash

# need to `pip install grpcio-tools`


python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ig.proto
