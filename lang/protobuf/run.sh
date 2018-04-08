#!/bin/bash

echo "Generating protobuf for go"
mkdir -p geo
protoc --go_out=geo geo.proto

echo "Running go program"
go run main.go
