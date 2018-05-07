#!/bin/bash

protoc -I../proto --go_out=plugins=grpc:ig ig.proto
